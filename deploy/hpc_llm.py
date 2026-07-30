#!/usr/bin/env python3
"""Run the forestry LLM analysis on an HPC GPU (Qwen2.5-7B via transformers).

Used two ways, both reusing the exact prompts from tree_analysis:
  * scene  — precomputed at segmentation time (A), cached to <task_dir>/scene_analysis.json
  * tree   — on-demand per-tree PBS job (B), written to <task_dir>/tree_<id>_<lang>.json

The model + libs live on the shared filesystem (downloaded once on the login node);
compute nodes run fully offline.

Usage:
  python hpc_llm.py --task-dir DIR --mode scene         [--lang en|zh] [--out FILE]
  python hpc_llm.py --task-dir DIR --mode tree --tree-id N [--lang en|zh] [--out FILE]
"""
import argparse
import json
import os
import sys
import time

MODEL_DIR = os.environ.get('HPC_LLM_MODEL', '/lustre1/work/c30636/models/qwen2.5-7b-instruct')

# tree_analysis lives one level up (deploy/); import its metrics + prompt builders.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.tree_analysis import (compute_tree_metrics, build_scene_prompt,
                                  build_tree_prompt)

_MODEL = _TOK = None


def _load_model():
    global _MODEL, _TOK
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    _TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
    # bfloat16 (not fp16) — fp16 logits can go inf/nan and break sampling on Qwen.
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map='cuda')
    print(f'[hpc_llm] model loaded in {time.time()-t0:.1f}s', flush=True)


def _generate(system, user, max_new_tokens=400):
    import torch
    _load_model()
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': user}]
    text = _TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(text, return_tensors='pt').to(_MODEL.device)
    t0 = time.time()
    with torch.no_grad():
        out = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=True, temperature=0.3, top_p=0.9,
                              pad_token_id=_TOK.eos_token_id)
    resp = _TOK.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f'[hpc_llm] generated {len(resp)} chars in {time.time()-t0:.1f}s', flush=True)
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-dir', required=True)
    ap.add_argument('--mode', required=True, choices=['scene', 'tree'])
    ap.add_argument('--tree-id', type=int, default=None)
    ap.add_argument('--lang', default='en')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ply = os.path.join(args.task_dir, 'result.ply')
    if not os.path.isfile(ply):
        print(f'ERROR: no result.ply in {args.task_dir}', file=sys.stderr)
        sys.exit(2)

    metrics = compute_tree_metrics(ply, cache_path=os.path.join(args.task_dir, 'trees.json'))

    if args.mode == 'scene':
        system, user = build_scene_prompt(metrics, lang=args.lang)
        analysis = user if system is None else _generate(system, user, max_new_tokens=500)
        out = args.out or os.path.join(args.task_dir, f'scene_analysis_{args.lang}.json')
        payload = {'task_id': os.path.basename(args.task_dir.rstrip('/')),
                   'analysis': analysis, 'n_trees': metrics['n_trees'], 'lang': args.lang}
    else:
        if args.tree_id is None:
            print('ERROR: --tree-id required for mode=tree', file=sys.stderr)
            sys.exit(2)
        system, user = build_tree_prompt(metrics, args.tree_id, lang=args.lang)
        analysis = _generate(system, user, max_new_tokens=400)
        out = args.out or os.path.join(args.task_dir, f'tree_{args.tree_id}_{args.lang}.json')
        payload = {'task_id': os.path.basename(args.task_dir.rstrip('/')),
                   'tree_id': args.tree_id, 'analysis': analysis, 'lang': args.lang}

    with open(out, 'w') as f:
        json.dump(payload, f)
    print(f'[hpc_llm] wrote {out}', flush=True)


if __name__ == '__main__':
    main()
