"""ForestFormer3D evaluation: run inference on test scenes and compute metrics.

Usage:
    # Evaluate pretrained model
    python tools/test.py \
        --checkpoint /path/to/epoch_3000_fix.pth \
        --data-root /path/to/ForAINetV2

    # Evaluate our trained model
    python tools/test.py \
        --checkpoint work_dirs/train_v1/epoch_3000.pth \
        --data-root /path/to/ForAINetV2 \
        --native-ckpt
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ff3d.model import ForestFormer3D
from ff3d.model.forestformer3d import load_pretrained
from ff3d.eval import evaluate_scene, aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True, help='ForAINetV2 data root')
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--output-dir', default=None, help='Save PLY results (optional)')
    parser.add_argument('--invariant-feat', action='store_true',
                        help='Use translation-invariant input feature (height above ground, xy=0) — must match training')
    parser.add_argument('--native-ckpt', action='store_true',
                        help='Checkpoint is from our training (not original mm* format)')
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model = ForestFormer3D(decoder_cfg=dict(
        num_layers=6, num_semantic_queries=3, in_channels=32,
        d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0,
        activation='gelu', fix_attention=True, objectness_flag=True, attn_mask=True))

    if args.native_ckpt:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    else:
        load_pretrained(model, args.checkpoint)

    if args.invariant_feat:
        model._invariant_feat = True

    model = model.cuda().eval()
    print(f"Model loaded from {args.checkpoint}"
          + ("  [invariant-feat]" if args.invariant_feat else "")
          + ("  [coarse-attn]" if os.environ.get('FF_COARSE_ATTN', '0') not in ('0', 'false', 'False') else ""))

    # Find test scenes
    split_file = os.path.join(args.data_root, 'meta_data', f'{args.split}_list.txt')
    with open(split_file) as f:
        scene_names = [l.strip() for l in f if l.strip()]
    print(f"Evaluating {len(scene_names)} {args.split} scenes\n")

    # Run inference + evaluation
    scene_results = []
    total_t0 = time.time()

    for i, name in enumerate(scene_names):
        pts = np.fromfile(
            os.path.join(args.data_root, 'points', f'{name}.bin'),
            dtype=np.float32).reshape(-1, 3)
        sem_gt = np.fromfile(
            os.path.join(args.data_root, 'semantic_mask', f'{name}.bin'),
            dtype=np.int64)
        ins_gt = np.fromfile(
            os.path.join(args.data_root, 'instance_mask', f'{name}.bin'),
            dtype=np.int64)

        points = torch.from_numpy(pts).float().cuda()
        out_path = os.path.join(args.output_dir, f'{name}.ply') if args.output_dir else None

        t0 = time.time()
        result = model.predict(points, output_path=out_path)
        dt = time.time() - t0

        # Evaluate
        r = evaluate_scene(result['semantic_pred'], result['instance_pred'], sem_gt, ins_gt)
        scene_results.append(r)

        n_trees = len(np.unique(result['instance_pred'][result['instance_pred'] >= 0]))
        print(f"  [{i+1}/{len(scene_names)}] {name}: {pts.shape[0]:>10,d} pts, "
              f"{dt:.1f}s, {n_trees} trees")

    total_time = time.time() - total_t0

    # Aggregate and print
    metrics = aggregate_metrics(scene_results)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({len(scene_names)} {args.split} scenes, {total_time:.1f}s)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<15s}: {v:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
