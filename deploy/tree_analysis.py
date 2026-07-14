"""Per-tree metrics + LLM health analysis for ForestFormer3D results.

Two layers:
  1. compute_tree_metrics(): parse result.ply -> per-instance geometry/structure
     stats (height, crown, wood/leaf ratio, ...). Pure numpy, cached to trees.json.
  2. LLM analysis via the in-cluster Ollama service (qwen2.5:7b-instruct):
     turn those metrics into a plain-language health assessment.

Semantic classes (configs/jpeaks_test.py): 0=ground, 1=wood, 2=leaf.
"""
import json
import os
import urllib.request

import numpy as np

GROUND, WOOD, LEAF = 0, 1, 2

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:7b-instruct')


def _read_result_ply(path):
    """Read result.ply -> (xyz, semantic, instance, score) arrays."""
    fields = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break
        col = {name: idx for idx, name in enumerate(fields)}
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, [col['x'], col['y'], col['z']]].astype(np.float64)
    sem = data[:, col['semantic_pred']].astype(np.int32)
    inst = data[:, col['instance_pred']].astype(np.int32)
    score = data[:, col['score']].astype(np.float32)
    return xyz, sem, inst, score


def compute_tree_metrics(ply_path, cache_path=None):
    """Compute per-tree metrics from a result PLY.

    Returns dict: {n_trees, ground_z, trees: [ {id, n_points, height, crown_width,
    crown_area_ratio, wood_points, leaf_points, leaf_wood_ratio, mean_score,
    base_z, top_z, center_x, center_y} ... ]}.
    """
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    xyz, sem, inst, score = _read_result_ply(ply_path)

    # Ground reference: median Z of ground-classified points (fallback: global min)
    ground_mask = sem == GROUND
    ground_z = float(np.median(xyz[ground_mask, 2])) if ground_mask.any() else float(xyz[:, 2].min())

    trees = []
    tree_ids = np.unique(inst[inst >= 0])
    for tid in tree_ids:
        m = inst == tid
        n = int(m.sum())
        if n < 10:
            continue
        pts = xyz[m]
        s = sem[m]
        z_base, z_top = float(pts[:, 2].min()), float(pts[:, 2].max())
        height = z_top - z_base
        dx = float(pts[:, 0].max() - pts[:, 0].min())
        dy = float(pts[:, 1].max() - pts[:, 1].min())
        crown_width = max(dx, dy)
        wood_n = int((s == WOOD).sum())
        leaf_n = int((s == LEAF).sum())
        trees.append({
            'id': int(tid),
            'n_points': n,
            'height_m': round(z_top - ground_z, 2),   # height above ground
            'trunk_height_m': round(height, 2),         # extent of the instance
            'crown_width_m': round(crown_width, 2),
            'crown_ratio': round(crown_width / height, 2) if height > 0.1 else None,
            'wood_points': wood_n,
            'leaf_points': leaf_n,
            'leaf_wood_ratio': round(leaf_n / wood_n, 2) if wood_n > 0 else None,
            'mean_score': round(float(score[m].mean()), 3),
            'center_x': round(float(pts[:, 0].mean()), 2),
            'center_y': round(float(pts[:, 1].mean()), 2),
            'base_z': round(z_base, 2),
        })

    # Sort tallest first
    trees.sort(key=lambda t: t['height_m'], reverse=True)

    result = {
        'n_trees': len(trees),
        'ground_z': round(ground_z, 2),
        'trees': trees,
    }
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(result, f)
    return result


def _ollama_chat(system, user, temperature=0.3, timeout=300):
    """Call the in-cluster Ollama chat API; return the assistant text."""
    payload = {
        'model': OLLAMA_MODEL,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'stream': False,
        'options': {'temperature': temperature},
    }
    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/chat',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data['message']['content']


def analyze_tree(metrics, tree_id, lang='en'):
    """LLM health assessment for one tree."""
    tree = next((t for t in metrics['trees'] if t['id'] == tree_id), None)
    if tree is None:
        raise ValueError(f'tree {tree_id} not found')

    system = (
        "You are a forestry expert analyzing individual trees from airborne "
        "LiDAR point-cloud segmentation. You receive geometric and structural "
        "metrics for one tree and assess its likely health and structure. "
        "Be concrete, note uncertainty, and ground every claim in the given "
        "metrics. Point clouds cannot show disease directly, so frame health "
        "as structural indicators (crown density, leaf/wood balance, form). "
        + ("Reply in Simplified Chinese." if lang == 'zh' else "Answer in English.")
    )
    user = (
        "Per-tree segmentation metrics (metres):\n"
        f"{json.dumps(tree, indent=2)}\n\n"
        "Field notes: wood_points = woody (trunk/branch) points, "
        "leaf_points = foliage points, leaf_wood_ratio = foliage/wood, "
        "crown_ratio = crown width / height, mean_score = segmentation "
        "confidence.\n\n"
        "Give: 1) a structural summary (height, crown form); 2) a structural "
        "inference of health (from leaf/wood ratio, canopy density, and "
        "confidence); 3) points that warrant field verification. "
        "Keep it under 120 words."
    )
    return _ollama_chat(system, user)


def analyze_scene(metrics, stats=None, lang='en'):
    """LLM summary for the whole scene."""
    trees = metrics['trees']
    if not trees:
        return "No tree instances detected; nothing to analyze."
    heights = [t['height_m'] for t in trees]
    lw = [t['leaf_wood_ratio'] for t in trees if t['leaf_wood_ratio'] is not None]
    summary = {
        'n_trees': metrics['n_trees'],
        'height_min': round(min(heights), 1),
        'height_max': round(max(heights), 1),
        'height_mean': round(float(np.mean(heights)), 1),
        'leaf_wood_ratio_mean': round(float(np.mean(lw)), 2) if lw else None,
        'tallest_trees': [
            {'id': t['id'], 'height_m': t['height_m'],
             'crown_width_m': t['crown_width_m']} for t in trees[:5]
        ],
    }
    system = (
        "You are a forestry expert summarizing a forest plot analyzed from "
        "airborne LiDAR. Give a concise stand-level assessment: size structure, "
        "canopy characteristics, and which individual trees warrant closer "
        "inspection. Ground claims in the metrics. "
        + ("Reply in Simplified Chinese." if lang == 'zh' else "Answer in English.")
    )
    user = (
        "Stand-level segmentation statistics:\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Give: 1) overall stand structure (height distribution, density); "
        "2) a general impression of canopy and health (from the mean leaf/wood "
        "ratio); 3) individual trees to prioritise for field verification "
        "(by structural anomaly). Keep it under 160 words."
    )
    return _ollama_chat(system, user)


def ollama_available():
    try:
        with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
