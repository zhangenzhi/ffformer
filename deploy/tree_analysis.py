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


_PLY_NP = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
           'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
           'short': '<i2', 'ushort': '<u2', 'char': 'i1', 'uchar': 'u1',
           'int8': 'i1', 'uint8': 'u1', 'uint16': '<u2'}


def _read_result_ply(path):
    """Read result.ply -> (xyz, semantic, instance, score). Supports both
    ASCII and binary_little_endian PLY."""
    fields, types = [], []
    fmt = 'ascii'
    header_bytes = 0
    with open(path, 'rb') as f:
        for raw in f:
            header_bytes += len(raw)
            line = raw.decode('ascii', 'ignore').strip()
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('property'):
                parts = line.split()
                types.append(parts[1])
                fields.append(parts[-1])
            elif line == 'end_header':
                break
    col = {name: idx for idx, name in enumerate(fields)}

    if fmt != 'ascii':
        rec_dtype = np.dtype([(fields[i], _PLY_NP.get(types[i], '<f4'))
                              for i in range(len(fields))])
        rec = np.fromfile(path, dtype=rec_dtype, offset=header_bytes)
        xyz = np.column_stack([rec['x'], rec['y'], rec['z']]).astype(np.float64)
        return (xyz, rec['semantic_pred'].astype(np.int32),
                rec['instance_pred'].astype(np.int32),
                rec['score'].astype(np.float32))

    with open(path, 'r') as f:
        for line in f:
            if line.strip() == 'end_header':
                break
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
    valid = inst >= 0
    if valid.any():
        # A single sort groups every tree's points contiguously, then each per-tree
        # statistic is one vectorized segment-reduction (reduceat) — O(N log N)
        # instead of an O(n_trees x N) boolean-mask scan per tree (which was reading
        # all ~150M points ~5000 times). Output is identical.
        inst_v = inst[valid]
        xv, yv, zv = xyz[valid, 0], xyz[valid, 1], xyz[valid, 2]
        sem_v = sem[valid]
        score_v = score[valid].astype(np.float64)

        order = np.argsort(inst_v, kind='stable')
        inst_s = inst_v[order]
        xs, ys, zs = xv[order], yv[order], zv[order]
        sem_s, score_s = sem_v[order], score_v[order]

        uniq, starts, counts = np.unique(inst_s, return_index=True, return_counts=True)
        zmin = np.minimum.reduceat(zs, starts); zmax = np.maximum.reduceat(zs, starts)
        xmin = np.minimum.reduceat(xs, starts); xmax = np.maximum.reduceat(xs, starts)
        ymin = np.minimum.reduceat(ys, starts); ymax = np.maximum.reduceat(ys, starts)
        xsum = np.add.reduceat(xs, starts); ysum = np.add.reduceat(ys, starts)
        ssum = np.add.reduceat(score_s, starts)
        wood = np.add.reduceat((sem_s == WOOD).astype(np.int64), starts)
        leaf = np.add.reduceat((sem_s == LEAF).astype(np.int64), starts)

        for i in range(len(uniq)):
            n = int(counts[i])
            if n < 10:
                continue
            z_base, z_top = float(zmin[i]), float(zmax[i])
            # Height = the tree's own vertical span. Using a single GLOBAL ground
            # median gave negative / tiny heights on sloped terrain (373 m relief),
            # so use each tree's base as its local ground reference.
            height = z_top - z_base
            crown_width = float(max(xmax[i] - xmin[i], ymax[i] - ymin[i]))
            wood_n, leaf_n = int(wood[i]), int(leaf[i])
            # leaf/wood ratio blows up (hundreds–thousands) when the stem is barely
            # segmented (few wood points) — require enough wood points and a plausible
            # ratio, else report unknown rather than noise.
            lw = None
            if wood_n >= 20 and leaf_n > 0:
                _r = leaf_n / wood_n
                lw = round(_r, 2) if _r <= 300 else None
            trees.append({
                'id': int(uniq[i]),
                'n_points': n,
                'height_m': round(height, 2),               # vertical extent ≈ tree height
                'crown_width_m': round(crown_width, 2),
                'crown_ratio': round(crown_width / height, 2) if height > 0.1 else None,
                'wood_points': wood_n,
                'leaf_points': leaf_n,
                'leaf_wood_ratio': lw,
                'mean_score': round(float(ssum[i] / n), 3),
                'center_x': round(float(xsum[i] / n), 2),
                'center_y': round(float(ysum[i] / n), 2),
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


# Semantic point colors (match the 3D viewer): ground / wood / leaf.
_SEM_RGB = np.array([[139, 119, 101], [210, 150, 60], [34, 180, 34]], dtype=np.uint8)


def build_tree_point_store(ply_path, bin_path, index_path):
    """Write every instance-assigned point, grouped by tree, as a 16 B/point blob
    (xyz float32 + rgb uint8 + 1 pad — same layout the viewer's tile parser reads)
    plus an index {tree_id: [byte_offset, n_points]}. One-time build per result;
    afterwards a single tree's FULL points are served by byte-range. Returns index."""
    if os.path.isfile(bin_path) and os.path.isfile(index_path):
        with open(index_path) as f:
            return json.load(f)
    xyz, sem, inst, _score = _read_result_ply(ply_path)
    valid = inst >= 0
    inst_v = inst[valid]
    xyz_v = np.ascontiguousarray(xyz[valid], dtype=np.float32)
    sem_v = np.clip(sem[valid], 0, 2)
    order = np.argsort(inst_v, kind='stable')
    inst_s = inst_v[order]
    xyz_s = np.ascontiguousarray(xyz_v[order])
    rgb_s = _SEM_RGB[sem_v[order]]
    n = len(inst_s)
    rec = np.zeros((n, 16), dtype=np.uint8)
    rec[:, 0:12] = xyz_s.view(np.uint8).reshape(n, 12)
    rec[:, 12:15] = rgb_s
    rec.tofile(bin_path)
    uniq, starts, counts = np.unique(inst_s, return_index=True, return_counts=True)
    index = {str(int(uniq[i])): [int(starts[i]) * 16, int(counts[i])]
             for i in range(len(uniq))}
    with open(index_path, 'w') as f:
        json.dump(index, f)
    return index


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


def build_tree_prompt(metrics, tree_id, lang='en'):
    """Return (system, user) for a single-tree assessment. Shared by the Ollama
    path (server) and the HPC-GPU path (hpc_llm.py) so both produce the same text."""
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
    return system, user


def analyze_tree(metrics, tree_id, lang='en'):
    """LLM health assessment for one tree (via Ollama)."""
    system, user = build_tree_prompt(metrics, tree_id, lang)
    return _ollama_chat(system, user)


def build_scene_prompt(metrics, stats=None, lang='en'):
    """Return (system, user) for a stand-level assessment, or (None, text) when
    there are no trees. Shared by the Ollama and HPC-GPU paths."""
    trees = metrics['trees']
    if not trees:
        return None, "No tree instances detected; nothing to analyze."
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
    return system, user


def analyze_scene(metrics, stats=None, lang='en'):
    """LLM summary for the whole scene (via Ollama)."""
    system, user = build_scene_prompt(metrics, stats, lang)
    if system is None:
        return user   # no-trees message
    return _ollama_chat(system, user)


def ollama_available():
    try:
        with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
