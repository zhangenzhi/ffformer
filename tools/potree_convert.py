#!/usr/bin/env python3
"""Convert ForestFormer3D PLY to octree tiles for streaming web visualization.

Builds a multi-resolution octree from the PLY file. Each node stores a fixed
budget of points; deeper nodes add progressive detail. The viewer loads
coarse nodes first and refines visible regions on demand.

Handles 100M+ point clouds with chunked I/O and numpy vectorization.

Usage:
    python tools/potree_convert.py work_dirs/jpeaks_area1/jpeaks_area1_test.ply
    python tools/potree_convert.py scene.ply -o viewer_output/ --budget 100000
    # Then: cd viewer_output && python -m http.server 8080
"""
import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

INSTANCE_PALETTE = np.array([
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
], dtype=np.uint8)

SEMANTIC_COLORS = np.array([
    [139, 119, 101],  # ground
    [210, 150, 60],   # wood
    [34, 180, 34],    # leaf
], dtype=np.uint8)

UNASSIGNED_COLOR = np.array([80, 80, 80], dtype=np.uint8)

# 16 bytes per point: xyz(3×f32=12) + rgb(3×u8=3) + flag(u8=1)
POINT_SIZE = 16


def parse_ply_header(filepath):
    """Parse PLY header, return field names, vertex count, header byte offset."""
    fields = []
    num_vertices = 0
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('element vertex'):
                num_vertices = int(stripped.split()[-1])
            elif stripped.startswith('property'):
                fields.append(stripped.split()[-1])
            elif stripped == 'end_header':
                header_end = f.tell()
                break
    col = {name: idx for idx, name in enumerate(fields)}
    return col, num_vertices, header_end, len(fields)


def load_ply_chunked(filepath, col, num_vertices, header_end, num_fields,
                     chunk_size=5_000_000):
    """Generator: yield chunks of (xyz, colors, flags) from PLY."""
    with open(filepath, 'r') as f:
        # Skip header
        for line in f:
            if line.strip() == 'end_header':
                break

        buf = []
        count = 0
        for line in f:
            if count >= num_vertices:
                break
            buf.append(line)
            count += 1
            if len(buf) >= chunk_size:
                yield _process_chunk(buf, col)
                buf = []
        if buf:
            yield _process_chunk(buf, col)


def _process_chunk(lines, col):
    """Convert text lines to numpy arrays."""
    data = np.loadtxt(lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyz = data[:, [col['x'], col['y'], col['z']]].astype(np.float32)

    n = len(data)
    colors = np.tile(UNASSIGNED_COLOR, (n, 1))
    flags = np.zeros(n, dtype=np.uint8)

    if 'instance_pred' in col:
        inst = data[:, col['instance_pred']].astype(np.int32)
        valid = inst >= 0
        colors[valid] = INSTANCE_PALETTE[inst[valid] % len(INSTANCE_PALETTE)]
        flags[valid] = 255

    if 'semantic_pred' in col:
        sem = data[:, col['semantic_pred']].astype(np.int32)
        ground = (~(flags > 0)) & (sem == 0)
        colors[ground] = SEMANTIC_COLORS[0]
        flags[ground] = 128

    return xyz, colors, flags


def octree_key(point, bbox_min, bbox_size, depth):
    """Compute octree node key string for a point at given depth."""
    # Normalize to [0, 1)
    norm = (point - bbox_min) / bbox_size
    norm = np.clip(norm, 0.0, 0.999999)

    key = 'r'
    for d in range(depth):
        scale = 2 ** (d + 1)
        ix = int(norm[0] * scale) % 2
        iy = int(norm[1] * scale) % 2
        iz = int(norm[2] * scale) % 2
        child = ix + iy * 2 + iz * 4
        key += str(child)
    return key


def octree_keys_batch(xyz, bbox_min, bbox_size, depth):
    """Vectorized octree key computation for a batch of points."""
    norm = (xyz - bbox_min) / bbox_size
    norm = np.clip(norm, 0.0, 0.999999)

    # Build keys level by level
    keys = np.full(len(xyz), 0, dtype=np.int64)  # numeric key
    for d in range(depth):
        scale = 2 ** (d + 1)
        ix = (norm[:, 0] * scale).astype(np.int64) % 2
        iy = (norm[:, 1] * scale).astype(np.int64) % 2
        iz = (norm[:, 2] * scale).astype(np.int64) % 2
        child = ix + iy * 2 + iz * 4
        keys = keys * 8 + child

    return keys


def numeric_key_to_string(nkey, depth):
    """Convert numeric octree key back to string like 'r012'."""
    if depth == 0:
        return 'r'
    digits = []
    for _ in range(depth):
        digits.append(str(nkey % 8))
        nkey //= 8
    return 'r' + ''.join(reversed(digits))


def pack_points(xyz, colors, flags):
    """Pack points into binary: xyz(3×f32) + rgb(3×u8) + flag(u8) = 16 bytes."""
    n = len(xyz)
    buf = bytearray(n * POINT_SIZE)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, POINT_SIZE)
    arr[:, 0:12] = xyz.astype(np.float32).view(np.uint8).reshape(n, 12)
    arr[:, 12:15] = colors.astype(np.uint8)
    arr[:, 15] = flags.astype(np.uint8)
    return bytes(buf)


def build_octree(filepath, output_dir, node_budget=50_000, max_depth=8):
    """Build octree from PLY file.

    Strategy:
    - First pass: compute bounding box
    - Second pass: assign each point to deepest octree node
    - For each node, randomly subsample to node_budget points
    - Parent nodes aggregate subsampled children
    """
    col, num_vertices, header_end, num_fields = parse_ply_header(filepath)
    print(f'Total points: {num_vertices:,}')
    print(f'Node budget: {node_budget:,}, max depth: {max_depth}')

    # --- Pass 1: bounding box ---
    print('Pass 1: Computing bounding box ...', end=' ', flush=True)
    t0 = time.time()
    bbox_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for xyz, _, _ in load_ply_chunked(filepath, col, num_vertices,
                                       header_end, num_fields):
        bbox_min = np.minimum(bbox_min, xyz.min(axis=0))
        bbox_max = np.maximum(bbox_max, xyz.max(axis=0))

    # Make cubic bounding box
    bbox_size = (bbox_max - bbox_min).max()
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_min = bbox_center - bbox_size / 2
    bbox_max = bbox_center + bbox_size / 2
    # Small padding
    bbox_min -= bbox_size * 0.001
    bbox_size *= 1.002

    print(f'{time.time() - t0:.1f}s')
    print(f'  BBox: [{bbox_min[0]:.1f}, {bbox_min[1]:.1f}, {bbox_min[2]:.1f}] - '
          f'[{bbox_max[0]:.1f}, {bbox_max[1]:.1f}, {bbox_max[2]:.1f}], '
          f'size={bbox_size:.1f}')

    # Determine appropriate depth
    # At depth d, there are 8^d leaf nodes. We want ~node_budget points per node.
    ideal_depth = 0
    for d in range(1, max_depth + 1):
        pts_per_leaf = num_vertices / (8 ** d)
        if pts_per_leaf < node_budget:
            ideal_depth = d
            break
    else:
        ideal_depth = max_depth
    print(f'  Using depth: {ideal_depth}')

    # --- Pass 2: distribute points to leaf nodes ---
    print('Pass 2: Distributing points to octree nodes ...')
    t0 = time.time()

    # We'll store point indices per leaf node, then subsample
    # For memory efficiency with 129M points, we write leaf bins to disk
    tiles_dir = os.path.join(output_dir, 'tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    # Track nodes: key -> {count, file_handle}
    node_files = {}
    node_counts = defaultdict(int)
    total_processed = 0

    for xyz, colors, flags in load_ply_chunked(filepath, col, num_vertices,
                                                header_end, num_fields):
        n = len(xyz)
        # Compute leaf keys
        leaf_keys = octree_keys_batch(xyz, bbox_min, bbox_size, ideal_depth)

        # Group by key and write to per-node temp files
        unique_keys = np.unique(leaf_keys)
        for uk in unique_keys:
            mask = leaf_keys == uk
            key_str = numeric_key_to_string(int(uk), ideal_depth)

            # Pack and append to file
            packed = pack_points(xyz[mask], colors[mask], flags[mask])

            if key_str not in node_files:
                fpath = os.path.join(tiles_dir, f'_tmp_{key_str}.bin')
                node_files[key_str] = open(fpath, 'wb')
            node_files[key_str].write(packed)
            node_counts[key_str] += int(mask.sum())

        total_processed += n
        pct = total_processed * 100 // num_vertices
        print(f'\r  {pct}% ({total_processed:,}/{num_vertices:,}) '
              f'nodes={len(node_counts)}  ', end='', flush=True)

    # Close temp files
    for fh in node_files.values():
        fh.close()

    print(f'\n  {time.time() - t0:.1f}s, {len(node_counts)} leaf nodes')

    # --- Pass 3: subsample and build hierarchy ---
    print('Pass 3: Building hierarchy ...')
    t0 = time.time()

    hierarchy = {}  # key -> {count, children}

    # Process leaf nodes: subsample to budget
    for key_str, count in sorted(node_counts.items()):
        tmp_path = os.path.join(tiles_dir, f'_tmp_{key_str}.bin')
        data = np.fromfile(tmp_path, dtype=np.uint8).reshape(-1, POINT_SIZE)

        if len(data) > node_budget:
            idx = np.random.choice(len(data), node_budget, replace=False)
            sampled = data[idx]
        else:
            sampled = data

        # Write final tile
        out_path = os.path.join(tiles_dir, f'{key_str}.bin')
        sampled.tobytes()
        with open(out_path, 'wb') as f:
            f.write(sampled.tobytes())

        hierarchy[key_str] = {
            'count': int(len(sampled)),
            'total': count,
        }

        # Clean up temp
        os.remove(tmp_path)

    # Build parent nodes by subsampling from children
    for depth in range(ideal_depth - 1, -1, -1):
        parent_keys = set()
        for key in hierarchy:
            if len(key) - 1 == depth:  # 'r' is depth 0, 'r0' is depth 1
                parent_keys.add(key)
            elif len(key) - 1 > depth:
                parent_keys.add(key[:depth + 1])

        for pkey in sorted(parent_keys):
            if pkey in hierarchy:
                continue  # Already exists

            # Collect subsampled points from children
            child_data = []
            child_total = 0
            for c in range(8):
                ckey = pkey + str(c)
                cpath = os.path.join(tiles_dir, f'{ckey}.bin')
                if os.path.exists(cpath):
                    cdata = np.fromfile(cpath, dtype=np.uint8).reshape(-1, POINT_SIZE)
                    child_data.append(cdata)
                    child_total += hierarchy.get(ckey, {}).get('total', len(cdata))

            if not child_data:
                continue

            all_child = np.concatenate(child_data)
            if len(all_child) > node_budget:
                idx = np.random.choice(len(all_child), node_budget, replace=False)
                sampled = all_child[idx]
            else:
                sampled = all_child

            out_path = os.path.join(tiles_dir, f'{pkey}.bin')
            with open(out_path, 'wb') as f:
                f.write(sampled.tobytes())

            hierarchy[pkey] = {
                'count': int(len(sampled)),
                'total': child_total,
            }

    # Write metadata
    # Compute children list for each node
    for key in hierarchy:
        children = []
        for c in range(8):
            ckey = key + str(c)
            if ckey in hierarchy:
                children.append(c)
        hierarchy[key]['children'] = children

    metadata = {
        'version': '1.0',
        'total_points': num_vertices,
        'bbox_min': bbox_min.tolist(),
        'bbox_size': float(bbox_size),
        'depth': ideal_depth,
        'node_budget': node_budget,
        'point_size': POINT_SIZE,
        'nodes': {k: v for k, v in sorted(hierarchy.items())},
    }

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  {time.time() - t0:.1f}s')
    print(f'  Total nodes: {len(hierarchy)}')
    total_tile_bytes = sum(
        os.path.getsize(os.path.join(tiles_dir, f'{k}.bin'))
        for k in hierarchy
    )
    print(f'  Total tile data: {total_tile_bytes / 1e6:.0f} MB')

    return metadata


def generate_viewer_html(output_dir, metadata):
    """Generate self-contained streaming point cloud viewer."""

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ForestFormer3D — Streaming Point Cloud Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;font-family:'Segoe UI',Arial,sans-serif;overflow:hidden;height:100vh;color:#c9d1d9}
#hdr{height:32px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;
  background:linear-gradient(90deg,#161b22,#0d1117,#161b22);
  border-bottom:1px solid #30363d;font-size:12px}
#hdr b{color:#e6edf3;letter-spacing:0.3px;font-size:14px}
#hdr .info{color:#8b949e;font-size:11px}
#wrap{height:calc(100vh - 32px);position:relative}
canvas{display:block;width:100%!important;height:100%!important}
#cp{position:absolute;right:10px;top:10px;z-index:20;
  background:rgba(22,27,34,0.92);backdrop-filter:blur(6px);
  border:1px solid #30363d;border-radius:8px;
  padding:10px 12px;font-size:11px;width:160px}
#cp label{display:block;margin:6px 0 2px;color:#8b949e;font-size:9px;
  text-transform:uppercase;letter-spacing:0.6px}
#cp input[type=range]{width:100%;accent-color:#58a6ff}
#cp select{width:100%;background:#0d1117;color:#c9d1d9;
  border:1px solid #30363d;border-radius:4px;padding:2px 4px;font-size:11px}
#cp button{width:100%;margin-top:8px;padding:4px;background:#238636;
  color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600}
#cp button:hover{background:#2ea043}
#stats{position:absolute;left:10px;bottom:10px;background:rgba(22,27,34,0.88);
  border:1px solid #30363d;border-radius:6px;padding:6px 14px;font-size:11px;color:#8b949e}
#lg{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
  background:rgba(22,27,34,0.88);backdrop-filter:blur(4px);border:1px solid #30363d;
  padding:4px 14px;border-radius:5px;z-index:10;font-size:11px;color:#c9d1d9;
  white-space:nowrap;display:flex;gap:12px}
.li{display:inline-flex;align-items:center;gap:4px}
.ld{width:9px;height:9px;border-radius:50%;display:inline-block;
  border:1px solid rgba(255,255,255,0.15)}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#8b949e;font-size:16px;z-index:30}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="hdr">
  <b>ForestFormer3D</b>
  <span class="info" id="hdrInfo">Loading...</span>
</div>
<div id="wrap">
  <canvas id="c0"></canvas>
  <div id="loading">Loading point cloud...</div>
</div>
<div id="cp">
  <label>Point Size</label>
  <input type="range" id="psize" min="3" max="80" value="25">
  <label>Background</label>
  <select id="bgsel">
    <option value="#161b22">Dark</option>
    <option value="#ffffff">White</option>
    <option value="#000000">Black</option>
    <option value="#f0efe8">Cream</option>
  </select>
  <label>Unassigned</label>
  <select id="unsel">
    <option value="dim">Dim</option>
    <option value="hide">Hide</option>
    <option value="show">Show</option>
  </select>
  <label>Detail Level</label>
  <input type="range" id="lod" min="1" max="20" value="10">
  <button id="resetBtn">Reset Camera</button>
</div>
<div id="stats"></div>
<div id="lg">
  <span class="li"><span class="ld" style="background:rgb(139,119,101)"></span>ground</span>
  <span class="li"><span class="ld" style="background:rgb(210,150,60)"></span>wood</span>
  <span class="li"><span class="ld" style="background:rgb(34,180,34)"></span>leaf</span>
  <span class="li"><span class="ld" style="background:#505050"></span>none</span>
</div>
<script>
(async function() {

// Load metadata
const meta = await fetch('metadata.json').then(r => r.json());
const POINT_SIZE = meta.point_size; // 16 bytes
const bboxMin = meta.bbox_min;
const bboxSize = meta.bbox_size;
const totalPoints = meta.total_points;
const nodes = meta.nodes;

document.getElementById('hdrInfo').textContent =
    totalPoints.toLocaleString() + ' points (streaming octree)';

// Three.js setup
const canvas = document.getElementById('c0');
const wrap = document.getElementById('wrap');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x161b22);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 10000);
camera.up.set(0, 0, 1);

// Center of bounding box
const center = new THREE.Vector3(
    bboxMin[0] + bboxSize/2,
    bboxMin[1] + bboxSize/2,
    bboxMin[2] + bboxSize/2
);
const R = bboxSize / 2;

// Shader
const vertShader = `
attribute float flag;
uniform float uSize;
uniform float uDimAlpha;
varying vec3 vColor;
varying float vAlpha;
void main() {
    vColor = color;
    vAlpha = flag > 0.3 ? 1.0 : uDimAlpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize * (400.0 / (-mv.z));
    gl_PointSize = max(gl_PointSize, 1.0);
    gl_Position = projectionMatrix * mv;
}`;
const fragShader = `
varying vec3 vColor;
varying float vAlpha;
void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = dot(p, p);
    if (d > 0.25) discard;
    float edge = smoothstep(0.25, 0.15, d);
    float shade = 1.0 - d * 0.4;
    gl_FragColor = vec4(vColor * shade, vAlpha * edge);
}`;

// Node management
const loadedNodes = {};  // key -> THREE.Points
const loadingNodes = new Set();
const MAX_CONCURRENT = 6;
let loadQueue = [];
let visiblePoints = 0;
let lodMultiplier = 1.0;

function getNodeBBox(key) {
    let min = [bboxMin[0], bboxMin[1], bboxMin[2]];
    let size = bboxSize;
    for (let i = 1; i < key.length; i++) {
        size /= 2;
        const c = parseInt(key[i]);
        min[0] += (c & 1) ? size : 0;
        min[1] += (c & 2) ? size : 0;
        min[2] += (c & 4) ? size : 0;
    }
    return {min, size};
}

function nodeScreenSize(key) {
    const bb = getNodeBBox(key);
    const cx = bb.min[0] + bb.size/2;
    const cy = bb.min[1] + bb.size/2;
    const cz = bb.min[2] + bb.size/2;
    const nodeCenter = new THREE.Vector3(cx, cy, cz);
    const dist = camera.position.distanceTo(nodeCenter);
    if (dist < 0.001) return 99999;
    // Projected size in pixels
    const fov = camera.fov * Math.PI / 180;
    const screenH = renderer.domElement.height;
    return (bb.size / dist) * screenH / (2 * Math.tan(fov/2));
}

function isInFrustum(key) {
    const bb = getNodeBBox(key);
    const cx = bb.min[0] + bb.size/2;
    const cy = bb.min[1] + bb.size/2;
    const cz = bb.min[2] + bb.size/2;
    // Rough sphere check
    const p = new THREE.Vector3(cx, cy, cz);
    const frustum = new THREE.Frustum();
    const projScreenMatrix = new THREE.Matrix4();
    projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(projScreenMatrix);
    const sphere = new THREE.Sphere(p, bb.size * 0.866);
    return frustum.intersectsSphere(sphere);
}

async function loadNode(key) {
    if (loadedNodes[key] || loadingNodes.has(key)) return;
    if (!nodes[key]) return;

    loadingNodes.add(key);

    try {
        const resp = await fetch('tiles/' + key + '.bin');
        if (!resp.ok) { loadingNodes.delete(key); return; }
        const buf = await resp.arrayBuffer();
        const arr = new Uint8Array(buf);
        const n = arr.length / POINT_SIZE;
        if (n === 0) { loadingNodes.delete(key); return; }

        const dv = new DataView(buf);
        const positions = new Float32Array(n * 3);
        const colors = new Float32Array(n * 3);
        const flags = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            const o = i * POINT_SIZE;
            positions[i*3]   = dv.getFloat32(o, true);
            positions[i*3+1] = dv.getFloat32(o+4, true);
            positions[i*3+2] = dv.getFloat32(o+8, true);
            colors[i*3]   = arr[o+12] / 255;
            colors[i*3+1] = arr[o+13] / 255;
            colors[i*3+2] = arr[o+14] / 255;
            flags[i] = arr[o+15] / 255;
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geo.setAttribute('flag', new THREE.Float32BufferAttribute(flags, 1));

        const mat = new THREE.ShaderMaterial({
            uniforms: {
                uSize: {value: R * 0.008},
                uDimAlpha: {value: 0.12}
            },
            vertexShader, fragmentShader: fragShader,
            vertexColors: true, transparent: true,
            depthWrite: true, depthTest: true
        });

        const points = new THREE.Points(geo, mat);
        points.userData = {key, count: n};
        scene.add(points);
        loadedNodes[key] = points;
        visiblePoints += n;
    } catch(e) {
        console.warn('Failed to load', key, e);
    }
    loadingNodes.delete(key);
}

function unloadNode(key) {
    const obj = loadedNodes[key];
    if (!obj) return;
    visiblePoints -= obj.userData.count;
    scene.remove(obj);
    obj.geometry.dispose();
    obj.material.dispose();
    delete loadedNodes[key];
}

// LOD update: decide which nodes to show/hide/load
const SCREEN_THRESHOLD = 150;  // pixels: if node > this, load children

function updateLOD() {
    const toLoad = [];
    const toKeep = new Set();
    const threshold = SCREEN_THRESHOLD * lodMultiplier;

    // BFS from root
    const queue = ['r'];
    while (queue.length > 0) {
        const key = queue.shift();
        if (!nodes[key]) continue;
        if (!isInFrustum(key)) continue;

        const ss = nodeScreenSize(key);
        const nodeInfo = nodes[key];

        if (ss > threshold && nodeInfo.children && nodeInfo.children.length > 0) {
            // This node is big on screen — load children for more detail
            for (const c of nodeInfo.children) {
                queue.push(key + c);
            }
            // Keep this node visible while children load
            toKeep.add(key);
        } else {
            // This node is fine at current detail
            toKeep.add(key);
        }
    }

    // Load needed nodes
    for (const key of toKeep) {
        if (!loadedNodes[key] && !loadingNodes.has(key)) {
            toLoad.push(key);
        }
    }

    // Unload nodes no longer needed (but keep parents while children load)
    for (const key of Object.keys(loadedNodes)) {
        if (!toKeep.has(key)) {
            // Check if any descendant is still needed
            let needed = false;
            for (const k of toKeep) {
                if (k.startsWith(key)) { needed = true; break; }
            }
            if (!needed) {
                unloadNode(key);
            }
        }
    }

    // Process load queue (limited concurrency)
    // Prioritize by screen size (largest first)
    toLoad.sort((a, b) => nodeScreenSize(b) - nodeScreenSize(a));
    const budget = MAX_CONCURRENT - loadingNodes.size;
    for (let i = 0; i < Math.min(budget, toLoad.length); i++) {
        loadNode(toLoad[i]);
    }
}

// Camera controls
let sph = {theta: -Math.PI/2, phi: Math.PI/3, radius: R * 3.5};
let tgt = center.clone();
let isDrag = false, isPan = false, lx = 0, ly = 0;

function updateCam() {
    const sp = Math.sin(sph.phi), cp = Math.cos(sph.phi);
    const st = Math.sin(sph.theta), ct = Math.cos(sph.theta);
    camera.position.set(
        tgt.x + sph.radius * sp * ct,
        tgt.y + sph.radius * sp * st,
        tgt.z + sph.radius * cp
    );
    camera.lookAt(tgt);
}
updateCam();
const initS = {...sph};
const initT = tgt.clone();

wrap.addEventListener('mousedown', e => {
    if (e.button === 0) isDrag = true;
    if (e.button === 2) isPan = true;
    lx = e.clientX; ly = e.clientY; e.preventDefault();
});
document.addEventListener('mousemove', e => {
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    if (isDrag) {
        sph.theta -= dx * 0.005;
        sph.phi -= dy * 0.005;
        sph.phi = Math.max(0.05, Math.min(Math.PI - 0.05, sph.phi));
        updateCam();
    }
    if (isPan) {
        const right = new THREE.Vector3(), up = new THREE.Vector3();
        right.crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
        up.copy(camera.up).normalize();
        const ps = sph.radius * 0.002;
        tgt.add(right.multiplyScalar(-dx * ps));
        tgt.add(up.multiplyScalar(dy * ps));
        updateCam();
    }
});
document.addEventListener('mouseup', () => { isDrag = false; isPan = false; });
wrap.addEventListener('wheel', e => {
    sph.radius *= (1 + e.deltaY * 0.001);
    sph.radius = Math.max(R * 0.05, Math.min(R * 30, sph.radius));
    updateCam(); e.preventDefault();
}, {passive: false});
wrap.addEventListener('contextmenu', e => e.preventDefault());

function onResize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (w < 1 || h < 1) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
}
window.addEventListener('resize', onResize);
onResize();

// UI
document.getElementById('psize').oninput = function() {
    const v = this.value / 25.0;
    for (const obj of Object.values(loadedNodes)) {
        obj.material.uniforms.uSize.value = R * 0.008 * v;
    }
};
document.getElementById('bgsel').onchange = function() {
    scene.background = new THREE.Color(this.value);
};
document.getElementById('unsel').onchange = function() {
    const alpha = this.value === 'hide' ? 0.0 : this.value === 'dim' ? 0.12 : 1.0;
    for (const obj of Object.values(loadedNodes)) {
        obj.material.uniforms.uDimAlpha.value = alpha;
    }
};
document.getElementById('lod').oninput = function() {
    lodMultiplier = 2.0 - this.value / 10.0;  // 1->1.0, 10->1.0, 20->0.0
    lodMultiplier = Math.max(0.2, lodMultiplier);
};
document.getElementById('resetBtn').onclick = function() {
    sph = {...initS};
    tgt.copy(initT);
    updateCam();
};

// Render loop
let lastLODUpdate = 0;
function animate(t) {
    requestAnimationFrame(animate);

    // Update LOD every 200ms
    if (t - lastLODUpdate > 200) {
        updateLOD();
        lastLODUpdate = t;

        // Update stats
        const loaded = Object.keys(loadedNodes).length;
        const total = Object.keys(nodes).length;
        document.getElementById('stats').textContent =
            visiblePoints.toLocaleString() + ' visible | ' +
            loaded + '/' + total + ' nodes | ' +
            loadingNodes.size + ' loading';
    }

    renderer.render(scene, camera);
}

// Initial load: start with root
document.getElementById('loading').style.display = 'none';
animate(0);

})();
</script>
</body>
</html>"""

    viewer_path = os.path.join(output_dir, 'index.html')
    with open(viewer_path, 'w') as f:
        f.write(html)
    return viewer_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PLY to streaming octree tiles for web viewing')
    parser.add_argument('input', help='PLY prediction file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: <input>_viewer/)')
    parser.add_argument('--budget', type=int, default=50_000,
                        help='Points per octree node (default: 50000)')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Max octree depth (default: 8)')
    args = parser.parse_args()

    output_dir = args.output or (
        os.path.splitext(args.input)[0] + '_viewer')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Input:  {args.input}')
    print(f'Output: {output_dir}')
    print()

    t_total = time.time()
    metadata = build_octree(args.input, output_dir,
                            node_budget=args.budget,
                            max_depth=args.max_depth)

    viewer_path = generate_viewer_html(output_dir, metadata)

    print()
    print(f'Total time: {time.time() - t_total:.0f}s')
    print(f'Viewer:     {viewer_path}')
    print()
    print(f'To view, run:')
    print(f'  cd {output_dir} && python -m http.server 8080')
    print(f'  Then open: http://localhost:8080')


if __name__ == '__main__':
    main()
