#!/usr/bin/env python3
"""Visualize ForestFormer3D PLY with Open3D, colored by instance/semantic.

Usage:
    python tools/vis_open3d.py work_dirs/jpeaks_area1/jpeaks_area1_test.ply
    python tools/vis_open3d.py scene.ply --mode semantic
    python tools/vis_open3d.py scene.ply --max-points 5000000
"""
import argparse
import sys
import time

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("pip install open3d")
    sys.exit(1)

INSTANCE_PALETTE = np.array([
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
], dtype=np.float64) / 255.0

SEMANTIC_COLORS = np.array([
    [139, 119, 101],  # ground
    [210, 150, 60],   # wood
    [34, 180, 34],    # leaf
], dtype=np.float64) / 255.0

UNASSIGNED = np.array([0.31, 0.31, 0.31])


def main():
    parser = argparse.ArgumentParser(description='Visualize PLY with Open3D')
    parser.add_argument('input', help='PLY prediction file')
    parser.add_argument('--mode', default='instance', choices=['instance', 'semantic'])
    parser.add_argument('--max-points', type=int, default=5_000_000,
                        help='Max points to load (default: 5M, 0 = all)')
    args = parser.parse_args()

    # Parse header
    fields = []
    num_vertices = 0
    header_lines = 0
    with open(args.input, 'r') as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith('element vertex'):
                num_vertices = int(s.split()[-1])
            elif s.startswith('property'):
                fields.append(s.split()[-1])
            elif s == 'end_header':
                break

    col = {name: idx for idx, name in enumerate(fields)}
    print(f'Loading {args.input} ({num_vertices:,} points) ...')

    # Load with numpy
    t0 = time.time()
    data = np.loadtxt(args.input, skiprows=header_lines, max_rows=num_vertices)
    print(f'  Parsed in {time.time() - t0:.1f}s')

    xyz = np.ascontiguousarray(data[:, [col['x'], col['y'], col['z']]]).astype(np.float64)

    # Downsample if needed
    if args.max_points > 0 and len(xyz) > args.max_points:
        print(f'  Downsampling {len(xyz):,} -> {args.max_points:,} ...')
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        data = data[idx]

    n = len(xyz)
    print(f'  Visualizing {n:,} points, mode={args.mode}')

    # Assign colors
    colors = np.tile(UNASSIGNED, (n, 1))

    if args.mode == 'instance' and 'instance_pred' in col:
        inst = data[:, col['instance_pred']].astype(np.int32)
        valid = inst >= 0
        colors[valid] = INSTANCE_PALETTE[inst[valid] % len(INSTANCE_PALETTE)]
        # Unassigned: use semantic
        if 'semantic_pred' in col:
            sem = data[:, col['semantic_pred']].astype(np.int32)
            for s in range(len(SEMANTIC_COLORS)):
                mask = (~valid) & (sem == s)
                colors[mask] = SEMANTIC_COLORS[s]
    elif args.mode == 'semantic' and 'semantic_pred' in col:
        sem = data[:, col['semantic_pred']].astype(np.int32)
        for s in range(len(SEMANTIC_COLORS)):
            mask = sem == s
            colors[mask] = SEMANTIC_COLORS[s]

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    n_trees = len(set(data[:, col['instance_pred']].astype(int)[
        data[:, col['instance_pred']].astype(int) >= 0])) if 'instance_pred' in col else 0
    print(f'  Trees: {n_trees}')
    print('  Controls: mouse drag=rotate, scroll=zoom, shift+drag=pan')

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f'ForestFormer3D — {n:,} points, {n_trees} trees ({args.mode})',
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == '__main__':
    main()
