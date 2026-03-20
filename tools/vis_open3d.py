#!/usr/bin/env python3
"""Visualize ForestFormer3D PLY with PyVista, colored by instance/semantic.

Works on Mac Apple Silicon, Linux, and Windows.

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
    import pyvista as pv
except ImportError:
    print("pip install pyvista")
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
    print(f'Loading {args.input} ({num_vertices:,} points)')
    print()
    timings = []
    t_total = time.time()

    # --- 1. Parse PLY ---
    t0 = time.time()
    try:
        import pandas as pd
        data = pd.read_csv(args.input, sep=r'\s+', header=None,
                           skiprows=header_lines, nrows=num_vertices,
                           dtype=np.float64, engine='c').values
    except ImportError:
        data = np.loadtxt(args.input, skiprows=header_lines, max_rows=num_vertices)
    dt = time.time() - t0
    timings.append(('Parse PLY', dt))

    # --- 2. Extract XYZ ---
    t0 = time.time()
    xyz = np.ascontiguousarray(data[:, [col['x'], col['y'], col['z']]]).astype(np.float64)
    dt = time.time() - t0
    timings.append(('Extract XYZ', dt))

    # --- 3. Downsample ---
    t0 = time.time()
    if args.max_points > 0 and len(xyz) > args.max_points:
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        data = data[idx]
    dt = time.time() - t0
    timings.append(('Downsample', dt))

    n = len(xyz)

    # --- 4. Assign colors ---
    t0 = time.time()
    colors = np.tile(UNASSIGNED, (n, 1))

    if args.mode == 'instance' and 'instance_pred' in col:
        inst = data[:, col['instance_pred']].astype(np.int32)
        valid = inst >= 0
        colors[valid] = INSTANCE_PALETTE[inst[valid] % len(INSTANCE_PALETTE)]
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
    dt = time.time() - t0
    timings.append(('Assign colors', dt))

    # --- 5. Count trees ---
    t0 = time.time()
    n_trees = len(set(data[:, col['instance_pred']].astype(int)[
        data[:, col['instance_pred']].astype(int) >= 0])) if 'instance_pred' in col else 0
    dt = time.time() - t0
    timings.append(('Count trees', dt))

    # --- 6. Build PyVista cloud ---
    t0 = time.time()
    cloud = pv.PolyData(xyz)
    cloud['rgb'] = (colors * 255).astype(np.uint8)
    dt = time.time() - t0
    timings.append(('Build PolyData', dt))

    # --- 7. Render ---
    t0 = time.time()
    pl = pv.Plotter(window_size=[1280, 800],
                     title=f'ForestFormer3D — {n:,} points, {n_trees} trees ({args.mode})')
    pl.set_background('#0d1117')
    pl.add_points(cloud, scalars='rgb', rgb=True, point_size=2.0,
                  render_points_as_spheres=False)
    dt = time.time() - t0
    timings.append(('Init renderer', dt))

    # --- Print timing summary ---
    dt_total = time.time() - t_total
    print(f'  {n:,} points, {n_trees} trees, mode={args.mode}')
    print()
    print(f'  {"Step":<20} {"Time":>8} {"Pct":>6}')
    print(f'  {"-"*20} {"-"*8} {"-"*6}')
    for name, dt in timings:
        pct = dt / dt_total * 100
        print(f'  {name:<20} {dt:>7.1f}s {pct:>5.1f}%')
    print(f'  {"-"*20} {"-"*8} {"-"*6}')
    print(f'  {"Total":<20} {dt_total:>7.1f}s {100.0:>5.1f}%')
    print()
    print('  Controls: mouse drag=rotate, scroll=zoom, right-drag=pan')

    pl.show()


if __name__ == '__main__':
    main()
