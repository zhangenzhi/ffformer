#!/usr/bin/env python3
"""Tile a large LAS file, run inference on each tile, and merge results.

Splits a large scene into manageable tiles, processes each tile through
ForestFormer3D, and stitches results back together.

Usage:
    # Step 1: Split into tiles
    python tools/tile_and_infer.py split /path/to/lidar.las --name jpeaks_area1 --tile-size 100

    # Step 2: Run inference on one tile
    python tools/tile_and_infer.py infer --name jpeaks_area1 --tile 0

    # Step 2b: Run inference on all tiles
    python tools/tile_and_infer.py infer --name jpeaks_area1 --all

    # Step 3: Merge all tile results
    python tools/tile_and_infer.py merge --name jpeaks_area1

    # Step 4: Visualize
    python tools/tile_and_infer.py viz --name jpeaks_area1

    # Or run everything:
    python tools/tile_and_infer.py run /path/to/lidar.las --name jpeaks_area1 --tile-size 100
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_split(args):
    """Split LAS into tiles."""
    import laspy

    print(f'Reading {args.input} ...')
    t0 = time.time()
    las = laspy.read(args.input)
    print(f'  {las.header.point_count:,} points in {time.time()-t0:.1f}s')

    xyz = np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ])

    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    zmin = xyz[:, 2].min()
    dx, dy = xmax - xmin, ymax - ymin
    print(f'  Extent: x=[{xmin:.1f},{xmax:.1f}] y=[{ymin:.1f},{ymax:.1f}] ({dx:.0f}x{dy:.0f}m)')

    tile_size = args.tile_size
    overlap = args.overlap
    # Compute tile grid
    nx = int(np.ceil(dx / tile_size))
    ny = int(np.ceil(dy / tile_size))
    print(f'  Tile grid: {nx}x{ny} = {nx*ny} tiles ({tile_size}m, overlap={overlap}m)')

    # Output directory
    out_dir = os.path.join(PROJECT_ROOT, 'work_dirs', f'tiles_{args.name}')
    os.makedirs(out_dir, exist_ok=True)

    tiles_meta = []
    for ix in range(nx):
        for iy in range(ny):
            tx0 = xmin + ix * tile_size - overlap
            tx1 = xmin + (ix + 1) * tile_size + overlap
            ty0 = ymin + iy * tile_size - overlap
            ty1 = ymin + (iy + 1) * tile_size + overlap

            # Core region (without overlap) for merging
            cx0 = xmin + ix * tile_size
            cx1 = min(xmin + (ix + 1) * tile_size, xmax)
            cy0 = ymin + iy * tile_size
            cy1 = min(ymin + (iy + 1) * tile_size, ymax)

            mask = ((xyz[:, 0] >= tx0) & (xyz[:, 0] <= tx1) &
                    (xyz[:, 1] >= ty0) & (xyz[:, 1] <= ty1))
            n_pts = mask.sum()
            if n_pts < 100:
                continue

            tile_id = len(tiles_meta)
            tile_xyz = xyz[mask].copy()
            tile_global_idx = np.where(mask)[0]

            # Normalize: center XY, shift Z
            mean_x = tile_xyz[:, 0].mean()
            mean_y = tile_xyz[:, 1].mean()
            tile_xyz_norm = tile_xyz.copy()
            tile_xyz_norm[:, 0] -= mean_x
            tile_xyz_norm[:, 1] -= mean_y
            tile_xyz_norm[:, 2] -= zmin
            tile_xyz_norm = tile_xyz_norm.astype(np.float32)

            # Save tile data
            tile_name = f'{args.name}_tile{tile_id:03d}_test'
            tile_dir = os.path.join(out_dir, f'tile{tile_id:03d}')
            os.makedirs(tile_dir, exist_ok=True)

            # Save raw bin for model
            pts_dir = os.path.join(PROJECT_ROOT, 'data', 'ForAINetV2', 'points')
            sem_dir = os.path.join(PROJECT_ROOT, 'data', 'ForAINetV2', 'semantic_mask')
            ins_dir = os.path.join(PROJECT_ROOT, 'data', 'ForAINetV2', 'instance_mask')
            for d in [pts_dir, sem_dir, ins_dir]:
                os.makedirs(d, exist_ok=True)

            tile_xyz_norm.tofile(os.path.join(pts_dir, f'{tile_name}.bin'))
            np.zeros(n_pts, dtype=np.int64).tofile(os.path.join(sem_dir, f'{tile_name}.bin'))
            np.zeros(n_pts, dtype=np.int64).tofile(os.path.join(ins_dir, f'{tile_name}.bin'))

            # Save global indices for merge
            np.save(os.path.join(tile_dir, 'global_idx.npy'), tile_global_idx)
            np.save(os.path.join(tile_dir, 'offsets.npy'),
                    np.array([mean_x, mean_y, zmin], dtype=np.float64))

            # Create pkl
            info = {
                'metainfo': {'categories': {'ground':0,'wood':1,'leaf':2},
                             'dataset':'ForAINetV2','info_version':'1.1'},
                'data_list': [{
                    'lidar_points': {'num_pts_feats': 3, 'lidar_path': f'{tile_name}.bin'},
                    'instances': [],
                    'pts_semantic_mask_path': f'{tile_name}.bin',
                    'pts_instance_mask_path': f'{tile_name}.bin',
                    'axis_align_matrix': np.eye(4).tolist(),
                }],
            }
            pkl_path = os.path.join(PROJECT_ROOT, 'data', 'ForAINetV2',
                                    f'forainetv2_oneformer3d_infos_{args.name}_tile{tile_id:03d}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(info, f)

            tiles_meta.append({
                'tile_id': tile_id,
                'tile_name': tile_name,
                'n_points': int(n_pts),
                'bounds': [float(tx0), float(ty0), float(tx1), float(ty1)],
                'core_bounds': [float(cx0), float(cy0), float(cx1), float(cy1)],
                'offsets': [float(mean_x), float(mean_y), float(zmin)],
                'pkl': os.path.basename(pkl_path),
            })
            print(f'  Tile {tile_id:3d}: ({ix},{iy}) {n_pts:>10,} pts  '
                  f'[{tx0:.0f},{ty0:.0f}]-[{tx1:.0f},{ty1:.0f}]')

    # Save meta
    meta = {
        'name': args.name,
        'input': args.input,
        'total_points': int(las.header.point_count),
        'tile_size': tile_size,
        'overlap': overlap,
        'n_tiles': len(tiles_meta),
        'global_zmin': float(zmin),
        'global_bounds': [float(xmin), float(ymin), float(xmax), float(ymax)],
        'tiles': tiles_meta,
    }
    meta_path = os.path.join(out_dir, 'tiles_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  {len(tiles_meta)} tiles saved -> {out_dir}')
    print(f'  Meta: {meta_path}')


def cmd_infer(args):
    """Run inference on tiles."""
    out_dir = os.path.join(PROJECT_ROOT, 'work_dirs', f'tiles_{args.name}')
    meta_path = os.path.join(out_dir, 'tiles_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    if args.all:
        tile_ids = list(range(meta['n_tiles']))
    else:
        tile_ids = [args.tile]

    for tid in tile_ids:
        tile = meta['tiles'][tid]
        tile_name = tile['tile_name']
        tile_dir = os.path.join(out_dir, f'tile{tid:03d}')
        result_ply = os.path.join(tile_dir, f'{tile_name}.ply')

        if os.path.exists(result_ply) and not args.force:
            print(f'Tile {tid}: already done, skip (use --force to redo)')
            continue

        print(f'\n{"="*60}')
        print(f'Tile {tid}/{meta["n_tiles"]-1}: {tile["n_points"]:,} points')
        print(f'{"="*60}')

        # Run test.py
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, 'tools', 'test.py'),
            os.path.join(PROJECT_ROOT, 'configs', 'jpeaks_test.py'),
            os.path.join(PROJECT_ROOT, 'work_dirs', 'clean_forestformer', 'epoch_3000_fix.pth'),
            '--work-dir', tile_dir,
            '--cfg-options',
            f'test_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_{args.name}_tile{tid:03d}.pkl',
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=3600)
        dt = time.time() - t0
        print(f'  Tile {tid} done in {dt:.0f}s (exit={proc.returncode})')

        if os.path.exists(result_ply):
            sz = os.path.getsize(result_ply) / 1e6
            print(f'  -> {result_ply} ({sz:.1f}MB)')
        else:
            print(f'  WARNING: no output PLY found')


def cmd_merge(args):
    """Merge all tile results into one PLY."""
    out_dir = os.path.join(PROJECT_ROOT, 'work_dirs', f'tiles_{args.name}')
    meta_path = os.path.join(out_dir, 'tiles_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    print(f'Merging {meta["n_tiles"]} tiles ...')
    N_total = meta['total_points']

    # Global arrays
    sem_pred = np.full(N_total, -1, dtype=np.int32)
    inst_pred = np.full(N_total, -1, dtype=np.int32)
    scores = np.full(N_total, -1.0, dtype=np.float32)
    max_instance_id = 0

    for tile in meta['tiles']:
        tid = tile['tile_id']
        tile_dir = os.path.join(out_dir, f'tile{tid:03d}')
        tile_name = tile['tile_name']
        ply_path = os.path.join(tile_dir, f'{tile_name}.ply')

        if not os.path.exists(ply_path):
            print(f'  Tile {tid}: no result, skip')
            continue

        global_idx = np.load(os.path.join(tile_dir, 'global_idx.npy'))

        # Parse tile PLY
        fields = []
        with open(ply_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('property'):
                    fields.append(line.split()[-1])
                elif line == 'end_header':
                    break
            col = {name: idx for idx, name in enumerate(fields)}
            rows = []
            for line in f:
                rows.append(line.strip().split())

        n = len(rows)
        # Only assign points in the core region (avoid overlap duplicates)
        core = tile['core_bounds']  # [cx0, cy0, cx1, cy1]
        offsets = tile['offsets']

        assigned = 0
        for i in range(n):
            gi = global_idx[i]
            s = int(rows[i][col['semantic_pred']])
            inst = int(rows[i][col['instance_pred']])
            sc = float(rows[i][col['score']])

            # Only keep if this point's score is better than existing
            if sc > scores[gi]:
                sem_pred[gi] = s
                inst_pred[gi] = inst + max_instance_id if inst >= 0 else -1
                scores[gi] = sc
                assigned += 1

        # Offset instance IDs for next tile
        tile_max = max(int(rows[i][col['instance_pred']]) for i in range(n)) if n > 0 else 0
        if tile_max >= 0:
            max_instance_id += tile_max + 1

        print(f'  Tile {tid}: {n:,} pts, assigned {assigned:,}')

    # Read original LAS for coordinates
    import laspy
    print(f'Reading original LAS for coordinates ...')
    las = laspy.read(meta['input'])
    xyz = np.column_stack([
        np.array(las.x, dtype=np.float32),
        np.array(las.y, dtype=np.float32),
        np.array(las.z, dtype=np.float32),
    ])

    # Write merged PLY
    merged_path = os.path.join(out_dir, f'{args.name}_merged.ply')
    print(f'Writing merged PLY ({N_total:,} points) ...')

    with open(merged_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {N_total}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property int semantic_pred\n')
        f.write('property int instance_pred\n')
        f.write('property float score\n')
        f.write('property int semantic_gt\n')
        f.write('property int instance_gt\n')
        f.write('end_header\n')
        for i in range(N_total):
            f.write(f'{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} '
                    f'{sem_pred[i]} {inst_pred[i]} {scores[i]} 0 0\n')

    sz = os.path.getsize(merged_path) / 1e9
    n_trees = len(set(inst_pred[inst_pred >= 0]))
    n_assigned = (inst_pred >= 0).sum()
    print(f'\n  -> {merged_path} ({sz:.1f}GB)')
    print(f'  Trees: {n_trees}')
    print(f'  Assigned: {n_assigned:,} / {N_total:,} ({100*n_assigned/N_total:.1f}%)')


def cmd_viz(args):
    """Visualize merged or single tile result."""
    out_dir = os.path.join(PROJECT_ROOT, 'work_dirs', f'tiles_{args.name}')

    if args.tile is not None:
        # Visualize single tile
        meta_path = os.path.join(out_dir, 'tiles_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        tile = meta['tiles'][args.tile]
        ply = os.path.join(out_dir, f'tile{args.tile:03d}', f'{tile["tile_name"]}.ply')
    else:
        ply = os.path.join(out_dir, f'{args.name}_merged.ply')

    if not os.path.exists(ply):
        print(f'PLY not found: {ply}')
        sys.exit(1)

    viz_script = os.path.join(PROJECT_ROOT, 'tools', 'visualize_ply.py')
    subprocess.run([sys.executable, viz_script, ply,
                    '--mode', 'instance', '--max-points', str(args.max_points)])


def cmd_run(args):
    """Run the full pipeline."""
    # Split
    args_split = argparse.Namespace(
        input=args.input, name=args.name,
        tile_size=args.tile_size, overlap=args.overlap)
    cmd_split(args_split)

    # Infer all
    args_infer = argparse.Namespace(
        name=args.name, all=True, tile=None, force=False)
    cmd_infer(args_infer)

    # Merge
    args_merge = argparse.Namespace(name=args.name)
    cmd_merge(args_merge)

    # Viz
    args_viz = argparse.Namespace(name=args.name, tile=None, max_points=300000)
    cmd_viz(args_viz)


def main():
    parser = argparse.ArgumentParser(description='Tile-based inference for large scenes')
    sub = parser.add_subparsers(dest='cmd')

    p_split = sub.add_parser('split', help='Split LAS into tiles')
    p_split.add_argument('input', help='Input LAS file')
    p_split.add_argument('--name', required=True)
    p_split.add_argument('--tile-size', type=float, default=100,
                         help='Tile size in meters (default: 100)')
    p_split.add_argument('--overlap', type=float, default=10,
                         help='Overlap between tiles in meters (default: 10)')

    p_infer = sub.add_parser('infer', help='Run inference on tile(s)')
    p_infer.add_argument('--name', required=True)
    p_infer.add_argument('--tile', type=int, help='Tile index')
    p_infer.add_argument('--all', action='store_true', help='Process all tiles')
    p_infer.add_argument('--force', action='store_true', help='Redo existing')

    p_merge = sub.add_parser('merge', help='Merge tile results')
    p_merge.add_argument('--name', required=True)

    p_viz = sub.add_parser('viz', help='Visualize results')
    p_viz.add_argument('--name', required=True)
    p_viz.add_argument('--tile', type=int, default=None, help='View single tile')
    p_viz.add_argument('--max-points', type=int, default=300000)

    p_run = sub.add_parser('run', help='Full pipeline: split → infer → merge → viz')
    p_run.add_argument('input', help='Input LAS file')
    p_run.add_argument('--name', required=True)
    p_run.add_argument('--tile-size', type=float, default=100)
    p_run.add_argument('--overlap', type=float, default=10)

    args = parser.parse_args()
    if args.cmd == 'split':
        cmd_split(args)
    elif args.cmd == 'infer':
        cmd_infer(args)
    elif args.cmd == 'merge':
        cmd_merge(args)
    elif args.cmd == 'viz':
        cmd_viz(args)
    elif args.cmd == 'run':
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
