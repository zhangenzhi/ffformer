#!/usr/bin/env python3
"""Convert LAS/LAZ LiDAR files to ForestFormer3D inference format.

Reads a LAS file, normalizes coordinates (center XY, shift Z to min=0),
and outputs the binary files + pkl needed by the model.

For large scenes, optionally splits into tiles to manage memory.

Usage:
    python tools/las2bin.py /path/to/lidar.las --name jpeaks_area1
    python tools/las2bin.py /path/to/lidar.las --name jpeaks_area1 --subsample 0.05
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

try:
    import laspy
except ImportError:
    print("Error: laspy required. Install with: pip install laspy")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAS to ForestFormer3D inference format')
    parser.add_argument('input', help='Input LAS/LAZ file')
    parser.add_argument('--name', required=True,
                        help='Scene name (e.g. jpeaks_area1)')
    parser.add_argument('--output-dir', default='data/ForAINetV2',
                        help='Output base directory (default: data/ForAINetV2)')
    parser.add_argument('--subsample', type=float, default=None,
                        help='Voxel size for subsampling in meters (e.g. 0.05). '
                             'None = keep all points')
    parser.add_argument('--max-points', type=int, default=None,
                        help='Max number of points (random subsample if exceeded)')
    args = parser.parse_args()

    print(f'Reading {args.input} ...')
    t0 = time.time()
    las = laspy.read(args.input)
    print(f'  Loaded in {time.time()-t0:.1f}s')
    print(f'  Points: {las.header.point_count:,}')
    print(f'  Format: {las.point_format.id}')
    print(f'  Fields: {list(las.point_format.dimension_names)}')

    # Extract XYZ
    xyz = np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ])
    print(f'  X: [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]')
    print(f'  Y: [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]')
    print(f'  Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]')
    print(f'  Extent: {xyz[:,0].ptp():.1f} x {xyz[:,1].ptp():.1f} x {xyz[:,2].ptp():.1f} m')

    # Optional voxel subsampling
    if args.subsample is not None:
        print(f'  Voxel subsampling at {args.subsample}m ...')
        voxel = args.subsample
        voxel_ids = np.floor(xyz / voxel).astype(np.int64)
        _, unique_idx = np.unique(voxel_ids, axis=0, return_index=True)
        xyz = xyz[unique_idx]
        print(f'  After subsampling: {len(xyz):,} points')

    # Optional random subsample
    if args.max_points is not None and len(xyz) > args.max_points:
        print(f'  Random subsampling to {args.max_points:,} ...')
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        xyz = xyz[idx]

    N = len(xyz)

    # Normalize: center XY, shift Z to min=0
    mean_x = xyz[:, 0].mean()
    mean_y = xyz[:, 1].mean()
    min_z = xyz[:, 2].min()
    offsets = np.array([mean_x, mean_y, min_z], dtype=np.float64)

    xyz[:, 0] -= mean_x
    xyz[:, 1] -= mean_y
    xyz[:, 2] -= min_z
    xyz = xyz.astype(np.float32)

    print(f'  Normalized: center=({mean_x:.2f}, {mean_y:.2f}), z_min={min_z:.2f}')
    print(f'  New ranges: x=[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] '
          f'y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] '
          f'z=[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]')

    # Create output directories
    scene_name = args.name + '_test'
    points_dir = os.path.join(args.output_dir, 'points')
    sem_dir = os.path.join(args.output_dir, 'semantic_mask')
    ins_dir = os.path.join(args.output_dir, 'instance_mask')
    instance_data_dir = os.path.join(args.output_dir, 'forainetv2_instance_data')
    for d in [points_dir, sem_dir, ins_dir, instance_data_dir]:
        os.makedirs(d, exist_ok=True)

    # Write binary files
    pts_path = os.path.join(points_dir, f'{scene_name}.bin')
    xyz.tofile(pts_path)
    print(f'  -> {pts_path} ({N:,} points, {os.path.getsize(pts_path)/1e6:.1f}MB)')

    # Dummy semantic/instance masks (all zeros — no GT)
    dummy_sem = np.zeros(N, dtype=np.int64)
    dummy_ins = np.zeros(N, dtype=np.int64)
    dummy_sem.tofile(os.path.join(sem_dir, f'{scene_name}.bin'))
    dummy_ins.tofile(os.path.join(ins_dir, f'{scene_name}.bin'))
    print(f'  -> semantic_mask/{scene_name}.bin (all zeros)')
    print(f'  -> instance_mask/{scene_name}.bin (all zeros)')

    # Write instance_data npy files
    prefix = os.path.join(instance_data_dir, scene_name)
    np.save(f'{prefix}_vert.npy', xyz)
    np.save(f'{prefix}_sem_label.npy', dummy_sem)
    np.save(f'{prefix}_ins_label.npy', dummy_ins)
    np.save(f'{prefix}_offsets.npy', offsets)
    np.save(f'{prefix}_axis_align_matrix.npy', np.eye(4))
    np.save(f'{prefix}_aligned_bbox.npy', np.zeros((0, 7)))
    np.save(f'{prefix}_unaligned_bbox.npy', np.zeros((0, 7)))
    print(f'  -> forainetv2_instance_data/{scene_name}_*.npy')

    # Create pkl info file
    info = {
        'metainfo': {'categories': {'ground': 0, 'wood': 1, 'leaf': 2},
                     'dataset': 'ForAINetV2',
                     'info_version': '1.1'},
        'data_list': [{
            'lidar_points': {
                'num_pts_feats': 3,
                'lidar_path': f'{scene_name}.bin',
            },
            'instances': [],
            'pts_semantic_mask_path': f'{scene_name}.bin',
            'pts_instance_mask_path': f'{scene_name}.bin',
            'axis_align_matrix': np.eye(4).tolist(),
        }],
    }
    pkl_name = f'forainetv2_oneformer3d_infos_{args.name}.pkl'
    pkl_path = os.path.join(args.output_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(info, f)
    print(f'  -> {pkl_path}')

    # Create test list
    meta_dir = os.path.join(args.output_dir, 'meta_data')
    os.makedirs(meta_dir, exist_ok=True)
    list_path = os.path.join(meta_dir, f'test_list_{args.name}.txt')
    with open(list_path, 'w') as f:
        f.write(f'{scene_name}\n')
    print(f'  -> {list_path}')

    print(f'\nDone! {N:,} points prepared as "{scene_name}"')
    print(f'\nTo run inference:')
    print(f'  python tools/test.py configs/jpeaks_test.py '
          f'work_dirs/clean_forestformer/epoch_3000_fix.pth '
          f'--work-dir work_dirs/jpeaks_output')


if __name__ == '__main__':
    main()
