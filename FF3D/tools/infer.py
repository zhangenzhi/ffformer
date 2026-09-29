"""
ForestFormer3D inference on LAS/PLY files.

Usage:
    python tools/infer.py \
        --input /path/to/lidar.las \
        --checkpoint /path/to/epoch_3000_fix.pth \
        --output results/output.ply
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_las(path):
    import laspy
    print(f"Loading {path} ...")
    las = laspy.read(path)
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    print(f"  {len(points):,d} points, "
          f"extent: {points[:, 0].ptp():.0f} x {points[:, 1].ptp():.0f} x {points[:, 2].ptp():.0f} m")

    offsets = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].min()],
                       dtype=np.float64)
    points[:, 0] -= offsets[0]
    points[:, 1] -= offsets[1]
    points[:, 2] -= offsets[2]
    return points.astype(np.float32), offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input LAS/LAZ/PLY file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--output", default=None, help="Output PLY path")
    parser.add_argument("--radius", type=float, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--query-num", type=int, default=300)
    parser.add_argument("--native-ckpt", action="store_true",
                        help="checkpoint is from our training ({'model':...}), not mm* format")
    parser.add_argument("--invariant-feat", action="store_true",
                        help="Translation-invariant input (height above ground, xy=0) — must match training (approach B)")
    parser.add_argument("--no-save", action="store_true",
                        help="skip PLY write — measure pure inference time")
    args = parser.parse_args()

    if args.no_save:
        args.output = None
    elif args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("results", f"{base}.ply")

    # Load point cloud
    points_np, offsets = load_las(args.input)

    # Build model
    from ff3d.model import ForestFormer3D
    from ff3d.model.forestformer3d import load_pretrained

    model = ForestFormer3D(
        in_channels=3, num_channels=32, voxel_size=args.voxel_size,
        query_point_num=args.query_num, radius=args.radius,
        decoder_cfg=dict(
            num_layers=6, num_semantic_queries=3, in_channels=32,
            d_model=256, num_heads=8, hidden_dim=1024,
            dropout=0.0, activation='gelu', fix_attention=True,
            objectness_flag=True, attn_mask=True))

    if args.native_ckpt:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    else:
        load_pretrained(model, args.checkpoint)
    if args.invariant_feat:
        model._invariant_feat = True
    model = model.cuda().eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,d}")

    # Run inference
    points_tensor = torch.from_numpy(points_np).float().cuda()

    t0 = time.time()
    result = model.predict(points_tensor, output_path=args.output)
    t_total = time.time() - t0

    # Summary
    sem = result['semantic_pred']
    ins = result['instance_pred']
    n_trees = len(np.unique(ins[ins >= 0]))
    print(f"\nInference time: {t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"Trees detected: {n_trees}")
    print(f"Semantic: ground={np.sum(sem == 0):,d}, wood={np.sum(sem == 1):,d}, leaf={np.sum(sem == 2):,d}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
