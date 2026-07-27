"""Operator-level inference profiler for ForestFormer3D.

Runs model.predict() under torch.profiler on a (optionally cropped) point cloud
and prints a per-operator CUDA-time breakdown plus a category summary
(attention / data-movement / spconv / mask / post). Self-contained — needs no
model instrumentation.

Usage (inside ffformer.sif, PYTHONPATH=<repo>/FF3D):
    python FF3D/tools/profile_inference.py \
        --input examples/sample_forest.laz \
        --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth
    # crop a dense 40 m chunk from a big tile and use fp16 decoder:
    python FF3D/tools/profile_inference.py --input area1.las --chunk 20 --fp16-decoder
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import laspy
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ff3d.model import ForestFormer3D
from ff3d.model.forestformer3d import load_pretrained

# CUDA-kernel-name substrings -> category (kernel view, device_type==CUDA)
_CATS = [
    ('matmul(attn+proj)', ['gemm', 'wmma', 'tensorop', 'cutlass', 'bmm', 'ampere', 'volta',
                           'SoftMax', 'softmax', 'fmha', 'flash', 'scaled_dot']),
    ('data-movement',     ['elementwise', 'Memcpy', 'CatArray', 'vectorized', 'copy',
                           'fill', 'direct_copy']),
    ('spconv',            ['cumm', 'Sparse', 'spconv', 'implicit_gemm']),
    ('mask/bitwise',      ['masked_fill', 'bitwise', 'Sigmoid', 'sigmoid', 'compare']),
    ('reduce/index',      ['reduce', 'unique', 'sort', 'scatter', 'gather', 'index',
                           'Sum', 'Mean', 'nonzero', 'arange', 'cumsum']),
]


def categorize(name):
    for cat, subs in _CATS:
        if any(s in name for s in subs):
            return cat
    return 'other'


def load_points(path, half):
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    if half:
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        m = (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
        x, y, z = x[m], y[m], z[m]
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    xyz[:, 0] -= xyz[:, 0].mean()
    xyz[:, 1] -= xyz[:, 1].mean()
    xyz[:, 2] -= xyz[:, 2].min()
    return torch.from_numpy(xyz.astype(np.float32)).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', default='work_dirs/clean_forestformer/epoch_3000_fix.pth')
    ap.add_argument('--chunk', type=float, default=None,
                    help='crop a centered 2*chunk metre square (dense stress test)')
    ap.add_argument('--rows', type=int, default=25)
    ap.add_argument('--fp16-decoder', action='store_true')
    args = ap.parse_args()

    if args.fp16_decoder:
        os.environ['FF_FP16_DECODER'] = '1'

    pts = load_points(args.input, args.chunk)
    area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
    print(f'{pts.shape[0]:,} pts, density {pts.shape[0] / area.item():.0f}/m2, '
          f'fp16_decoder={args.fp16_decoder}', flush=True)

    model = ForestFormer3D(decoder_cfg=dict(
        num_layers=6, num_semantic_queries=3, in_channels=32, d_model=256,
        num_heads=8, hidden_dim=1024, dropout=0.0, activation='gelu',
        fix_attention=True, objectness_flag=True, attn_mask=True))
    load_pretrained(model, args.checkpoint)
    model = model.cuda().eval()

    with torch.no_grad():                        # warm caches
        model.predict(pts)
    torch.cuda.synchronize()

    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            out = model.predict(pts)
        torch.cuda.synchronize()
    wall = time.time() - t0
    ins = out['instance_pred']
    print(f'profiled predict {wall:.1f}s, trees {len(np.unique(ins[ins >= 0]))}\n', flush=True)

    # Sum only device-side kernel events (device_type == CUDA); the CPU aten
    # dispatchers carry the same time and would double-count.
    ka = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    total = sum(e.self_cuda_time_total for e in ka) / 1000.0  # ms
    cats = {}
    for e in ka:
        cats.setdefault(categorize(e.key), 0.0)
        cats[categorize(e.key)] += e.self_cuda_time_total / 1000.0

    print('===== CATEGORY SUMMARY (self CUDA) =====', flush=True)
    for cat, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f'  {cat:<16}{ms:>9.1f} ms  {100 * ms / total:>5.1f}%', flush=True)
    print(f'  {"TOTAL":<16}{total:>9.1f} ms', flush=True)

    print(f'\n===== TOP {args.rows} OPS (self CUDA) =====', flush=True)
    for e in sorted(ka, key=lambda e: -e.self_cuda_time_total)[:args.rows]:
        ms = e.self_cuda_time_total / 1000.0
        print(f'  {e.key[:46]:<48}{ms:>8.1f} ms {100 * ms / total:>5.1f}%  '
              f'[{categorize(e.key)}] x{e.count}', flush=True)


if __name__ == '__main__':
    main()
