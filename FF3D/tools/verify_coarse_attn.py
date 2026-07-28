"""Forward-verification + speed probe for the coarse-attention decoder path.

Runs model.predict() on a dense cropped chunk twice — baseline (fine attention)
and FF_COARSE_ATTN=1 (iterative cross-attn over a coarse U-Net scale, final mask
projected back to fine) — and reports wall time + tree count for each.

The coarse projections (input_proj_coarse / x_mask_coarse) are UNTRAINED, so the
coarse run's instance count is meaningless garbage — the point here is only to
confirm the shapes flow end-to-end without error and to measure the decoder-side
speedup the architecture buys. Accuracy comes after fine-tuning.

Usage (inside ffformer.sif, PYTHONPATH=<repo>/FF3D):
    python FF3D/tools/verify_coarse_attn.py --input area1.las --chunk 20
"""
import argparse, os, sys, time
import numpy as np, torch, laspy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ff3d.model import ForestFormer3D
from ff3d.model.forestformer3d import load_pretrained


def load_points(path, half):
    las = laspy.read(path)
    x = np.asarray(las.x, np.float64); y = np.asarray(las.y, np.float64); z = np.asarray(las.z, np.float64)
    if half:
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        m = (np.abs(x-cx) <= half) & (np.abs(y-cy) <= half); x, y, z = x[m], y[m], z[m]
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    xyz -= [xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].min()]
    return torch.from_numpy(xyz.astype(np.float32)).cuda()


def build():
    model = ForestFormer3D(decoder_cfg=dict(num_layers=6, num_semantic_queries=3, in_channels=32,
        d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0, activation='gelu',
        fix_attention=True, objectness_flag=True, attn_mask=True))
    load_pretrained(model, args.checkpoint)
    return model.cuda().eval()


def run(model, pts, tag):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.predict(pts)
    torch.cuda.synchronize(); wall = time.time() - t0
    ins = out['instance_pred']
    print(f'[{tag}] wall {wall:.1f}s  trees {len(np.unique(ins[ins>=0]))}', flush=True)
    return wall


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', default='work_dirs/clean_forestformer/epoch_3000_fix.pth')
    ap.add_argument('--chunk', type=float, default=20)
    ap.add_argument('--scale', type=int, default=2)
    args = ap.parse_args()

    pts = load_points(args.input, args.chunk)
    print(f'{pts.shape[0]:,} pts, coarse scale={args.scale}', flush=True)

    model = build()
    with torch.no_grad(): model.predict(pts)   # warm

    os.environ.pop('FF_COARSE_ATTN', None)
    base = run(model, pts, 'baseline(fine)')

    os.environ['FF_COARSE_ATTN'] = '1'
    os.environ['FF_COARSE_SCALE'] = str(args.scale)
    model2 = build()   # fresh so _coarse_attn re-reads env cleanly
    with torch.no_grad(): model2.predict(pts)  # warm
    coarse = run(model2, pts, 'coarse-attn')

    print(f'\nspeedup {base/coarse:.2f}x  ({base:.1f}s -> {coarse:.1f}s)', flush=True)
    print('NOTE: coarse tree count is garbage (untrained proj) — shapes/speed only', flush=True)
