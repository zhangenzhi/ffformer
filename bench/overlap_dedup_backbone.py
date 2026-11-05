"""Overlap-dedup backbone measurement: run the sparse-conv backbone ONCE per big
tile (invariant feature -> tile-size-independent), instead of once per 16 m sliding
window (~51x overlap). Measures the backbone half of overlap-dedup: time + (via
nvidia-smi dmon in the PBS wrapper) GPU utilisation. Decode stays per-window."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/lustre1/work/c30636/ffformer/FF3D')
import spconv.pytorch as spconv
from ff3d.model import ForestFormer3D


def load_las(path):
    import laspy
    las = laspy.read(path)
    pts = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    off = np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].min()])
    pts[:, 0] -= off[0]; pts[:, 1] -= off[1]; pts[:, 2] -= off[2]
    return pts.astype(np.float32), off

CKPT = os.environ.get('DEDUP_CKPT', 'work_dirs/b_scratch/epoch_2100.pth')
INPUT = os.environ.get('DEDUP_INPUT', '/lustre1/work/c30636/dataset/forest3d/lidar_area1.las')
TILE = float(os.environ.get('DEDUP_TILE', 100.0))

model = ForestFormer3D(
    in_channels=3, num_channels=32, voxel_size=0.2, query_point_num=300, radius=16,
    decoder_cfg=dict(num_layers=6, num_semantic_queries=3, in_channels=32, d_model=256,
                     num_heads=8, hidden_dim=1024, dropout=0.0, activation='gelu',
                     fix_attention=True, objectness_flag=True, attn_mask=True))
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model._invariant_feat = True
model = model.cuda().eval()
print(f"model loaded: {CKPT}", flush=True)

pts_np, _ = load_las(INPUT)
pts = torch.from_numpy(pts_np).float().cuda()
N = len(pts)
minxy = pts[:, :2].min(0)[0]
tid = torch.floor((pts[:, :2] - minxy) / TILE).long()
tid_flat = tid[:, 0] * 1_000_000 + tid[:, 1]
uniq = torch.unique(tid_flat)
print(f"{N:,} points, {len(uniq)} tiles of {TILE:.0f} m", flush=True)

torch.cuda.synchronize(); t0 = time.time()
nvox = 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for u in uniq:
        tp = pts[tid_flat == u]
        if len(tp) < 100:
            continue
        coords, feats, inv, spatial = model.collate([tp])
        x_sparse = spconv.SparseConvTensor(feats, coords, spatial, 1)
        fine, coarse = model.extract_feat(x_sparse, return_coarse=True)
        nvox += int(coords.shape[0])
torch.cuda.synchronize()
dt = time.time() - t0
print(f"TILE-ONCE BACKBONE: {dt:.1f}s over {len(uniq)} tiles, {nvox:,} fine voxels", flush=True)
print(f"  vs standard sliding-window backbone (~205s on this scene) = {205.0/dt:.0f}x", flush=True)
