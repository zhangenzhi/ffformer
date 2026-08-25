"""Grid-based utilities: voxel sampling and query selection."""
import os
import torch
import numpy as np


def fps_queries(feat, n_target):
    """ForestFormer3D BASELINE query selection — their exact implementation.

    FPS in the learned discriminative-EMBEDDING space (ISA-guided), via
    torch_cluster.fps, exactly as oneformer3d.py:
        fps(embed_logits[tree_indices], batch, ratio=query_point_num / N_tree)
    `feat` is the (N_tree, D) embedding at tree points. Do not micro-optimise.
    """
    from torch_cluster import fps as _tc_fps
    N = feat.size(0)
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=feat.device)
    feat = feat.float().contiguous()                   # torch_cluster kernel needs fp32
    ratio = min(n_target / N, 1.0)
    batch = torch.zeros(N, dtype=torch.long, device=feat.device)
    return _tc_fps(feat, batch, ratio=ratio)


def _part1by1(n):
    """Interleave the low 16 bits of n with zeros (Morton bit-spread)."""
    n = n & 0xFFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def _morton2d(gx, gy):
    """Z-order (Morton) code of non-negative int grid coords (<= 16 bits/axis)."""
    return _part1by1(gx) | (_part1by1(gy) << 1)


def adaptive_queries(tree_xyz, n_target, feat=None, pool_mult=3, min_grid=0.4):
    """Density-adaptive query selection: fast like grid, spread like FPS.

    Works in one of two 2-D coordinate systems:
      * spatial (feat=None): XY-only grid, per-cell canopy-top (max-z) rep.
      * EMBEDDING (feat given): PCA-project the learned discriminative embedding
        to 2-D and grid THERE — points of the same tree cluster in embedding
        space, so one-rep-per-cell ≈ one query per instance (ISA-guided, like the
        paper's embedding-FPS) but separates trees by feature, not just position.

    Either way: (1) fine grid → ~pool_mult*n_target candidate cells so dense
    clusters each get a candidate; (2) if the pool overflows the budget, thin by
    an even stride along a Morton (Z-order) curve — fully-vectorised, sync-free,
    no 300-step FPS loop.
    """
    N = tree_xyz.size(0)
    n_target = min(n_target, N)
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=tree_xyz.device)
    if feat is not None:
        # project embedding to its top-2 principal axes (cheap: DxD eigdecomp).
        # Force fp32 — autocast/AMP would push the matmul to half and eigh has no
        # half CUDA kernel.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            e = feat.float()
            e = e - e.mean(0, keepdim=True)
            cov = (e.t() @ e) / max(N - 1, 1)
            _, evecs = torch.linalg.eigh(cov)          # ascending eigenvalues
            coord = e @ evecs[:, -2:]                   # (N, 2) top-2 components
        floor_cell = 0.0                                # no metric floor in embed space
        rep_key = None                                  # no canopy-top; first-occurrence rep
    else:
        coord = tree_xyz[:, :2]
        floor_cell = min_grid
        rep_key = tree_xyz[:, 2]                         # canopy-top representative
    rng = (coord.max(0)[0] - coord.min(0)[0]).clamp(min=1e-6)
    target_cells = min(pool_mult * n_target, N)
    cell = max((rng[0] * rng[1] / target_cells).sqrt().item(), floor_cell)
    gc = torch.floor(coord / cell).to(torch.int64)
    uniq, inv = torch.unique(gc, return_inverse=True, dim=0)
    ncell = uniq.size(0)
    # per-cell representative: last write wins after sorting by rep_key
    order = torch.argsort(rep_key) if rep_key is not None \
        else torch.arange(N - 1, -1, -1, device=tree_xyz.device)
    best = torch.full((ncell,), -1, dtype=torch.long, device=tree_xyz.device)
    best[inv[order]] = order
    cand = best[best >= 0]
    if cand.numel() > n_target:
        cgc = gc[cand]
        gx = (cgc[:, 0] - cgc[:, 0].min())
        gy = (cgc[:, 1] - cgc[:, 1].min())
        mcode = _morton2d(gx, gy)
        msort = torch.argsort(mcode)
        pick = (torch.arange(n_target, device=tree_xyz.device).float()
                * (cand.numel() / n_target)).long().clamp_(max=cand.numel() - 1)
        cand = cand[msort][pick]
    return cand


def select_queries(tree_xyz, n_target, feat=None):
    """Dispatch query selection by FF_QUERY_MODE env (grid|fps|adaptive).

    feat: (N_tree, D) embedding at tree points — required for the 'fps' baseline
    (ForestFormer3D runs FPS in embedding space); ignored by grid/adaptive.
    """
    mode = os.environ.get('FF_QUERY_MODE', 'grid').lower()
    if mode == 'fps':
        return fps_queries(feat if feat is not None else tree_xyz, n_target)
    if mode == 'adaptive':                              # spatial (XY) adaptive
        return adaptive_queries(tree_xyz, n_target)
    if mode == 'adaptive_emb':                          # embedding-space adaptive
        return adaptive_queries(tree_xyz, n_target, feat=feat)
    return grid_subsample_queries(tree_xyz, n_target)


def grid_sample(points, indices, grid_size):
    """Voxel-downsample point cloud by averaging points in each voxel.

    Args:
        points: (N, 3) GPU tensor.
        indices: (N,) original point indices.
        grid_size: float, voxel size.

    Returns:
        vox_points: (M, 3) voxel centers.
        vox_indices: (M,) one representative index per voxel.
        inverse: (N,) mapping from input points to voxel indices [0, M).
    """
    voxel = torch.floor(points / grid_size).to(torch.int32)
    uniq, inverse = torch.unique(voxel, return_inverse=True, dim=0)
    M = uniq.size(0)

    ones = torch.ones_like(inverse, dtype=points.dtype)
    sum_xyz = torch.zeros((M, points.size(1)), device=points.device, dtype=points.dtype)
    cnt_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)
    sum_xyz.index_add_(0, inverse, points)
    cnt_xyz.index_add_(0, inverse, ones)
    vox_points = sum_xyz / cnt_xyz.unsqueeze(1)

    vox_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
    vox_indices.index_copy_(0, inverse, indices)

    return vox_points, vox_indices, inverse


def grid_subsample_queries(tree_xyz, n_target, min_grid=0.5):
    """Select spatially uniform query points via adaptive grid subsampling.

    Args:
        tree_xyz: (N_tree, 3) coordinates of tree voxels.
        n_target: target number of queries (e.g. 300).
        min_grid: minimum grid size in meters.

    Returns:
        selected_indices: (Q,) indices into tree_xyz, Q <= n_target.
    """
    n_target = min(n_target, tree_xyz.size(0))
    xyz_range = tree_xyz.max(0)[0] - tree_xyz.min(0)[0]
    vol = (xyz_range[0] * xyz_range[1] * xyz_range[2]).clamp(min=1.0)
    grid_q = max((vol / n_target).pow(1.0 / 3.0).item(), min_grid)

    grid_coords = torch.floor(tree_xyz / grid_q).to(torch.int32)
    _, inv_q = torch.unique(grid_coords, return_inverse=True, dim=0)
    n_cells = inv_q.max().item() + 1

    # Pick one point per cell (first occurrence via reverse scatter)
    cell_rep = torch.full((n_cells,), -1, dtype=torch.long, device=tree_xyz.device)
    arange_rev = torch.arange(tree_xyz.size(0) - 1, -1, -1, device=tree_xyz.device)
    cell_rep.scatter_(0, inv_q.flip(0), arange_rev)
    cell_rep = cell_rep[cell_rep >= 0]

    if cell_rep.size(0) > n_target:
        perm = torch.randperm(cell_rep.size(0), device=tree_xyz.device)[:n_target]
        cell_rep = cell_rep[perm]

    return cell_rep
