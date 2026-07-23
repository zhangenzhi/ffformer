"""Grid-based utilities: voxel sampling and query selection."""
import torch
import numpy as np


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
