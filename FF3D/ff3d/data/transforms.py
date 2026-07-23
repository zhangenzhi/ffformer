"""Data transforms: cylinder crop, grid sample, augmentation.

All transforms operate on dicts with numpy arrays (no mm* dependency).
"""
import numpy as np
import torch


def cylinder_crop(points, pts_instance_mask, pts_semantic_mask, radius):
    """Random cylinder crop from a point cloud.

    Returns dict with cropped points and masks, or None if empty.
    """
    center = points[np.random.randint(len(points))]
    dist_sq = np.sum(np.square(points[:, :2] - center[:2]), axis=1)
    choices = np.where(dist_sq < radius ** 2)[0]

    if len(choices) == 0:
        return None

    cropped_pts = points[choices].copy()

    # Instance mask: background (semantic==0) → instance=-1
    instance_mask = pts_semantic_mask != 0  # bool: True = foreground
    inst = pts_instance_mask.copy()
    inst[~instance_mask] = -1
    original_inst = inst.copy()

    inst = inst[choices]
    instance_mask_cropped = instance_mask[choices]

    # Remap instance IDs to continuous
    idxs = np.unique(inst)
    mapping = np.zeros(np.max(idxs) + 2, dtype=int)
    new_idxs = np.arange(len(idxs))
    if idxs[0] == -1:
        mapping[idxs] = new_idxs - 1
    else:
        mapping[idxs] = new_idxs
    inst = mapping[inst]

    # Vote labels: offset from each point to its instance center (vectorized)
    vote_label = np.zeros((len(choices), 3), dtype=np.float32)
    ratio_inspoint = {}

    fg_idxs = idxs[idxs != -1]
    if len(fg_idxs) > 0:
        # Vectorized ratio: count per instance in original and cropped
        orig_counts = np.bincount(original_inst[original_inst >= 0],
                                  minlength=fg_idxs.max() + 1)
        crop_mapped = inst[inst >= 0]
        crop_counts = np.bincount(crop_mapped, minlength=len(fg_idxs))
        for mapped_id in range(len(fg_idxs)):
            orig_id = fg_idxs[mapped_id]
            ratio_inspoint[mapped_id] = (crop_counts[mapped_id] / orig_counts[orig_id]
                                         if orig_counts[orig_id] > 0 else 0)

        # Vectorized instance centers: min/max per instance from full cloud
        pts3 = points[:, :3]
        n_inst = len(fg_idxs)
        # Map original instance IDs → mapped IDs [0..n_inst); -1 for instances not in crop
        max_orig_id = int(original_inst.max()) + 1
        orig_to_mapped = np.full(max_orig_id, -1, dtype=np.int32)
        orig_to_mapped[fg_idxs] = np.arange(n_inst)

        fg_mask = (original_inst >= 0) & (orig_to_mapped[np.clip(original_inst, 0, max_orig_id - 1)] >= 0)
        fg_pts = pts3[fg_mask]
        fg_mapped = orig_to_mapped[original_inst[fg_mask]]

        inst_min = np.full((n_inst, 3), np.inf,  dtype=np.float32)
        inst_max = np.full((n_inst, 3), -np.inf, dtype=np.float32)
        np.minimum.at(inst_min, fg_mapped, fg_pts)
        np.maximum.at(inst_max, fg_mapped, fg_pts)
        inst_centers = 0.5 * (inst_min + inst_max)  # (n_inst, 3)

        # Assign vote labels: for each cropped foreground point, center - pos
        cyl_fg_mask = inst >= 0
        cyl_fg_ids = inst[cyl_fg_mask]  # mapped IDs in cropped cloud
        vote_label[cyl_fg_mask] = inst_centers[cyl_fg_ids] - cropped_pts[cyl_fg_mask, :3]

    sem = pts_semantic_mask[choices].copy()

    return {
        'points': cropped_pts,
        'pts_instance_mask': inst,
        'pts_semantic_mask': sem,
        'instance_mask': instance_mask_cropped,
        'vote_label': vote_label,
        'ratio_inspoint': ratio_inspoint,
    }


def grid_sample_np(sample, grid_size):
    """Voxel downsampling: keep one random point per voxel."""
    points = sample['points']
    voxel = np.floor(points / grid_size).astype(np.int32)

    # Linearize 3D voxel coords → 1D key (much faster than np.unique on 2D)
    voxel_min = voxel.min(axis=0)
    v = voxel - voxel_min
    voxel_max = v.max(axis=0).astype(np.int64) + 1
    hash_key = (v[:, 0].astype(np.int64) * voxel_max[1] * voxel_max[2]
                + v[:, 1].astype(np.int64) * voxel_max[2]
                + v[:, 2].astype(np.int64))

    # Random point per voxel: shuffle then take first occurrence in 1D sort
    perm = np.random.permutation(len(points))
    _, first_idx = np.unique(hash_key[perm], return_index=True)
    choices = perm[first_idx]

    sample['points'] = points[choices]
    for key in ['pts_instance_mask', 'pts_semantic_mask', 'instance_mask', 'vote_label']:
        if key in sample and sample[key] is not None:
            sample[key] = sample[key][choices]

    # Remap instance IDs after downsampling
    inst = sample['pts_instance_mask']
    idxs = np.unique(inst)
    mapping = np.zeros(np.max(idxs) + 2, dtype=int)
    new_idxs = np.arange(len(idxs))
    if idxs[0] == -1:
        mapping[idxs] = new_idxs - 1
    else:
        mapping[idxs] = new_idxs
    sample['pts_instance_mask'] = mapping[inst]

    return sample


def point_sample(sample, max_points):
    """Random subsample if exceeding max_points."""
    N = len(sample['points'])
    if N <= max_points:
        return sample

    choices = np.random.choice(N, max_points, replace=False)
    sample['points'] = sample['points'][choices]
    for key in ['pts_instance_mask', 'pts_semantic_mask', 'instance_mask', 'vote_label']:
        if key in sample and sample[key] is not None:
            sample[key] = sample[key][choices]

    # Remap instance IDs
    inst = sample['pts_instance_mask']
    idxs = np.unique(inst)
    mapping = np.zeros(np.max(idxs) + 2, dtype=int)
    new_idxs = np.arange(len(idxs))
    if idxs[0] == -1:
        mapping[idxs] = new_idxs - 1
    else:
        mapping[idxs] = new_idxs
    sample['pts_instance_mask'] = mapping[inst]

    return sample


def random_flip(sample, p=0.5):
    """Random BEV-horizontal flip (matches mmdet3d RandomFlip3D horizontal: flips Y)."""
    if np.random.rand() > p:
        return sample
    sample['points'][:, 1] = -sample['points'][:, 1]
    if 'vote_label' in sample:
        sample['vote_label'][:, 1] = -sample['vote_label'][:, 1]
    return sample


def global_rot_scale_trans(sample, rot_range=(-np.pi, np.pi),
                           scale_range=(0.8, 1.2), trans_std=0.1):
    """Random rotation, scaling, and translation."""
    points = sample['points']

    # Rotation around Z axis
    angle = np.random.uniform(*rot_range)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0],
                     [sin_a, cos_a, 0],
                     [0, 0, 1]], dtype=np.float32)
    points[:, :3] = points[:, :3] @ rot.T

    # Scale
    scale = np.random.uniform(*scale_range)
    points[:, :3] *= scale

    # Translation
    trans = np.random.normal(0, trans_std, 3).astype(np.float32)
    points[:, :3] += trans

    sample['points'] = points

    # Vote labels also need rotation + scale (but not translation)
    if 'vote_label' in sample:
        sample['vote_label'] = sample['vote_label'] @ rot.T * scale

    return sample
