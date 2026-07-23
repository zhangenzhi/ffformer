"""FOR-instanceV2 dataset for training and validation."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import cylinder_crop, grid_sample_np, point_sample, random_flip, global_rot_scale_trans


class ForAINetV2Dataset(Dataset):
    """FOR-instanceV2 point cloud dataset.

    Each sample is a randomly cropped cylinder from one scene.

    Args:
        data_root: Root directory containing points/, instance_mask/, semantic_mask/.
        split_file: Path to train/val/test list txt file.
        radius: Cylinder crop radius (default 16).
        grid_size: Voxel downsampling size (default 0.2).
        max_points: Maximum points per sample (default 640000).
        augment: Whether to apply data augmentation (default True).
        num_classes: Number of semantic classes (default 3).
    """

    def __init__(self, data_root, split_file, radius=16, grid_size=0.2,
                 max_points=640000, augment=True, num_classes=3):
        self.data_root = data_root
        self.radius = radius
        self.grid_size = grid_size
        self.max_points = max_points
        self.augment = augment
        self.num_classes = num_classes

        with open(split_file) as f:
            self.scene_names = [line.strip() for line in f if line.strip()]

        # Preload all data into memory to avoid per-iter disk I/O
        print(f"Preloading {len(self.scene_names)} scenes into memory...")
        self._cache = []
        for name in self.scene_names:
            pts = np.fromfile(os.path.join(data_root, 'points', f'{name}.bin'),
                              dtype=np.float32).reshape(-1, 3)
            ins = np.fromfile(os.path.join(data_root, 'instance_mask', f'{name}.bin'),
                              dtype=np.int64)
            sem = np.fromfile(os.path.join(data_root, 'semantic_mask', f'{name}.bin'),
                              dtype=np.int64)
            self._cache.append((pts, ins, sem))
        print(f"  Loaded {sum(p.nbytes + i.nbytes + s.nbytes for p, i, s in self._cache) / 1e9:.1f} GB")

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        points, pts_instance_mask, pts_semantic_mask = self._cache[idx]

        # Cylinder crop
        sample = cylinder_crop(points, pts_instance_mask, pts_semantic_mask, self.radius)
        if sample is None:
            # Retry with different random center
            return self.__getitem__(np.random.randint(len(self)))

        # Grid sample (voxelization)
        sample = grid_sample_np(sample, self.grid_size)

        # Point sample (cap at max_points)
        sample = point_sample(sample, self.max_points)

        # Skip if too few instances
        if len(np.unique(sample['pts_instance_mask'])) < 2:
            return self.__getitem__(np.random.randint(len(self)))

        # Instance class mapping
        sample = self._inst_class_mapping(sample)

        # Data augmentation
        if self.augment:
            sample = random_flip(sample, p=0.5)
            sample = global_rot_scale_trans(sample, rot_range=(-np.pi, np.pi),
                                            scale_range=(0.8, 1.2), trans_std=0.1)

        # Convert to tensors
        return {
            'points': torch.from_numpy(sample['points']).float(),
            'pts_instance_mask': torch.from_numpy(sample['pts_instance_mask']).long(),
            'pts_semantic_mask': torch.from_numpy(sample['pts_semantic_mask']).long(),
            'gt_labels_3d': torch.from_numpy(sample['gt_labels_3d']).long(),
            'instance_mask': torch.from_numpy(sample['instance_mask']),
            'vote_label': torch.from_numpy(sample['vote_label']).float(),
            'ratio_inspoint': sample['ratio_inspoint'],
        }

    def _inst_class_mapping(self, sample):
        """Map instance labels to continuous IDs and compute gt_labels_3d."""
        pts_inst = np.array(sample['pts_instance_mask'])
        pts_sem = sample['pts_semantic_mask']

        # Mark ignore class
        pts_inst[pts_sem == self.num_classes] = -1
        pts_sem[pts_sem == self.num_classes] = -1

        # Remap instance IDs to continuous
        idxs = np.unique(pts_inst)
        mapping = np.zeros(np.max(idxs) + 2, dtype=int)
        new_idxs = np.arange(len(idxs))
        if idxs[0] == -1:
            mapping[idxs] = new_idxs - 1
            new_idxs = new_idxs[:-1]
        else:
            mapping[idxs] = new_idxs
        pts_inst = mapping[pts_inst]

        # Compute per-instance semantic label
        gt_labels = np.zeros(len(new_idxs), dtype=int)
        for inst in new_idxs:
            gt_labels[inst] = pts_sem[pts_inst == inst][0]

        sample['pts_instance_mask'] = pts_inst
        sample['pts_semantic_mask'] = pts_sem
        sample['gt_labels_3d'] = gt_labels
        return sample


def collate_fn(batch):
    """Custom collate: keep variable-size tensors as lists."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'points': [b['points'] for b in batch],
        'pts_instance_mask': [b['pts_instance_mask'] for b in batch],
        'pts_semantic_mask': [b['pts_semantic_mask'] for b in batch],
        'gt_labels_3d': [b['gt_labels_3d'] for b in batch],
        'instance_mask': [b['instance_mask'] for b in batch],
        'vote_label': [b['vote_label'] for b in batch],
        'ratio_inspoint': [b['ratio_inspoint'] for b in batch],
    }
