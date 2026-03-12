# Data Format

## Input Data

### PLY File Format
Input point clouds are PLY files with the following fields:
- `x`, `y`, `z`: 3D coordinates (float)
- `semantic_seg`: Semantic label per point (int, 1-indexed: 1=ground, 2=wood, 3=leaf)
- `treeID`: Instance/tree ID per point (int, 0=unannotated)

### Directory Layout
```
data/ForAINetV2/
├── train_val_data/          # Training and validation PLY files
│   ├── scene_001.ply
│   ├── scene_002.ply
│   └── ...
├── test_data/               # Test PLY files
│   ├── test_scene_001.ply
│   └── ...
├── meta_data/
│   ├── train_list.txt       # One scan name per line (without .ply)
│   ├── val_list.txt
│   ├── test_list.txt
│   └── test_list_initial.txt  # For multi-pass inference
└── forainetv2_instance_data/  # Generated preprocessed data
    ├── scene_001_vert.npy
    ├── scene_001_sem_label.npy
    ├── scene_001_ins_label.npy
    ├── scene_001_offsets.npy
    ├── scene_001_aligned_bbox.npy
    ├── scene_001_unaligned_bbox.npy
    ├── scene_001_axis_align_matrix.npy
    └── ...
```

## Preprocessing

1. Coordinates are normalized: X,Y centered at mean, Z shifted to min=0
2. Offsets (mean_x, mean_y, min_z) are saved for inverse transform
3. Semantic labels are converted to 0-indexed (subtract 1)
4. Background (ground) instance IDs are set to 0
5. Bounding boxes are extracted for non-background instances

## Output Format

Predictions are saved as PLY files with:
- Point coordinates (original space, offsets applied back)
- Predicted instance IDs per point
- Predicted semantic labels per point
