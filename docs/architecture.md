# ForestFormer3D Architecture

## Overview

ForestFormer3D is a unified end-to-end framework for individual tree segmentation and semantic segmentation from forest LiDAR 3D point clouds. It builds on the OneFormer3D architecture with forest-specific adaptations.

## Pipeline

```
Input Point Cloud (PLY)
    │
    ▼
Preprocessing (Normalize, Cylinder Crop, Grid Sample)
    │
    ▼
SpConv U-Net Backbone (Sparse Convolution)
    │
    ▼
ForAINetv2 Query Decoder (6-layer Transformer)
    │
    ▼
Instance + Semantic Prediction Heads
    │
    ▼
Score-based Block Merging (Inference)
    │
    ▼
Output: Instance Masks + Semantic Labels
```

## Components

### 1. SpConv U-Net Backbone
- 5 resolution levels with increasing channels: [32, 64, 96, 128, 160]
- Sparse convolution for efficient 3D processing
- Returns multi-scale features

### 2. ForAINetv2 Query Decoder (X-Aware Query)
- 6 transformer layers with cross-attention and self-attention
- 256-dim hidden features, 8 attention heads, 1024-dim FFN
- Content-aware query generation from tree points via FPS
- ISA-guided query point selection

### 3. Loss Functions
- **Semantic**: S3DIS-style semantic criterion (weight=0.2)
- **Instance**: One-to-many matching with dice loss + mask loss
- **Discriminative**: Embedding-based loss for instance separation

### 4. Inference Strategy
- Cylindrical region processing for large-scale scenes
- Sequential cylinder scanning with configurable radius (default 16m)
- Score-based instance mask merging across overlapping regions
- Multi-pass "bluepoints" inference for dense forest scenes

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Voxel size | 0.2m |
| Cylinder radius | 16m |
| Query points | 300 |
| Max points per sample | 640,000 |
| Transformer layers | 6 |
| Hidden dim | 256 |
| Attention heads | 8 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.05) |
| Scheduler | PolyLR (power=0.9, 450K iters) |
| Training epochs | 3000 |
| Batch size | 2 |

## Semantic Classes

| ID | Class | Type |
|----|-------|------|
| 0 | ground | stuff |
| 1 | wood | thing |
| 2 | leaf | thing |
