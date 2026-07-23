# ForestFormer3D-pure: Project Structure Plan

## Goal
Pure PyTorch reimplementation of ForestFormer3D, no mm* dependencies.
Only external deps: torch, spconv, MinkowskiEngine (collate only), numpy, laspy, plyfile.

## Directory Structure

```
ForestFormer3D-pure/
├── configs/
│   └── default.yaml                 # Single YAML config (replaces .py config)
│
├── ff3d/                             # Main package
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── forestformer3d.py         # Main model class (~300 lines)
│   │   ├── backbone.py              # SpConvUNet (~250 lines, from spconv_unet.py)
│   │   ├── decoder.py               # Transformer decoder (~350 lines, from query_decoder.py)
│   │   └── heads.py                 # Embed + BiSemantic + prediction heads (~100 lines)
│   │
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── instance_loss.py         # mask BCE + dice + score loss (~200 lines)
│   │   ├── semantic_loss.py         # semantic criterion (~80 lines)
│   │   ├── discriminative_loss.py   # embedding clustering loss (~120 lines)
│   │   └── matcher.py              # One2Many matcher (~30 lines)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # ForAINetV2 dataset (~100 lines)
│   │   ├── transforms.py           # CylinderCrop, GridSample, augmentations (~300 lines)
│   │   ├── collate.py              # Sparse tensor collation (~50 lines)
│   │   └── preprocess.py           # LAS/PLY → .bin preprocessing (~100 lines)
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py              # F1, PQ, mIoU evaluation (~200 lines)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── nms.py                  # Matrix NMS (~50 lines)
│       ├── ply_io.py               # PLY read/write with RGB (~50 lines)
│       └── grid_sample.py          # Grid subsampling utility (~30 lines)
│
├── tools/
│   ├── train.py                     # Training script (~150 lines)
│   ├── test.py                      # Full evaluation (~80 lines)
│   ├── infer.py                     # Inference on LAS/PLY without labels (~100 lines)
│   ├── preprocess_data.py           # Data preprocessing pipeline (~80 lines)
│   └── visualize.py                 # Visualization tools (~100 lines)
│
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **No registry/config magic**: Direct Python imports + YAML config via dataclass
2. **Model in ~300 lines**: Extract only ForAINetV2OneFormer3D_XAwarequery, split into backbone/decoder/heads
3. **Training loop in plain PyTorch**: ~150 lines with epoch/iter tracking, checkpointing
4. **Data pipeline**: Standard PyTorch Dataset + DataLoader, transforms as simple functions
5. **Inference**: Direct model.predict() without Runner/hooks overhead
6. **All optimizations included**: grid_inverse, grid_subsampling, binary PLY, periodic empty_cache

## Migration Order

Phase 1: Model + Inference (can verify against pretrained weights)
  - ff3d/model/ (backbone, decoder, heads, main model)
  - ff3d/utils/ (nms, ply_io, grid_sample)
  - tools/infer.py
  - Load pretrained weights → run inference → compare F1

Phase 2: Data + Training
  - ff3d/data/ (dataset, transforms, collate)
  - ff3d/loss/ (all losses)
  - tools/train.py
  - Train from scratch → compare F1 with original

Phase 3: Evaluation + Polish
  - ff3d/eval/
  - tools/test.py, visualize.py
  - README, requirements.txt

## Estimated Total: ~2500 lines (vs original 9804 lines, 75% reduction)
