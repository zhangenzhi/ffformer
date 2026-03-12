# FFFormer - ForestFormer3D Baseline

A unified framework for end-to-end segmentation of forest LiDAR 3D point clouds, based on [ForestFormer3D](https://bxiang233.github.io/FF3D/) (ICCV 2025 Oral).

This repository implements the baseline for individual tree instance segmentation and semantic segmentation from forest LiDAR data, using tree-structured tokenization for 3D forest point cloud compression and large-scale forest reconstruction.

## Project Structure

```
ffformer/
├── configs/                          # Model training configurations
│   └── oneformer3d_qs_radius16_qp300_2many.py
├── data/
│   └── ForAINetV2/                  # Dataset directory
│       ├── train_val_data/          # Training/validation PLY files
│       ├── test_data/               # Test PLY files
│       ├── meta_data/               # Train/val/test split lists
│       ├── batch_load_ForAINetV2_data.py
│       └── load_forainetv2_data.py
├── oneformer3d/                      # Core model implementation
│   ├── oneformer3d.py               # Main model (ForAINetV2OneFormer3D)
│   ├── spconv_unet.py              # SpConv U-Net backbone
│   ├── query_decoder.py            # Transformer query decoder
│   ├── instance_criterion.py       # Instance segmentation losses
│   ├── semantic_criterion.py       # Semantic segmentation losses
│   ├── transforms_3d.py            # Data augmentation transforms
│   └── ...
├── tools/                           # Training and inference scripts
│   ├── train.py
│   ├── test.py
│   ├── create_data_forainetv2.py
│   ├── fix_spconv_checkpoint.py
│   └── inference_bluepoint.sh
├── docs/                            # Documentation
├── Dockerfile                       # Docker environment setup
└── README.md
```

## Setup

### Docker (Recommended)

```bash
docker build -t ffformer-image .
docker run --gpus all --shm-size=128g -d -p 127.0.0.1:49211:22 \
  -v $(pwd):/workspace \
  --name ffformer-container ffformer-image
```

### Dependencies

Core requirements:
- PyTorch 1.13.1 + CUDA 11.6
- mmengine==0.7.3, mmdet==3.0.0, mmsegmentation==1.0.0, mmdetection3d
- mmcv==2.0.0
- spconv-cu116==2.3.6
- MinkowskiEngine
- torch-points-kernels==0.7.0
- torch-cluster, torch-scatter
- open3d, laspy

After installing dependencies, apply required patches:
```bash
# Fix torch-points-kernels
pip uninstall torch-points-kernels -y
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

# Replace mmengine/mmdet3d files
cp replace_mmdetection_files/loops.py <mmengine_path>/runner/
cp replace_mmdetection_files/base_model.py <mmengine_path>/model/base_model/
cp replace_mmdetection_files/transforms_3d.py <mmdet3d_path>/datasets/transforms/
```

## Data Preparation

1. Download FOR-instanceV2 dataset and place PLY files in `data/ForAINetV2/train_val_data/` and `data/ForAINetV2/test_data/`.

2. Create train/val/test split lists in `data/ForAINetV2/meta_data/`:
   - `train_list.txt`
   - `val_list.txt`
   - `test_list.txt`

3. Preprocess point clouds:
```bash
cd data/ForAINetV2
pip install laspy "laspy[lazrs]"
python batch_load_ForAINetV2_data.py
cd ../..
python tools/create_data_forainetv2.py forainetv2
```

## Training

```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  --work-dir work_dirs/forestformer_baseline
```

Hardware: Tested on single A100 GPU. Reduce `radius` in config for smaller VRAM.

## Inference

```bash
# Fix checkpoint weights (spconv format)
python tools/fix_spconv_checkpoint.py \
  --in-path work_dirs/.../epoch_N.pth \
  --out-path work_dirs/.../epoch_N_fix.pth

# Run inference
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/.../epoch_N_fix.pth

# Multi-pass inference for dense scenes
bash tools/inference_bluepoint.sh
```

## Model Architecture

- **Backbone**: SpConv U-Net with 5 resolution levels
- **Decoder**: ForAINetv2QueryDecoder with 6 transformer layers, 256-dim, 8 heads
- **Key innovations**:
  - ISA-guided query point selection via farthest point sampling
  - Score-based block merging for large-scene inference
  - One-to-many association mechanism for training
- **Classes**: ground (0), wood (1), leaf (2)
- **Input**: 3D point clouds, voxelized at 0.2m resolution, cylinder-cropped (radius=16m)

## Citation

```bibtex
@inproceedings{xiang2025forestformer3d,
  title={ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds},
  author={Xiang, Binbin and Wielgosz, Maciej and Puliti, Stefano and Kr{\'a}{\v{l}}, Kamil and Kr{\r{u}}{\v{c}}ek, Martin and Missarov, Azim and Astrup, Rasmus},
  booktitle={ICCV},
  year={2025}
}
```

## License

CC BY-NC 4.0 (inherited from OneFormer3D)
