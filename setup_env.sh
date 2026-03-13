#!/bin/bash
# FFFormer environment setup script for HPC server
# Target: CUDA 12.8 / H100
# Usage: bash setup_env.sh

set -e

ENV_NAME="ffformer"

echo "=== Step 1: Create conda environment ==="
conda create -n $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "=== Step 2: Install PyTorch 2.1.0 + CUDA 12.1 ==="
# PyTorch 2.1 supports H100 natively; cu121 is the closest stable build to CUDA 12.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Pin numpy and setuptools before OpenMMLab ==="
pip install "numpy<2" setuptools

echo "=== Step 4: Install OpenMMLab stack ==="
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmengine==0.7.3
pip install mmdet==3.0.0
pip install mmsegmentation==1.0.0
pip install --no-build-isolation git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

echo "=== Step 5: Install MinkowskiEngine ==="
conda install openblas-devel -c anaconda -y 2>/dev/null || true
# H100 = sm_90
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4
pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 -v --no-deps \
    --global-option="--blas=openblas" \
    --global-option="--force_cuda"

echo "=== Step 6: Install torch-scatter ==="
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "=== Step 7: Install torch-cluster ==="
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "=== Step 8: Install torch-points-kernels ==="
pip install torch-points-kernels==0.7.0

echo "=== Step 9: Install spconv ==="
pip install spconv-cu120==2.3.6

echo "=== Step 10: Install segmentator ==="
cd /tmp
rm -rf segmentator
git clone https://github.com/Karbo123/segmentator.git
cd segmentator/csrc
git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696
mkdir -p build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; libdir=sysconfig.get_config_var('LIBDIR'); print(os.path.join(libdir, 'libpython3.10.so'))") \
    -DCMAKE_INSTALL_PREFIX=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
make -j$(nproc)
make install
cd /tmp && rm -rf segmentator

echo "=== Step 11: Install remaining dependencies ==="
pip install \
    open3d==0.17.0 \
    plyfile==1.0.2 \
    laspy "laspy[lazrs]" \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    numpy==1.24.1 \
    pandas==2.0.1 \
    matplotlib==3.5.2 \
    tensorboard==2.15.1 \
    trimesh==3.21.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    terminaltables==3.1.10 \
    pycocotools==2.0.6 \
    Shapely==1.8.5 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    numba==0.57.0

echo "=== Step 12: Replace mmengine/mmdet3d files ==="
MMENGINE_PATH=$(python -c "import mmengine; import os; print(os.path.dirname(mmengine.__file__))")
MMDET3D_PATH=$(python -c "import mmdet3d; import os; print(os.path.dirname(mmdet3d.__file__))")
WORKSPACE=$(cd "$(dirname "$0")" && pwd)

if [ -d "$WORKSPACE/replace_mmdetection_files" ]; then
    echo "Applying patches from replace_mmdetection_files/"
    cp "$WORKSPACE/replace_mmdetection_files/loops.py" "$MMENGINE_PATH/runner/loops.py" 2>/dev/null && echo "  - patched loops.py" || echo "  - loops.py not found, skipping"
    cp "$WORKSPACE/replace_mmdetection_files/base_model.py" "$MMENGINE_PATH/model/base_model/base_model.py" 2>/dev/null && echo "  - patched base_model.py" || echo "  - base_model.py not found, skipping"
    cp "$WORKSPACE/replace_mmdetection_files/transforms_3d.py" "$MMDET3D_PATH/datasets/transforms/transforms_3d.py" 2>/dev/null && echo "  - patched transforms_3d.py" || echo "  - transforms_3d.py not found, skipping"
else
    echo "WARNING: replace_mmdetection_files/ not found. You may need to patch mmengine/mmdet3d manually."
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  conda activate $ENV_NAME"
echo "Train with:     PYTHONPATH=\$(pwd) CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir work_dirs/exp1"
