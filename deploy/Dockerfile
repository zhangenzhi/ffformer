# ForestFormer3D Cloud Deployment Image
# Mirrors the verified Singularity environment (ffformer.def) for Docker/cloud
#
# Build:
#   docker build -t forestformer3d:latest -f deploy/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v ./work_dirs/clean_forestformer:/weights:ro \
#     forestformer3d:latest

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=5

# ── System dependencies ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
    git ninja-build cmake wget curl \
    build-essential python3-dev python3-pip \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda (matches Singularity env) ─────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    rm -f /opt/conda/compiler_compat/ld

# ── PyTorch 2.1.0 + CUDA 12.1 ──────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Pin numpy + setuptools before OpenMMLab ─────────────────────────
RUN pip install --no-cache-dir --only-binary=:all: numpy==1.24.4 setuptools

# ── OpenMMLab stack (--no-deps to prevent dependency pollution) ─────
RUN pip install --no-cache-dir \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

RUN pip install --no-cache-dir --no-deps mmengine==0.7.3 && \
    pip install --no-cache-dir --no-deps mmdet==3.0.0 && \
    pip install --no-cache-dir --no-deps mmsegmentation==1.0.0 && \
    pip install --no-cache-dir --no-deps --no-build-isolation \
        git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

# ── Patch mmengine & mmdet3d ────────────────────────────────────────
COPY replace_mmdetection_files/ /tmp/patches/
RUN SITE_PACKAGES=$(pip show mmengine | grep "^Location:" | awk '{print $2}') && \
    cp /tmp/patches/loops.py      "$SITE_PACKAGES/mmengine/runner/loops.py" && \
    cp /tmp/patches/base_model.py "$SITE_PACKAGES/mmengine/model/base_model/base_model.py" && \
    cp /tmp/patches/transforms_3d.py "$SITE_PACKAGES/mmdet3d/datasets/transforms/transforms_3d.py" && \
    rm -rf /tmp/patches && \
    echo "Patched mmengine and mmdet3d successfully"

# ── MinkowskiEngine (H100 = sm_90) ──────────────────────────────────
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    for f in src/*.cu src/3rdparty/*.cuh; do \
        [ -f "$f" ] && sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/sort.h>\n#include <thrust/unique.h>\n#include <thrust/remove.h>\n#include <thrust/reduce.h>\n#include <thrust/iterator/zip_iterator.h>\n#include <thrust/tuple.h>\n#include <thrust/system_error.h>' "$f"; \
    done && \
    MAX_JOBS=4 FORCE_CUDA=1 BLAS=openblas \
    python setup.py install --blas=openblas --force_cuda && \
    cd / && rm -rf /tmp/MinkowskiEngine

# ── torch-scatter + torch-cluster ───────────────────────────────────
RUN pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html && \
    pip install --no-cache-dir \
    torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# ── numpy (re-pin) + scipy + numba ──────────────────────────────────
RUN pip install --no-cache-dir --only-binary=:all: numpy==1.24.4 && \
    pip install --no-cache-dir scipy==1.10.1 numba==0.57.0 llvmlite==0.40.0

# ── torch-points-kernels (from source with CUDA) ───────────────────
RUN cd /tmp && \
    git clone https://github.com/nicolas-chaulet/torch-points-kernels.git && \
    cd torch-points-kernels && \
    git checkout v0.7.0 && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace && \
    pip install --no-cache-dir --no-deps . && \
    cd / && rm -rf /tmp/torch-points-kernels

# ── spconv (cu120 forward-compatible with 12.8 driver) ──────────────
RUN pip install --no-cache-dir spconv-cu120==2.3.6

# ── segmentator (C++/CUDA from source) ──────────────────────────────
RUN git clone https://github.com/Karbo123/segmentator.git /tmp/segmentator && \
    cd /tmp/segmentator/csrc && \
    git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 && \
    sed -i '/^project(/a set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; libdir=sysconfig.get_config_var('LIBDIR'); print(os.path.join(libdir, 'libpython3.10.so'))") \
        -DCMAKE_INSTALL_PREFIX=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') && \
    make -j$(nproc) && make install && \
    SITE_PKG=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') && \
    rm -rf "$SITE_PKG/segmentator" && \
    mkdir -p "$SITE_PKG/segmentator/csrc/build" && \
    cp /tmp/segmentator/__init__.py /tmp/segmentator/main.py /tmp/segmentator/utils.py "$SITE_PKG/segmentator/" && \
    cp /tmp/segmentator/csrc/build/libsegmentator.so "$SITE_PKG/segmentator/csrc/build/" && \
    touch "$SITE_PKG/segmentator/csrc/__init__.py" "$SITE_PKG/segmentator/csrc/build/__init__.py" && \
    cd / && rm -rf /tmp/segmentator

# ── Remaining Python packages ───────────────────────────────────────
RUN pip install --no-cache-dir \
    open3d==0.17.0 \
    plyfile==1.0.2 \
    laspy "laspy[lazrs]" \
    scikit-learn==1.2.2 \
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
    opencv-python-headless==4.7.0.72

# ── API server dependencies ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.0 \
    python-multipart

# ── Application code ────────────────────────────────────────────────
WORKDIR /app

# Application code (rebuild: 2026-03-24-v2)
COPY oneformer3d/ /app/oneformer3d/
COPY configs/ /app/configs/
COPY tools/ /app/tools/
COPY deploy/ /app/deploy/
COPY mmcv_compat.py /app/

ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/weights/epoch_3000_fix.pth
ENV CONFIG_PATH=/app/configs/jpeaks_test.py
ENV RESULTS_DIR=/tmp/ffformer_results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "deploy.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
