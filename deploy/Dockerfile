# ForestFormer3D Deployment Image
# Based on the working Singularity environment
#
# Build:
#   docker build -t forestformer3d:latest -f deploy/Dockerfile .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v ./work_dirs/clean_forestformer:/weights \
#     forestformer3d:latest

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Python deps
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    setuptools \
    laspy[lazrs] \
    open3d==0.17.0 \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.0 \
    python-multipart \
    scipy \
    numba

# OpenMMLab stack
RUN pip3 install --no-cache-dir \
    mmengine==0.7.3 \
    mmcv==2.0.1 \
    mmdet==3.0.0 \
    mmsegmentation==1.0.0

RUN pip3 install --no-cache-dir \
    mmdet3d==1.1.0 --no-build-isolation

# Sparse convolution backends
RUN pip3 install --no-cache-dir \
    spconv-cu121==2.3.6

RUN pip3 install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# MinkowskiEngine (build from source for CUDA 12.1)
RUN pip3 install --no-cache-dir ninja && \
    pip3 install --no-cache-dir \
    -U git+https://github.com/NVIDIA/MinkowskiEngine.git \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda" || true

WORKDIR /app

# Copy project code
COPY oneformer3d/ /app/oneformer3d/
COPY configs/ /app/configs/
COPY tools/ /app/tools/
COPY deploy/ /app/deploy/
COPY mmcv_compat.py /app/
COPY replace_mmdetection_files/ /app/replace_mmdetection_files/

# Default checkpoint location (mount at runtime)
ENV CHECKPOINT_PATH=/weights/epoch_3000_fix.pth
ENV CONFIG_PATH=/app/configs/jpeaks_test.py
ENV RESULTS_DIR=/tmp/ffformer_results

EXPOSE 8000

CMD ["uvicorn", "deploy.server:app", "--host", "0.0.0.0", "--port", "8000"]
