FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# System dependencies
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev cmake \
    && apt-get install -y build-essential python3-dev python3-pip \
    && apt-get install -y --no-install-recommends libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Pin numpy and setuptools first (avoid numpy 2.x and missing pkg_resources)
RUN pip install numpy==1.24.1 "setuptools<70"

# Install OpenMMLab stack
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN pip install mmengine==0.7.3 mmdet==3.0.0 mmsegmentation==1.0.0
RUN pip install --no-build-isolation \
    git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61

# Install MinkowskiEngine (H100 = sm_90)
RUN apt-get update && apt-get -y install libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4 \
    pip install --no-build-isolation --no-deps \
    git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
    --global-option="--blas=openblas" \
    --global-option="--force_cuda"

# torch-scatter & torch-cluster (pre-built wheels)
RUN pip install torch-scatter torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# torch-points-kernels
RUN pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

# spconv for CUDA 12
RUN pip install spconv-cu120==2.3.6

# segmentator
RUN git clone https://github.com/Karbo123/segmentator.git /tmp/segmentator \
    && cd /tmp/segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.10.so')") \
        -DCMAKE_INSTALL_PREFIX=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    && make && make install \
    && rm -rf /tmp/segmentator

# Remaining Python packages
RUN pip install \
    open3d==0.17.0 \
    plyfile==1.0.2 \
    laspy "laspy[lazrs]" \
    scipy==1.10.1 \
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
    opencv-python==4.7.0.72 \
    numba==0.57.0

ENV PYTHONPATH=/workspace
CMD ["bash"]
