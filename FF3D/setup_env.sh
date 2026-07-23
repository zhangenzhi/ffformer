#!/bin/bash
#
# ForestFormer3D-pure environment setup
# Target: H100 (SM90), latest packages
#
# Run on login node:
#   bash setup_env.sh
#
# Current env (forest3D): torch=2.0.1+cu118, spconv=2.3.8, ME=0.5.4
# New env (ff3d):         torch=2.6+cu124,  spconv latest, ME latest
#
set -e

ENV_NAME="ff3d"
CUDA_TAG="cu124"  # CUDA 12.4, matches driver 595.58

echo "============================================"
echo " Creating environment: $ENV_NAME"
echo " CUDA: 12.4, GPU: H100 (SM90)"
echo "============================================"

# ── Step 1: Create conda env ──
conda create -n $ENV_NAME python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# ── Step 2: PyTorch (latest stable + CUDA 12.4) ──
echo ""
echo "=== [1/6] PyTorch ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_TAG}

# ── Step 3: spconv ──
echo ""
echo "=== [2/6] spconv ==="
pip install spconv-${CUDA_TAG}

# ── Step 4: MinkowskiEngine ──
echo ""
echo "=== [3/6] MinkowskiEngine ==="
# ME is tricky — try pip wheel first, then source build
conda install -n $ENV_NAME openblas-devel -c anaconda -y 2>/dev/null || true
pip install ninja
pip install MinkowskiEngine 2>/dev/null || {
    echo "  pip install failed, trying source build..."
    export CUDA_HOME=$(python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME or '/usr/local/cuda')")
    export MAX_JOBS=4
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine.git \
        --install-option="--blas=openblas" \
        --install-option="--force_cuda" 2>/dev/null || {
        echo "  Source build also failed. Will try alternative collation."
        echo "  (MinkowskiEngine is only used for batch_sparse_collate — can be replaced)"
    }
}

# ── Step 5: torch_scatter + torch_cluster ──
echo ""
echo "=== [4/6] torch_scatter + torch_cluster ==="
pip install torch_scatter torch_cluster 2>/dev/null || {
    # If pip wheels not available, install from PyG repo
    TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    pip install torch_scatter torch_cluster \
        -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html
}

# ── Step 6: Other dependencies ──
echo ""
echo "=== [5/6] Other dependencies ==="
pip install numpy scipy laspy "laspy[lazrs]" plyfile tqdm pyyaml

# ── Step 7: Flash Attention ──
echo ""
echo "=== [6/6] Flash Attention ==="
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "  flash-attn build failed (needs matching CUDA toolkit)."
    echo "  PyTorch SDPA will still use Flash Attention 2 on H100 automatically."
}

# ── Verify ──
echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}')

import spconv
print(f'spconv: {spconv.__version__}')

try:
    import MinkowskiEngine as ME
    print(f'MinkowskiEngine: {ME.__version__}')
except ImportError:
    print('MinkowskiEngine: NOT INSTALLED (need alternative collation)')

try:
    import torch_scatter
    print(f'torch_scatter: {torch_scatter.__version__}')
except ImportError:
    print('torch_scatter: NOT INSTALLED')

try:
    import torch_cluster
    print(f'torch_cluster: {torch_cluster.__version__}')
except ImportError:
    print('torch_cluster: NOT INSTALLED')

import numpy; print(f'numpy: {numpy.__version__}')
import laspy; print(f'laspy: {laspy.__version__}')

try:
    from flash_attn import flash_attn_func
    print(f'flash-attn: OK')
except ImportError:
    print('flash-attn: not installed (SDPA handles it)')

# Quick sanity: SDPA with flash backend
if torch.cuda.is_available():
    q = torch.randn(1, 8, 300, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, 40000, 32, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, 40000, 32, device='cuda', dtype=torch.float16)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        try:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            print(f'SDPA Flash Attention: OK (output shape {out.shape})')
        except Exception as e:
            print(f'SDPA Flash Attention: FAILED ({e})')
            # Fallback test
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print(f'SDPA mem-efficient fallback: OK')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate:  conda activate $ENV_NAME"
echo " Train:     python tools/train.py --data-root ..."
echo " Test:      python tools/test.py --checkpoint ..."
echo " Infer:     python tools/infer.py --input file.las --checkpoint ..."
echo "============================================"
