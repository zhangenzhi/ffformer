#!/usr/bin/env python3
"""Environment verification script for FFFormer / ForestFormer3D.
Run inside the Singularity container:
    singularity exec --nv --bind /lustre1/work/c30636/ffformer:/workspace ffformer.sif \
        python /workspace/test_env.py
"""
import sys
import os
sys.path.insert(0, os.environ.get("WORKSPACE", "/workspace"))
import mmcv_compat  # patch mmcv version before mmdet/mmseg/mmdet3d imports
import importlib

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []

def check(name, fn):
    try:
        info = fn()
        print(f"  {PASS} {name}: {info}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False))

# ── 1. Python version ────────────────────────────────────────────────
print("\n=== Python ===")
check("Python version", lambda: sys.version.split()[0])

# ── 2. CUDA / GPU ───────────────────────────────────────────────────
print("\n=== CUDA / GPU ===")
def _torch_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available (did you use --nv?)")
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
check("PyTorch + CUDA", _torch_cuda)

# ── 3. Core dependencies ────────────────────────────────────────────
print("\n=== Core packages ===")
packages = [
    ("numpy", None),
    ("scipy", None),
    ("numba", None),
    ("open3d", None),
    ("sklearn", "scikit-learn"),
    ("pandas", None),
    ("matplotlib", None),
    ("tensorboard", None),
    ("cv2", "opencv"),
    ("plyfile", None),
    ("laspy", None),
    ("trimesh", None),
]
for mod, label in packages:
    label = label or mod
    check(label, lambda m=mod: getattr(importlib.import_module(m), "__version__", "ok"))

# ── 4. OpenMMLab stack ──────────────────────────────────────────────
print("\n=== OpenMMLab ===")
for mod in ["mmcv", "mmengine", "mmdet", "mmsegmentation", "mmdet3d"]:
    def _check(m=mod):
        pkg = importlib.import_module(m if m != "mmsegmentation" else "mmseg")
        return getattr(pkg, "__version__", "imported ok")
    check(mod, _check)

# ── 5. Sparse / 3D backends ─────────────────────────────────────────
print("\n=== Sparse / 3D backends ===")

def _minkowski():
    import MinkowskiEngine as ME
    return ME.__version__
check("MinkowskiEngine", _minkowski)

def _torch_scatter():
    import torch_scatter
    return getattr(torch_scatter, "__version__", "imported ok")
check("torch-scatter", _torch_scatter)

def _torch_cluster():
    import torch_cluster
    return getattr(torch_cluster, "__version__", "imported ok")
check("torch-cluster", _torch_cluster)

def _spconv():
    import spconv
    return spconv.__version__
check("spconv", _spconv)

def _segmentator():
    import segmentator
    return "imported ok"
check("segmentator", _segmentator)

def _tpk():
    import torch_points_kernels
    return getattr(torch_points_kernels, "__version__", "imported ok")
check("torch-points-kernels", _tpk)

# ── 6. Project imports (oneformer3d) ─────────────────────────────────
print("\n=== Project: oneformer3d ===")

def _oneformer3d():
    import oneformer3d
    names = [n for n in dir(oneformer3d) if not n.startswith("_")]
    return f"{len(names)} exports"
check("oneformer3d package", _oneformer3d)

def _config_parse():
    from mmengine.config import Config
    cfg = Config.fromfile("/workspace/configs/oneformer3d_qs_radius16_qp300_2many.py")
    return f"model={cfg.model.type}"
check("config parse", _config_parse)

# ── 7. Quick CUDA kernel smoke test ─────────────────────────────────
print("\n=== CUDA kernel smoke test ===")

def _cuda_smoke():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU")
    x = torch.randn(100, 3, device="cuda")
    y = x @ x.T  # basic matmul
    assert y.shape == (100, 100)
    return f"matmul ok on {x.device}"
check("basic CUDA matmul", _cuda_smoke)

def _me_cuda_smoke():
    import torch
    import MinkowskiEngine as ME
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU")
    coords = torch.randint(0, 100, (50, 4)).int()
    coords[:, 0] = 0  # batch index
    feats = torch.randn(50, 16).float()
    x = ME.SparseTensor(coordinates=coords.cuda(), features=feats.cuda())
    return f"SparseTensor on {x.device}, shape={x.F.shape}"
check("MinkowskiEngine CUDA", _me_cuda_smoke)

def _spconv_cuda_smoke():
    import torch
    import spconv.pytorch as spconv
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU")
    spatial_shape = [128, 128, 128]
    indices = torch.randint(0, 128, (20, 4)).int().cuda()
    indices[:, 0] = 0
    features = torch.randn(20, 16).float().cuda()
    sp = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size=1)
    return f"SparseConvTensor ok, dense_shape={sp.dense().shape}"
check("spconv CUDA", _spconv_cuda_smoke)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
failed = [n for n, ok in results if not ok]
if failed:
    print(f"  {FAIL} {passed}/{total} passed. Failed: {', '.join(failed)}")
else:
    print(f"  {PASS} All {total}/{total} checks passed!")
print("=" * 50 + "\n")

sys.exit(0 if not failed else 1)
