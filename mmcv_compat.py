"""Runtime patches for the FFFormer Singularity container.

1. Patch mmcv version string to bypass strict upper-bound checks in
   mmdet 3.0.0, mmseg 1.0.0, and mmdet3d 1.1.0.
2. Stub out lyft_dataset_sdk (unused optional mmdet3d dependency).

Import this module *before* importing mmdet / mmseg / mmdet3d / oneformer3d.
"""
import sys
import types

# ── 1. mmcv version compat ───────────────────────────────────────────
import mmcv

_real_version = mmcv.__version__
if _real_version == "2.1.0":
    mmcv.__version__ = "2.0.1"

# ── 2. Stub optional mmdet3d dependencies (not needed for forest data) ──
def _make_stub_tree(prefixes):
    """Register stub modules for all given dotted names."""
    for mod_name in prefixes:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            # Make any attribute access return None (for imported names)
            class _StubModule(types.ModuleType):
                def __getattr__(self, name):
                    return None
            stub.__class__ = _StubModule
            sys.modules[mod_name] = stub

_make_stub_tree([
    # lyft
    "lyft_dataset_sdk",
    "lyft_dataset_sdk.eval",
    "lyft_dataset_sdk.eval.detection",
    "lyft_dataset_sdk.eval.detection.mAP_evaluation",
    "lyft_dataset_sdk.lyftdataset",
    "lyft_dataset_sdk.utils",
    "lyft_dataset_sdk.utils.data_classes",
    "lyft_dataset_sdk.utils.geometry_utils",
    # nuscenes
    "nuscenes", "nuscenes.nuscenes",
    "nuscenes.eval", "nuscenes.eval.common",
    "nuscenes.eval.common.config", "nuscenes.eval.common.loaders",
    "nuscenes.eval.common.utils",
    "nuscenes.eval.detection",
    "nuscenes.eval.detection.config",
    "nuscenes.eval.detection.evaluate",
    "nuscenes.eval.detection.data_classes",
    "nuscenes.eval.detection.utils",
    "nuscenes.eval.detection.render",
    "nuscenes.eval.detection.algo",
    "nuscenes.eval.detection.constants",
    "nuscenes.eval.lidarseg",
    "nuscenes.eval.lidarseg.evaluate",
    "nuscenes.eval.lidarseg.utils",
    "nuscenes.utils", "nuscenes.utils.data_classes",
    "nuscenes.utils.geometry_utils",
    "nuscenes.utils.map_mask",
    "nuscenes.lidarseg",
    "nuscenes.lidarseg.lidarseg_utils",
    # waymo (another optional)
    "waymo_open_dataset",
])


def restore():
    """Restore the real mmcv version after all mmlab imports are done."""
    mmcv.__version__ = _real_version
