"""ForestFormer3D inference engine — pure-PyTorch backend.

Drop-in replacement for the old mmengine/mmdet3d-based engine. It wraps the
pure-PyTorch reimplementation (the `ff3d` package under FF3D/) and keeps the
exact same public interface, so hpc_run_task.py and server.py need no changes:

    engine = ForestFormerEngine(checkpoint_path='.../epoch_3000_fix.pth')
    result = engine.predict(xyz, output_ply_path=..., global_zmin=...)
    # result = {
    #     'points': (N,3),           # original coordinates
    #     'offsets': (3,),           # [mean_x, mean_y, min_z] used to normalize
    #     'semantic_pred': (N,),     # 0=ground, 1=wood, 2=leaf, -1=unvoted
    #     'instance_pred': (N,),     # -1=unassigned, 0+ = tree id
    #     'instance_scores': (N,),   # confidence per point
    # }

The pure model does full-scene sliding-window inference internally, so a single
predict() call handles whatever point cloud it is given (a tile, or a whole
scene). The `config_path` argument is accepted for backward compatibility and
ignored — the pure model carries its own architecture config.
"""
import os
import sys
import time

import numpy as np
import torch

# Project root = repo root; the pure package lives under FF3D/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF3D_ROOT = os.path.join(PROJECT_ROOT, 'FF3D')
for _p in (PROJECT_ROOT, FF3D_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Architecture config — must match the checkpoint (same as tools/infer.py).
_DECODER_CFG = dict(
    num_layers=6, num_semantic_queries=3, in_channels=32,
    d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0,
    activation='gelu', fix_attention=True, objectness_flag=True, attn_mask=True)


def _write_result_ply(path, xyz, semantic, instance, scores):
    """Write an ASCII PLY matching hpc_run_task's expected schema.

    Coordinates are written as given (callers pass normalized coords so the
    downstream tile-preview restorer can re-add the offsets).
    """
    N = len(xyz)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int semantic_pred\nproperty int instance_pred\n"
        "property float score\nend_header\n"
    )
    data = np.column_stack([
        xyz,
        semantic.astype(np.float64),
        instance.astype(np.float64),
        scores.astype(np.float64),
    ])
    with open(path, 'w') as f:
        f.write(header)
    with open(path, 'ab') as f:
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %.4f')


class ForestFormerEngine:
    """Stateful inference engine backed by the pure-PyTorch ForestFormer3D."""

    def __init__(self, config_path=None, checkpoint_path=None, device='cuda:0'):
        # config_path kept for API compatibility with the old mm* engine; unused.
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PROJECT_ROOT, 'work_dirs', 'clean_forestformer', 'epoch_3000_fix.pth')
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load the model on first predict call."""
        if self._model is not None:
            return

        from ff3d.model import ForestFormer3D
        from ff3d.model.forestformer3d import load_pretrained

        print(f'[Engine] Building pure ForestFormer3D')
        model = ForestFormer3D(decoder_cfg=_DECODER_CFG)
        print(f'[Engine] Loading checkpoint: {self.checkpoint_path}')
        load_pretrained(model, self.checkpoint_path)
        n = sum(p.numel() for p in model.parameters())
        self._model = model.to(self.device).eval()
        print(f'[Engine] Model loaded on {self.device} ({n:,} params)')

    def predict(self, xyz, output_ply_path=None, global_zmin=None, progress_cb=None):
        """Run inference on a point cloud.

        Args:
            xyz: numpy array (N, 3), raw coordinates.
            output_ply_path: optional path to save a result PLY (normalized coords).
            global_zmin: if provided, use as the Z offset instead of the local
                         min-z (keeps tiles in a shared vertical frame).
            progress_cb: optional callable(done, total) invoked per sliding
                         window, for driving a progress bar during whole-scene
                         inference.

        Returns:
            dict with keys: points, offsets, semantic_pred, instance_pred,
            instance_scores.
        """
        self._ensure_loaded()

        xyz = np.asarray(xyz, dtype=np.float64)
        N = len(xyz)
        assert xyz.shape == (N, 3), f"Expected (N,3), got {xyz.shape}"

        # Normalize: center XY, shift Z to (global) min — same as tools/infer.py.
        mean_x, mean_y = xyz[:, 0].mean(), xyz[:, 1].mean()
        min_z = global_zmin if global_zmin is not None else xyz[:, 2].min()
        xyz_norm = xyz.copy()
        xyz_norm[:, 0] -= mean_x
        xyz_norm[:, 1] -= mean_y
        xyz_norm[:, 2] -= min_z
        xyz_norm = xyz_norm.astype(np.float32)

        points = torch.from_numpy(xyz_norm).float().to(self.device)

        print(f'[Engine] Running inference on {N:,} points ...')
        t0 = time.time()
        use_amp = os.environ.get('FF_USE_AMP', '1') not in ('0', 'false', 'False')
        with torch.no_grad():
            out = self._model.predict(points, progress_cb=progress_cb, use_amp=use_amp)
        dt = time.time() - t0
        print(f'[Engine] Inference done in {dt:.1f}s')

        semantic = np.asarray(out['semantic_pred'], dtype=np.int64)
        instance = np.asarray(out['instance_pred'], dtype=np.int64)
        scores = np.asarray(out['instance_scores'], dtype=np.float32)

        if output_ply_path:
            _write_result_ply(output_ply_path, xyz_norm, semantic, instance, scores)

        return {
            'points': xyz,
            'points_normalized': xyz_norm,
            'offsets': np.array([mean_x, mean_y, min_z]),
            'semantic_pred': semantic,
            'instance_pred': instance,
            'instance_scores': scores,
        }


def predict_las(las_path, output_ply=None, subsample=0.05, max_points=None):
    """Convenience: LAS file → predictions (mostly for local/manual testing)."""
    import laspy

    print(f'Reading {las_path} ...')
    las = laspy.read(las_path)
    xyz = np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ])
    print(f'  {len(xyz):,} points')

    if subsample is not None:
        voxel_ids = np.floor(xyz / subsample).astype(np.int64)
        _, idx = np.unique(voxel_ids, axis=0, return_index=True)
        xyz = xyz[idx]
        print(f'  After {subsample}m voxel subsampling: {len(xyz):,} points')

    if max_points and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        print(f'  After random subsampling: {len(xyz):,} points')

    engine = ForestFormerEngine()
    return engine.predict(xyz, output_ply_path=output_ply)
