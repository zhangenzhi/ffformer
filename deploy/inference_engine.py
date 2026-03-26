"""ForestFormer3D inference engine — standalone model wrapper.

Encapsulates the full pipeline: load checkpoint → accept point cloud →
run cylindrical-region inference → return predictions.

Can be used directly in Python or as the backend for the API server.
"""
import os
import sys
import time
import tempfile

import numpy as np
import torch

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Patch mmcv version before any mmlab imports
import mmcv_compat  # noqa: E402, F401

from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner


class ForestFormerEngine:
    """Stateful inference engine for ForestFormer3D.

    Usage:
        engine = ForestFormerEngine(
            config_path='configs/jpeaks_test.py',
            checkpoint_path='work_dirs/clean_forestformer/epoch_3000_fix.pth',
        )
        result = engine.predict(xyz_array)  # (N,3) float32
        # result = {
        #     'points': (N,3),
        #     'semantic_pred': (N,),     # 0=ground, 1=wood, 2=leaf
        #     'instance_pred': (N,),     # -1=unassigned, 0+ = tree id
        #     'instance_scores': (N,),   # confidence per point
        # }
    """

    def __init__(self, config_path=None, checkpoint_path=None, device='cuda:0'):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'configs', 'jpeaks_test.py')
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PROJECT_ROOT, 'work_dirs', 'clean_forestformer', 'epoch_3000_fix.pth')

        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self._runner = None
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load the model on first predict call."""
        if self._model is not None:
            return

        print(f'[Engine] Loading config: {self.config_path}')
        cfg = Config.fromfile(self.config_path)

        # Load checkpoint (no spconv permutation needed)
        print(f'[Engine] Loading checkpoint: {self.checkpoint_path}')
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            torch.save(checkpoint, tmp.name)
            cfg.load_from = tmp.name

        if cfg.model.get('test_cfg') is None:
            cfg.model.test_cfg = ConfigDict()

        cfg.work_dir = tempfile.mkdtemp(prefix='ffformer_')
        cfg.model.test_cfg['output_dir'] = cfg.work_dir

        # Create dummy dataset so Runner can initialize without real data
        self._dummy_dir = tempfile.mkdtemp(prefix='ffformer_dummy_')
        self._create_dummy_data(self._dummy_dir)
        cfg.test_dataloader.dataset.data_root = self._dummy_dir + '/'
        cfg.test_dataloader.dataset.ann_file = 'info.pkl'

        # Ensure project root stays in sys.path (mmengine Runner may reset it)
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        os.environ['PYTHONPATH'] = PROJECT_ROOT + ':' + os.environ.get('PYTHONPATH', '')

        runner = Runner.from_cfg(cfg)
        self._runner = runner
        self._model = runner.model
        self._model.eval()
        print(f'[Engine] Model loaded on {self.device}')

    @staticmethod
    def _create_dummy_data(tmpdir):
        """Create minimal dummy dataset files for Runner initialization."""
        import pickle
        scene_name = 'dummy_test'
        for subdir in ('points', 'semantic_mask', 'instance_mask'):
            os.makedirs(os.path.join(tmpdir, subdir), exist_ok=True)
        # 3 dummy points
        np.array([[0, 0, 0]], dtype=np.float32).tofile(
            os.path.join(tmpdir, 'points', f'{scene_name}.bin'))
        np.zeros(1, dtype=np.int64).tofile(
            os.path.join(tmpdir, 'semantic_mask', f'{scene_name}.bin'))
        np.zeros(1, dtype=np.int64).tofile(
            os.path.join(tmpdir, 'instance_mask', f'{scene_name}.bin'))
        info = {
            'metainfo': {'categories': {'ground': 0, 'wood': 1, 'leaf': 2},
                         'dataset': 'ForAINetV2', 'info_version': '1.1'},
            'data_list': [{
                'lidar_points': {'num_pts_feats': 3,
                                 'lidar_path': f'{scene_name}.bin'},
                'instances': [],
                'pts_semantic_mask_path': f'{scene_name}.bin',
                'pts_instance_mask_path': f'{scene_name}.bin',
                'axis_align_matrix': np.eye(4).tolist(),
            }],
        }
        with open(os.path.join(tmpdir, 'info.pkl'), 'wb') as f:
            pickle.dump(info, f)

    def predict(self, xyz, output_ply_path=None, global_zmin=None):
        """Run inference on a point cloud.

        Args:
            xyz: numpy array (N, 3) float32/float64, raw coordinates.
                 Will be normalized internally (center XY, shift Z).
            output_ply_path: optional path to save result PLY.
            global_zmin: if provided, use this as Z offset instead of tile's
                         local min_z. Matches tile_and_infer.py normalization.

        Returns:
            dict with keys: points, semantic_pred, instance_pred, instance_scores
        """
        self._ensure_loaded()

        xyz = np.asarray(xyz, dtype=np.float64)
        N = len(xyz)
        assert xyz.shape == (N, 3), f"Expected (N,3), got {xyz.shape}"

        # Normalize: center XY per tile, shift Z to global min
        mean_x, mean_y = xyz[:, 0].mean(), xyz[:, 1].mean()
        min_z = global_zmin if global_zmin is not None else xyz[:, 2].min()
        xyz_norm = xyz.copy()
        xyz_norm[:, 0] -= mean_x
        xyz_norm[:, 1] -= mean_y
        xyz_norm[:, 2] -= min_z
        xyz_norm = xyz_norm.astype(np.float32)

        # Prepare temporary data files
        import pickle
        tmpdir = tempfile.mkdtemp(prefix='ffformer_data_')
        scene_name = 'input_scene_test'

        # Write bin files
        os.makedirs(os.path.join(tmpdir, 'points'), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, 'semantic_mask'), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, 'instance_mask'), exist_ok=True)

        xyz_norm.tofile(os.path.join(tmpdir, 'points', f'{scene_name}.bin'))
        np.zeros(N, dtype=np.int64).tofile(
            os.path.join(tmpdir, 'semantic_mask', f'{scene_name}.bin'))
        np.zeros(N, dtype=np.int64).tofile(
            os.path.join(tmpdir, 'instance_mask', f'{scene_name}.bin'))

        # Write pkl
        info = {
            'metainfo': {'categories': {'ground': 0, 'wood': 1, 'leaf': 2},
                         'dataset': 'ForAINetV2', 'info_version': '1.1'},
            'data_list': [{
                'lidar_points': {'num_pts_feats': 3,
                                 'lidar_path': f'{scene_name}.bin'},
                'instances': [],
                'pts_semantic_mask_path': f'{scene_name}.bin',
                'pts_instance_mask_path': f'{scene_name}.bin',
                'axis_align_matrix': np.eye(4).tolist(),
            }],
        }
        pkl_path = os.path.join(tmpdir, 'info.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(info, f)

        # Reconfigure dataloader with actual data
        cfg = self._runner.cfg
        cfg.test_dataloader.dataset.data_root = tmpdir + '/'
        cfg.test_dataloader.dataset.ann_file = 'info.pkl'

        if output_ply_path:
            out_dir = os.path.dirname(output_ply_path) or '.'
            cfg.model.test_cfg['output_dir'] = out_dir

        # Rebuild test dataloader with new data (cfg changes don't affect existing loader)
        from mmengine.runner import Runner as _Runner
        self._runner._test_dataloader = _Runner.build_dataloader(cfg.test_dataloader)

        # Run
        print(f'[Engine] Running inference on {N:,} points ...')
        t0 = time.time()

        try:
            self._runner.test()
        except (IndexError, Exception) as e:
            # Evaluator may crash (no GT) but PLY is already saved
            if 'out of bounds' in str(e) or 'IndexError' in str(type(e).__name__):
                print(f'[Engine] Evaluator error (expected, no GT): {e}')
            else:
                raise

        dt = time.time() - t0
        print(f'[Engine] Inference done in {dt:.1f}s')

        # Parse output PLY
        result = {
            'points': xyz,
            'points_normalized': xyz_norm,
            'offsets': np.array([mean_x, mean_y, min_z]),
            'semantic_pred': None,
            'instance_pred': None,
            'instance_scores': None,
        }

        # Find output PLY
        out_dir = cfg.model.test_cfg.get('output_dir', cfg.work_dir)
        ply_candidates = [
            os.path.join(out_dir, f'{scene_name}.ply'),
            os.path.join(out_dir, 'input_scene_test.ply'),
        ]
        for ply_path in ply_candidates:
            if os.path.exists(ply_path):
                result.update(self._parse_output_ply(ply_path))
                if output_ply_path and ply_path != output_ply_path:
                    import shutil
                    shutil.copy2(ply_path, output_ply_path)
                break

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

        return result

    @staticmethod
    def _parse_output_ply(ply_path):
        """Parse the output PLY file into arrays."""
        fields = []
        with open(ply_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('property'):
                    fields.append(line.split()[-1])
                elif line == 'end_header':
                    break

            rows = []
            for line in f:
                rows.append(line.strip().split())

        col = {name: idx for idx, name in enumerate(fields)}
        n = len(rows)
        result = {}
        if 'semantic_pred' in col:
            result['semantic_pred'] = np.array(
                [int(rows[i][col['semantic_pred']]) for i in range(n)])
        if 'instance_pred' in col:
            result['instance_pred'] = np.array(
                [int(rows[i][col['instance_pred']]) for i in range(n)])
        if 'score' in col:
            result['instance_scores'] = np.array(
                [float(rows[i][col['score']]) for i in range(n)])
        return result


def predict_las(las_path, output_ply=None, subsample=0.05, max_points=None):
    """Convenience function: LAS file → predictions.

    Args:
        las_path: path to .las/.laz file
        output_ply: optional output PLY path
        subsample: voxel size for subsampling (meters), None to keep all
        max_points: maximum number of points

    Returns:
        dict with prediction arrays
    """
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
