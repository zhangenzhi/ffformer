"""ForestFormer3D REST API Server.

FastAPI-based server for forest point cloud segmentation.

Endpoints:
    POST /predict              — Upload LAS/PLY, returns task_id immediately
    GET  /task/{task_id}/status — Poll task progress
    GET  /task/{task_id}/files  — List task files (input + results)
    DELETE /task/{task_id}      — Delete task and its files
    GET  /health               — Health check
    GET  /result/{task_id}/ply  — Download result PLY
    GET  /result/{task_id}/json — Download result JSON stats
    GET  /files                 — List all task files
    GET  /tasks                 — List all tasks

Usage:
    uvicorn deploy.server:app --host 0.0.0.0 --port 8000

    # Or with auto-reload for development:
    uvicorn deploy.server:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import json
import uuid
import shutil
import tempfile
import time
import threading
import asyncio
import multiprocessing
from pathlib import Path

# Ensure project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Patch mmcv version BEFORE any mmlab imports (must be early)
import mmcv_compat  # noqa: E402, F401

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

# Use 'fork' (default on Linux). Safe because parent process never initializes CUDA
# — model loading happens only in the subprocess.
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # already set

app = FastAPI(
    title="ForestFormer3D API",
    description="Forest LiDAR point cloud instance & semantic segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
WORK_DIR = os.environ.get('WORK_DIR', os.path.join(PROJECT_ROOT, 'work_dirs'))
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(WORK_DIR, 'results'))
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Lazy-loaded engine (loads model on first request)
_engine = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the dashboard UI (no cache to always get latest code)."""
    html_path = os.path.join(DEPLOY_DIR, 'dashboard.html')
    with open(html_path, 'r') as f:
        content = f.read()
    return HTMLResponse(content=content, headers={
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
    })


def get_engine():
    global _engine
    if _engine is None:
        from deploy.inference_engine import ForestFormerEngine
        config = os.environ.get('CONFIG_PATH',
                                os.path.join(PROJECT_ROOT, 'configs', 'jpeaks_test.py'))
        ckpt = os.environ.get('CHECKPOINT_PATH',
                              os.path.join(PROJECT_ROOT, 'work_dirs',
                                           'clean_forestformer', 'epoch_3000_fix.pth'))
        _engine = ForestFormerEngine(config_path=config, checkpoint_path=ckpt)
    return _engine


# --- Task tracking (shared across processes via Manager) ---
_manager = multiprocessing.Manager()
tasks = _manager.dict()


PROGRESS_STEPS = [
    ('uploading', 'Uploading file', 0),
    ('reading', 'Reading point cloud', 10),
    ('splitting', 'Splitting into tiles', 15),
    ('inferring', 'Running inference', 20),
    ('merging', 'Merging results', 85),
    ('saving', 'Saving results', 88),
    ('tiling', 'Building viewer tiles', 92),
    ('completed', 'Completed', 100),
]

STEP_NAMES = {s[0]: s[1] for s in PROGRESS_STEPS}
STEP_PROGRESS = {s[0]: s[2] for s in PROGRESS_STEPS}


def _get_gpu_utilization():
    """Get real-time GPU/system utilization via nvidia-smi + psutil.

    IMPORTANT: Do NOT use torch.cuda.* here — it initializes CUDA in the parent
    process and breaks fork()-ed subprocess GPU access.
    """
    stats = {}
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,'
             'memory.used,memory.total',
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            stats['gpu_util_pct'] = float(parts[0])
            stats['mem_util_pct'] = float(parts[1])
            stats['temp_c'] = float(parts[2])
            stats['vram_reserved_gb'] = round(float(parts[3]) / 1024, 2)
            stats['vram_total_gb'] = round(float(parts[4]) / 1024, 1)
    except Exception:
        pass
    try:
        import psutil
        stats['ram_used_pct'] = round(psutil.virtual_memory().percent, 1)
        stats['cpu_used_pct'] = round(psutil.cpu_percent(interval=0), 1)
    except Exception:
        pass
    return stats


def _update_task_progress(task_id, step, progress=None, **extra):
    """Update task progress atomically (Manager dict needs full reassignment)."""
    if task_id not in tasks:
        return
    t = dict(tasks[task_id])  # copy from Manager proxy
    now = time.time()

    # Track per-step timing
    step_times = dict(t.get('step_times', {}))
    prev_step = t.get('step')
    if prev_step and prev_step != step and prev_step in step_times:
        # Close out previous step
        st = dict(step_times[prev_step])
        st['end'] = now
        st['duration'] = round(now - st['start'], 1)
        step_times[prev_step] = st
    if step not in step_times:
        step_times[step] = {'start': now, 'end': None, 'duration': None}
    t['step_times'] = step_times

    t['step'] = step
    t['step_label'] = STEP_NAMES.get(step, step)
    t['progress'] = progress if progress is not None else STEP_PROGRESS.get(step, 0)
    t['updated'] = now
    if step in ('completed', 'failed'):
        t['status'] = step
    else:
        t['status'] = 'processing'
    t.update(extra)
    tasks[task_id] = t  # atomic reassignment to Manager dict



def _log_task(task_id, t_start, message):
    """Append a timestamped log message to the task's log list."""
    elapsed = round(time.time() - t_start, 1)
    entry = f"[{elapsed}s] {message}"
    if task_id in tasks:
        t = dict(tasks[task_id])
        log = list(t.get('log', []))
        log.append(entry)
        t['log'] = log
        tasks[task_id] = t


def _voxel_subsample(xyz, subsample):
    """Voxel subsample a point cloud. Tries Open3D C++ (fastest), then numpy fallback."""
    # Open3D C++ implementation — 10x+ faster than numpy for large N
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd_down = pcd.voxel_down_sample(voxel_size=subsample)
        return np.asarray(pcd_down.points)
    except (ImportError, Exception):
        pass

    # Numpy fallback: 1D voxel hash + sort-based first-occurrence
    N = len(xyz)
    voxel_ids = np.floor(xyz / subsample).astype(np.int64)
    v_min = voxel_ids.min(axis=0)
    voxel_ids -= v_min
    dims = voxel_ids.max(axis=0) + 1
    flat = (voxel_ids[:, 0] * dims[1] * dims[2] +
            voxel_ids[:, 1] * dims[2] +
            voxel_ids[:, 2])
    sort_idx = flat.argsort()
    flat_sorted = flat[sort_idx]
    keep = np.empty(N, dtype=bool)
    keep[0] = True
    keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
    return xyz[sort_idx[keep]]


def _save_result_ply(path, xyz, semantic, instance, scores):
    """Write merged result to PLY file (vectorized, fast for millions of points)."""
    N = len(xyz)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int semantic_pred\nproperty int instance_pred\n"
        "property float score\nend_header\n"
    )
    # Build data array and write in one shot with np.savetxt
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




def _run_inference_background(tasks_proxy, task_id, input_path, suffix, tile_size, overlap):
    """Tile-based full-resolution inference — mirrors tools/tile_and_infer.py exactly.

    Each tile runs tools/test.py as a fresh subprocess (clean Runner/model each time).
    This matches the proven tile_and_infer.py pipeline.
    """
    global tasks
    tasks = tasks_proxy
    t_start = time.time()
    import pickle
    import subprocess as sp
    try:
        # ── Step 1: Read ──
        _update_task_progress(task_id, 'reading')
        _log_task(task_id, t_start, "Reading point cloud...")
        if suffix in ('.las', '.laz'):
            import laspy
            las = laspy.read(input_path)
            xyz = np.column_stack([
                np.array(las.x, dtype=np.float64),
                np.array(las.y, dtype=np.float64),
                np.array(las.z, dtype=np.float64),
            ])
            del las
        else:
            xyz = _read_ply_xyz(input_path)

        N = len(xyz)
        xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
        ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
        zmin = xyz[:, 2].min()
        dx, dy = xmax - xmin, ymax - ymin
        _log_task(task_id, t_start, f"{N:,} points, extent {dx:.0f}x{dy:.0f}m")

        # ── Step 2: Split into tiles (same as tools/tile_and_infer.py cmd_split) ──
        _update_task_progress(task_id, 'splitting', progress=15)
        task_dir = os.path.dirname(input_path)
        task_name = f'task_{task_id}'

        nx = max(1, int(np.ceil(dx / tile_size)))
        ny = max(1, int(np.ceil(dy / tile_size)))
        _log_task(task_id, t_start, f"Tile grid: {nx}x{ny} ({tile_size}m, overlap={overlap}m)")

        # Prepare data dirs (same paths as tile_and_infer.py)
        data_root = os.path.join(task_dir, 'data')
        pts_dir = os.path.join(data_root, 'points')
        sem_dir = os.path.join(data_root, 'semantic_mask')
        ins_dir = os.path.join(data_root, 'instance_mask')
        for d in [pts_dir, sem_dir, ins_dir]:
            os.makedirs(d, exist_ok=True)

        tiles_meta = []
        for ix in range(nx):
            for iy in range(ny):
                tx0 = xmin + ix * tile_size - overlap
                tx1 = xmin + (ix + 1) * tile_size + overlap
                ty0 = ymin + iy * tile_size - overlap
                ty1 = ymin + (iy + 1) * tile_size + overlap
                cx0 = xmin + ix * tile_size
                cx1 = min(xmin + (ix + 1) * tile_size, xmax)
                cy0 = ymin + iy * tile_size
                cy1 = min(ymin + (iy + 1) * tile_size, ymax)

                mask = ((xyz[:, 0] >= tx0) & (xyz[:, 0] <= tx1) &
                        (xyz[:, 1] >= ty0) & (xyz[:, 1] <= ty1))
                n_pts = int(mask.sum())
                if n_pts < 100:
                    continue

                tid = len(tiles_meta)
                tile_xyz = xyz[mask].copy()
                tile_global_idx = np.where(mask)[0]

                # Normalize (same as tile_and_infer.py)
                mean_x = tile_xyz[:, 0].mean()
                mean_y = tile_xyz[:, 1].mean()
                tile_xyz_norm = tile_xyz.copy()
                tile_xyz_norm[:, 0] -= mean_x
                tile_xyz_norm[:, 1] -= mean_y
                tile_xyz_norm[:, 2] -= zmin
                tile_xyz_norm = tile_xyz_norm.astype(np.float32)

                tile_name = f'{task_name}_tile{tid:03d}_test'
                tile_dir_i = os.path.join(task_dir, f'tile{tid:03d}')
                os.makedirs(tile_dir_i, exist_ok=True)

                # Write bin files
                tile_xyz_norm.tofile(os.path.join(pts_dir, f'{tile_name}.bin'))
                np.zeros(n_pts, dtype=np.int64).tofile(os.path.join(sem_dir, f'{tile_name}.bin'))
                np.zeros(n_pts, dtype=np.int64).tofile(os.path.join(ins_dir, f'{tile_name}.bin'))

                # Save global indices for merge
                np.save(os.path.join(tile_dir_i, 'global_idx.npy'), tile_global_idx)

                # Create pkl
                info = {
                    'metainfo': {'categories': {'ground': 0, 'wood': 1, 'leaf': 2},
                                 'dataset': 'ForAINetV2', 'info_version': '1.1'},
                    'data_list': [{
                        'lidar_points': {'num_pts_feats': 3, 'lidar_path': f'{tile_name}.bin'},
                        'instances': [],
                        'pts_semantic_mask_path': f'{tile_name}.bin',
                        'pts_instance_mask_path': f'{tile_name}.bin',
                        'axis_align_matrix': np.eye(4).tolist(),
                    }],
                }
                pkl_name = f'info_tile{tid:03d}.pkl'
                pkl_path = os.path.join(data_root, pkl_name)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(info, f)

                tiles_meta.append({
                    'tile_id': tid,
                    'tile_name': tile_name,
                    'n_points': n_pts,
                    'core_bounds': [float(cx0), float(cy0), float(cx1), float(cy1)],
                    'offsets': [float(mean_x), float(mean_y), float(zmin)],
                    'pkl': pkl_name,
                    'tile_dir': tile_dir_i,
                })

        n_tiles = len(tiles_meta)
        _log_task(task_id, t_start, f"{n_tiles} non-empty tiles")
        _update_task_progress(task_id, 'splitting', progress=18,
                              stats={'n_original': N, 'n_processed': N,
                                     'n_tiles': n_tiles, 'tile_size': tile_size})

        # ── Step 3: Infer each tile via tools/test.py subprocess ──
        config_path = os.environ.get('CONFIG_PATH',
                                     os.path.join(PROJECT_ROOT, 'configs', 'jpeaks_test.py'))
        ckpt_path = os.environ.get('CHECKPOINT_PATH',
                                   os.path.join(PROJECT_ROOT, 'work_dirs',
                                                'clean_forestformer', 'epoch_3000_fix.pth'))
        test_script = os.path.join(PROJECT_ROOT, 'tools', 'test.py')
        t_infer_start = time.time()

        for i, tile in enumerate(tiles_meta):
            progress = 20 + int(65 * i / n_tiles)
            _update_task_progress(task_id, 'inferring', progress=progress,
                                  stats={'n_original': N, 'n_processed': N,
                                         'n_tiles': n_tiles, 'current_tile': i + 1,
                                         'tile_points': tile['n_points']})
            _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles}: {tile['n_points']:,} pts")

            # Run tools/test.py as subprocess (fresh Runner each time)
            cmd = [
                sys.executable, test_script,
                config_path, ckpt_path,
                '--work-dir', tile['tile_dir'],
                '--cfg-options',
                f'test_dataloader.dataset.data_root={data_root}/',
                f'test_dataloader.dataset.ann_file={tile["pkl"]}',
            ]
            t0 = time.time()
            proc = sp.run(cmd, cwd=PROJECT_ROOT, timeout=7200,
                          capture_output=True, text=True)
            dt = round(time.time() - t0, 1)

            # Check for output PLY
            result_ply = os.path.join(tile['tile_dir'], f'{tile["tile_name"]}.ply')
            if os.path.exists(result_ply):
                _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles}: done in {dt}s")
            else:
                _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles}: no PLY output (exit={proc.returncode})")
                if proc.stderr:
                    _log_task(task_id, t_start, f"  stderr: {proc.stderr[-500:]}")

        inference_time = round(time.time() - t_infer_start, 1)

        # ── Step 4: Merge (same as tools/tile_and_infer.py cmd_merge) ──
        _update_task_progress(task_id, 'merging', progress=85)
        _log_task(task_id, t_start, "Merging tile results...")

        sem_pred = np.full(N, -1, dtype=np.int32)
        inst_pred = np.full(N, -1, dtype=np.int32)
        scores_arr = np.full(N, -1.0, dtype=np.float32)
        max_instance_id = 0

        for tile in tiles_meta:
            tid = tile['tile_id']
            result_ply = os.path.join(tile['tile_dir'], f'{tile["tile_name"]}.ply')
            if not os.path.exists(result_ply):
                _log_task(task_id, t_start, f"  Tile {tid}: no result, skip")
                continue

            global_idx = np.load(os.path.join(tile['tile_dir'], 'global_idx.npy'))

            # Parse PLY
            fields = []
            with open(result_ply, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('property'):
                        fields.append(line.split()[-1])
                    elif line == 'end_header':
                        break
                col = {name: idx for idx, name in enumerate(fields)}
                rows = []
                for line in f:
                    rows.append(line.strip().split())

            n = len(rows)
            assigned = 0
            for j in range(n):
                gi = global_idx[j]
                s = int(rows[j][col['semantic_pred']])
                inst = int(rows[j][col['instance_pred']])
                sc = float(rows[j][col['score']])

                if sc > scores_arr[gi]:
                    sem_pred[gi] = s
                    inst_pred[gi] = (inst + max_instance_id) if inst >= 0 else -1
                    scores_arr[gi] = sc
                    assigned += 1

            tile_max = max(int(rows[j][col['instance_pred']]) for j in range(n)) if n > 0 else 0
            if tile_max >= 0:
                max_instance_id += tile_max + 1
            _log_task(task_id, t_start, f"  Tile {tid}: {n:,} pts, assigned {assigned:,}")

        n_trees = len(set(inst_pred[inst_pred >= 0].tolist()))
        n_assigned = int((inst_pred >= 0).sum())
        _log_task(task_id, t_start, f"Merged: {n_trees} trees, {n_assigned:,}/{N:,} assigned")

        # ── Save ──
        _update_task_progress(task_id, 'saving', progress=88)
        output_ply = os.path.join(task_dir, 'result.ply')
        _save_result_ply(output_ply, xyz, sem_pred, inst_pred, scores_arr.astype(np.float32))

        stats = {
            'n_original': N,
            'n_processed': N,
            'n_tiles': n_tiles,
            'tile_size': tile_size,
            'inference_time_s': inference_time,
            'n_trees': n_trees,
            'n_assigned': n_assigned,
        }
        if sem_pred is not None:
            stats['semantic'] = {
                'ground': int((sem_pred == 0).sum()),
                'wood': int((sem_pred == 1).sum()),
                'leaf': int((sem_pred == 2).sum()),
            }

        _log_task(task_id, t_start, f"Done: {n_trees} trees found in {inference_time}s")
        _log_task(task_id, t_start, "Saving results...")

        # Save stats JSON
        stats_path = os.path.join(task_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Generate octree tiles for streaming visualization
        _log_task(task_id, t_start, "Building octree tiles for streaming viewer...")
        _update_task_progress(task_id, 'tiling', progress=92)
        viewer_dir = os.path.join(task_dir, 'viewer')
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from tools.potree_convert import build_octree
            build_octree(output_ply, viewer_dir, node_budget=50_000, max_depth=8)
            _log_task(task_id, t_start, "Octree tiles generated")
            stats['has_viewer'] = True
        except Exception as tile_err:
            _log_task(task_id, t_start, f"Tile generation failed (non-fatal): {tile_err}")
            stats['has_viewer'] = False

        # Save inference log
        log_path = os.path.join(task_dir, 'inference.log')
        with open(log_path, 'w') as f:
            f.write('\n'.join(tasks[task_id].get('log', [])) + '\n')

        _update_task_progress(
            task_id, 'completed', progress=100,
            stats=stats,
            result_ply=output_ply,
            completed=time.time(),
        )

    except Exception as e:
        _log_task(task_id, t_start, f"ERROR: {str(e)}")
        # Save log even on failure
        task_dir = os.path.dirname(input_path)
        log_path = os.path.join(task_dir, 'inference.log')
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(tasks[task_id].get('log', [])) + '\n')
        except Exception:
            pass
        _update_task_progress(task_id, 'failed', progress=0, error=str(e))


@app.get("/health")
def health():
    import platform
    info = {
        "status": "ok",
        "model_loaded": any(t.get('status') == 'processing' for t in tasks.values()),
        "gpu_available": False,
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    # IMPORTANT: Do NOT call torch.cuda.* here — it initializes CUDA in the parent
    # process, which breaks fork()-ed subprocess GPU access. Use nvidia-smi only.
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import subprocess
        # Query all GPU info via nvidia-smi (no CUDA context needed)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,count,memory.total,memory.used,memory.free,'
             'utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            mem_total = float(parts[2]) / 1024  # MiB -> GiB
            mem_used = float(parts[3]) / 1024
            # Get CUDA version from nvidia-smi header
            cuda_ver = ''
            smi_out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if smi_out.returncode == 0:
                for line in smi_out.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        cuda_ver = line.split('CUDA Version:')[1].strip().split()[0]
                        break
            info["gpu_available"] = True
            info["gpu"] = {
                "name": parts[0],
                "count": int(parts[1]),
                "memory_total_gb": round(mem_total, 1),
                "memory_reserved_gb": round(mem_used, 1),
                "memory_allocated_gb": round(mem_used, 1),
                "memory_used_pct": round(mem_used / mem_total * 100, 1) if mem_total else 0,
                "cuda_version": cuda_ver,
                "utilization_pct": float(parts[5]),
                "memory_util_pct": float(parts[6]),
                "temperature_c": float(parts[7]),
                "power_w": float(parts[8]),
                "power_limit_w": float(parts[9]),
            }
    except Exception:
        pass
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["system"]["ram_total_gb"] = round(mem.total / 1e9, 1)
        info["system"]["ram_used_gb"] = round(mem.used / 1e9, 1)
        info["system"]["ram_used_pct"] = round(mem.percent, 1)
        info["system"]["cpu_used_pct"] = round(psutil.cpu_percent(interval=0), 1)
    except ImportError:
        pass
    # Checkpoint info
    ckpt = os.environ.get('CHECKPOINT_PATH', '')
    if ckpt and os.path.isfile(ckpt):
        info["checkpoint"] = {
            "path": ckpt,
            "filename": os.path.basename(ckpt),
            "size_mb": round(os.path.getsize(ckpt) / 1e6, 1),
        }
    return info


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tile_size: float = 100,
    overlap: float = 10,
):
    """Upload a LAS/LAZ/PLY file for tree instance segmentation.

    Full-resolution tiled inference: splits into large tiles, runs the complete
    model pipeline on each tile, merges results by score competition.

    Args:
        file: LAS, LAZ, or PLY point cloud file
        tile_size: Tile size in meters (default 100m)
        overlap: Overlap between tiles in meters (default 10m)

    Returns:
        JSON with task_id and status URL
    """
    task_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ('.las', '.laz', '.ply'):
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .las, .laz, or .ply")

    # Create task entry immediately
    tasks[task_id] = {
        'status': 'processing',
        'step': 'uploading',
        'step_label': 'Uploading file',
        'progress': 0,
        'filename': file.filename,
        'created': time.time(),
        'updated': time.time(),
    }

    # Save upload
    task_dir = os.path.join(RESULTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    input_path = os.path.join(task_dir, f'input{suffix}')

    with open(input_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    _update_task_progress(task_id, 'uploading', progress=5)

    # Launch background process for inference (separate process = no GIL blocking API)
    proc = multiprocessing.Process(
        target=_run_inference_background,
        args=(tasks, task_id, input_path, suffix, tile_size, overlap),
        daemon=True,
    )
    proc.start()

    return JSONResponse({
        'task_id': task_id,
        'status': 'processing',
        'status_url': f'/task/{task_id}/status',
    })


@app.get("/task/{task_id}/status")
def task_status(task_id: str):
    """Get current task progress.

    Returns:
        JSON with status, step, step_label, progress (0-100), elapsed time, stats (if completed), error (if failed)
    """
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]
    now = time.time()
    result = {
        'task_id': task_id,
        'status': task['status'],
        'step': task.get('step', 'unknown'),
        'step_label': task.get('step_label', ''),
        'progress': task.get('progress', 0),
        'filename': task.get('filename', ''),
        'elapsed': round(now - task.get('created', now), 1),
        'step_times': task.get('step_times', {}),
    }

    # Include real-time GPU/system stats for active tasks
    if task['status'] == 'processing':
        result['hw'] = _get_gpu_utilization()
        result['stats'] = task.get('stats', {})

    if task['status'] == 'completed':
        result['stats'] = task.get('stats', {})
        result['download_url'] = f'/result/{task_id}/ply'
        result['json_url'] = f'/result/{task_id}/json'

    if task['status'] == 'failed':
        result['error'] = task.get('error', 'Unknown error')

    return JSONResponse(result)




@app.delete("/task/{task_id}")
def delete_task(task_id: str):
    """Delete a task and all its files."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    # Don't delete tasks that are currently processing
    if tasks[task_id]['status'] == 'processing':
        raise HTTPException(400, "Cannot delete a task that is still processing")

    # Remove files
    task_dir = os.path.join(RESULTS_DIR, task_id)
    if os.path.isdir(task_dir):
        shutil.rmtree(task_dir, ignore_errors=True)

    # Remove from tracking
    del tasks[task_id]

    return JSONResponse({'status': 'deleted', 'task_id': task_id})


@app.get("/files")
def list_all_files():
    """List all task files (inputs + results) with sizes and timestamps."""
    all_files = []
    for tid, task in tasks.items():
        task_dir = os.path.join(RESULTS_DIR, tid)
        if not os.path.isdir(task_dir):
            continue
        for fname in os.listdir(task_dir):
            fpath = os.path.join(task_dir, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                is_result = fname.startswith('result')
                all_files.append({
                    'task_id': tid,
                    'filename': fname,
                    'original_name': task.get('filename', ''),
                    'type': 'result' if is_result else 'input',
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'download_url': f'/result/{tid}/ply' if fname == 'result.ply' else None,
                    'status': task.get('status', 'unknown'),
                })
    # Sort newest first
    all_files.sort(key=lambda f: f['modified'], reverse=True)
    return JSONResponse(all_files)


@app.get("/viewer/{task_id}/metadata.json")
def viewer_metadata(task_id: str):
    """Serve octree metadata for streaming viewer."""
    meta_path = os.path.join(RESULTS_DIR, task_id, 'viewer', 'metadata.json')
    if not os.path.exists(meta_path):
        raise HTTPException(404, "Viewer tiles not available for this task")
    return FileResponse(meta_path, media_type='application/json')


@app.get("/viewer/{task_id}/tiles/{node_key}.bin")
def viewer_tile(task_id: str, node_key: str):
    """Serve a single octree tile binary."""
    # Sanitize node_key to prevent path traversal
    if not all(c in 'r01234567' for c in node_key):
        raise HTTPException(400, "Invalid tile key")
    tile_path = os.path.join(RESULTS_DIR, task_id, 'viewer', 'tiles', f'{node_key}.bin')
    if not os.path.exists(tile_path):
        raise HTTPException(404, "Tile not found")
    return FileResponse(tile_path, media_type='application/octet-stream')


@app.get("/result/{task_id}/ply")
def download_ply(task_id: str):
    """Download/stream result PLY file as text (for viewer) or binary."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    if task['status'] != 'completed':
        raise HTTPException(400, f"Task status: {task['status']}")
    # Return as text so the JS viewer can parse it
    return FileResponse(
        task['result_ply'],
        media_type='application/octet-stream',
        filename=f'{task_id}_result.ply')


@app.get("/result/{task_id}/json")
def download_json(task_id: str):
    """Get result as JSON (stats + per-point predictions)."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    task = tasks[task_id]
    if task['status'] != 'completed':
        raise HTTPException(400, f"Task status: {task['status']}")
    return JSONResponse({
        'task_id': task_id,
        'stats': task.get('stats', {}),
    })


@app.get("/tasks")
def list_tasks():
    """List all tasks with their current status and progress."""
    result = {}
    for tid, t in tasks.items():
        result[tid] = {
            k: v for k, v in t.items()
            if k not in ('result_ply', 'log')
        }
    return result


# --- Model management endpoints ---

@app.get("/models")
def list_models():
    """Scan for available model checkpoints (.pth files)."""
    current_ckpt = os.environ.get('CHECKPOINT_PATH', '')
    scan_dirs = [WORK_DIR, '/workspace/data/', '/weights/']
    # Also scan the directory containing the current checkpoint
    if current_ckpt:
        ckpt_dir = os.path.dirname(current_ckpt)
        if ckpt_dir and ckpt_dir not in scan_dirs:
            scan_dirs.append(ckpt_dir)

    models = []
    seen = set()
    for d in scan_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for fname in files:
                if fname.endswith('.pth'):
                    fpath = os.path.join(root, fname)
                    if fpath in seen:
                        continue
                    seen.add(fpath)
                    try:
                        stat = os.stat(fpath)
                        models.append({
                            'path': fpath,
                            'filename': fname,
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'current': (fpath == current_ckpt),
                        })
                    except OSError:
                        pass

    models.sort(key=lambda m: m['modified'], reverse=True)
    return JSONResponse(models)


from pydantic import BaseModel as _BaseModel

class _ModelSwitchRequest(_BaseModel):
    path: str

@app.post("/config/model")
def switch_model_endpoint(req: _ModelSwitchRequest):
    """Switch the active model checkpoint.

    JSON body: {path: "/workspace/data/epoch_3000_fix.pth"}
    """
    global _engine
    new_path = req.path
    if not os.path.isfile(new_path):
        raise HTTPException(404, f"Checkpoint not found: {new_path}")
    if not new_path.endswith('.pth'):
        raise HTTPException(400, "Path must point to a .pth file")
    os.environ['CHECKPOINT_PATH'] = new_path
    _engine = None  # Reset so next predict reloads with new checkpoint
    return JSONResponse({
        'status': 'ok',
        'checkpoint': new_path,
        'message': 'Model will be loaded on next inference request.',
    })


@app.get("/data")
def list_data():
    """List available LiDAR data files on the server."""
    scan_dirs = [WORK_DIR, '/workspace/data/']
    data_files = []
    seen = set()
    valid_ext = {'.las', '.laz', '.ply'}
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_ext:
                    fpath = os.path.join(root, fname)
                    if fpath in seen:
                        continue
                    seen.add(fpath)
                    try:
                        stat = os.stat(fpath)
                        data_files.append({
                            'path': fpath,
                            'filename': fname,
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'type': ext.lstrip('.'),
                        })
                    except OSError:
                        pass
    data_files.sort(key=lambda f: f['modified'], reverse=True)
    return JSONResponse(data_files)


@app.get("/outputs")
def list_outputs():
    """List existing inference output directories."""
    outputs = []
    if not os.path.isdir(RESULTS_DIR):
        return JSONResponse(outputs)

    for tid in os.listdir(RESULTS_DIR):
        task_dir = os.path.join(RESULTS_DIR, tid)
        if not os.path.isdir(task_dir):
            continue

        # Determine status from task tracking or files on disk
        task_info = tasks.get(tid, {})
        status = task_info.get('status', 'unknown')
        if status == 'unknown':
            if os.path.isfile(os.path.join(task_dir, 'result.ply')):
                status = 'completed'

        # Gather stats
        stats = task_info.get('stats', {})
        stats_file = os.path.join(task_dir, 'stats.json')
        if not stats and os.path.isfile(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception:
                pass

        # List files in directory
        dir_files = []
        for fname in os.listdir(task_dir):
            fpath = os.path.join(task_dir, fname)
            if os.path.isfile(fpath):
                fstat = os.stat(fpath)
                download_url = None
                if fname == 'result.ply':
                    download_url = f'/result/{tid}/ply'
                elif fname == 'stats.json':
                    download_url = f'/result/{tid}/json'
                elif fname == 'inference.log':
                    download_url = f'/task/{tid}/log'
                dir_files.append({
                    'name': fname,
                    'size': fstat.st_size,
                    'download_url': download_url,
                })

        outputs.append({
            'task_id': tid,
            'filename': task_info.get('filename', tid),
            'status': status,
            'stats': stats,
            'files': dir_files,
        })

    outputs.sort(key=lambda o: tasks.get(o['task_id'], {}).get('created', 0), reverse=True)
    return JSONResponse(outputs)


@app.get("/task/{task_id}/log")
def task_log(task_id: str):
    """Get the inference log for a task."""
    # First try in-memory log
    if task_id in tasks and tasks[task_id].get('log'):
        return PlainTextResponse('\n'.join(tasks[task_id]['log']))

    # Fall back to log file on disk
    log_path = os.path.join(RESULTS_DIR, task_id, 'inference.log')
    if os.path.isfile(log_path):
        with open(log_path, 'r') as f:
            return PlainTextResponse(f.read())

    raise HTTPException(404, "Log not found for this task")


class _DataInferenceRequest(_BaseModel):
    data_path: str
    tile_size: float = 100
    overlap: float = 10

@app.post("/predict/data")
def predict_data(req: _DataInferenceRequest):
    """Start inference on a server-side data file (no upload needed).

    JSON body: {data_path: "/workspace/data/scan.las", tile_size: 100, overlap: 10}
    """
    data_path = req.data_path
    if not os.path.isfile(data_path):
        raise HTTPException(404, f"Data file not found: {data_path}")

    suffix = os.path.splitext(data_path)[1].lower()
    if suffix not in ('.las', '.laz', '.ply'):
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .las, .laz, or .ply")

    task_id = str(uuid.uuid4())[:8]
    task_dir = os.path.join(RESULTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Symlink or copy to task dir
    input_path = os.path.join(task_dir, f'input{suffix}')
    try:
        os.symlink(data_path, input_path)
    except OSError:
        shutil.copy2(data_path, input_path)

    filename = os.path.basename(data_path)
    tasks[task_id] = {
        'status': 'processing',
        'step': 'reading',
        'step_label': 'Reading point cloud',
        'progress': 10,
        'filename': filename,
        'source_path': data_path,
        'created': time.time(),
        'updated': time.time(),
    }

    proc = multiprocessing.Process(
        target=_run_inference_background,
        args=(tasks, task_id, input_path, suffix, req.tile_size, req.overlap),
        daemon=True,
    )
    proc.start()

    return JSONResponse({
        'task_id': task_id,
        'status': 'processing',
        'filename': filename,
        'status_url': f'/task/{task_id}/status',
    })


def _read_ply_xyz(path):
    """Read xyz from a PLY file."""
    fields = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break
        col = {name: idx for idx, name in enumerate(fields)}
        rows = []
        for line in f:
            vals = line.strip().split()
            rows.append([float(vals[col['x']]), float(vals[col['y']]),
                         float(vals[col['z']])])
    return np.array(rows, dtype=np.float64)
