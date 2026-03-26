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
    """Serve the dashboard UI."""
    html_path = os.path.join(DEPLOY_DIR, 'dashboard.html')
    with open(html_path, 'r') as f:
        return f.read()


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


# --- Task tracking ---
tasks = {}  # task_id -> {status, step, progress, result_path, stats, error, ...}


PROGRESS_STEPS = [
    ('uploading', 'Uploading file', 0),
    ('reading', 'Reading point cloud', 10),
    ('subsampling', 'Subsampling points', 25),
    ('splitting', 'Splitting into tiles', 30),
    ('inferring', 'Running inference', 35),
    ('merging', 'Merging tile results', 85),
    ('saving', 'Saving results', 88),
    ('tiling', 'Building viewer tiles', 92),
    ('completed', 'Completed', 100),
]

STEP_NAMES = {s[0]: s[1] for s in PROGRESS_STEPS}
STEP_PROGRESS = {s[0]: s[2] for s in PROGRESS_STEPS}


def _get_gpu_utilization():
    """Get real-time GPU utilization (lightweight, no GIL issues)."""
    stats = {}
    try:
        import torch
        if torch.cuda.is_available():
            stats['vram_allocated_gb'] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            stats['vram_reserved_gb'] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
            stats['vram_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu',
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            stats['gpu_util_pct'] = float(parts[0])
            stats['mem_util_pct'] = float(parts[1])
            stats['temp_c'] = float(parts[2])
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
    """Update task progress atomically."""
    if task_id not in tasks:
        return
    tasks[task_id]['step'] = step
    tasks[task_id]['step_label'] = STEP_NAMES.get(step, step)
    tasks[task_id]['progress'] = progress if progress is not None else STEP_PROGRESS.get(step, 0)
    tasks[task_id]['updated'] = time.time()
    if step in ('completed', 'failed'):
        tasks[task_id]['status'] = step
    else:
        tasks[task_id]['status'] = 'processing'
    tasks[task_id].update(extra)


def _log_task(task_id, t_start, message):
    """Append a timestamped log message to the task's log list."""
    elapsed = round(time.time() - t_start, 1)
    entry = f"[{elapsed}s] {message}"
    if task_id in tasks:
        tasks[task_id].setdefault('log', []).append(entry)


def _save_result_ply(path, xyz, semantic, instance, scores):
    """Write merged result to PLY file."""
    N = len(xyz)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int semantic_pred\n")
        f.write("property int instance_pred\n")
        f.write("property float score\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{semantic[i]} {instance[i]} {scores[i]:.4f}\n")


def _split_into_tiles(xyz, tile_size, overlap):
    """Split point cloud into spatial tiles with overlap.

    Returns list of (tile_indices, tile_center_xy) for each tile.
    """
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
    stride = tile_size - overlap

    tiles = []
    y = y_min
    while y < y_max:
        x = x_min
        while x < x_max:
            mask = (
                (xyz[:, 0] >= x) & (xyz[:, 0] < x + tile_size) &
                (xyz[:, 1] >= y) & (xyz[:, 1] < y + tile_size)
            )
            indices = np.where(mask)[0]
            if len(indices) > 0:
                center = np.array([x + tile_size / 2, y + tile_size / 2])
                tiles.append((indices, center))
            x += stride
        y += stride
    return tiles


def _merge_tile_results(xyz, tile_results, tiles):
    """Merge per-tile predictions. For overlapping points, the tile whose center
    is closest wins (reduces boundary artifacts)."""
    N = len(xyz)
    semantic = np.full(N, -1, dtype=np.int64)
    instance = np.full(N, -1, dtype=np.int64)
    scores = np.zeros(N, dtype=np.float32)
    best_dist = np.full(N, np.inf, dtype=np.float64)

    instance_offset = 0  # offset instance IDs to avoid collisions across tiles

    for (tile_idx, tile_center), result in zip(tiles, tile_results):
        if result is None:
            continue

        sem_pred = result.get('semantic_pred')
        inst_pred = result.get('instance_pred')
        inst_scores = result.get('instance_scores')

        # Distance of each point to tile center (XY only)
        dist = np.sqrt(
            (xyz[tile_idx, 0] - tile_center[0]) ** 2 +
            (xyz[tile_idx, 1] - tile_center[1]) ** 2
        )

        # Only overwrite points where this tile is closer to center
        closer = dist < best_dist[tile_idx]
        update_idx = tile_idx[closer]

        if sem_pred is not None:
            semantic[update_idx] = sem_pred[closer]

        if inst_pred is not None:
            # Offset instance IDs (but keep -1 as unassigned)
            inst_tile = inst_pred.copy()
            inst_tile[inst_tile >= 0] += instance_offset
            instance[update_idx] = inst_tile[closer]

        if inst_scores is not None:
            scores[update_idx] = inst_scores[closer]

        best_dist[tile_idx[closer]] = dist[closer]

        # Update offset for next tile
        if inst_pred is not None and len(inst_pred[inst_pred >= 0]) > 0:
            instance_offset = max(instance_offset, inst_pred.max() + 1 + instance_offset)

    return semantic, instance, scores


def _run_inference_background(task_id, input_path, suffix, subsample, tile_size):
    """Run the full inference pipeline in a background thread with tiled inference."""
    t_start = time.time()
    tasks[task_id].setdefault('log', [])
    overlap = tile_size * 0.2  # 20% overlap between tiles
    try:
        # Read point cloud
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
        else:  # .ply
            xyz = _read_ply_xyz(input_path)

        n_original = len(xyz)
        _log_task(task_id, t_start, f"{n_original:,} points loaded")

        # Voxel subsample (density reduction only, keeps full coverage)
        _update_task_progress(task_id, 'subsampling', progress=25)
        if subsample > 0:
            _log_task(task_id, t_start, f"Voxel subsampling at {subsample}m...")
            voxel_ids = np.floor(xyz / subsample).astype(np.int64)
            _, idx = np.unique(voxel_ids, axis=0, return_index=True)
            xyz = xyz[idx]

        n_subsampled = len(xyz)
        _log_task(task_id, t_start, f"{n_subsampled:,} points after voxel subsampling")

        # Split into spatial tiles
        _update_task_progress(task_id, 'splitting', progress=30)
        extent_x = xyz[:, 0].max() - xyz[:, 0].min()
        extent_y = xyz[:, 1].max() - xyz[:, 1].min()
        _log_task(task_id, t_start, f"Point cloud extent: {extent_x:.1f} x {extent_y:.1f} m")

        tiles = _split_into_tiles(xyz, tile_size, overlap)
        n_tiles = len(tiles)
        _log_task(task_id, t_start, f"Split into {n_tiles} tiles ({tile_size}m, {overlap:.0f}m overlap)")
        _update_task_progress(task_id, 'splitting', progress=32,
                              stats={'n_original': n_original, 'n_processed': n_subsampled,
                                     'n_tiles': n_tiles, 'tile_size': tile_size})

        # Run inference on each tile
        engine = get_engine()
        task_dir = os.path.dirname(input_path)
        tile_results = []
        t_infer_start = time.time()

        for i, (tile_idx, tile_center) in enumerate(tiles):
            tile_xyz = xyz[tile_idx]
            tile_n = len(tile_xyz)
            progress = 35 + int(50 * i / n_tiles)  # 35-85% across tiles

            _update_task_progress(task_id, 'inferring', progress=progress,
                                  stats={'n_original': n_original, 'n_processed': n_subsampled,
                                         'n_tiles': n_tiles, 'current_tile': i + 1,
                                         'tile_points': tile_n})
            _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles}: {tile_n:,} points "
                      f"at ({tile_center[0]:.0f}, {tile_center[1]:.0f})")

            try:
                tile_ply = os.path.join(task_dir, f'tile_{i}.ply')
                result = engine.predict(tile_xyz, output_ply_path=tile_ply)
                tile_results.append(result)

                tile_trees = 0
                if result.get('instance_pred') is not None:
                    tile_trees = len(set(result['instance_pred'][result['instance_pred'] >= 0]))
                _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles}: {tile_trees} trees found")
            except Exception as tile_err:
                _log_task(task_id, t_start, f"Tile {i+1}/{n_tiles} failed: {tile_err}")
                tile_results.append(None)

        inference_time = time.time() - t_infer_start

        # Merge tile results
        _update_task_progress(task_id, 'merging', progress=85)
        _log_task(task_id, t_start, "Merging tile results...")
        semantic, instance, scores = _merge_tile_results(xyz, tile_results, tiles)

        # Save merged result PLY
        _update_task_progress(task_id, 'saving', progress=88)
        output_ply = os.path.join(task_dir, 'result.ply')
        _save_result_ply(output_ply, xyz, semantic, instance, scores)

        # Clean up temporary tile PLY files
        for i in range(n_tiles):
            tile_ply = os.path.join(task_dir, f'tile_{i}.ply')
            if os.path.exists(tile_ply):
                os.remove(tile_ply)

        # Stats
        stats = {
            'n_original': n_original,
            'n_processed': n_subsampled,
            'n_tiles': n_tiles,
            'tile_size': tile_size,
            'inference_time_s': round(inference_time, 1),
        }
        if semantic is not None:
            stats['semantic'] = {
                'ground': int((semantic == 0).sum()),
                'wood': int((semantic == 1).sum()),
                'leaf': int((semantic == 2).sum()),
            }
        if instance is not None:
            n_trees = len(set(instance[instance >= 0].tolist()))
            stats['n_trees'] = n_trees
            stats['n_assigned'] = int((instance >= 0).sum())

        _log_task(task_id, t_start, f"Inference complete, {stats.get('n_trees', 0)} trees found across {n_tiles} tiles")
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
        "model_loaded": _engine is not None,
        "gpu_available": False,
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    try:
        import torch
        info["gpu_available"] = torch.cuda.is_available()
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_total = props.total_memory
            mem_reserved = torch.cuda.memory_reserved(0)
            mem_allocated = torch.cuda.memory_allocated(0)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory_total_gb": round(mem_total / 1e9, 1),
                "memory_reserved_gb": round(mem_reserved / 1e9, 1),
                "memory_allocated_gb": round(mem_allocated / 1e9, 1),
                "memory_used_pct": round(mem_reserved / mem_total * 100, 1) if mem_total else 0,
                "cuda_version": torch.version.cuda,
            }
            # GPU utilization via nvidia-smi (works even during inference)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                     '--format=csv,noheader,nounits', '-i', '0'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(',')]
                    info["gpu"]["utilization_pct"] = float(parts[0])
                    info["gpu"]["memory_util_pct"] = float(parts[1])
                    info["gpu"]["temperature_c"] = float(parts[2])
                    info["gpu"]["power_w"] = float(parts[3])
                    info["gpu"]["power_limit_w"] = float(parts[4])
            except Exception:
                pass
    except ImportError:
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
    subsample: float = 0.05,
    tile_size: float = 30.0,
):
    """Upload a LAS/LAZ/PLY file for tree instance segmentation.

    Returns immediately with a task_id. Poll /task/{task_id}/status for progress.
    Large point clouds are automatically split into spatial tiles for inference.

    Args:
        file: LAS, LAZ, or PLY point cloud file
        subsample: Voxel subsampling size in meters (0 = no subsampling)
        tile_size: Spatial tile size in meters for tiled inference (default 30m)

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

    # Launch background thread for inference
    thread = threading.Thread(
        target=_run_inference_background,
        args=(task_id, input_path, suffix, subsample, tile_size),
        daemon=True,
    )
    thread.start()

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
    result = {
        'task_id': task_id,
        'status': task['status'],
        'step': task.get('step', 'unknown'),
        'step_label': task.get('step_label', ''),
        'progress': task.get('progress', 0),
        'filename': task.get('filename', ''),
        'elapsed': round(time.time() - task.get('created', time.time()), 1),
    }

    # Include real-time GPU/system stats for active tasks
    if task['status'] == 'processing':
        result['hw'] = _get_gpu_utilization()

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
    subsample: float = 0.05
    tile_size: float = 30.0

@app.post("/predict/data")
def predict_data(req: _DataInferenceRequest):
    """Start inference on a server-side data file (no upload needed).

    JSON body: {data_path: "/workspace/data/scan.las", subsample: 0.05, tile_size: 30}
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

    thread = threading.Thread(
        target=_run_inference_background,
        args=(task_id, input_path, suffix, req.subsample, req.tile_size),
        daemon=True,
    )
    thread.start()

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
