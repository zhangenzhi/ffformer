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
    ('subsampling', 'Subsampling points', 30),
    ('inferring', 'Running inference', 50),
    ('saving', 'Saving results', 88),
    ('tiling', 'Building viewer tiles', 92),
    ('completed', 'Completed', 100),
]

STEP_NAMES = {s[0]: s[1] for s in PROGRESS_STEPS}
STEP_PROGRESS = {s[0]: s[2] for s in PROGRESS_STEPS}


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


def _run_inference_background(task_id, input_path, suffix, subsample, max_points):
    """Run the full inference pipeline in a background thread."""
    t_start = time.time()
    tasks[task_id].setdefault('log', [])
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

        # Subsample
        _update_task_progress(task_id, 'subsampling', progress=30)
        if subsample > 0:
            _log_task(task_id, t_start, f"Subsampling with voxel size {subsample}m...")
            voxel_ids = np.floor(xyz / subsample).astype(np.int64)
            _, idx = np.unique(voxel_ids, axis=0, return_index=True)
            xyz = xyz[idx]

        if max_points > 0 and len(xyz) > max_points:
            _log_task(task_id, t_start, f"Subsampling to {max_points:,}...")
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]

        n_processed = len(xyz)
        _log_task(task_id, t_start, f"{n_processed:,} points after subsampling")

        # Inference
        _update_task_progress(task_id, 'inferring', progress=50)
        _log_task(task_id, t_start, "Running inference...")
        task_dir = os.path.dirname(input_path)
        output_ply = os.path.join(task_dir, 'result.ply')
        engine = get_engine()
        t0 = time.time()
        result = engine.predict(xyz, output_ply_path=output_ply)
        inference_time = time.time() - t0

        # Save results
        _update_task_progress(task_id, 'saving', progress=90)

        # Stats
        sem = result.get('semantic_pred')
        inst = result.get('instance_pred')
        stats = {
            'n_original': n_original,
            'n_processed': n_processed,
            'inference_time_s': round(inference_time, 1),
        }
        if sem is not None:
            stats['semantic'] = {
                'ground': int((sem == 0).sum()),
                'wood': int((sem == 1).sum()),
                'leaf': int((sem == 2).sum()),
            }
        if inst is not None:
            n_trees = len(set(inst[inst >= 0]))
            stats['n_trees'] = n_trees
            stats['n_assigned'] = int((inst >= 0).sum())

        _log_task(task_id, t_start, f"Inference complete, {stats.get('n_trees', 0)} trees found")
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
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "gpu_available": True,  # Will be checked at runtime
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    subsample: float = 0.05,
    max_points: int = 500000,
):
    """Upload a LAS/LAZ/PLY file for tree instance segmentation.

    Returns immediately with a task_id. Poll /task/{task_id}/status for progress.

    Args:
        file: LAS, LAZ, or PLY point cloud file
        subsample: Voxel subsampling size in meters (0 = no subsampling)
        max_points: Maximum number of points (0 = no limit)

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
        args=(task_id, input_path, suffix, subsample, max_points),
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
    max_points: int = 500000

@app.post("/predict/data")
def predict_data(req: _DataInferenceRequest):
    """Start inference on a server-side data file (no upload needed).

    JSON body: {data_path: "/workspace/data/scan.las", subsample: 0.05, max_points: 500000}
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
        args=(task_id, input_path, suffix, req.subsample, req.max_points),
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
