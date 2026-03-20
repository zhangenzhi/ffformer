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
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/tmp/ffformer_results')
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
    ('saving', 'Saving results', 90),
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


def _run_inference_background(task_id, input_path, suffix, subsample, max_points):
    """Run the full inference pipeline in a background thread."""
    try:
        # Read point cloud
        _update_task_progress(task_id, 'reading')
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

        # Subsample
        _update_task_progress(task_id, 'subsampling', progress=30)
        if subsample > 0:
            voxel_ids = np.floor(xyz / subsample).astype(np.int64)
            _, idx = np.unique(voxel_ids, axis=0, return_index=True)
            xyz = xyz[idx]

        if max_points > 0 and len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]

        n_processed = len(xyz)

        # Inference
        _update_task_progress(task_id, 'inferring', progress=50)
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

        # Save stats JSON
        stats_path = os.path.join(task_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        _update_task_progress(
            task_id, 'completed', progress=100,
            stats=stats,
            result_ply=output_ply,
            completed=time.time(),
        )

    except Exception as e:
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
            if k not in ('result_ply',)
        }
    return result


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
