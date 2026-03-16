"""ForestFormer3D REST API Server.

FastAPI-based server for forest point cloud segmentation.

Endpoints:
    POST /predict          — Upload LAS/PLY, get segmentation results
    GET  /health           — Health check
    GET  /result/{task_id} — Download result PLY/JSON

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
tasks = {}  # task_id -> {status, result_path, stats, error, ...}


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

    Args:
        file: LAS, LAZ, or PLY point cloud file
        subsample: Voxel subsampling size in meters (0 = no subsampling)
        max_points: Maximum number of points (0 = no limit)

    Returns:
        JSON with task_id, stats, and download links
    """
    task_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ('.las', '.laz', '.ply'):
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .las, .laz, or .ply")

    # Save upload
    task_dir = os.path.join(RESULTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    input_path = os.path.join(task_dir, f'input{suffix}')

    with open(input_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    tasks[task_id] = {
        'status': 'processing',
        'filename': file.filename,
        'created': time.time(),
    }

    # Process
    try:
        # Read point cloud
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
        if subsample > 0:
            voxel_ids = np.floor(xyz / subsample).astype(np.int64)
            _, idx = np.unique(voxel_ids, axis=0, return_index=True)
            xyz = xyz[idx]

        if max_points > 0 and len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]

        n_processed = len(xyz)

        # Inference
        output_ply = os.path.join(task_dir, 'result.ply')
        engine = get_engine()
        t0 = time.time()
        result = engine.predict(xyz, output_ply_path=output_ply)
        inference_time = time.time() - t0

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

        tasks[task_id].update({
            'status': 'completed',
            'stats': stats,
            'result_ply': output_ply,
        })

        return JSONResponse({
            'task_id': task_id,
            'status': 'completed',
            'stats': stats,
            'download_url': f'/result/{task_id}/ply',
            'json_url': f'/result/{task_id}/json',
        })

    except Exception as e:
        tasks[task_id].update({'status': 'failed', 'error': str(e)})
        raise HTTPException(500, f"Inference failed: {e}")


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
    """List all tasks."""
    return {tid: {k: v for k, v in t.items() if k != 'result_ply'}
            for tid, t in tasks.items()}


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
