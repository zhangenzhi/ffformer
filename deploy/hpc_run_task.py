"""HPC-side inference worker — runs inside singularity on a PBS GPU node.

Standalone version of the server's tiled inference: no FastAPI/multiprocessing
dependencies. Progress is reported by atomically rewriting status.json in the
task directory; the K8s pod polls that file over SFTP.

Usage (inside ffformer.sif, PYTHONPATH=/workspace):
    python deploy/hpc_run_task.py \
        --input /workspace/deploy_jobs/<id>/input.las \
        --task-dir /workspace/deploy_jobs/<id> \
        --tile-size 100 --overlap 10
"""
import argparse
import json
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


class StatusReporter:
    """Write progress to status.json (atomic rename) + inference.log."""

    def __init__(self, task_dir):
        self.task_dir = task_dir
        self.status_path = os.path.join(task_dir, 'status.json')
        self.log_path = os.path.join(task_dir, 'inference.log')
        self.t_start = time.time()
        self._lock = threading.Lock()  # update() is called from the GPU sampler too
        self.state = {
            'step': 'starting', 'progress': 0, 'stats': {},
            'completed_tiles': 0, 'error': None, 'updated': time.time(),
        }
        self._flush()

    def update(self, step=None, progress=None, **extra):
        with self._lock:
            if step is not None:
                self.state['step'] = step
            if progress is not None:
                self.state['progress'] = progress
            self.state.update(extra)
            self.state['updated'] = time.time()
            self._flush()

    def log(self, message):
        elapsed = round(time.time() - self.t_start, 1)
        line = f"[{elapsed}s] {message}"
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def _flush(self):
        tmp = self.status_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.state, f)
        os.replace(tmp, self.status_path)


def _gpu_stats():
    """Capture compute-node GPU telemetry via nvidia-smi (no CUDA context)."""
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,'
             'memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            return None
        p = [x.strip() for x in r.stdout.strip().split(',')]
        return {
            'gpu_name': p[0],
            'gpu_util_pct': float(p[1]),
            'vram_reserved_gb': round(float(p[2]) / 1024, 2),
            'vram_total_gb': round(float(p[3]) / 1024, 1),
            'temp_c': float(p[4]),
            'power_w': float(p[5]),
        }
    except Exception:
        return None


def _read_ply_xyz(path):
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


def _save_result_ply(path, xyz, semantic, instance, scores):
    """Write a binary little-endian PLY. Fast + compact — ASCII np.savetxt is
    ~1-2h and ~30GB for a 600M-point cloud; binary is seconds and ~14GB."""
    N = len(xyz)
    rec = np.empty(N, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                             ('semantic_pred', '<i4'), ('instance_pred', '<i4'),
                             ('score', '<f4')])
    rec['x'] = xyz[:, 0]
    rec['y'] = xyz[:, 1]
    rec['z'] = xyz[:, 2]
    rec['semantic_pred'] = semantic
    rec['instance_pred'] = instance
    rec['score'] = scores
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int semantic_pred\nproperty int instance_pred\n"
        "property float score\nend_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        rec.tofile(f)


def _generate_tile_preview(ply_path, offsets, preview_path):
    """Read model output PLY (normalized coords), restore original coordinates."""
    mean_x, mean_y, zmin = offsets
    fields = []
    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break
        col = {name: idx for idx, name in enumerate(fields)}
        rows = [line.strip().split() for line in f]

    if not rows:
        return

    n = len(rows)
    xyz = np.zeros((n, 3), dtype=np.float64)
    sem = np.zeros(n, dtype=np.int64)
    inst = np.full(n, -1, dtype=np.int64)
    scores = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rows):
        xyz[i, 0] = float(r[col['x']]) + mean_x
        xyz[i, 1] = float(r[col['y']]) + mean_y
        xyz[i, 2] = float(r[col['z']]) + zmin
        if 'semantic_pred' in col:
            sem[i] = int(r[col['semantic_pred']])
        if 'instance_pred' in col:
            inst[i] = int(r[col['instance_pred']])
        if 'score' in col:
            scores[i] = float(r[col['score']])
    _save_result_ply(preview_path, xyz, sem, inst, scores)


def run(input_path, task_dir, tile_size, overlap, config_path, checkpoint_path):
    rep = StatusReporter(task_dir)
    suffix = os.path.splitext(input_path)[1].lower()

    # Sample GPU telemetry continuously in the background. Sampling only at tile
    # boundaries catches the GPU idle between tiles and reports 0% utilization;
    # a steady 2s poll reflects the real load during inference.
    node = os.environ.get('HOSTNAME') or os.uname().nodename
    _stop_hw = threading.Event()

    def _hw_sampler():
        while not _stop_hw.wait(2):
            s = _gpu_stats()
            if s:
                rep.update(hw={'node': node, **s})
    _hw_thread = threading.Thread(target=_hw_sampler, daemon=True)
    _hw_thread.start()

    try:
        # ── Step 1: Read ──
        rep.update('reading', 10)
        rep.log("Reading point cloud...")
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
        zmin = float(xyz[:, 2].min())
        dx, dy = xmax - xmin, ymax - ymin
        rep.log(f"{N:,} points, extent {dx:.0f}x{dy:.0f}m")

        # ── Step 2: Load model ONCE ──
        from deploy.inference_engine import ForestFormerEngine
        rep.log("Loading model...")
        engine = ForestFormerEngine(config_path=config_path,
                                    checkpoint_path=checkpoint_path)
        rep.log("Model loaded")

        # ── Step 3: Inference ──
        # The pure engine does full-scene sliding-window inference internally,
        # so external tiling is redundant for normal scenes and only wastes time
        # re-processing overlap regions. Feed the whole cloud in one predict();
        # tile only as a memory-safety fallback for very large clouds.
        WHOLE_MAX = int(os.environ.get('FF_WHOLE_SCENE_MAX', 4_000_000))
        PREVIEW_MAX = int(os.environ.get('FF_PREVIEW_MAX', 2_000_000))
        sem_pred = np.full(N, -1, dtype=np.int32)
        inst_pred = np.full(N, -1, dtype=np.int32)
        scores_arr = np.full(N, -1.0, dtype=np.float32)
        t_infer_start = time.time()

        if N <= WHOLE_MAX:
            # ── Whole-scene inference (default, fp16 AMP) ──
            base_stats = {'n_original': N, 'n_processed': N,
                          'n_tiles': 1, 'tile_size': tile_size, 'mode': 'whole'}
            rep.update('inferring', 20, stats=base_stats)
            rep.log(f"Whole-scene inference on {N:,} points...")

            _last_pct = [-1]

            def _cb(done, total):
                pct = int(100 * done / max(total, 1))
                if pct != _last_pct[0]:
                    _last_pct[0] = pct
                    rep.update('inferring', 20 + int(65 * done / max(total, 1)),
                               stats={**base_stats, 'regions_done': done,
                                      'regions_total': total})

            result = engine.predict(xyz, output_ply_path=None,
                                    global_zmin=zmin, progress_cb=_cb)
            sem_pred = np.asarray(result['semantic_pred'], dtype=np.int32)
            inst_pred = np.asarray(result['instance_pred'], dtype=np.int32)
            scores_arr = np.asarray(result['instance_scores'], dtype=np.float32)
        else:
            # ── Tiled fallback for very large clouds (bounds CPU accumulators) ──
            rep.update('splitting', 15)
            nx = max(1, int(np.ceil(dx / tile_size)))
            ny = max(1, int(np.ceil(dy / tile_size)))
            rep.log(f"Large cloud ({N:,} pts > {WHOLE_MAX:,}); tiling {nx}x{ny} "
                    f"({tile_size}m, overlap={overlap}m)")

            tiles = []
            for ix in range(nx):
                for iy in range(ny):
                    tx0 = xmin + ix * tile_size - overlap
                    tx1 = xmin + (ix + 1) * tile_size + overlap
                    ty0 = ymin + iy * tile_size - overlap
                    ty1 = ymin + (iy + 1) * tile_size + overlap
                    mask = ((xyz[:, 0] >= tx0) & (xyz[:, 0] <= tx1) &
                            (xyz[:, 1] >= ty0) & (xyz[:, 1] <= ty1))
                    n_pts = int(mask.sum())
                    if n_pts < 100:
                        continue
                    tiles.append({'global_idx': np.where(mask)[0], 'n_points': n_pts})

            n_tiles = len(tiles)
            rep.log(f"{n_tiles} non-empty tiles")
            base_stats = {'n_original': N, 'n_processed': N,
                          'n_tiles': n_tiles, 'tile_size': tile_size, 'mode': 'tiled'}
            rep.update('splitting', 18, stats=base_stats)

            max_instance_id = 0
            for i, tile in enumerate(tiles):
                progress = 20 + int(65 * i / n_tiles)
                rep.update('inferring', progress,
                           stats={**base_stats, 'current_tile': i + 1,
                                  'tile_points': tile['n_points']})
                rep.log(f"Tile {i+1}/{n_tiles}: {tile['n_points']:,} pts")

                tile_xyz = xyz[tile['global_idx']]
                # Per-tile previews are a UI nicety; skip them (and the per-tile
                # PLY write) for large tiles to avoid huge ASCII dumps.
                tile_ply = (os.path.join(task_dir, f'tile_{i}.ply')
                            if tile['n_points'] <= PREVIEW_MAX else None)
                try:
                    t0 = time.time()
                    result = engine.predict(tile_xyz, output_ply_path=tile_ply,
                                            global_zmin=zmin)
                    dt = round(time.time() - t0, 1)

                    tile_sem = result.get('semantic_pred')
                    tile_inst = result.get('instance_pred')
                    tile_scores = result.get('instance_scores')

                    if tile_inst is not None and tile_scores is not None:
                        # Vectorized score-based merge (a per-point Python loop
                        # here would be ~15M iters/tile on dense clouds).
                        gi = tile['global_idx']
                        ts = np.asarray(tile_scores, dtype=np.float32)
                        better = ts > scores_arr[gi]
                        sel = gi[better]
                        scores_arr[sel] = ts[better]
                        if tile_sem is not None:
                            sem_pred[sel] = np.asarray(tile_sem)[better].astype(np.int32)
                        ti = np.asarray(tile_inst)[better]
                        inst_pred[sel] = np.where(ti >= 0, ti + max_instance_id, -1).astype(np.int32)
                        tile_max = int(tile_inst.max()) if len(tile_inst) > 0 else 0
                        if tile_max >= 0:
                            max_instance_id += tile_max + 1

                    tile_trees = len(set(tile_inst[tile_inst >= 0].tolist())) if tile_inst is not None else 0
                    rep.log(f"Tile {i+1}/{n_tiles}: {tile_trees} trees in {dt}s")

                    if tile_ply:
                        offsets = [float(tile_xyz[:, 0].mean()),
                                   float(tile_xyz[:, 1].mean()), zmin]
                        try:
                            _generate_tile_preview(
                                tile_ply, offsets,
                                os.path.join(task_dir, f'tile_{i}_preview.ply'))
                        except Exception:
                            pass
                        if os.path.exists(tile_ply):
                            os.remove(tile_ply)

                    rep.update('inferring', progress, completed_tiles=i + 1)
                except Exception as e:
                    rep.log(f"Tile {i+1}/{n_tiles} failed: {e}")

        inference_time = round(time.time() - t_infer_start, 1)

        # ── Step 5: Merge + save ──
        rep.update('merging', 85)
        n_trees = len(set(inst_pred[inst_pred >= 0].tolist()))
        n_assigned = int((inst_pred >= 0).sum())
        rep.log(f"Merged: {n_trees} trees, {n_assigned:,}/{N:,} assigned")

        rep.update('saving', 88)
        output_ply = os.path.join(task_dir, 'result.ply')
        _save_result_ply(output_ply, xyz, sem_pred, inst_pred,
                         scores_arr.astype(np.float32))

        stats = {
            **base_stats,
            'inference_time_s': inference_time,
            'n_trees': n_trees,
            'n_assigned': n_assigned,
            'semantic': {
                'ground': int((sem_pred == 0).sum()),
                'wood': int((sem_pred == 1).sum()),
                'leaf': int((sem_pred == 2).sum()),
            },
        }
        rep.log(f"Done: {n_trees} trees found in {inference_time}s")

        # ── Step 6: Viewer octree ──
        rep.update('tiling', 92, stats=stats)
        rep.log("Building octree tiles for streaming viewer...")
        viewer_dir = os.path.join(task_dir, 'viewer')
        try:
            from tools.potree_convert import build_octree
            build_octree(output_ply, viewer_dir, node_budget=50_000, max_depth=8)
            stats['has_viewer'] = True
            rep.log("Octree tiles generated")
        except Exception as tile_err:
            stats['has_viewer'] = False
            rep.log(f"Tile generation failed (non-fatal): {tile_err}")

        with open(os.path.join(task_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        rep.update('completed', 100, stats=stats)
        rep.log("Task completed")
        return 0

    except Exception as e:
        import traceback
        rep.log(f"ERROR: {e}")
        rep.log(traceback.format_exc())
        rep.update('failed', 0, error=str(e))
        return 1
    finally:
        _stop_hw.set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--task-dir', required=True)
    p.add_argument('--tile-size', type=float, default=100)
    p.add_argument('--overlap', type=float, default=10)
    p.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'configs', 'jpeaks_test.py'))
    p.add_argument('--checkpoint', default=os.path.join(
        PROJECT_ROOT, 'work_dirs', 'clean_forestformer', 'epoch_3000_fix.pth'))
    args = p.parse_args()
    sys.exit(run(args.input, args.task_dir, args.tile_size, args.overlap,
                 args.config, args.checkpoint))


if __name__ == '__main__':
    main()
