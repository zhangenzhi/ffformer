"""Push-mode HPC backend — the K8s pod drives inference on the HPC cluster.

Flow (runs in a background process, same entry signature as the local backend):
    1. SFTP the uploaded point cloud to {HPC_WORKDIR}/deploy_jobs/{task_id}/
    2. Generate a PBS job script and `qsub` it (singularity + GPU node)
    3. Poll qstat + the job's status.json; stream progress/log into the task dict;
       download tile previews incrementally so the dashboard's live view works
    4. On completion, download result_bundle.tar and extract into the local
       task dir so all existing download/viewer endpoints work unchanged.

Connectivity: pod → HPC login node over the campus-internal network
(verified ~113 MB/s via SFTP). Auth: dedicated ed25519 key mounted as a
K8s Secret (HPC_KEY_PATH).
"""
import json
import os
import posixpath
import stat as statmod
import tarfile
import time

# --- Configuration (env-driven, defaults match the current clusters) ---
HPC_HOST = os.environ.get('HPC_HOST', '172.31.20.1')
HPC_PORT = int(os.environ.get('HPC_PORT', '22'))
HPC_USER = os.environ.get('HPC_USER', 'c30746')
HPC_KEY_PATH = os.environ.get('HPC_KEY_PATH', '/tmp/ffformer_key')
HPC_WORKDIR = os.environ.get('HPC_WORKDIR', '/lustre1/work/c30636/ffformer')
HPC_SIF = os.environ.get('HPC_SIF', posixpath.join(HPC_WORKDIR, 'ffformer.sif'))
HPC_QUEUE = os.environ.get('HPC_QUEUE', 'c30636g')
HPC_SELECT = os.environ.get('HPC_SELECT', '1:ngpus=1')
HPC_WALLTIME = os.environ.get('HPC_WALLTIME', '06:00:00')
HPC_GROUP = os.environ.get('HPC_GROUP', 'c30636')
HPC_POLL_INTERVAL = float(os.environ.get('HPC_POLL_INTERVAL', '10'))
QSUB = os.environ.get('HPC_QSUB', '/opt/pbs/bin/qsub')
QSTAT = os.environ.get('HPC_QSTAT', '/opt/pbs/bin/qstat')

STEP_LABELS = {
    'uploading': 'Uploading file',
    'submitting': 'Submitting job to HPC',
    'queued': 'Queued on HPC GPU cluster',
    'reading': 'Reading point cloud',
    'splitting': 'Splitting into tiles',
    'inferring': 'Running inference (HPC GPU)',
    'merging': 'Merging results',
    'saving': 'Saving results',
    'tiling': 'Building viewer tiles',
    'downloading': 'Downloading results',
    'completed': 'Completed',
    'failed': 'Failed',
}

PBS_TEMPLATE = """#!/bin/bash
#PBS -q {queue}
#PBS -N ff_{task_id}
#PBS -l select={select}
#PBS -l walltime={walltime}
#PBS -W group_list={group}
#PBS -j oe
#PBS -o {rdir}/pbs_out.log

module load singularity 2>/dev/null || true
SING=$(command -v singularity || command -v apptainer)

# PBS exports GPU UUIDs in CUDA_VISIBLE_DEVICES; some libs parse it as int.
# The job's cgroup only exposes the assigned GPU, so index 0 is always correct.
export CUDA_VISIBLE_DEVICES=0

cd {workdir}
"$SING" exec --nv --bind {workdir}:/workspace --pwd /workspace {sif} \\
    python deploy/hpc_run_task.py \\
        --input /workspace/deploy_jobs/{task_id}/input{suffix} \\
        --task-dir /workspace/deploy_jobs/{task_id} \\
        --tile-size {tile_size} --overlap {overlap}
rc=$?

# Bundle results into one tar for a single fast SFTP download
cd {rdir}
files=""
for f in result.ply stats.json inference.log status.json; do
    [ -f "$f" ] && files="$files $f"
done
[ -d viewer ] && files="$files viewer"
for f in tile_*_preview.ply; do
    [ -f "$f" ] && files="$files $f"
done
[ -n "$files" ] && tar cf result_bundle.tar $files

exit $rc
"""


# --- Task-dict helpers (same semantics as server.py, kept import-free) ---

tasks = None  # set by run_hpc_inference from the Manager proxy


def _update(task_id, step, progress=None, **extra):
    if task_id not in tasks:
        return
    t = dict(tasks[task_id])
    now = time.time()
    step_times = dict(t.get('step_times', {}))
    prev = t.get('step')
    if prev and prev != step and prev in step_times:
        st = dict(step_times[prev])
        st['end'] = now
        st['duration'] = round(now - st['start'], 1)
        step_times[prev] = st
    if step not in step_times:
        step_times[step] = {'start': now, 'end': None, 'duration': None}
    t['step_times'] = step_times
    t['step'] = step
    t['step_label'] = STEP_LABELS.get(step, step)
    if progress is not None:
        t['progress'] = progress
    t['updated'] = now
    t['status'] = step if step in ('completed', 'failed') else 'processing'
    t.update(extra)
    tasks[task_id] = t


def _log(task_id, t_start, message):
    if task_id not in tasks:
        return
    entry = f"[{round(time.time() - t_start, 1)}s] {message}"
    t = dict(tasks[task_id])
    log = list(t.get('log', []))
    log.append(entry)
    t['log'] = log
    tasks[task_id] = t


# --- SSH helpers ---

def _connect(retries=3):
    import paramiko
    last_err = None
    for attempt in range(retries):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(HPC_HOST, port=HPC_PORT, username=HPC_USER,
                           key_filename=HPC_KEY_PATH, timeout=15,
                           banner_timeout=30)
            return client
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"SSH to {HPC_USER}@{HPC_HOST} failed: {last_err}")


def _exec(client, cmd, timeout=60):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    rc = out.channel.recv_exit_status()
    return rc, out.read().decode(errors='replace'), err.read().decode(errors='replace')


def _sftp_read(sftp, path):
    with sftp.open(path, 'r') as f:
        return f.read().decode(errors='replace')


def _job_state(client, job_id):
    """Return PBS job state char (Q/R/E/F/H/...) or '?' if unknown."""
    rc, out, _ = _exec(client, f"{QSTAT} -xf {job_id} 2>/dev/null | grep -m1 'job_state'")
    if rc == 0 and '=' in out:
        return out.split('=')[1].strip()[:1]
    return '?'


# --- Main entry (multiprocessing.Process target) ---

def run_hpc_inference(tasks_proxy, task_id, input_path, suffix, tile_size, overlap):
    global tasks
    tasks = tasks_proxy
    t_start = time.time()
    local_task_dir = os.path.dirname(input_path)
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', task_id)

    client = None
    try:
        _update(task_id, 'uploading', progress=3)
        _log(task_id, t_start, f"Connecting to HPC {HPC_USER}@{HPC_HOST}...")
        client = _connect()
        sftp = client.open_sftp()

        # ── Upload input ──
        _exec(client, f"mkdir -p {rdir}")
        size_mb = os.path.getsize(input_path) / 1e6
        _log(task_id, t_start, f"Uploading input ({size_mb:.1f} MB) to {rdir}...")
        t0 = time.time()
        sftp.put(input_path, posixpath.join(rdir, f'input{suffix}'))
        dt = time.time() - t0
        _log(task_id, t_start, f"Upload done ({size_mb / max(dt, 0.01):.0f} MB/s)")
        _update(task_id, 'uploading', progress=6)

        # ── Generate + submit PBS job ──
        _update(task_id, 'submitting', progress=7)
        script = PBS_TEMPLATE.format(
            queue=HPC_QUEUE, select=HPC_SELECT, walltime=HPC_WALLTIME,
            group=HPC_GROUP,
            workdir=HPC_WORKDIR, sif=HPC_SIF, rdir=rdir, task_id=task_id,
            suffix=suffix, tile_size=tile_size, overlap=overlap,
        )
        with sftp.open(posixpath.join(rdir, 'job.pbs'), 'w') as f:
            f.write(script)

        rc, out, err = _exec(client, f"cd {HPC_WORKDIR} && {QSUB} {rdir}/job.pbs")
        if rc != 0:
            raise RuntimeError(f"qsub failed: {err.strip() or out.strip()}")
        job_id = out.strip().splitlines()[-1]
        _log(task_id, t_start, f"Submitted PBS job {job_id} (queue {HPC_QUEUE})")
        _update(task_id, 'queued', progress=8, hpc_job_id=job_id)

        # ── Poll until done ──
        status_rpath = posixpath.join(rdir, 'status.json')
        log_rpath = posixpath.join(rdir, 'inference.log')
        log_offset = 0
        fetched_previews = set()
        deadline = time.time() + _walltime_seconds(HPC_WALLTIME) + 1800
        remote = {}

        while True:
            time.sleep(HPC_POLL_INTERVAL)
            if time.time() > deadline:
                raise TimeoutError("HPC job exceeded walltime + grace period")

            state = _job_state(client, job_id)

            # Stream new log lines into the task log
            try:
                st = sftp.stat(log_rpath)
                if st.st_size > log_offset:
                    with sftp.open(log_rpath, 'r') as f:
                        f.seek(log_offset)
                        new = f.read(st.st_size - log_offset).decode(errors='replace')
                    log_offset = st.st_size
                    for line in new.strip().splitlines():
                        _log(task_id, t_start, f"[hpc] {line}")
            except FileNotFoundError:
                pass

            # Read remote progress
            try:
                remote = json.loads(_sftp_read(sftp, status_rpath))
                step = remote.get('step', 'inferring')
                if step not in ('starting',):
                    _update(task_id, step,
                            progress=remote.get('progress'),
                            stats=remote.get('stats', {}),
                            completed_tiles=remote.get('completed_tiles', 0))
            except FileNotFoundError:
                if state == 'Q':
                    _update(task_id, 'queued', progress=8)
            except Exception:
                pass  # transient partial read

            # Fetch newly completed tile previews for the live dashboard
            try:
                for fname in sftp.listdir(rdir):
                    if (fname.startswith('tile_') and fname.endswith('_preview.ply')
                            and fname not in fetched_previews):
                        sftp.get(posixpath.join(rdir, fname),
                                 os.path.join(local_task_dir, fname))
                        fetched_previews.add(fname)
            except Exception:
                pass

            if remote.get('step') == 'failed':
                raise RuntimeError(f"HPC job failed: {remote.get('error', 'unknown')}")

            if state == 'F':
                if remote.get('step') == 'completed':
                    break
                # Job ended without completing — surface PBS stderr
                err_tail = ''
                try:
                    err_tail = _sftp_read(sftp, posixpath.join(rdir, 'pbs_out.log'))[-2000:]
                except Exception:
                    pass
                raise RuntimeError(
                    f"PBS job {job_id} finished but task incomplete "
                    f"(last step: {remote.get('step', 'none')}). stderr tail:\n{err_tail}")

        # ── Download results ──
        _update(task_id, 'downloading', progress=94)
        bundle_r = posixpath.join(rdir, 'result_bundle.tar')
        bundle_l = os.path.join(local_task_dir, 'result_bundle.tar')
        # tar is written by the PBS epilogue right after python exits; wait briefly
        for _ in range(30):
            try:
                bsize = sftp.stat(bundle_r).st_size
                break
            except FileNotFoundError:
                time.sleep(2)
        else:
            raise FileNotFoundError(f"result_bundle.tar not found in {rdir}")

        _log(task_id, t_start, f"Downloading results ({bsize / 1e6:.1f} MB)...")
        t0 = time.time()
        sftp.get(bundle_r, bundle_l)
        dt = time.time() - t0
        _log(task_id, t_start, f"Download done ({bsize / 1e6 / max(dt, 0.01):.0f} MB/s)")

        with tarfile.open(bundle_l) as tar:
            tar.extractall(local_task_dir)
        os.remove(bundle_l)

        output_ply = os.path.join(local_task_dir, 'result.ply')
        if not os.path.isfile(output_ply):
            raise FileNotFoundError("result.ply missing from result bundle")

        stats = remote.get('stats', {})
        stats_file = os.path.join(local_task_dir, 'stats.json')
        if os.path.isfile(stats_file):
            with open(stats_file) as f:
                stats = json.load(f)

        _log(task_id, t_start, f"Task completed on HPC (job {job_id})")
        _update(task_id, 'completed', progress=100,
                stats=stats, result_ply=output_ply, completed=time.time())

    except Exception as e:
        _log(task_id, t_start, f"ERROR: {e}")
        try:
            with open(os.path.join(local_task_dir, 'inference.log'), 'w') as f:
                f.write('\n'.join(tasks[task_id].get('log', [])) + '\n')
        except Exception:
            pass
        _update(task_id, 'failed', progress=0, error=str(e))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _walltime_seconds(walltime):
    parts = [int(p) for p in walltime.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
