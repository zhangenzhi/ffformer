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
import base64
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
        --input /workspace/deploy_jobs/{jobdir}/input{suffix} \\
        --task-dir /workspace/deploy_jobs/{jobdir} \\
        --tile-size {tile_size} --overlap {overlap} --model {model}
rc=$?

# Bundle results into one compressed tar for a single fast SFTP download.
# The link to the pod is capped at ~1 Gbps; ASCII PLY compresses ~3x, so
# pigz/gzip raises effective transfer speed accordingly.
cd {rdir}
files=""
for f in result.ply stats.json inference.log status.json; do
    [ -f "$f" ] && files="$files $f"
done
[ -d viewer ] && files="$files viewer"
for f in tile_*_preview.ply; do
    [ -f "$f" ] && files="$files $f"
done
if command -v pigz >/dev/null 2>&1; then
    [ -n "$files" ] && tar cf - $files | pigz -1 > result_bundle.tar.gz
else
    [ -n "$files" ] && tar czf result_bundle.tar.gz $files
fi

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

def run_hpc_inference(tasks_proxy, task_id, input_path, suffix, tile_size, overlap,
                      model='accurate'):
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
        hpc_suffix = _stage_input(client, sftp, input_path, suffix, rdir,
                                  log=lambda m: _log(task_id, t_start, m))
        _update(task_id, 'uploading', progress=6)
        _submit_poll_download(client, sftp, task_id, task_id, local_task_dir,
                              t_start, hpc_suffix, tile_size, overlap, model)
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


def _stage_input(client, sftp, input_path, suffix, rdir, log=lambda m: None):
    """Compress LAS->LAZ (pod<->HPC link is ~1 Gbps; LAZ is ~3x smaller) and
    SFTP the input into rdir. Returns the on-HPC file suffix."""
    _exec(client, f"mkdir -p {rdir}")
    upload_path, upload_suffix = input_path, suffix
    if suffix == '.las':
        try:
            import laspy
            laz_path = os.path.join(os.path.dirname(input_path), 'input.laz')
            t0 = time.time()
            laspy.read(input_path).write(laz_path)
            ratio = os.path.getsize(input_path) / max(os.path.getsize(laz_path), 1)
            log(f"Compressed LAS->LAZ {ratio:.1f}x in {time.time() - t0:.1f}s")
            upload_path, upload_suffix = laz_path, '.laz'
        except Exception as e:
            log(f"LAZ compression skipped ({e}), sending raw")
    size_mb = os.path.getsize(upload_path) / 1e6
    log(f"Uploading input ({size_mb:.1f} MB) to {rdir}...")
    t0 = time.time()
    sftp.put(upload_path, posixpath.join(rdir, f'input{upload_suffix}'))
    log(f"Upload done ({size_mb / max(time.time() - t0, 0.01):.0f} MB/s)")
    if upload_path != input_path:
        os.remove(upload_path)
    return upload_suffix


def _submit_poll_download(client, sftp, task_id, jobdir, local_task_dir, t_start,
                          hpc_suffix, tile_size, overlap, model='accurate'):
    """Generate the PBS job over the input already in deploy_jobs/{jobdir},
    qsub it, stream progress/log/tiles, then download the result bundle."""
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', jobdir)

    # Clear any stale outputs (a pre-staged dir may be reused) but keep the input.
    _exec(client, "cd %s && rm -rf status.json inference.log result.ply stats.json "
                  "viewer tile_*_preview.ply result_bundle.tar.gz pbs_out.log "
                  "2>/dev/null; true" % rdir)

    _update(task_id, 'submitting', progress=7)
    script = PBS_TEMPLATE.format(
        queue=HPC_QUEUE, select=HPC_SELECT, walltime=HPC_WALLTIME,
        group=HPC_GROUP, workdir=HPC_WORKDIR, sif=HPC_SIF, rdir=rdir,
        jobdir=jobdir, task_id=task_id, suffix=hpc_suffix,
        tile_size=tile_size, overlap=overlap, model=model,
    )
    with sftp.open(posixpath.join(rdir, 'job.pbs'), 'w') as f:
        f.write(script)

    rc, out, err = _exec(client, f"cd {HPC_WORKDIR} && {QSUB} {rdir}/job.pbs")
    if rc != 0:
        raise RuntimeError(f"qsub failed: {err.strip() or out.strip()}")
    job_id = out.strip().splitlines()[-1]
    _log(task_id, t_start, f"Submitted PBS job {job_id} (queue {HPC_QUEUE})")
    _update(task_id, 'queued', progress=8, hpc_job_id=job_id)

    if True:
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
                    extra = {}
                    if remote.get('hw'):
                        extra['hw'] = {**remote['hw'], 'job_id': job_id}
                    _update(task_id, step,
                            progress=remote.get('progress'),
                            stats=remote.get('stats', {}),
                            completed_tiles=remote.get('completed_tiles', 0),
                            **extra)
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
        bundle_r = posixpath.join(rdir, 'result_bundle.tar.gz')
        bundle_l = os.path.join(local_task_dir, 'result_bundle.tar.gz')
        # tar is written by the PBS epilogue right after python exits; wait briefly
        for _ in range(30):
            try:
                bsize = sftp.stat(bundle_r).st_size
                break
            except FileNotFoundError:
                time.sleep(2)
        else:
            raise FileNotFoundError(f"result_bundle.tar.gz not found in {rdir}")

        _log(task_id, t_start, f"Downloading results ({bsize / 1e6:.1f} MB)...")
        t0 = time.time()
        sftp.get(bundle_r, bundle_l)
        dt = time.time() - t0
        _log(task_id, t_start, f"Download done ({bsize / 1e6 / max(dt, 0.01):.0f} MB/s)")

        with tarfile.open(bundle_l, 'r:gz') as tar:
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


# --- Import-time staging (data lands on the pod and HPC before segmentation) ---

def stream_import_to_hpc(ds_id, suffix, chunk_queue, state):
    """Relay an upload straight to the HPC as it arrives.

    Runs in a writer thread: the /import request feeds byte chunks into
    chunk_queue (None ends it) while this thread SFTP-writes them to the HPC,
    so browser->pod and pod->HPC overlap. No LAZ compression (that needs the
    whole file first) — worth it only because pod->HPC is fast enough that
    overlap beats the ~3x compression saving.
    """
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', ds_id)
    client = None
    try:
        client = _connect()
        sftp = client.open_sftp()
        _exec(client, f"mkdir -p {rdir}")
        with sftp.open(posixpath.join(rdir, f'input{suffix}'), 'wb') as rf:
            rf.set_pipelined(True)  # pipeline writes — critical for throughput
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                rf.write(chunk)
        state['hpc_suffix'] = suffix
        state['ok'] = True
    except Exception as e:
        state['error'] = str(e)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


HPC_DATA_DIRS = os.environ.get('HPC_DATA_DIRS', 'examples')


def list_hpc_data():
    """List point-cloud files sitting on the HPC (the 'server data' the
    dashboard shows) — datasets already on the cluster, ready to segment."""
    client = _connect()
    try:
        files = []
        for d in HPC_DATA_DIRS.split(':'):
            base = posixpath.join(HPC_WORKDIR, d.strip())
            rc, out, _ = _exec(
                client, f"find {base} -maxdepth 3 -type f "
                        r"\( -name '*.las' -o -name '*.laz' -o -name '*.ply' \) "
                        r"-printf '%s\t%p\n' 2>/dev/null")
            for line in out.strip().splitlines():
                if '\t' not in line:
                    continue
                size, p = line.split('\t', 1)
                files.append({'path': p, 'filename': posixpath.basename(p),
                              'size': int(size),
                              'type': p.rsplit('.', 1)[-1].lower()})
        files.sort(key=lambda f: f['filename'])
        return files
    finally:
        client.close()


def list_hpc_datasets():
    """Rebuild the uploaded-dataset index from the HPC. Every
    deploy_jobs/<id>/input.* is a dataset the user uploaded — the data persists on
    the HPC even when the server's in-memory index is wiped (restart / redeploy).
    One SSH round-trip emits: id|suffix|size|has_result|meta_json per dataset."""
    base = posixpath.join(HPC_WORKDIR, 'deploy_jobs')
    cmd = (
        f"for f in $(find {base} -maxdepth 2 -name 'input.*' 2>/dev/null); do "
        f"  d=$(dirname \"$f\"); id=$(basename \"$d\"); "
        f"  sz=$(stat -c%s \"$f\" 2>/dev/null); "
        f"  res=$([ -f \"$d/result.ply\" ] && echo 1 || echo 0); "
        f"  meta=$(cat \"$d/meta.json\" 2>/dev/null | tr -d '\\n'); "
        f"  echo \"$id|$f|$sz|$res|$meta\"; "
        f"done")
    client = _connect()
    try:
        rc, out, _ = _exec(client, cmd)
        rows = []
        for line in out.strip().splitlines():
            parts = line.split('|', 4)
            if len(parts) < 5:
                continue
            ds_id, path, sz, res, meta = parts
            ext = path.rsplit('.', 1)[-1].lower()
            entry = {'dataset_id': ds_id, 'path': path, 'hpc_suffix': '.' + ext,
                     'size': int(sz) if sz.isdigit() else 0,
                     'filename': f'{ds_id}.{ext}',
                     'has_result': res == '1'}
            if meta.strip():
                try:
                    entry.update(json.loads(meta))
                except Exception:
                    pass
            rows.append(entry)
        return rows
    finally:
        client.close()


def list_hpc_results():
    """Rebuild the completed-task index from the HPC. Every deploy_jobs/<id> with a
    status.json (or result.ply) is a finished segmentation whose result persists on
    the HPC even when the server's /tmp results are wiped on restart. One SSH pass
    emits id|has_viewer|base64(status.json)|base64(meta.json)."""
    base = posixpath.join(HPC_WORKDIR, 'deploy_jobs')
    cmd = (
        f"for d in {base}/*/; do id=$(basename \"$d\"); "
        f"  if [ -f \"$d/status.json\" ] && [ -f \"$d/result.ply\" ]; then "
        f"    st=$(base64 -w0 < \"$d/status.json\" 2>/dev/null); "
        f"    mt=$(base64 -w0 < \"$d/meta.json\" 2>/dev/null); "
        f"    hv=$([ -d \"$d/viewer\" ] && echo 1 || echo 0); "
        f"    echo \"$id|$hv|$st|$mt\"; fi; done")
    client = _connect()
    try:
        rc, out, _ = _exec(client, cmd)
        rows = []
        for line in out.strip().splitlines():
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            tid, hv, st_b64, mt_b64 = parts
            try:
                status = json.loads(base64.b64decode(st_b64)) if st_b64 else {}
            except Exception:
                status = {}
            name = tid
            if mt_b64:
                try:
                    name = json.loads(base64.b64decode(mt_b64)).get('filename', tid)
                except Exception:
                    pass
            rows.append({'task_id': tid, 'filename': name, 'has_viewer': hv == '1',
                         'stats': status.get('stats', {}),
                         'step': status.get('step', 'completed')})
        return rows
    finally:
        client.close()


def fetch_result_bundle(task_id, local_dir, want_ply=True):
    """Download + extract a finished task's result bundle from the HPC into
    local_dir, so the viewer/result endpoints (which read the pod filesystem) work
    for results restored after a restart. Returns True on success."""
    if not task_id or '/' in task_id or '..' in task_id:
        raise ValueError(f"bad task id: {task_id}")
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', task_id)
    os.makedirs(local_dir, exist_ok=True)
    client = _connect()
    try:
        import tarfile
        sftp = client.open_sftp()
        # Prefer the prebuilt bundle (contains result.ply + stats + viewer).
        rc, out, _ = _exec(client, f"test -f {rdir}/result_bundle.tar.gz && echo y || echo n")
        if out.strip().endswith('y'):
            local_tar = os.path.join(local_dir, 'result_bundle.tar.gz')
            sftp.get(posixpath.join(rdir, 'result_bundle.tar.gz'), local_tar)
            with tarfile.open(local_tar) as tf:
                tf.extractall(local_dir)
            os.remove(local_tar)
        else:
            # Individual files. The (possibly multi-GB) result.ply is only pulled
            # when the caller needs it (download) — opening the streaming viewer
            # needs just the octree below. Also skip it if already local.
            fetch_files = ['stats.json', 'status.json']
            if want_ply:
                fetch_files = ['result.ply'] + fetch_files
            for fn in fetch_files:
                lp = os.path.join(local_dir, fn)
                if fn == 'result.ply' and os.path.isfile(lp):
                    continue
                try:
                    sftp.get(posixpath.join(rdir, fn), lp)
                except Exception:
                    pass
        # Ensure the streaming octree viewer is present — it is what lets large
        # results be previewed online (without it the client falls back to loading
        # the whole PLY and freezes). Pack it on the HPC and pull a single tar to
        # avoid thousands of tiny tile transfers. No-op if there is no viewer/ dir.
        if not os.path.isfile(os.path.join(local_dir, 'viewer', 'metadata.json')):
            rc, out2, _ = _exec(
                client,
                f"cd {rdir} && test -d viewer && tar czf viewer_bundle.tar.gz viewer && echo y || echo n")
            if out2.strip().endswith('y'):
                local_vtar = os.path.join(local_dir, 'viewer_bundle.tar.gz')
                try:
                    sftp.get(posixpath.join(rdir, 'viewer_bundle.tar.gz'), local_vtar)
                    with tarfile.open(local_vtar) as tf:
                        tf.extractall(local_dir)
                finally:
                    if os.path.isfile(local_vtar):
                        os.remove(local_vtar)
                    _exec(client, f"rm -f {rdir}/viewer_bundle.tar.gz")
        return (os.path.isfile(os.path.join(local_dir, 'result.ply')) or
                os.path.isfile(os.path.join(local_dir, 'viewer', 'metadata.json')))
    finally:
        client.close()


def delete_hpc_dataset(ds_id):
    """Remove a dataset's directory (input + any results) from the HPC so it does
    not reappear on the next index rebuild. ds_id is validated to be a bare id."""
    if not ds_id or '/' in ds_id or '..' in ds_id:
        raise ValueError(f"bad dataset id: {ds_id}")
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', ds_id)
    client = _connect()
    try:
        _exec(client, f"rm -rf {rdir}")
    finally:
        client.close()


def write_dataset_meta(ds_id, meta):
    """Persist dataset metadata (filename, format, created) next to the input on
    the HPC so list_hpc_datasets can restore nice names after a restart."""
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', ds_id)
    client = _connect()
    try:
        sftp = client.open_sftp()
        _exec(client, f"mkdir -p {rdir}")
        with sftp.open(posixpath.join(rdir, 'meta.json'), 'w') as f:
            f.write(json.dumps(meta))
    except Exception as e:
        print(f"[hpc] write_dataset_meta({ds_id}) failed: {e}")
    finally:
        client.close()


def stage_hpc_file(ds_id, src_path, suffix=None):
    """Turn an existing HPC file into a dataset by copying it into the dataset
    dir inside the HPC filesystem (instant, no transfer). Returns the suffix."""
    suffix = suffix or ('.' + src_path.rsplit('.', 1)[-1].lower())
    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', ds_id)
    client = _connect()
    try:
        rc, out, err = _exec(
            client, f"test -f '{src_path}' && mkdir -p {rdir} && "
                    f"cp '{src_path}' {rdir}/input{suffix} && echo ok")
        if rc != 0 or 'ok' not in out:
            raise RuntimeError(err.strip() or out.strip() or
                               f"file not found: {src_path}")
    finally:
        client.close()
    return suffix


def stage_to_hpc(datasets_proxy, dataset_id, input_path, suffix):
    """Compress + push a freshly-imported dataset to the HPC so segmentation
    can start instantly later. Updates the datasets dict, does NOT qsub."""
    def upd(**kw):
        d = dict(datasets_proxy[dataset_id])
        d.update(kw)
        d['updated'] = time.time()
        datasets_proxy[dataset_id] = d

    rdir = posixpath.join(HPC_WORKDIR, 'deploy_jobs', dataset_id)
    client = None
    try:
        upd(hpc_stage='staging', hpc_progress=10)
        client = _connect()
        sftp = client.open_sftp()
        hpc_suffix = _stage_input(client, sftp, input_path, suffix, rdir)
        upd(hpc_stage='ready', hpc_ready=True, hpc_suffix=hpc_suffix,
            hpc_progress=100)
    except Exception as e:
        upd(hpc_stage='failed', error=str(e))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def run_hpc_inference_prestaged(tasks_proxy, task_id, dataset_id, hpc_suffix,
                                tile_size, overlap, model='accurate'):
    """Segment a dataset already staged on the HPC — skips upload, qsub only."""
    global tasks
    tasks = tasks_proxy
    t_start = time.time()
    results_dir = os.environ.get('RESULTS_DIR', '/tmp/ffformer_results')
    local_task_dir = os.path.join(results_dir, task_id)
    os.makedirs(local_task_dir, exist_ok=True)
    client = None
    try:
        _update(task_id, 'queued', progress=5)
        _log(task_id, t_start, "Data already on HPC; submitting job...")
        client = _connect()
        sftp = client.open_sftp()
        _submit_poll_download(client, sftp, task_id, dataset_id, local_task_dir,
                              t_start, hpc_suffix, tile_size, overlap, model)
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
