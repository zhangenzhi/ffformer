#!/usr/bin/env python3
"""grand1 worker — the HPC-side half of the grand1-push architecture.

Runs on grand1 (the HPC login node). The web pod no longer reaches the HPC over
the (retired) management-network SSH; instead the two sides share state through
the ffformer PVC, which grand1 reaches over the approved .17 channel:

  browser -> pod -> PVC(_imports/<ds>/input.*)        [upload, no SSH]
  pod                -> PVC(results/<task>/request.json)  [run request]
  grand1 (this)      <- PVC(request + input)  over .17   [pull]
  grand1             -> HPC deploy_jobs/<task>/ + qsub    [run, local to HPC]
  grand1             -> PVC(results/<task>/*) over .17    [push results + status]
  pod                <- PVC(results/<task>/status.json)   [progress + result]

Everything grand1 does toward the cloud is grand1-initiated (the only working
direction). deploy_jobs and qsub are local to grand1, so no SSH anywhere.

Credentials: WEBDAV_URL / WEBDAV_USER / WEBDAV_PASS env, or a creds file
(default deploy/.webdav-creds) of one line "user:pass" (URL falls back to the
public .17 endpoint). Single-instance guarded by a pidfile.
"""
import os
import sys
import json
import time
import signal
import shutil
import subprocess

REPO = os.environ.get('FFFORMER_REPO', '/lustre1/work/c30636/ffformer')
sys.path.insert(0, REPO)
# hpc_backend's module level is import-free of paramiko, so this is safe and
# keeps the PBS template / cluster constants in one place.
from deploy.hpc_backend import (  # noqa: E402
    PBS_TEMPLATE, HPC_WORKDIR, HPC_SIF, HPC_QUEUE, HPC_SELECT, HPC_WALLTIME,
    HPC_GROUP, QSUB, QSTAT)

DEPLOY_JOBS = os.path.join(HPC_WORKDIR, 'deploy_jobs')
STATE_DIR = os.environ.get('FFWORKER_STATE', os.path.join(REPO, '.worker_state'))
POLL = float(os.environ.get('FFWORKER_POLL', '15'))
PROGRESS_PUSH = float(os.environ.get('FFWORKER_PROGRESS_PUSH', '20'))
JOB_TIMEOUT = float(os.environ.get('FFWORKER_JOB_TIMEOUT', str(20 * 3600)))
PIDFILE = os.path.join(STATE_DIR, 'worker.pid')

WEBDAV_URL = os.environ.get('WEBDAV_URL', 'http://133.50.39.17:443')
_CREDS_FILE = os.environ.get('FFWORKER_CREDS', os.path.join(REPO, 'deploy', '.webdav-creds'))


def _load_creds():
    u = os.environ.get('WEBDAV_USER')
    p = os.environ.get('WEBDAV_PASS')
    if not (u and p) and os.path.isfile(_CREDS_FILE):
        line = open(_CREDS_FILE).read().strip()
        u, _, p = line.partition(':')
    if not (u and p):
        sys.exit(f"[worker] no webdav creds (set WEBDAV_USER/PASS or {_CREDS_FILE})")
    obs = subprocess.run(['rclone', 'obscure', p], capture_output=True,
                         text=True, check=True).stdout.strip()
    return u, obs


WUSER, WOBS = _load_creds()


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _rc(args, **kw):
    """Run rclone against the .17 webdav (root = the PVC results dir)."""
    base = ['rclone', '--webdav-url', WEBDAV_URL, '--webdav-user', WUSER,
            '--webdav-pass', WOBS]
    return subprocess.run(base + args, capture_output=True, text=True, **kw)


def _R(path):
    """Remote path on the .17 webdav (root = the PVC results dir)."""
    return ':webdav:' + path.lstrip('/')


def rc_cat(remote):
    r = _rc(['cat', _R(remote)])
    return r.stdout if r.returncode == 0 else None


def rc_pull(remote, localdir):
    os.makedirs(localdir, exist_ok=True)
    return _rc(['copy', _R(remote), localdir, '--transfers', '4',
                '--multi-thread-streams', '8']).returncode == 0


def rc_push_file(localfile, remotedir):
    return _rc(['copy', localfile, _R(remotedir), '--transfers', '4',
                '--multi-thread-streams', '8']).returncode == 0


def rc_push_dir(localdir, remotedir):
    return _rc(['copy', localdir, _R(remotedir), '--transfers', '4',
                '--multi-thread-streams', '8']).returncode == 0


def push_status(task_id, obj):
    """Write status.json into PVC results/<task_id>/ so the pod/UI can see it."""
    tmp = os.path.join(STATE_DIR, f'{task_id}.status.json')
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    rc_push_file(tmp, task_id)
    try:
        os.rename(tmp, os.path.join(STATE_DIR, f'{task_id}.status.json'))
    except OSError:
        pass


def is_done_local(task_id):
    return os.path.isfile(os.path.join(STATE_DIR, f'{task_id}.done'))


def mark_done(task_id):
    open(os.path.join(STATE_DIR, f'{task_id}.done'), 'w').close()


def qsub_job(task_id, ds_id, suffix, tile_size, overlap, model):
    rdir = os.path.join(DEPLOY_JOBS, task_id)
    os.makedirs(rdir, exist_ok=True)
    script = PBS_TEMPLATE.format(
        queue=HPC_QUEUE, task_id=task_id, select=HPC_SELECT,
        walltime=HPC_WALLTIME, group=HPC_GROUP, rdir=rdir, workdir=HPC_WORKDIR,
        sif=HPC_SIF, jobdir=task_id, suffix=suffix, tile_size=tile_size,
        overlap=overlap, model=model)
    sp = os.path.join(rdir, 'job.pbs')
    with open(sp, 'w') as f:
        f.write(script)
    r = subprocess.run([QSUB, sp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"qsub failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def job_alive(jobid):
    r = subprocess.run([QSTAT, jobid], capture_output=True, text=True)
    if r.returncode != 0:
        return False  # unknown to PBS = finished/purged
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].split('.')[0] == jobid.split('.')[0]:
            return parts[-2] not in ('C', 'F', 'E')  # completing/finished/exiting
    return True


def read_local_status(task_id):
    p = os.path.join(DEPLOY_JOBS, task_id, 'status.json')
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


RESULT_FILES = ['result.ply', 'stats.json', 'status.json', 'inference.log']


def push_results(task_id):
    rdir = os.path.join(DEPLOY_JOBS, task_id)
    for fn in RESULT_FILES:
        fp = os.path.join(rdir, fn)
        if os.path.isfile(fp):
            rc_push_file(fp, task_id)
    for fn in os.listdir(rdir):
        if fn.startswith('tile_') and fn.endswith('_preview.ply'):
            rc_push_file(os.path.join(rdir, fn), task_id)
    vdir = os.path.join(rdir, 'viewer')
    if os.path.isdir(vdir):
        rc_push_dir(vdir, f'{task_id}/viewer')


def process(task_id):
    log(f"claim task {task_id}")
    req_raw = rc_cat(f'{task_id}/request.json')
    if not req_raw:
        log(f"  no request.json for {task_id}, skip")
        return
    req = json.loads(req_raw)
    ds_id = req['dataset_id']
    suffix = req.get('suffix') or '.laz'
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    tile_size = req.get('tile_size', 100)
    overlap = req.get('overlap', 10)
    model = req.get('model', 'accurate')

    # already completed on a previous run?
    st = read_local_status(task_id) or {}
    remote_st = rc_cat(f'{task_id}/status.json')
    if remote_st:
        try:
            if json.loads(remote_st).get('step') == 'completed':
                mark_done(task_id)
                log(f"  {task_id} already completed, skip")
                return
        except ValueError:
            pass

    # In-flight guard: if we already submitted a job for this task and it is
    # still queued/running, don't resubmit (e.g. after a worker restart).
    jobf = os.path.join(STATE_DIR, f'{task_id}.job')
    if os.path.isfile(jobf):
        jid = open(jobf).read().strip()
        if jid and job_alive(jid):
            log(f"  {task_id} already submitted ({jid}), still queued/running — skip")
            return
        os.remove(jobf)  # job gone and not completed → allow a retry

    push_status(task_id, {'step': 'staging', 'progress': 8, 'stats': {}})
    rdir = os.path.join(DEPLOY_JOBS, task_id)
    os.makedirs(rdir, exist_ok=True)
    # pull the input from the PVC (grand1-initiated over .17)
    log(f"  pull input _imports/{ds_id}/input{suffix}")
    ok = rc_pull(f'_imports/{ds_id}/input{suffix}', rdir)
    inp = os.path.join(rdir, f'input{suffix}')
    if not ok or not os.path.isfile(inp) or os.path.getsize(inp) == 0:
        push_status(task_id, {'step': 'failed', 'progress': 0, 'stats': {},
                              'error': 'input pull from PVC failed'})
        mark_done(task_id)
        log(f"  input pull failed for {task_id}")
        return

    push_status(task_id, {'step': 'queued', 'progress': 10, 'stats': {}})
    jobid = qsub_job(task_id, ds_id, suffix, tile_size, overlap, model)
    with open(jobf, 'w') as f:
        f.write(jobid)
    log(f"  qsub {task_id} -> {jobid}")

    start = time.time()
    last_push = 0.0
    while True:
        time.sleep(POLL)
        local = read_local_status(task_id)
        step = (local or {}).get('step')
        if step in ('completed', 'failed'):
            break
        if time.time() - last_push >= PROGRESS_PUSH and local:
            push_status(task_id, local)
            last_push = time.time()
        if not job_alive(jobid):
            # give the fs a moment, then decide by status.json
            time.sleep(5)
            local = read_local_status(task_id)
            if (local or {}).get('step') in ('completed', 'failed'):
                break
            push_status(task_id, {'step': 'failed', 'progress': 0,
                                  'stats': (local or {}).get('stats', {}),
                                  'error': 'HPC job ended without completion'})
            mark_done(task_id)
            log(f"  {task_id} job ended without completion")
            return
        if time.time() - start > JOB_TIMEOUT:
            push_status(task_id, {'step': 'failed', 'progress': 0, 'stats': {},
                                  'error': 'job timeout'})
            mark_done(task_id)
            log(f"  {task_id} timed out")
            return

    log(f"  push results for {task_id} (final step={step})")
    push_results(task_id)
    mark_done(task_id)
    log(f"  done {task_id}")


def find_pending():
    """task_ids that have a request.json but no completed status yet. Depth is
    capped at 2 so we don't recurse the deep viewer/tiles trees under each result."""
    out = []
    r = _rc(['lsf', ':webdav:', '-R', '--max-depth', '2'])
    if r.returncode != 0:
        return out
    for line in r.stdout.splitlines():
        if line.endswith('/request.json'):
            tid = line.split('/', 1)[0]
            if not is_done_local(tid):
                out.append(tid)
    return out


def _write_pid():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.isfile(PIDFILE):
        try:
            old = int(open(PIDFILE).read().strip())
            os.kill(old, 0)
            sys.exit(f"[worker] already running (pid {old})")
        except (ValueError, OSError):
            pass
    open(PIDFILE, 'w').write(str(os.getpid()))


def main():
    _write_pid()
    log(f"grand1 worker up. webdav={WEBDAV_URL} deploy_jobs={DEPLOY_JOBS} poll={POLL}s")
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *a: stop.update(v=True))
    while not stop['v']:
        try:
            for tid in find_pending():
                try:
                    process(tid)
                except Exception as ex:
                    log(f"  ERROR processing {tid}: {ex}")
        except Exception as ex:
            log(f"loop error: {ex}")
        for _ in range(int(POLL)):
            if stop['v']:
                break
            time.sleep(1)
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    log("worker stopped")


if __name__ == '__main__':
    main()
