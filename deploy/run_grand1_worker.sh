#!/bin/bash
# Launch the grand1 worker (see deploy/grand1_worker.py) as a detached daemon.
# Run on grand1 (the HPC login node). Single-GPU 'sg' queue submissions.
#
#   deploy/run_grand1_worker.sh          # start (no-op if already running)
#   deploy/run_grand1_worker.sh stop     # stop
#   deploy/run_grand1_worker.sh restart  # restart
#   deploy/run_grand1_worker.sh status   # show pid + log tail
set -u
REPO="${FFFORMER_REPO:-/lustre1/work/c30636/ffformer}"
cd "$REPO" || exit 1

# PBS submission target for inference jobs: single-GPU shared queue.
export HPC_QUEUE="${HPC_QUEUE:-sg}"
export HPC_GROUP="${HPC_GROUP:-c30746}"      # sg jobs run under the personal group
export HPC_SELECT="${HPC_SELECT:-1:ngpus=1}"
export HPC_WALLTIME="${HPC_WALLTIME:-06:00:00}"

LOG="$REPO/worker.log"
PIDFILE="$REPO/.worker_state/worker.pid"

_pid() { [ -f "$PIDFILE" ] && cat "$PIDFILE" 2>/dev/null; }

case "${1:-start}" in
  stop|restart)
    pkill -f 'deploy/grand1_worker.py' 2>/dev/null && echo "stopped"
    rm -f "$PIDFILE"
    [ "$1" = stop ] && exit 0
    ;;
esac

if pgrep -f 'deploy/grand1_worker.py' >/dev/null 2>&1 && [ "${1:-start}" = start ]; then
  echo "already running (pid $(pgrep -f 'deploy/grand1_worker.py' | head -1))"
  exit 0
fi

if [ "${1:-start}" = status ]; then
  echo "pid: $(pgrep -f 'deploy/grand1_worker.py' | head -1 || echo none)"
  tail -8 "$LOG" 2>/dev/null
  exit 0
fi

setsid nohup python3 deploy/grand1_worker.py >> "$LOG" 2>&1 < /dev/null &
disown
echo "launched (queue=$HPC_QUEUE group=$HPC_GROUP); log: $LOG"
