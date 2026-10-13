#!/bin/bash
# Launch the grand1 worker (see deploy/grand1_worker.py) as a detached daemon.
# Run on grand1 (the HPC login node). Single-GPU 'sg' queue submissions.
#
#   deploy/run_grand1_worker.sh          # start detached (nohup)
#   deploy/run_grand1_worker.sh tmux     # start inside a tmux session 'ffworker'
#   deploy/run_grand1_worker.sh fg       # run in the foreground (used by tmux)
#   deploy/run_grand1_worker.sh stop     # stop
#   deploy/run_grand1_worker.sh restart  # restart (detached)
#   deploy/run_grand1_worker.sh status   # show pid + log tail
# tmux:  attach with 'tmux attach -t ffworker'  (detach: Ctrl-b then d)
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

# Liveness via the pidfile the worker maintains (grand1_worker.py _write_pid).
# pgrep -f is unreliable here: it matches any shell whose command line mentions
# the script name, including the one invoking this launcher.
_pid() { cat "$PIDFILE" 2>/dev/null; }
_running() { local p; p="$(_pid)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

TMUX_SESSION="${FFWORKER_TMUX:-ffworker}"

case "${1:-start}" in
  stop|restart)
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
    pkill -f 'python3 deploy/grand1_worker.py' 2>/dev/null && echo "stopped"
    rm -f "$PIDFILE"
    [ "$1" = stop ] && exit 0
    ;;
  fg)
    # Run in the foreground (env already exported above) — used inside tmux.
    # Tee so the tmux pane shows live logs and worker.log keeps a copy.
    python3 deploy/grand1_worker.py 2>&1 | tee -a "$LOG"
    ;;
  tmux)
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      echo "tmux session '$TMUX_SESSION' already exists (attach: tmux attach -t $TMUX_SESSION)"
      exit 0
    fi
    if _running; then
      echo "a worker is already running (pid $(_pid)); stop it first (run_grand1_worker.sh stop)"
      exit 1
    fi
    # 'fg' re-runs this script from the top, which re-exports the sg defaults.
    tmux new-session -d -s "$TMUX_SESSION" -c "$REPO" \
      "bash '$REPO/deploy/run_grand1_worker.sh' fg"
    echo "started in tmux '$TMUX_SESSION' (queue=$HPC_QUEUE group=$HPC_GROUP); attach: tmux attach -t $TMUX_SESSION"
    exit 0
    ;;
esac

if _running && [ "${1:-start}" = start ]; then
  echo "already running (pid $(_pid))"
  exit 0
fi

if [ "${1:-start}" = status ]; then
  if _running; then echo "pid: $(_pid) (running)"; else echo "pid: none"; fi
  tail -8 "$LOG" 2>/dev/null
  exit 0
fi

setsid nohup python3 deploy/grand1_worker.py >> "$LOG" 2>&1 < /dev/null &
disown
echo "launched (queue=$HPC_QUEUE group=$HPC_GROUP); log: $LOG"
