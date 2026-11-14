#!/bin/bash
# Start the GNN dashboard in a detached tmux session (bound to 127.0.0.1:8765).
set -euo pipefail
SESSION=gnn-dashboard
ROOT=/workspace/Our_Pipeline
PY=$ROOT/.venv-exp1/bin/python
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (attach: tmux attach -t $SESSION)"
  exit 0
fi
mkdir -p "$ROOT/experiments/gnn/logs"
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "$PY experiments/gnn/dashboard_simple.py --host ${DASH_HOST:-0.0.0.0} --port ${DASH_PORT:-10100} --root $ROOT >> $ROOT/experiments/gnn/logs/dashboard.log 2>&1"
echo "Started '$SESSION' -> http://0.0.0.0:${DASH_PORT:-10100} (public: http://$PUBLIC_IPADDR:$VAST_TCP_PORT_10100) (log: $ROOT/experiments/gnn/logs/dashboard.log)"
