#!/bin/bash
set -e

export HOME=${HOME:-/home/appuser}
cd /app

# Pull latest code from GitHub (code-only update, no env rebuild)
echo "[entrypoint] Pulling latest code from GitHub..."
git config --global --add safe.directory /app 2>/dev/null || true
git fetch origin main --depth=1 2>&1 && \
git reset --hard origin/main 2>&1 && \
echo "[entrypoint] Code updated to $(git rev-parse --short HEAD)" || \
echo "[entrypoint] Git pull failed, using image code"

# Start API server
echo "[entrypoint] Starting ForestFormer3D API server..."
exec uvicorn deploy.server:app --host 0.0.0.0 --port 8000 --workers 1
