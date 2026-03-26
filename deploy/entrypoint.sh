#!/bin/bash
set -e

export HOME=${HOME:-/home/appuser}
cd /app

# Pull latest code from GitHub (code-only update, no env rebuild)
echo "[entrypoint] Pulling latest code from GitHub..."
git config --global --add safe.directory /app 2>/dev/null || true

# Ensure we're on main branch, then fetch + reset
git checkout -B main 2>/dev/null || true
if git fetch origin main --depth=1 2>&1; then
    git reset --hard origin/main 2>&1
    echo "[entrypoint] Code updated to $(git rev-parse --short HEAD)"
else
    echo "[entrypoint] Git fetch failed, using image code ($(git rev-parse --short HEAD 2>/dev/null || echo 'no commits'))"
fi

# Start API server
echo "[entrypoint] Starting ForestFormer3D API server..."
exec uvicorn deploy.server:app --host 0.0.0.0 --port 8000 --workers 1
