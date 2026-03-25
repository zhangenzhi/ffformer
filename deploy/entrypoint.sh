#!/bin/bash
set -e

cd /app

# Pull latest code from GitHub (code-only update, no env rebuild)
echo "[entrypoint] Pulling latest code from GitHub..."
git config --global --add safe.directory /app
git fetch origin main --depth=1 2>/dev/null && \
git reset --hard origin/main 2>/dev/null && \
echo "[entrypoint] Code updated to $(git rev-parse --short HEAD)" || \
echo "[entrypoint] Git pull failed, using image code"

# Start API server
echo "[entrypoint] Starting ForestFormer3D API server..."
exec uvicorn deploy.server:app --host 0.0.0.0 --port 8000 --workers 1
