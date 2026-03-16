#!/bin/bash
# Quick debug test with single scene (Yuchen - smallest test scene)
# Run inside Singularity container with GPU

set -e

WORKSPACE=/workspace
cd $WORKSPACE

# Fix: mmengine can't parse GPU UUID in CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

echo "=== Debug inference test ==="
echo "Scene: Yuchen_2023_dls_merged_230209_panoptic_test"
echo ""

python tools/test.py \
    configs/debug_test.py \
    work_dirs/clean_forestformer/epoch_3000_fix.pth \
    --work-dir work_dirs/debug_output \
    2>&1 | tee work_dirs/debug_output.log

echo ""
echo "=== Done. Check work_dirs/debug_output.log for DEBUG prints ==="
