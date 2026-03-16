#!/bin/bash
# Full test inference on all 29 scenes (no spconv permutation)
set -e

WORKSPACE=/workspace
cd $WORKSPACE

export CUDA_VISIBLE_DEVICES=0

echo "=== Full inference test (29 scenes, no spconv permute) ==="

python tools/test.py \
    configs/oneformer3d_qs_radius16_qp300_2many.py \
    work_dirs/clean_forestformer/epoch_3000_fix.pth \
    --work-dir work_dirs/test_output_fixed \
    2>&1 | tee work_dirs/test_output_fixed.log

echo ""
echo "=== Done ==="
