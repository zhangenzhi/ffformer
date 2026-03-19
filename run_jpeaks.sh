#!/bin/bash
# End-to-end pipeline: LAS → convert → inference → visualize
# Usage: singularity exec --nv --bind ... ffformer.sif bash run_jpeaks.sh [area1|area2]
set -e

AREA=${1:-area1}
WORKSPACE=/workspace
cd $WORKSPACE
export CUDA_VISIBLE_DEVICES=0

LAS_DIR=/data/dataset/forest3d
LAS_FILE=$LAS_DIR/lidar_${AREA}.las
SCENE_NAME=jpeaks_${AREA}
OUTPUT_DIR=work_dirs/jpeaks_${AREA}

echo "============================================"
echo "  ForestFormer3D — JPEAKS ${AREA}"
echo "============================================"

# Step 1: Convert LAS to model format
echo ""
echo "[Step 1/3] Converting LAS → bin ..."
if [ ! -f "data/ForAINetV2/points/${SCENE_NAME}_test.bin" ]; then
    python tools/las2bin.py "$LAS_FILE" \
        --name "$SCENE_NAME" \
        --subsample 0.05
else
    echo "  Already converted, skipping."
fi

# Step 2: Run inference
echo ""
echo "[Step 2/3] Running inference ..."
mkdir -p "$OUTPUT_DIR"

# Update config to point to correct pkl
python tools/test.py \
    configs/jpeaks_test.py \
    work_dirs/clean_forestformer/epoch_3000_fix.pth \
    --work-dir "$OUTPUT_DIR" \
    --cfg-options "test_dataloader.dataset.ann_file=forainetv2_oneformer3d_infos_${SCENE_NAME}.pkl" \
    2>&1 | tee "${OUTPUT_DIR}/inference.log" || true
# Note: evaluator will error (no GT) but PLY is saved before that

echo ""
echo "[Step 3/4] Generating quick preview (5M points) ..."
PLY_FILE=$(ls ${OUTPUT_DIR}/${SCENE_NAME}_test.ply 2>/dev/null || echo "")
if [ -z "$PLY_FILE" ]; then
    echo "  ERROR: No PLY output found. Check ${OUTPUT_DIR}/inference.log"
    exit 1
fi

# Quick preview HTML (downsampled for fast viewing)
python tools/visualize_ply.py "$PLY_FILE" --mode instance

echo ""
echo "[Step 4/4] Building full-resolution streaming viewer ..."
# Octree tiles for full 100M+ point streaming
python tools/potree_convert.py "$PLY_FILE" -o "${OUTPUT_DIR}/${SCENE_NAME}_viewer"

echo ""
echo "============================================"
echo "  Done! Output files:"
echo "============================================"
ls -lh ${OUTPUT_DIR}/${SCENE_NAME}*.ply ${OUTPUT_DIR}/${SCENE_NAME}*.html 2>/dev/null
echo ""
echo "Quick preview: open the .html files in a browser."
echo ""
echo "Full-resolution viewer:"
echo "  cd ${OUTPUT_DIR}/${SCENE_NAME}_viewer && python -m http.server 8080"
echo "  Then open: http://localhost:8080"
