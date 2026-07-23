#!/bin/bash
# Smoke test of the pure-PyTorch FF3D reimplementation: run tools/infer.py on
# the bundled sample point cloud inside ffformer.sif (which already carries
# torch/spconv/MinkowskiEngine/torch_scatter/torch_cluster).
#
#   qsub FF3D/run_infer_test.sh
#
#PBS -q c30636g
#PBS -N ff3d_infer
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30636
#PBS -j oe
#PBS -o /lustre1/work/c30636/ffformer/FF3D/infer_test.log

set -e
REPO=/lustre1/work/c30636/ffformer
cd "$REPO"
mkdir -p FF3D/results

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
module load singularity 2>/dev/null || true
SING=$(command -v singularity || command -v apptainer)

"$SING" exec --nv --bind "$REPO":/workspace --pwd /workspace "$REPO/ffformer.sif" \
    env PYTHONPATH=/workspace/FF3D python /workspace/FF3D/tools/infer.py \
        --input /workspace/examples/sample_forest.laz \
        --checkpoint /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth \
        --output /workspace/FF3D/results/sample_forest.ply

echo "=== infer.py exit: $? ==="
ls -la "$REPO/FF3D/results/" 2>/dev/null
