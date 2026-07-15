#!/bin/bash
# End-to-end pipeline smoke test on a small bundled sample.
#
# Runs the exact HPC inference path (tile split -> per-tile segmentation ->
# score-based merge -> result.ply + stats.json + streaming viewer octree) on
# examples/sample_forest.laz (600k pts, 120x120m -> 2x2 tiles). Finishes in a
# few minutes on one GPU. Submit from an HPC login node:
#
#   qsub examples/run_example.sh
#
# Then watch examples/out/inference.log and inspect examples/out/ for
# result.ply, stats.json and viewer/.
#
#PBS -q c30636g
#PBS -N ff_example
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30636
#PBS -j oe
#PBS -o /lustre1/work/c30636/ffformer/examples/out/pbs_out.log

set -e
REPO=/lustre1/work/c30636/ffformer
cd "$REPO"
mkdir -p examples/out

export CUDA_VISIBLE_DEVICES=0
module load singularity 2>/dev/null || true
SING=$(command -v singularity || command -v apptainer)

"$SING" exec --nv --bind "$REPO":/workspace --pwd /workspace "$REPO/ffformer.sif" \
    python deploy/hpc_run_task.py \
        --input /workspace/examples/sample_forest.laz \
        --task-dir /workspace/examples/out \
        --tile-size 100 --overlap 10

echo "Done. Outputs in examples/out/ (result.ply, stats.json, viewer/)."
