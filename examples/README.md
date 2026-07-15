# Example: end-to-end pipeline smoke test

A small bundled sample to verify the whole ForestFormer3D pipeline works.

- **`sample_forest.laz`** — 600k points over a 120×120 m forest plot
  (subsampled from a real scan). At `tile_size=100, overlap=10` it splits
  into a 2×2 tile grid, so it exercises the full path — tiling, per-tile
  segmentation, and score-based merge — not just a single tile. Runs in a
  few minutes on one GPU.

  > The `.laz` is **not** committed to git (point-cloud data doesn't belong
  > in the repo). It lives alongside the code on the HPC at
  > `/lustre1/work/c30636/ffformer/examples/sample_forest.laz`. The scripts
  > below reference it there; for the dashboard, grab it from that path.

## Option A — through the dashboard (full pipeline: import → segment → analyze)

1. Open the dashboard and sign in.
2. **Data Import**: drag in `sample_forest.laz`. It uploads to the pod and
   streams to the HPC in parallel; the card shows **HPC Ready**.
3. Click **Send to Segment**, keep the defaults (tile 100 m / overlap 10 m),
   press **Start Segmentation**. Watch the tiles render and the stats fill in.
4. **Analysis**: open the tree table and charts, and run the per-tree /
   stand-level LLM assessment.

Expected: ~a few hundred trees detected, most points assigned, a viewable
3D result.

## Option B — headless on an HPC login node (inference only)

```bash
cd /lustre1/work/c30636/ffformer
mkdir -p examples/out
qsub examples/run_example.sh
# watch progress:
tail -f examples/out/inference.log
```

Outputs land in `examples/out/`:

- `result.ply` — per-point semantic + instance predictions
- `stats.json` — tree count, per-class point counts, timing
- `viewer/` — octree tiles for the streaming 3D viewer
- `inference.log` — per-tile log

A successful run reports every tile with a tree count (not `failed`) and a
non-zero `n_trees` / `n_assigned` in `stats.json`.

> `examples/out/` is git-ignored — it holds generated results, not source.
