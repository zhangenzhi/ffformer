---
name: project_status_march2026
description: ForestFormer3D project status - inference fixed, visualization done, reconstruction deferred
type: project
---

Inference pipeline is working correctly after fixing spconv weight permutation bug (2026-03-16). Single scene (Yuchen) verified with 19 trees detected. Full 29-scene test not yet run.

Key fix: checkpoint weights were already in correct spconv 2.x format, the permutation in test.py was WRONG and had to be removed.

Visualization: Three.js dual-panel viewer working (point cloud mode). Mesh reconstruction attempted but Poisson creates deformed blobs for trees - user decided to defer 3D reconstruction.

**Why:** User wants to focus on the core segmentation results first, reconstruction is not a priority now.

**How to apply:** Don't spend time on 3D reconstruction or 3DGS unless user explicitly asks. Next step is likely running full 29-scene inference and evaluating metrics.
