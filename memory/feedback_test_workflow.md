---
name: test_before_full_run
description: User prefers testing single scene end-to-end (including visualization) before running full dataset
type: feedback
---

Always test with a single small scene first, verifying the entire pipeline (inference + visualization) works end-to-end, before running on the full dataset.

**Why:** Avoid wasting HPC GPU time on long runs that might fail or produce wrong results.

**How to apply:** When running inference or any batch job, first do one scene, check the output (PLY + visualization), confirm with user, then proceed to full run.
