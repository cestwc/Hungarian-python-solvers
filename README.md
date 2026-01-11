# Hungarian Solvers (dynamic/phung)

This folder contains several Hungarian-algorithm variants, including warm-started and incremental updates for dynamic assignment problems.

## Files

- `hung.py`: baseline Hungarian solver.
- `hung2.py`: experimental variant (see file for details).
- `hung3.py`: warm-started Hungarian (non-verbose).
- `hung4.py`: warm-started Hungarian with verbose tracing.
- `hung5.py`: warm-started Hungarian with detailed stats and dual-change counters.
- `hung6.py`: **single-edge incremental update** (forbid one matched edge, then do one augmentation).
- `hung7.py`: **multi-edge incremental update** (forbid multiple matched edges, then do one augmentation per freed row).

## Key ideas

- **Warm start (hung3/4/5)**: reuse dual potentials `u, v` from a previous solve. This reduces dual movement and can cut work, but still runs a full `N`-augmentation Hungarian loop.
- **Incremental update (hung6)**: if only one matched edge is forbidden/deleted, remove it and do exactly one augmentation. This is typically much faster than a full re-solve.
- **Multi-augmentation update (hung7)**: if multiple matched edges are forbidden, remove all of them and do one augmentation per freed row in a single pass.

## Usage examples

### Full solve with warm start (hung5)

```python
from dynamic.phung import hung5

assignment, total_cost, u, v, stats = hung5.hungarian_warm_start_verbose(
    C, u0=None, v0=None, repair_iters=2, verbose=False
)
```

### Single-edge incremental repair (hung6)

```python
from dynamic.phung import hung6

new_row_to_col, new_cost, new_u, new_v, stats = hung6.forbid_edge_and_reoptimize_one_augmentation(
    C, row_to_col, u, v, forbid_edge=(r, c)
)
```

### Multi-edge incremental repair (hung7)

```python
from dynamic.phung import hung7

new_row_to_col, new_cost, new_u, new_v, stats = hung7.forbid_edges_and_reoptimize_multi_augmentation(
    C, row_to_col, u, v, forbid_edges=[(r1, c1), (r2, c2)]
)
```

## Benchmark (random costs)

These timings compare four approaches after forbidding `k` edges that were in the current optimal matching. Times are in seconds (mean of 5 trials).

Columns:
- `full`: full recompute on modified matrix `C'` (no warm start).
- `full_warm`: full recompute on `C'` with warm-start duals from the original problem.
- `inc_seq`: sequential single-edge repairs (one `hung6` call per forbidden edge).
- `inc_multi`: one `hung7` call (multi-augmentation in a single pass).

```
n=60  k=1  full=0.002978  full_warm=0.002172  inc_seq=0.000202  inc_multi=0.000194
n=60  k=2  full=0.003025  full_warm=0.002448  inc_seq=0.000713  inc_multi=0.000597
n=60  k=5  full=0.003205  full_warm=0.002713  inc_seq=0.001286  inc_multi=0.000836
n=60  k=10 full=0.002813  full_warm=0.002537  inc_seq=0.002800  inc_multi=0.001291

n=100 k=1  full=0.008217  full_warm=0.006308  inc_seq=0.000833  inc_multi=0.000757
n=100 k=2  full=0.008853  full_warm=0.006728  inc_seq=0.001433  inc_multi=0.001166
n=100 k=5  full=0.008766  full_warm=0.007153  inc_seq=0.003167  inc_multi=0.001829
n=100 k=10 full=0.008901  full_warm=0.007967  inc_seq=0.006678  inc_multi=0.004364

n=150 k=1  full=0.021909  full_warm=0.014514  inc_seq=0.001596  inc_multi=0.001577
n=150 k=2  full=0.021956  full_warm=0.014927  inc_seq=0.002528  inc_multi=0.002333
n=150 k=5  full=0.022610  full_warm=0.016805  inc_seq=0.007340  inc_multi=0.004691
n=150 k=10 full=0.024318  full_warm=0.017198  inc_seq=0.015998  inc_multi=0.006783
```

## Notes and assumptions

- The incremental solvers (`hung6`, `hung7`) assume a **square** cost matrix and a **perfect matching** at the start.
- Dual potentials `u, v` must be feasible for the original cost matrix. Increasing costs preserves feasibility.
- If a forbidden edge is **not** in the current matching, `hung6` returns immediately (no change). `hung7` also returns immediately if none of the forbidden edges were matched.
- For large `k`, a full recompute can become competitive; try both if unsure.
