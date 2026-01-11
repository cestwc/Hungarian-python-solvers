# Hungarian Python Solvers

This folder contains several Hungarian-algorithm variants, including warm-started and incremental updates for dynamic assignment problems.

## Solver overview

| File | Purpose | Example command |
| --- | --- | --- |
| `hung.py` | Baseline Hungarian solver | `python hung.py` |
| `hung2.py` | Experimental variant | `python hung2.py` |
| `hung3.py` | Warm-started Hungarian (non-verbose) | `python hung3.py` |
| `hung4.py` | Warm-started Hungarian (verbose tracing) | `python hung4.py` |
| `hung5.py` | Warm-started Hungarian with stats and dual-change counters | `python hung5.py` |
| `hung6.py` | Single-edge incremental update | `python hung6.py` |
| `hung7.py` | Multi-edge incremental update | `python hung7.py` |
| `hung8.py` | Forbid + force edges (forced edges already in matching) | `python hung8.py` |
| `hung9.py` | Forbid + force edges (forced edges can be new) | `python hung9.py` |

## Quick API snippets

These snippets assume you are running from this directory.

### hung5

```python
import hung5

C = [[2, 3], [4, 1]]
assignment, total_cost, u, v, stats = hung5.hungarian_warm_start_verbose(
    C, u0=None, v0=None, repair_iters=2, verbose=False
)
print(total_cost)
```

### hung6

```python
import hung6

C = [[2, 3], [4, 1]]
row_to_col = [0, 1]
u = [0, 0]
v = [0, 0]
new_row_to_col, new_cost, new_u, new_v, stats = hung6.forbid_edge_and_reoptimize_one_augmentation(
    C, row_to_col, u, v, forbid_edge=(0, 0)
)
print(new_cost)
```

### hung7

```python
import hung7

C = [[2, 3], [4, 1]]
row_to_col = [0, 1]
u = [0, 0]
v = [0, 0]
new_row_to_col, new_cost, new_u, new_v, stats = hung7.forbid_edges_and_reoptimize_multi_augmentation(
    C, row_to_col, u, v, forbid_edges=[(0, 0)]
)
print(new_cost)
```

### hung8

```python
import hung8

C = [[2, 3], [4, 1]]
row_to_col = [0, 1]
u = [0, 0]
v = [0, 0]
new_row_to_col, new_cost, new_u, new_v, stats = hung8.forbid_and_force_edges_reoptimize(
    C,
    row_to_col,
    u,
    v,
    forbid_edges=[(0, 0)],
    force_edges=[(1, 1)],
)
print(new_cost)
```

### hung9

```python
import hung9

C = [[2, 3], [4, 1]]
row_to_col = [0, 1]
u = [0, 0]
v = [0, 0]
new_row_to_col, new_cost, new_u, new_v, stats = hung9.forbid_and_force_edges_reoptimize_allow_new_forces(
    C,
    row_to_col,
    u,
    v,
    forbid_edges=[(0, 0)],
    force_edges=[(0, 1)],
)
print(new_cost)
```

## Key ideas

- Warm start (hung3/4/5): reuse dual potentials `u, v` from a previous solve. This reduces dual movement and can cut work, but still runs a full `N`-augmentation Hungarian loop.
- Incremental update (hung6): if one matched edge is forbidden/deleted, remove it and do exactly one augmentation.
- Multi-augmentation update (hung7): if multiple matched edges are forbidden, remove all of them and do one augmentation per freed row.
- Forced edges (hung8/9): fix some matched edges, then re-optimize the remaining subproblem.

## Benchmark (random costs)

These timings compare four approaches after forbidding `k` edges that were in the current optimal matching. Times are in seconds (mean of 5 trials).

| n | k | full | full_warm | inc_seq | inc_multi |
| --- | --- | --- | --- | --- | --- |
| 60 | 1 | 0.002978 | 0.002172 | 0.000202 | 0.000194 |
| 60 | 2 | 0.003025 | 0.002448 | 0.000713 | 0.000597 |
| 60 | 5 | 0.003205 | 0.002713 | 0.001286 | 0.000836 |
| 60 | 10 | 0.002813 | 0.002537 | 0.002800 | 0.001291 |
| 100 | 1 | 0.008217 | 0.006308 | 0.000833 | 0.000757 |
| 100 | 2 | 0.008853 | 0.006728 | 0.001433 | 0.001166 |
| 100 | 5 | 0.008766 | 0.007153 | 0.003167 | 0.001829 |
| 100 | 10 | 0.008901 | 0.007967 | 0.006678 | 0.004364 |
| 150 | 1 | 0.021909 | 0.014514 | 0.001596 | 0.001577 |
| 150 | 2 | 0.021956 | 0.014927 | 0.002528 | 0.002333 |
| 150 | 5 | 0.022610 | 0.016805 | 0.007340 | 0.004691 |
| 150 | 10 | 0.024318 | 0.017198 | 0.015998 | 0.006783 |

Column meanings:
- `full`: full recompute on modified matrix `C'` (no warm start).
- `full_warm`: full recompute on `C'` with warm-start duals from the original problem.
- `inc_seq`: sequential single-edge repairs (one `hung6` call per forbidden edge).
- `inc_multi`: one `hung7` call (multi-augmentation in a single pass).

## Notes and assumptions

- Incremental solvers (`hung6`, `hung7`, `hung8`, `hung9`) assume a square cost matrix and a perfect matching at the start.
- Dual potentials `u, v` must be feasible for the original cost matrix. Increasing costs preserves feasibility.
- If a forbidden edge is not in the current matching, `hung6` returns immediately (no change). `hung7` returns immediately if none of the forbidden edges were matched.
- For large `k`, a full recompute can become competitive; try both if unsure.
