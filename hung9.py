from typing import List, Tuple, Dict, Any, Optional, Iterable


def compute_total_cost(C: List[List[float]], row_to_col: List[int]) -> float:
    return sum(C[i][j] for i, j in enumerate(row_to_col))


def p_to_row_to_col(p: List[int]) -> List[int]:
    """Convert Hungarian p array (1-based, col->row) to row->col (0-based)."""
    n = len(p) - 1
    row_to_col = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        if i <= 0:
            raise ValueError("p has an unmatched column; expected perfect matching.")
        row_to_col[i - 1] = j - 1
    if any(x == -1 for x in row_to_col):
        raise ValueError("p does not represent a perfect matching (some row unmatched).")
    return row_to_col


def forbid_and_force_edges_reoptimize_allow_new_forces(
    C: List[List[float]],
    row_to_col: List[int],
    u: List[float],
    v: List[float],
    forbid_edges: Iterable[Tuple[int, int]],
    force_edges: Iterable[Tuple[int, int]],
    bigM: Optional[float] = None,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Tuple[List[int], float, List[float], List[float], Dict[str, Any]]:
    """
    Incremental strategy with forbidden + forced edges, allowing forced edges
    that are NOT currently in the matching.

    Steps:
      - Enforce forced edges by swapping them into the matching, freeing
        the conflicting row/col assignments.
      - Forbid edges by setting their cost to bigM.
      - Solve the remaining free rows/cols with one Hungarian augmentation
        per freed row (reduced subproblem).

    Assumptions:
      - Square problem (n x n)
      - row_to_col is a perfect matching before changes
      - u, v are dual potentials (length n each), dual-feasible for original C
        After increasing costs, they remain feasible for the unconstrained problem.
    """
    n = len(C)
    if n == 0 or any(len(row) != n for row in C):
        raise ValueError("C must be square (n x n).")
    if len(row_to_col) != n or len(u) != n or len(v) != n:
        raise ValueError("row_to_col, u, v must have length n.")

    forbid_edges = list(forbid_edges)
    force_edges = list(force_edges)
    force_edge_set = set(force_edges)

    # Validate force edges
    seen_rows = set()
    seen_cols = set()
    for r, c in force_edges:
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError("force_edges must be within matrix bounds.")
        if r in seen_rows:
            raise ValueError("force_edges contains duplicate rows.")
        if c in seen_cols:
            raise ValueError("force_edges contains duplicate columns.")
        seen_rows.add(r)
        seen_cols.add(c)

    forced_rows = set(seen_rows)
    forced_cols = set(seen_cols)

    for r, c in forbid_edges:
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError("forbid_edges must be within matrix bounds.")
        if (r, c) in force_edge_set:
            raise ValueError("forbid_edges conflicts with force_edges.")

    # Choose bigM if not provided
    if bigM is None:
        cmax = max(max(row) for row in C)
        bigM = cmax * n * 10.0 + 1.0

    # Build C' with forbidden edges penalized
    Cprime = [row[:] for row in C]
    for r, c in forbid_edges:
        Cprime[r][c] = bigM

    # Enforce forced edges by swapping into matching
    row_to_col_work = row_to_col[:]
    col_to_row = [-1] * n
    for i, j in enumerate(row_to_col_work):
        if j < 0 or j >= n:
            raise ValueError("row_to_col must be a perfect matching.")
        col_to_row[j] = i

    enforced_new_forces: List[Tuple[int, int]] = []

    for r, c in force_edges:
        if row_to_col_work[r] == c:
            continue
        r2 = col_to_row[c]
        if r2 == -1:
            raise ValueError("Internal error: column is unassigned in perfect matching.")
        if r2 in forced_rows and (r2, c) not in force_edge_set:
            raise ValueError("Forced edge conflicts with another forced row.")

        c1 = row_to_col_work[r]

        row_to_col_work[r] = c
        col_to_row[c] = r

        row_to_col_work[r2] = -1
        col_to_row[c1] = -1

        enforced_new_forces.append((r, c))

    # Ensure all forced edges are satisfied
    for r, c in force_edges:
        if row_to_col_work[r] != c:
            raise ValueError("Failed to enforce forced edges.")

    # Build reduced index maps (exclude forced rows/cols)
    free_rows = [i for i in range(n) if i not in forced_rows]
    free_cols = [j for j in range(n) if j not in forced_cols]
    m = len(free_rows)
    if m != len(free_cols):
        raise ValueError("Forced rows/cols are inconsistent; remaining problem is not square.")

    # If everything is forced, return early
    if m == 0:
        stats = {
            "augmentations": 0,
            "inner_iterations_total": 0,
            "dual_update_count": 0,
            "dual_delta_sum": 0.0,
            "forbidden_edges": forbid_edges,
            "forced_edges": force_edges,
            "bigM": bigM,
            "note": "All rows/cols forced; no re-optimization needed.",
        }
        total = compute_total_cost(Cprime, row_to_col_work)
        return row_to_col_work, total, u[:], v[:], stats

    row_map = {r: i for i, r in enumerate(free_rows)}
    col_map = {c: j for j, c in enumerate(free_cols)}

    # Build reduced cost matrix
    Cfree = [[Cprime[r][c] for c in free_cols] for r in free_rows]

    # Build reduced matching (may be partial)
    row_to_col_free = [-1] * m
    for r in free_rows:
        c = row_to_col_work[r]
        if c == -1:
            continue
        if c in forced_cols:
            raise ValueError("Free row matched to forced column.")
        row_to_col_free[row_map[r]] = col_map[c]

    # Reduced duals
    u_free = [u[r] for r in free_rows]
    v_free = [v[c] for c in free_cols]

    # Apply forbids in reduced problem
    forbid_edges_in_free: List[Tuple[int, int]] = []
    for r, c in forbid_edges:
        if r in row_map and c in col_map:
            rr = row_map[r]
            cc = col_map[c]
            Cfree[rr][cc] = bigM
            forbid_edges_in_free.append((rr, cc))

    # Identify freed rows/cols by removing forbidden edges used in free matching
    partial_row_to_col = row_to_col_free[:]
    freed_rows: List[int] = []
    freed_cols: List[int] = []
    for rr, cc in forbid_edges_in_free:
        if partial_row_to_col[rr] == cc:
            partial_row_to_col[rr] = -1
            freed_rows.append(rr)
            freed_cols.append(cc)

    # Also include rows that became free due to forcing new edges
    for rr, c in enumerate(partial_row_to_col):
        if c == -1 and rr not in freed_rows:
            freed_rows.append(rr)

    # Build p from partial matching
    p = [0] * (m + 1)
    for i in range(m):
        j = partial_row_to_col[i]
        if j != -1:
            p[j + 1] = i + 1

    free_cols_count = sum(1 for j in range(1, m + 1) if p[j] == 0)
    if len(freed_rows) != free_cols_count:
        raise ValueError("Free rows/cols mismatch after enforcing constraints.")

    if not freed_rows:
        total = compute_total_cost(Cprime, row_to_col_work)
        stats = {
            "augmentations": 0,
            "inner_iterations_total": 0,
            "dual_update_count": 0,
            "dual_delta_sum": 0.0,
            "forbidden_edges": forbid_edges,
            "forced_edges": force_edges,
            "bigM": bigM,
            "note": "Matching is already complete in the free subproblem.",
        }
        return row_to_col_work, total, u[:], v[:], stats

    # Load duals into 1-index arrays
    U = [0.0] * (m + 1)
    V = [0.0] * (m + 1)
    for i in range(1, m + 1):
        U[i] = u_free[i - 1]
        V[i] = v_free[i - 1]

    way = [0] * (m + 1)
    INF = 10**18

    def log(msg: str):
        if verbose:
            print(msg)

    inner_iterations_total = 0
    dual_update_count = 0
    dual_delta_sum = 0.0
    per_augmentation_iters: List[int] = []
    per_augmentation_dual_updates: List[int] = []

    for r_free in freed_rows:
        i = r_free + 1
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)

        inner_iters = 0
        dual_updates_i = 0

        log(f"Start augmentation from free row r={r_free}")

        while True:
            inner_iters += 1
            inner_iterations_total += 1
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = Cfree[i0 - 1][j - 1] - U[i0] - V[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            if delta < 0 and delta > -1e-9:
                delta = 0.0

            if delta > eps:
                dual_update_count += 1
                dual_updates_i += 1
                dual_delta_sum += delta

            log(
                f"  step {inner_iters}: j1={j1-1} (1-based {j1}), delta={delta:g}, "
                f"dual_change={'YES' if delta > eps else 'no'}"
            )

            for j in range(0, m + 1):
                if used[j]:
                    U[p[j]] += delta
                    V[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                log(f"  reached free column c={j0-1}; augmenting path")
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

        per_augmentation_iters.append(inner_iters)
        per_augmentation_dual_updates.append(dual_updates_i)

    # Extract new matching for free subproblem
    new_row_to_col_free = p_to_row_to_col(p)

    # Reconstruct full matching with forced edges preserved
    new_row_to_col = row_to_col_work[:]
    for r in free_rows:
        rr = row_map[r]
        cc = new_row_to_col_free[rr]
        new_row_to_col[r] = free_cols[cc]

    if any(x == -1 for x in new_row_to_col):
        raise RuntimeError("Reconstruction failed; matching is not complete.")

    total_cost = compute_total_cost(Cprime, new_row_to_col)

    # Merge updated duals back (free rows/cols only)
    new_u = u[:]
    new_v = v[:]
    for r in free_rows:
        rr = row_map[r]
        new_u[r] = U[rr + 1]
    for c in free_cols:
        cc = col_map[c]
        new_v[c] = V[cc + 1]

    stats: Dict[str, Any] = {
        "augmentations": len(freed_rows),
        "inner_iterations_total": inner_iterations_total,
        "dual_update_count": dual_update_count,
        "dual_delta_sum": dual_delta_sum,
        "per_augmentation_iters": per_augmentation_iters,
        "per_augmentation_dual_updates": per_augmentation_dual_updates,
        "forbidden_edges": forbid_edges,
        "forced_edges": force_edges,
        "enforced_new_forces": enforced_new_forces,
        "forbidden_edges_in_free": forbid_edges_in_free,
        "bigM": bigM,
        "note": "Duals are updated for free rows/cols only.",
    }
    return new_row_to_col, total_cost, new_u, new_v, stats


if __name__ == "__main__":
    C = [
        [2, 3, 1, 1],
        [5, 8, 3, 2],
        [4, 9, 5, 1],
        [8, 7, 8, 4],
    ]

    row_to_col = [0, 2, 3, 1]
    u = [2.0, 4.0, 4.0, 7.0]
    v = [0.0, 0.0, -1.0, -3.0]

    # Force an edge not in the current matching
    forbids = [(2, 3)]
    forces = [(0, 1)]

    new_match, new_cost, new_u, new_v, stats = forbid_and_force_edges_reoptimize_allow_new_forces(
        C, row_to_col, u, v, forbid_edges=forbids, force_edges=forces, verbose=True
    )

    print("\nOld row->col:", row_to_col)
    print("New row->col:", new_match)
    print("Stats:", stats)
    print("New total cost (with forbidden edges penalized):", new_cost)
