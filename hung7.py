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


def forbid_edges_and_reoptimize_multi_augmentation(
    C: List[List[float]],
    row_to_col: List[int],
    u: List[float],
    v: List[float],
    forbid_edges: Iterable[Tuple[int, int]],
    bigM: Optional[float] = None,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Tuple[List[int], float, List[float], List[float], Dict[str, Any]]:
    """
    Incremental strategy for forbidding MULTIPLE edges:
      - Forbid each edge (r,c) by setting its cost to bigM
      - Remove all forbidden edges that are currently used by the matching
      - Re-optimize by running ONE Hungarian augmentation per freed row,
        warm-started with the current duals u,v and reusing the rest of the matching.

    Assumptions:
      - Square problem (n x n)
      - row_to_col is a perfect matching before forbidding
      - u, v are dual potentials (length n each), dual-feasible for original C
        After increasing costs, they remain feasible.

    Returns:
      new_row_to_col (perfect matching),
      total_cost w.r.t. modified cost matrix C',
      new_u, new_v,
      stats (including inner_iterations_total and dual_update_count).
    """
    n = len(C)
    if n == 0 or any(len(row) != n for row in C):
        raise ValueError("C must be square (n x n).")
    if len(row_to_col) != n or len(u) != n or len(v) != n:
        raise ValueError("row_to_col, u, v must have length n.")

    # Choose bigM if not provided
    if bigM is None:
        cmax = max(max(row) for row in C)
        bigM = cmax * n * 10.0 + 1.0

    # Create modified cost matrix C'
    Cprime = [row[:] for row in C]
    forbid_edges = list(forbid_edges)
    for r, c in forbid_edges:
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError("forbid_edges must be within matrix bounds.")
        Cprime[r][c] = bigM

    # Remove forbidden edges that are currently used
    partial_row_to_col = row_to_col[:]
    freed_rows: List[int] = []
    freed_cols: List[int] = []
    for r, c in forbid_edges:
        if partial_row_to_col[r] == c:
            partial_row_to_col[r] = -1
            freed_rows.append(r)
            freed_cols.append(c)

    # If no matched edges were removed, matching is still feasible; return early
    if not freed_rows:
        stats = {
            "note": "No forbidden edge was in the current matching; matching unchanged.",
            "augmentations": 0,
            "inner_iterations_total": 0,
            "dual_update_count": 0,
            "dual_delta_sum": 0.0,
            "forbidden_edges": forbid_edges,
            "bigM": bigM,
        }
        total = compute_total_cost(Cprime, row_to_col)
        return row_to_col[:], total, u[:], v[:], stats

    # Build p (col->row) from the partial matching: p[j]=row, with freed columns having p=0
    p = [0] * (n + 1)
    for i in range(n):
        j = partial_row_to_col[i]
        if j != -1:
            p[j + 1] = i + 1

    # Load duals into 1-index arrays (Hungarian convention). v[0] exists and stays 0.
    U = [0.0] * (n + 1)
    V = [0.0] * (n + 1)
    for i in range(1, n + 1):
        U[i] = u[i - 1]
        V[i] = v[i - 1]

    way = [0] * (n + 1)
    INF = 10**18

    def log(msg: str):
        if verbose:
            print(msg)

    inner_iterations_total = 0
    dual_update_count = 0
    dual_delta_sum = 0.0
    per_augmentation_iters: List[int] = []
    per_augmentation_dual_updates: List[int] = []

    # One augmentation per freed row
    for r_free in freed_rows:
        i = r_free + 1
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)

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

            for j in range(1, n + 1):
                if not used[j]:
                    cur = Cprime[i0 - 1][j - 1] - U[i0] - V[j]
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

            log(f"  step {inner_iters}: j1={j1-1} (1-based {j1}), delta={delta:g}, "
                f"dual_change={'YES' if delta > eps else 'no'}")

            for j in range(0, n + 1):
                if used[j]:
                    U[p[j]] += delta
                    V[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                log(f"  reached free column c={j0-1}; augmenting path")
                break

        # Augment along the path
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

        per_augmentation_iters.append(inner_iters)
        per_augmentation_dual_updates.append(dual_updates_i)

    # Extract new perfect matching
    new_row_to_col = p_to_row_to_col(p)

    # Compute total cost under modified matrix C'
    total_cost = compute_total_cost(Cprime, new_row_to_col)

    # Output updated duals (drop index 0)
    new_u = [U[i] for i in range(1, n + 1)]
    new_v = [V[j] for j in range(1, n + 1)]

    stats: Dict[str, Any] = {
        "augmentations": len(freed_rows),
        "inner_iterations_total": inner_iterations_total,
        "dual_update_count": dual_update_count,
        "dual_delta_sum": dual_delta_sum,
        "per_augmentation_iters": per_augmentation_iters,
        "per_augmentation_dual_updates": per_augmentation_dual_updates,
        "forbidden_edges": forbid_edges,
        "forbidden_edges_in_matching": list(zip(freed_rows, freed_cols)),
        "bigM": bigM,
    }
    return new_row_to_col, total_cost, new_u, new_v, stats


if __name__ == "__main__":
    C = [
        [2, 3, 1, 1],
        [5, 8, 3, 2],
        [4, 9, 5, 1],
        [8, 7, 8, 4],
    ]

    # Suppose current optimum is row->col = [0,2,3,1]
    row_to_col = [0, 2, 3, 1]
    u = [2.0, 4.0, 4.0, 7.0]
    v = [0.0, 0.0, -1.0, -3.0]

    # Forbid two edges that are currently used: (1,2) and (2,3)
    forbids = [(1, 2), (2, 3)]

    new_match, new_cost, new_u, new_v, stats = forbid_edges_and_reoptimize_multi_augmentation(
        C, row_to_col, u, v, forbid_edges=forbids, verbose=True
    )

    print("\nOld row->col:", row_to_col)
    print("New row->col:", new_match)
    print("Stats:", stats)
    print("New total cost (with forbidden edges penalized):", new_cost)
