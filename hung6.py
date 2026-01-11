from typing import List, Tuple, Dict, Any, Optional

def compute_total_cost(C: List[List[float]], row_to_col: List[int]) -> float:
    return sum(C[i][j] for i, j in enumerate(row_to_col))

def row_to_col_to_p(row_to_col: List[int]) -> List[int]:
    """
    Convert row->col matching (0-based) into p array (1-based) where:
      p[j] = row assigned to column j, and p[0] is scratch (Hungarian convention).
    """
    n = len(row_to_col)
    p = [0] * (n + 1)
    for i, j in enumerate(row_to_col):
        if j < 0 or j >= n:
            raise ValueError("row_to_col must be a perfect matching with columns in [0..n-1].")
        p[j + 1] = i + 1
    if any(pj == 0 for pj in p[1:]):
        raise ValueError("row_to_col does not represent a perfect matching (some column unmatched).")
    return p

def p_to_row_to_col(p: List[int]) -> List[int]:
    """Inverse of row_to_col_to_p."""
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

def forbid_edge_and_reoptimize_one_augmentation(
    C: List[List[float]],
    row_to_col: List[int],
    u: List[float],
    v: List[float],
    forbid_edge: Tuple[int, int],
    bigM: Optional[float] = None,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Tuple[List[int], float, List[float], List[float], Dict[str, Any]]:
    """
    FULLY incremental strategy:
      - Forbid one edge (r,c) by setting its cost to bigM
      - Remove it from the current optimal matching (if present)
      - Re-optimize by running EXACTLY ONE Hungarian augmentation, warm-started with current duals u,v,
        reusing the rest of the matching.

    Assumptions:
      - Square problem (n x n)
      - row_to_col is a perfect matching before forbidding
      - u, v are dual potentials (length n each). They should be dual-feasible for the original C.
        After increasing one cost, they remain feasible.

    Returns:
      new_row_to_col (perfect matching),
      total_cost w.r.t. modified cost matrix C',
      new_u, new_v,
      stats (including inner_iterations and dual_update_count for this single augmentation)
    """
    n = len(C)
    if n == 0 or any(len(row) != n for row in C):
        raise ValueError("C must be square (n x n).")
    if len(row_to_col) != n or len(u) != n or len(v) != n:
        raise ValueError("row_to_col, u, v must have length n.")

    r_forbid, c_forbid = forbid_edge
    if not (0 <= r_forbid < n and 0 <= c_forbid < n):
        raise ValueError("forbid_edge must be within matrix bounds.")

    # Choose bigM if not provided
    if bigM is None:
        cmax = max(max(row) for row in C)
        bigM = cmax * n * 10.0 + 1.0  # safe "very large" for minimization

    # Create modified cost matrix C' (copy-on-write row copy for the forbidden entry)
    Cprime = [row[:] for row in C]
    Cprime[r_forbid][c_forbid] = bigM

    # Start from existing matching, remove forbidden edge if it was used
    old_col = row_to_col[r_forbid]
    if old_col != c_forbid:
        # The forbidden edge isn't currently used; the current matching remains feasible.
        # Still, the *optimal* solution might change, but the "critical question" case assumes it was in the solution.
        # We'll still run a 0-augmentation "repair" only if you want; here we just return current matching.
        stats = {
            "note": "Forbidden edge was not in the current matching; matching unchanged.",
            "inner_iterations": 0,
            "dual_update_count": 0,
        }
        total = compute_total_cost(Cprime, row_to_col)
        return row_to_col[:], total, u[:], v[:], stats

    # Remove (r_forbid, c_forbid) from matching => one free row (r_forbid) and one free column (c_forbid)
    partial_row_to_col = row_to_col[:]
    partial_row_to_col[r_forbid] = -1

    # Build p (col->row) from the partial matching: p[j]=row, with one free column (c_forbid+1) having p=0
    p = [0] * (n + 1)
    for i in range(n):
        j = partial_row_to_col[i]
        if j != -1:
            p[j + 1] = i + 1
    # Sanity: forbidden column should be free
    if p[c_forbid + 1] != 0:
        raise RuntimeError("Internal error: forbidden column is not free after removing edge.")

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

    # One augmentation from the now-free row
    i = r_forbid + 1
    p[0] = i
    j0 = 0
    minv = [INF] * (n + 1)
    used = [False] * (n + 1)

    inner_iters = 0
    dual_update_count = 0
    dual_delta_sum = 0.0

    log(f"Start one-augmentation repair from free row r={r_forbid}, forbidden edge ({r_forbid},{c_forbid}) cost={bigM}")

    while True:
        inner_iters += 1
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

        # Numerical safety
        if delta < 0 and delta > -1e-9:
            delta = 0.0

        if delta > eps:
            dual_update_count += 1
            dual_delta_sum += delta

        log(f"  step {inner_iters}: j1={j1-1} (1-based {j1}), delta={delta:g}, dual_change={'YES' if delta>eps else 'no'}")

        # Update duals
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

    # Extract new perfect matching
    new_row_to_col = p_to_row_to_col(p)

    # Compute total cost under modified matrix C'
    total_cost = compute_total_cost(Cprime, new_row_to_col)

    # Output updated duals (drop index 0)
    new_u = [U[i] for i in range(1, n + 1)]
    new_v = [V[j] for j in range(1, n + 1)]

    stats: Dict[str, Any] = {
        "inner_iterations": inner_iters,
        "dual_update_count": dual_update_count,
        "dual_delta_sum": dual_delta_sum,
        "forbidden_edge": (r_forbid, c_forbid),
        "bigM": bigM,
    }
    return new_row_to_col, total_cost, new_u, new_v, stats


def row_to_col_to_X(row_to_col: List[int]) -> List[List[int]]:
    n = len(row_to_col)
    X = [[0] * n for _ in range(n)]
    for i, j in enumerate(row_to_col):
        X[i][j] = 1
    return X


if __name__ == "__main__":
    C = [
        [2, 3, 1, 1],
        [5, 8, 3, 2],
        [4, 9, 5, 1],
        [8, 7, 8, 4],
    ]

    # Suppose current optimum is row->col = [0,2,3,1] (from earlier)
    row_to_col = [0, 2, 3, 1]

    # And we have some duals (example set that is feasible/optimal for the original)
    u = [2.0, 4.0, 4.0, 7.0]
    v = [0.0, 0.0, -1.0, -3.0]

    # Forbid an edge that is currently used: (2,3)
    forbid = (2, 3)

    new_match, new_cost, new_u, new_v, stats = forbid_edge_and_reoptimize_one_augmentation(
        C, row_to_col, u, v, forbid_edge=forbid, verbose=True
    )

    print("\nOld row->col:", row_to_col)
    print("New row->col:", new_match)
    print("Stats:", stats)
    print("New total cost (with forbidden edge penalized):", new_cost)

    X = row_to_col_to_X(new_match)
    print("\nNew X:")
    for row in X:
        print(row)
