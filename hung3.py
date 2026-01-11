from typing import List, Optional, Tuple
import math

def _pad_to_square(C: List[List[float]], pad_value: float = 0.0):
    """Pad rectangular cost matrix to square with pad_value. Returns (Cpad, n_rows, n_cols)."""
    m = len(C)
    n = len(C[0]) if m else 0
    N = max(m, n)
    Cpad = [row[:] + [pad_value] * (N - n) for row in C]
    for _ in range(N - m):
        Cpad.append([pad_value] * N)
    return Cpad, m, n

def _repair_duals_to_feasible(
    C: List[List[float]],
    u: List[float],
    v: List[float],
    iters: int = 2,
) -> Tuple[List[float], List[float]]:
    """
    Make (u,v) dual-feasible for minimization assignment:
        u_i + v_j <= C_ij for all i,j
    by alternating tightening steps:
        u_i := min_j (C_ij - v_j)
        v_j := min_i (C_ij - u_i)
    This guarantees feasibility after each update.
    """
    n = len(C)
    u = u[:]
    v = v[:]

    # Ensure correct sizes
    if len(u) != n or len(v) != n:
        raise ValueError("u0 and v0 must be length N (after padding to square).")

    # Alternating projection-ish tightening
    for _ in range(max(1, iters)):
        # Update u to be as large as possible while keeping feasibility given v
        for i in range(n):
            u[i] = min(C[i][j] - v[j] for j in range(n))
        # Update v similarly given u
        for j in range(n):
            v[j] = min(C[i][j] - u[i] for i in range(n))

    # Optional normalization (doesn't affect feasibility):
    # shifting u by +t and v by -t keeps u+v unchanged.
    # We keep as-is.

    return u, v

def hungarian_warm_start(
    cost_matrix: List[List[float]],
    u0: Optional[List[float]] = None,
    v0: Optional[List[float]] = None,
    repair_iters: int = 2,
    pad_value: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float, List[float], List[float]]:
    """
    Hungarian (shortest augmenting path) with optional warm-start duals.

    Args:
        cost_matrix: m x n cost matrix (minimization)
        u0, v0: optional guessed duals (potentials). If rectangular, provide for the padded square size,
                OR pass None and we will start from zeros.
        repair_iters: how many tightening rounds to repair guessed duals into feasibility.
        pad_value: padding cost for rectangular matrices. If None, uses a safe large value.

    Returns:
        assignment: list of (row, col) pairs for original matrix shape (only real rows/cols)
        total_cost: objective value for original matrix
        u, v: final dual potentials for the padded square problem (length N each)
    """
    if not cost_matrix or not cost_matrix[0]:
        return [], 0.0, [], []

    m = len(cost_matrix)
    n = len(cost_matrix[0])

    # Choose padding value for rectangular case:
    # We want dummy assignments to be unattractive for minimization.
    # Safe choice: max(C) * 10 + 1 (or 0 if already square).
    flat = [x for row in cost_matrix for x in row]
    Cmax = max(flat)
    if pad_value is None:
        pad_value = 0.0 if m == n else (Cmax * 10.0 + 1.0)

    Cpad, m0, n0 = _pad_to_square(cost_matrix, pad_value=pad_value)
    N = len(Cpad)

    # Initialize duals (0-indexed here, but Hungarian core below uses 1-index arrays)
    if u0 is None:
        u_init = [0.0] * N
    else:
        u_init = u0[:]
    if v0 is None:
        v_init = [0.0] * N
    else:
        v_init = v0[:]

    # Repair to feasibility (if warm-start provided; also fine for zeros)
    u_init, v_init = _repair_duals_to_feasible(Cpad, u_init, v_init, iters=repair_iters)

    # Standard shortest augmenting path Hungarian (1-indexed arrays)
    INF = 10**18
    u = [0.0] * (N + 1)
    v = [0.0] * (N + 1)

    # Load warm-start into 1-indexed arrays
    for i in range(1, N + 1):
        u[i] = u_init[i - 1]
        v[i] = v_init[i - 1]

    p = [0] * (N + 1)     # p[j] = row matched to column j
    way = [0] * (N + 1)

    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (N + 1)
        used = [False] * (N + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, N + 1):
                if not used[j]:
                    cur = Cpad[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Update duals
            for j in range(0, N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augment
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build assignment row->col for padded problem
    row_to_col = [-1] * N
    for j in range(1, N + 1):
        row_to_col[p[j] - 1] = j - 1

    # Extract assignment for original shape (ignore dummy rows/cols)
    assignment = []
    total_cost = 0.0
    for i in range(m0):
        j = row_to_col[i]
        if 0 <= j < n0:
            assignment.append((i, j))
            total_cost += cost_matrix[i][j]
        else:
            # Row matched to dummy column: happens when m > n.
            # That's expected in rectangular padding; it means "unassigned" in original.
            pass

    u_out = [u[i] for i in range(1, N + 1)]
    v_out = [v[j] for j in range(1, N + 1)]
    return assignment, total_cost, u_out, v_out


def assignment_to_X(m: int, n: int, assignment: List[Tuple[int, int]]) -> List[List[int]]:
    X = [[0]*n for _ in range(m)]
    for i, j in assignment:
        X[i][j] = 1
    return X

C = [
    [2, 3, 1, 1],
    [5, 8, 3, 2],
    [4, 9, 5, 1],
    [8, 7, 8, 4],
]

# A "guess": could come from a previous similar instance, a heuristic, or zeros.
u0_guess = [0, 0, 0, 0]
v0_guess = [0, 0, 0, 0]

assignment, total_cost, u, v = hungarian_warm_start(C, u0=u0_guess, v0=v0_guess, repair_iters=2)

X = assignment_to_X(len(C), len(C[0]), assignment)

print("Assignment:", assignment)
print("X:")
for row in X:
    print(row)
print("Cost:", total_cost)
