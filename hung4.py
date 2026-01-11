from typing import List, Optional, Tuple, Dict, Any

def _pad_to_square(C: List[List[float]], pad_value: float = 0.0):
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
    Tighten duals so u_i + v_j <= C_ij for all i,j.
    Alternating updates guarantee feasibility after each full round.
    """
    n = len(C)
    if len(u) != n or len(v) != n:
        raise ValueError("u0 and v0 must be length N (after padding to square).")

    u = u[:]
    v = v[:]

    rounds = max(0, iters)
    for _ in range(rounds):
        for i in range(n):
            u[i] = min(C[i][j] - v[j] for j in range(n))
        for j in range(n):
            v[j] = min(C[i][j] - u[i] for i in range(n))

    return u, v

def hungarian_warm_start_verbose(
    cost_matrix: List[List[float]],
    u0: Optional[List[float]] = None,
    v0: Optional[List[float]] = None,
    repair_iters: int = 2,
    pad_value: Optional[float] = None,
    eps: float = 1e-12,
    verbose: bool = True,
    max_print_steps: Optional[int] = 2000,  # prevent runaway logs
) -> Tuple[List[Tuple[int, int]], float, List[float], List[float], Dict[str, Any]]:
    """
    Hungarian (shortest augmenting path) with optional warm-start duals
    and explicit iteration/step tracing.

    Returns:
      assignment (on original shape),
      total_cost,
      final u, v (length N each, for padded square),
      stats dict containing iteration counts and deltas.
    """
    if not cost_matrix or not cost_matrix[0]:
        return [], 0.0, [], [], {"message": "Empty matrix"}

    m = len(cost_matrix)
    n = len(cost_matrix[0])
    flat = [x for row in cost_matrix for x in row]
    Cmax = max(flat)

    if pad_value is None:
        pad_value = 0.0 if m == n else (Cmax * 10.0 + 1.0)

    Cpad, m0, n0 = _pad_to_square(cost_matrix, pad_value=pad_value)
    N = len(Cpad)

    # Initialize guessed duals (0-indexed)
    u_init = [0.0] * N if u0 is None else u0[:]
    v_init = [0.0] * N if v0 is None else v0[:]

    # Feasibility repair (recommended when warm-starting)
    u_init, v_init = _repair_duals_to_feasible(Cpad, u_init, v_init, iters=repair_iters)

    # Stats collection
    stats: Dict[str, Any] = {
        "N": N,
        "augmentations": N,          # outer loop count (one per row)
        "inner_iterations_total": 0, # total inner while-loop iterations
        "inner_iterations_per_i": [],# list length N
        "delta_sequence": [],        # all deltas applied (in order)
        "delta_per_i": [],           # list of lists per i
        "repair_iters": repair_iters,
        "pad_value": pad_value,
    }

    INF = 10**18

    # 1-index arrays (classic implementation)
    u = [0.0] * (N + 1)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        u[i] = u_init[i - 1]
        v[i] = v_init[i - 1]

    p = [0] * (N + 1)   # p[j] = matched row for column j
    way = [0] * (N + 1)

    printed = 0
    def log(msg: str):
        nonlocal printed
        if not verbose:
            return
        if max_print_steps is not None and printed >= max_print_steps:
            return
        print(msg)
        printed += 1

    # Optional: check feasibility of initial duals
    # (u_i + v_j <= C_ij). Because we repaired, should pass up to eps.
    infeas_count = 0
    max_violation = 0.0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            viol = (u[i] + v[j]) - Cpad[i - 1][j - 1]
            if viol > eps:
                infeas_count += 1
                if viol > max_violation:
                    max_violation = viol
    stats["initial_dual_infeasibilities"] = infeas_count
    stats["initial_dual_max_violation"] = max_violation

    log(f"=== Hungarian warm-start (N={N}) ===")
    log(f"repair_iters={repair_iters}, pad_value={pad_value}")
    log(f"initial dual infeasibilities: {infeas_count}, max_violation={max_violation:g}")

    # Main algorithm
    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (N + 1)
        used = [False] * (N + 1)

        iters_i = 0
        deltas_i: List[float] = []

        log(f"\n--- Augmentation i={i}/{N} (start) ---")

        while True:
            iters_i += 1
            stats["inner_iterations_total"] += 1

            used[j0] = True
            i0 = p[j0]

            delta = INF
            j1 = 0

            # Relax edges / maintain shortest augmenting path potentials
            for j in range(1, N + 1):
                if not used[j]:
                    cur = Cpad[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Numerical safety: delta should be >= 0 for feasible duals;
            # allow tiny negatives due to float noise
            if delta < 0 and delta > -1e-9:
                delta = 0.0

            deltas_i.append(delta)
            stats["delta_sequence"].append(delta)

            log(f"  step {iters_i}: chosen next col j1={j1}, delta={delta:g}")

            # Update dual variables
            for j in range(0, N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            # If column j0 is free, we can augment
            if p[j0] == 0:
                log(f"  reached free column j0={j0}; augmenting path will be applied.")
                break

        # Augment matching along the path
        log("  augmenting...")
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

        stats["inner_iterations_per_i"].append(iters_i)
        stats["delta_per_i"].append(deltas_i)

        log(f"--- Augmentation i={i}/{N} (done); inner iters={iters_i} ---")

    # Build padded assignment (row -> col)
    row_to_col = [-1] * N
    for j in range(1, N + 1):
        row_to_col[p[j] - 1] = j - 1

    # Extract assignment for original shape
    assignment = []
    total_cost = 0.0
    for i in range(m0):
        j = row_to_col[i]
        if 0 <= j < n0:
            assignment.append((i, j))
            total_cost += cost_matrix[i][j]

    # Final duals (0-indexed)
    u_out = [u[i] for i in range(1, N + 1)]
    v_out = [v[j] for j in range(1, N + 1)]

    stats["total_cost"] = total_cost
    stats["assignment"] = assignment

    log("\n=== Done ===")
    log(f"assignment (original): {assignment}")
    log(f"total_cost: {total_cost:g}")
    log(f"inner_iterations_total: {stats['inner_iterations_total']}")

    return assignment, total_cost, u_out, v_out, stats

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

# Case A: no warm start (defaults to zeros)
a_asg, a_cost, a_u, a_v, a_stats = hungarian_warm_start_verbose(
    C, u0=None, v0=None, repair_iters=0, verbose=False
)

# Case B: some guessed duals (example guesses)
u_guess = [2, 4, 4, 7]
v_guess = [0, 0, -1, -3]
b_asg, b_cost, b_u, b_v, b_stats = hungarian_warm_start_verbose(
    C, u0=u_guess, v0=v_guess, repair_iters=1, verbose=False
)

print("Case A inner iterations total:", a_stats["inner_iterations_total"])
print("Case B inner iterations total:", b_stats["inner_iterations_total"])
print("Case A per augmentation:", a_stats["inner_iterations_per_i"])
print("Case B per augmentation:", b_stats["inner_iterations_per_i"])
