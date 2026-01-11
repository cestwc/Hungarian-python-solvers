from typing import List, Tuple

def hungarian_min(cost: List[List[float]]) -> Tuple[List[int], List[float], List[float], float]:
    """
    Hungarian algorithm (minimization) for square matrices.
    Returns:
      assignment: list where assignment[i] = assigned column for row i (0-based)
      u: dual potentials for rows
      v: dual potentials for cols
      min_cost: total minimum cost
    """
    n = len(cost)
    if n == 0 or any(len(row) != n for row in cost):
        raise ValueError("Cost matrix must be non-empty and square (n x n).")

    INF = 10**18

    # 1-indexed arrays to match the standard concise implementation
    u = [0.0] * (n + 1)     # row potentials
    v = [0.0] * (n + 1)     # col potentials
    p = [0] * (n + 1)       # p[j] = row assigned to column j
    way = [0] * (n + 1)     # predecessor columns for augmenting path reconstruction

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]  # current row
            delta = INF
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Update potentials
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augmenting: flip assignments along the found path
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build row->col assignment from p[col]=row
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1

    min_cost = sum(cost[i][assignment[i]] for i in range(n))

    # Duals (drop index 0)
    u_vec = u[1:]
    v_vec = v[1:]

    return assignment, u_vec, v_vec, min_cost


def print_matrix(mat: List[List[float]], title: str) -> None:
    print(title)
    for row in mat:
        print("  " + " ".join(f"{x:>6g}" for x in row))
    print()


def assignment_to_X(assignment: List[int]) -> List[List[int]]:
    n = len(assignment)
    X = [[0] * n for _ in range(n)]
    for i, j in enumerate(assignment):
        X[i][j] = 1
    return X


if __name__ == "__main__":
    C = [
        [2, 3, 1, 1],
        [5, 8, 3, 2],
        [4, 9, 5, 1],
        [8, 7, 8, 4],
    ]

    assignment, u, v, min_cost = hungarian_min(C)
    X = assignment_to_X(assignment)

    print_matrix(C, "Cost matrix C:")
    print_matrix(X, "Solution (assignment) matrix X (1 means selected):")

    print("Assignment (row -> col, 0-based):", assignment)
    print("Minimum total cost:", min_cost)
    print()

    print("Dual potentials u (rows):", u)
    print("Dual potentials v (cols):", v)

    # Optional: verify dual feasibility and complementary slackness on assigned edges
    # Check: u[i] + v[j] <= C[i][j] for all i,j and equality for assigned pairs.
    ok = True
    for i in range(len(C)):
        for j in range(len(C)):
            if u[i] + v[j] - C[i][j] > 1e-9:
                ok = False
    print("\nDual feasibility check (u_i + v_j <= C_ij):", "OK" if ok else "FAILED")

    eq_ok = True
    for i, j in enumerate(assignment):
        if abs((u[i] + v[j]) - C[i][j]) > 1e-9:
            eq_ok = False
    print("Complementary slackness on chosen edges (u_i+v_j == C_ij):", "OK" if eq_ok else "FAILED")
