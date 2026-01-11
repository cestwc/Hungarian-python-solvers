import numpy as np

def hungarian_algorithm(cost_matrix):
    cost = np.array(cost_matrix, dtype=float)
    n = cost.shape[0]

    cost -= cost.min(axis=1).reshape(n, 1)
    cost -= cost.min(axis=0)

    starred = np.zeros_like(cost, dtype=bool)
    primed = np.zeros_like(cost, dtype=bool)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if cost[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                starred[i, j] = True
                row_covered[i] = True
                col_covered[j] = True

    row_covered[:] = False
    col_covered[:] = False

    def cover_columns():
        for j in range(n):
            if np.any(starred[:, j]):
                col_covered[j] = True

    cover_columns()

    while col_covered.sum() < n:
        done = False
        while not done:
            found = False
            for i in range(n):
                if row_covered[i]:
                    continue
                for j in range(n):
                    if cost[i, j] == 0 and not col_covered[j]:
                        primed[i, j] = True
                        found = True

                        star_col = np.where(starred[i])[0]
                        if len(star_col) == 0:
                            path = [(i, j)]
                            r, c = i, j
                            while True:
                                star_row = np.where(starred[:, c])[0]
                                if len(star_row) == 0:
                                    break
                                r = star_row[0]
                                path.append((r, c))
                                c = np.where(primed[r])[0][0]
                                path.append((r, c))

                            for r, c in path:
                                starred[r, c] = not starred[r, c]

                            primed[:] = False
                            row_covered[:] = False
                            col_covered[:] = False
                            cover_columns()
                            done = True
                            break
                        else:
                            row_covered[i] = True
                            col_covered[star_col[0]] = False
                        break
                if found:
                    break
            if not found:
                uncovered_rows = ~row_covered
                uncovered_cols = ~col_covered

                if np.any(uncovered_rows) and np.any(uncovered_cols):
                    min_uncovered = cost[np.ix_(uncovered_rows, uncovered_cols)].min()
                    cost[row_covered] += min_uncovered
                    cost[:, uncovered_cols] -= min_uncovered
                else:
                    done = True

    assignment = [(i, np.where(starred[i])[0][0]) for i in range(n)]
    total_cost = sum(cost_matrix[i][j] for i, j in assignment)

    return assignment, total_cost
    
C = [
    [2, 3, 1, 1],
    [5, 8, 3, 2],
    [4, 9, 5, 1],
    [8, 7, 8, 4]
]

assignment, total_cost = hungarian_algorithm(C)

n = len(C)

# Build solution matrix X
X = [[0]*n for _ in range(n)]
for i, j in assignment:
    X[i][j] = 1

print("Cost matrix C:")
for row in C:
    print(row)

print("\nSolution matrix X:")
for row in X:
    print(row)

print("\nMinimum total cost:", total_cost)
