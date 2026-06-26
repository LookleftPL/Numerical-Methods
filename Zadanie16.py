import numpy as np

def evaluate_system(x, y):
    eq1 = 2 * x ** 2 + y ** 2 - 2
    eq2 = (x - 0.5) ** 2 + (y - 1) ** 2 - 0.25

    return np.array([eq1, eq2])

def evaluate_jacobian(x, y):
    dg1_dx = 4 * x
    dg1_dy = 2 * y
    dg2_dx = 2 * (x - 0.5)
    dg2_dy = 2 * (y - 1)
    return np.array([[dg1_dx, dg1_dy],[dg2_dx, dg2_dy]])

def solve_linear_2x2(matrix_J, vector_G):
    a = matrix_J[0, 0]
    b = matrix_J[0, 1]
    c = matrix_J[1, 0]
    d = matrix_J[1, 1]
    det = a * d - b * c
    rhs1 = -vector_G[0]
    rhs2 = -vector_G[1]
    # Wzory Cramera
    delta_x = (rhs1 * d - b * rhs2) / det
    delta_y = (a * rhs2 - rhs1 * c) / det

    return np.array([delta_x, delta_y])

def newton_multivariate_solver(start_x, start_y, tolerance=1e-8, max_iter=50):
    x = start_x
    y = start_y
    for i in range(max_iter):
        G = evaluate_system(x, y)
        J = evaluate_jacobian(x, y)

        delta = solve_linear_2x2(J, G)
        x = x + delta[0]
        y = y + delta[1]
        if np.sqrt(delta[0] ** 2 + delta[1] ** 2) < tolerance:
            return np.array([x, y])
    return None

guesses_x = [0.0, 1.0, -0.5]
guesses_y = [1.5, 0.5, 1.0]

found_solutions = []

for i in range(len(guesses_x)):
    gx = guesses_x[i]
    gy = guesses_y[i]

    sol = newton_multivariate_solver(gx, gy)
    if sol is not None:
        is_new = True
        for j in range(len(found_solutions)):
            existing = found_solutions[j]
            dist = np.sqrt((sol[0] - existing[0]) ** 2 + (sol[1] - existing[1]) ** 2)
            if dist < 1e-5:
                is_new = False

        if is_new:
            found_solutions.append(sol)

for i in range(len(found_solutions)):
    sol = found_solutions[i]
    x_val = sol[0]
    y_val = sol[1]

    res = evaluate_system(x_val, y_val)
    err = np.sqrt(res[0] ** 2 + res[1] ** 2)

    print(f"Rozwiązanie {i + 1}:")
    print(f"x = {x_val:.8f}")
    print(f"y = {y_val:.8f}")
    print(f"Błąd (norma reszt): {err:.2e}")
