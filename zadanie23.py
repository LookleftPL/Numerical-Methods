import math
import numpy as np
import random

def rosenbrock_4d(vec):
    x1, x2, x3, x4 = vec
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2 + \
        100 * (x3 - x2 ** 2) ** 2 + 100 * (x4 - x3 ** 2) ** 2

def bracket_minimum(func, start_alpha=0.0, step=0.1):
    a, b = start_alpha, start_alpha + step
    fa, fb = func(a), func(b)
    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        step = -step
    c = b + step
    fc = func(c)
    while fc < fb:
        a, b, fa, fb = b, c, fb, fc
        step *= 1.618
        c = b + step
        fc = func(c)
    return (a, c)


def brent_solver(func, bracket_pair, tol=1e-7):
    a, b = bracket_pair
    if a > b: a, b = b, a
    invphi = (3 - math.sqrt(5)) / 2
    x = w = v = a + invphi * (b - a)
    fx = fv = fw = func(x)
    d = e = 0.0
    m = 0.5 * (a + b)

    for _ in range(100):
        if abs(x - m) <= 2.0 * tol - 0.5 * (b - a): break
        if (b - a) < tol: break
        para_ok = False
        if abs(e) > tol:
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r)
            if q > 0: p = -p
            q = abs(q)
            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                e = d;
                d = p / q;
                u = x + d
                if (u - a) < 2 * tol or (b - u) < 2 * tol: d = math.copysign(tol, m - x)
                para_ok = True
        if not para_ok:
            if x < m:
                e = b - x
            else:
                e = a - x
            d = invphi * e
        if abs(d) >= tol:
            u = x + d
        else:
            u = x + math.copysign(tol, d)
        fu = func(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw;
            w, fw = x, fx;
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw; w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        m = 0.5 * (a + b)
    return x, fx

def powell_method(start_point, tol=1e-6, max_iters=200):
    x_curr = np.array(start_point)
    n = len(start_point)
    directions = np.eye(n)

    iter_count = 0

    for k in range(max_iters):
        iter_count += 1
        x_start_iter = x_curr.copy()

        for i in range(n):
            d = directions[i]

            def line_func(alpha): return rosenbrock_4d(x_curr + alpha * d)

            bracket = bracket_minimum(line_func, step=0.05)  # Mniejszy krok startowy
            alpha_opt, _ = brent_solver(line_func, bracket)
            x_curr = x_curr + alpha_opt * d

        new_direction = x_curr - x_start_iter
        move_dist = np.linalg.norm(new_direction)

        if move_dist < tol:
            break

        def line_func_new(alpha):
            return rosenbrock_4d(x_curr + alpha * new_direction)

        bracket = bracket_minimum(line_func_new, step=0.05)
        alpha_opt, _ = brent_solver(line_func_new, bracket)
        x_curr = x_curr + alpha_opt * new_direction

        directions = np.vstack([directions[1:], new_direction])

    return x_curr, rosenbrock_4d(x_curr), iter_count


def solver(start_point):
    current_x = start_point
    total_iters = 0
    for restart in range(4):
        new_x, val, iters = powell_method(current_x, tol=1e-6)
        total_iters += iters

        dist = np.linalg.norm(np.array(new_x) - np.array(current_x))
        if dist < 1e-4:
            return new_x, val, total_iters

        current_x = new_x

    return current_x, val, total_iters

print("Minimalizacja funkcji Rosenbrocka (Metoda Powella)")
random.seed(42)
num_attempts = 3
bounds = (-2.0, 2.0)

print(f"{'Punkt Startowy':<45} | {'Znalezione Minimum':<45} | {'Wartość':<10} | {'Iteracje'}")
print("-" * 120)

for i in range(num_attempts):
    start_vec = [random.uniform(*bounds) for j in range(4)]

    min_vec, min_val, iters = solver(start_vec)

    start_str = f"[{start_vec[0]:.2f}, {start_vec[1]:.2f}, {start_vec[2]:.2f}, {start_vec[3]:.2f}]"
    end_str = f"[{min_vec[0]:.4f}, {min_vec[1]:.4f}, {min_vec[2]:.4f}, {min_vec[3]:.4f}]"

    print(f"{start_str:<45} | {end_str:<45} | {min_val:<10.2e} | {iters}")