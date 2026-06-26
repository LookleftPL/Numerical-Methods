import math
import numpy as np
import random

def func(vec):
    x, y = vec
    term1 = 0.25 * x ** 4
    term2 = y ** 2
    term3 = -0.5 * x ** 2
    term4 = 0.125 * x
    term5 = 0.0625 * (x - y)
    return term1 + term2 + term3 + term4 + term5

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

            def line_func(alpha): return func(x_curr + alpha * d)

            bracket = bracket_minimum(line_func, step=0.05)
            alpha_opt, _ = brent_solver(line_func, bracket)
            x_curr = x_curr + alpha_opt * d

        new_direction = x_curr - x_start_iter
        if np.linalg.norm(new_direction) < tol:
            break

        def line_func_new(alpha):
            return func(x_curr + alpha * new_direction)

        bracket = bracket_minimum(line_func_new, step=0.05)
        alpha_opt, _ = brent_solver(line_func_new, bracket)
        x_curr = x_curr + alpha_opt * new_direction
        directions = np.vstack([directions[1:], new_direction])

    return x_curr, func(x_curr)

print("Liczba losowych punktów startowych: 128\n")

random.seed(12345)
num_attempts = 128
bounds = (-3.0, 3.0)

found_minima = []
tolerance = 1e-3

for i in range(num_attempts):
    start_vec = [random.uniform(*bounds) for i in range(2)]

    min_vec, min_val = powell_method(start_vec)

    is_new = True
    for m in found_minima:
        dist = np.linalg.norm(min_vec - m['coords'])
        if dist < tolerance:
            m['hits'] += 1
            is_new = False
            break

    if is_new:
        found_minima.append({'coords': min_vec, 'value': min_val, 'hits': 1})

found_minima.sort(key=lambda x: x['value'])

print(f"{'Znalezione Minimum (x, y)':<35} | {'Wartość f(x,y)':<15} | {'Trafienia'}")
print("-" * 80)

for m in found_minima:
    coords_str = f"[{m['coords'][0]:.5f}, {m['coords'][1]:.5f}]"
    val_str = f"{m['value']:.8f}"

    percent = (m['hits'] / num_attempts) * 100
    hits_str = f"{m['hits']}/128 ({percent:.1f}%)"

    print(f"{coords_str:<35} | {val_str:<15} | {hits_str}")