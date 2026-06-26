import math

def f(x):
    return 0.25 * x ** 4 - 0.5 * x ** 2 - 0.0625 * x
def df(x):
    return x ** 3 - x - 0.0625
def ddf(x):
    return 3 * x ** 2 - 1
def scan_for_bracket(start, end, samples=1000):
    step = (end - start) / samples

    best_x = start
    min_val = f(start)

    for i in range(1, samples + 1):
        curr_x = start + i * step
        curr_val = f(curr_x)
        if curr_val < min_val:
            min_val = curr_val
            best_x = curr_x

    return best_x - step, best_x + step

def golden_section_solver(a, b, tol=1e-6):
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi ~= 0.618
    invphi2 = (3 - math.sqrt(5)) / 2  # (1-invphi) ~= 0.382

    x1 = a + invphi2 * (b - a)
    x2 = a + invphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    iters = 0
    while (b - a) > tol:
        iters += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + invphi2 * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + invphi * (b - a)
            f2 = f(x2)
    xmin = (a + b) / 2
    return xmin, f(xmin), iters

def brent_solver(a, b, tol=1e-6):
    invphi = (3 - math.sqrt(5)) / 2

    x = w = v = a + invphi * (b - a)
    fx = fw = fv = f(x)

    d = e = 0.0
    iters = 0

    m = 0.5 * (a + b)
    while abs(x - m) > tol - 0.5 * (b - a):
        if max(x - a, b - x) < tol:
            break
        iters += 1
        para_ok = False
        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0: p = -p
            q = abs(q)
            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                e = d
                d = p / q
                u = x + d
                # Jeśli u jest zbyt blisko krawędzi, robimy mały krok
                if (u - a) < 2 * tol or (b - u) < 2 * tol:
                    d = math.copysign(tol, m - x)
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

        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

        m = 0.5 * (a + b)
    return x, fx, iters


RANGE_START = -3.0
RANGE_END = 3.0
SAMPLES = 1000
EPSILON = 1e-6

print(f"Minimalizacja f(x) na przedziale [{RANGE_START}, {RANGE_END}]")
iso_a, iso_b = scan_for_bracket(RANGE_START, RANGE_END, SAMPLES)
print(f"Znaleziono wąski przedział startowy: [{iso_a:.4f}, {iso_b:.4f}]")

gs_x, gs_y, gs_iter = golden_section_solver(iso_a, iso_b, EPSILON)
print(f"Metoda Złotego Podziału:")
print(f"x_min = {gs_x:.10f}")
print(f"y_min = {gs_y:.10f}")
print(f"Iteracje: {gs_iter}")

br_x, br_y, br_iter = brent_solver(iso_a, iso_b, EPSILON)
print(f"\nMetoda Brenta:")
print(f"x_min = {br_x:.10f}")
print(f"y_min = {br_y:.10f}")
print(f"Iteracje: {br_iter}")

check_df = abs(df(br_x))
check_ddf = ddf(br_x)
print(f"Sprawdzenie pochodnych dla wyniku Brenta:")
print(f"|f'(x)| = {check_df:.2e}")
print(f"f''(x)  = {check_ddf:.4f}")
if check_df < 1e-5 and check_ddf > 0:
    print("\nPunkt jest poprawnym minimum lokalnym.")
    if br_iter < gs_iter:
        print(f"Metoda Brenta była szybsza o {gs_iter - br_iter} iteracji.")
