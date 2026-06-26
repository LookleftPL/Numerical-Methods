import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 7 - 2 * x ** 5 + 3 * x ** 4 - x + 1

def brent_solver(a, b, tol=1e-6):
    invphi = (3 - math.sqrt(5)) / 2
    x = w = v = a + invphi * (b - a)
    fx = fv = fw = f(x)
    d = e = 0.0

    iters = 0
    m = 0.5 * (a + b)
    tol2 = 2.0 * tol

    while abs(x - m) > (tol2 - 0.5 * (b - a)):
        iters += 1

        if (b - a) < tol:
            break

        para_ok = False

        # Próba interpolacji parabolicznej
        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)

            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                e = d
                d = p / q
                u = x + d

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

RANGE_START = 0.0
RANGE_END = 1.0
TOLERANCE = 1e-6

print(f"Minimalizacja wielomianu W(x) na przedziale [{RANGE_START}, {RANGE_END}]")

br_x, br_y, br_iter = brent_solver(RANGE_START, RANGE_END, TOLERANCE)
print("\nMetoda Brenta:")
print(f"x_min = {br_x:.10f}")
print(f"y_min = {br_y:.10f}")
print(f"Iteracje: {br_iter}")

x_plot = np.linspace(-0.2, 1.2, 200)
y_plot = [f(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='W(x) = x^7 - 2x^5 + 3x^4 - x + 1', color='blue')
plt.plot(br_x, br_y, 'ro', label=f'Minimum ({br_x:.4f}, {br_y:.4f})', zorder=5)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('Minimalizacja wielomianu interpolacyjnego (Zadanie 18)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()