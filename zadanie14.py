import numpy as np
import matplotlib.pyplot as plt

def integrand_task14(t):
    numerator = 1 + t
    denominator = t ** 2 + 0.04
    return np.cos(numerator / denominator) * np.exp(-t ** 2)


def calculate_dynamic_A(epsilon):
    limit = np.sqrt(-np.log(epsilon))
    A = np.ceil(limit) + 1.0
    return A


def romberg_integration(func, a, b, tolerance=1e-8, max_iter=20):
    R = []
    h = b - a

    trapezoid_0 = 0.5 * h * (func(a) + func(b))
    R.append([trapezoid_0])

    for k in range(1, max_iter):
        # Rekurencyjny wzór trapezów
        num_new_points = 2 ** (k - 1)
        h_k = (b - a) / (2 ** k)

        indices = np.arange(1, num_new_points + 1)
        new_points = a + (2 * indices - 1) * h_k

        sum_new = np.sum(func(new_points))
        t_k0 = 0.5 * R[k - 1][0] + h_k * sum_new

        current_row = [t_k0]

        # Ekstrapolacja Richardsona
        for m in range(1, k + 1):
            t_curr_m = current_row[m - 1]
            t_prev_m = R[k - 1][m - 1]

            correction = (t_curr_m - t_prev_m) / (4 ** m - 1)
            t_km = t_curr_m + correction
            current_row.append(t_km)

        R.append(current_row)

        # Warunek stopu (na przekątnej)
        if k >= 1:
            diff = abs(R[k][k] - R[k - 1][k - 1])
            if diff < tolerance:
                return R[k][k], diff, k

    return R[-1][-1], abs(R[-1][-1] - R[-2][-2]), max_iter


target_accuracy = 1e-8

A = calculate_dynamic_A(target_accuracy)
print(f"Zakres całkowania: [{-A}, {A}]")

limit_val, error, iters = romberg_integration(integrand_task14, -A, A, tolerance=target_accuracy)
print(f"\nObliczona granica (lim x->inf F(x)): {limit_val:.10f}")
print(f"Liczba iteracji: {iters}")

x_vals = np.linspace(-A, A, 500)
y_vals = integrand_task14(x_vals)
F_vals = np.cumsum(y_vals) * (x_vals[1] - x_vals[0])

plt.plot(x_vals, F_vals, label="F(x)")
plt.axhline(limit_val, color='r', linestyle='--', label=f"Granica: {limit_val:.5f}")
plt.legend()
plt.title("Wykres F(x)")
plt.grid(True)
plt.show()