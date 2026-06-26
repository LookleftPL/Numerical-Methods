import numpy as np

def transformed_integrand(t):
    numerator = np.pi * (1 + t)
    denominator = 1 + t ** 4
    jacobian = 2 * t
    return np.sin(numerator / denominator) * np.exp(-t ** 2) * jacobian


def find_cutoff_A(epsilon):
    A = -np.log(epsilon)
    return np.ceil(A)


def romberg_integration(func, a, b, tolerance=1e-7, max_iter=14):
    R = []
    h = b - a

    # Krok 0
    trapezoid_0 = 0.5 * h * (func(a) + func(b))
    R.append([trapezoid_0])

    print(f"{'k':<3}")
    print(f"{0:<3} | {trapezoid_0:.12f}")

    for k in range(1, max_iter):
        #Rekurencyjny wzór trapezów
        num_new_points = 2 ** (k - 1)
        h_k = (b - a) / (2 ** k)

        indices = np.arange(1, num_new_points + 1)
        new_points = a + (2 * indices - 1) * h_k

        sum_new = np.sum(func(new_points))
        t_k0 = 0.5 * R[k - 1][0] + h_k * sum_new

        current_row = [t_k0]

        #Ekstrapolacja Richardsona
        for m in range(1, k + 1):
            t_curr_m = current_row[m - 1]
            t_prev_m = R[k - 1][m - 1]

            correction = (t_curr_m - t_prev_m) / (4 ** m - 1)
            t_km = t_curr_m + correction
            current_row.append(t_km)

        R.append(current_row)

        row_str = " | ".join([f"{val:.12f}" for val in current_row])
        print(f"{k:<3} | {row_str}")

        #Warunek stopu
        if k >= 1:
            diff = abs(R[k][k] - R[k - 1][k - 1])
            if diff < tolerance:
                print("-" * 150)
                return R[k][k], diff, k

    return R[-1][-1], abs(R[-1][-1] - R[-2][-2]), max_iter



target_accuracy = 1e-7
# 1. Wyznaczenie A
A = find_cutoff_A(target_accuracy)
tail_error = np.exp(-A)

print(f"1.Punkt odcięcia ogona całki:")
print(f"Wyznaczone A: {A:.0f}")
print(f"Błąd obcięcia (e^-A): {tail_error:.2e}")

    # 2. Zmiana granic
upper_limit_t = np.sqrt(A)

print(f"\n2. Zastosowane podstawienie x = t^2:")
print(f"Nowy przedział całkowania dla t: [0, {upper_limit_t:.6f}]")

print(f"\n3. Tabela Romberga:")
romberg_tol = target_accuracy / 2

result, error_romberg, iterations = romberg_integration(transformed_integrand,0,upper_limit_t,tolerance=romberg_tol)

total_error = tail_error + error_romberg

print(f"\n4. Wyniki końcowe:")
print(f"Wartość całki: {result:.12f}")
print(f"Liczba iteracji: {iterations}")
print(f"Całkowity błąd: {total_error:.2e}")

if total_error < target_accuracy:
    print("\nOsiągnięto zadaną dokładność 1e-7.")




