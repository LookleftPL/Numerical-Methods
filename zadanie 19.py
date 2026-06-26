import numpy as np

def calculate_L0(vessels):
    LO = []
    for i in range(0, len(vessels)):
        current_denominator = 1
        current_numerator_coeffs = [1]

        for j in range(0, len(vessels)):
            if i == j:
                continue
            current_denominator *= (vessels[i] - vessels[j])

        LO.append(current_denominator)
    return LO

def get_lagrange_poly_coeffs(x_nodes, y_nodes):
    n = len(x_nodes)
    final_coeffs = np.zeros(n)

    for i in range(n):
        Li_coeffs = np.zeros(n)
        Li_coeffs[0] = 1.0
        poly_degree = 0

        denominator = 1.0

        for j in range(n):
            if i == j:
                continue
            xj = x_nodes[j]
            denominator *= (x_nodes[i] - xj)

            new_coeffs = np.zeros(n)
            new_coeffs[1:poly_degree + 2] += Li_coeffs[0:poly_degree + 1]
            new_coeffs[0:poly_degree + 1] -= xj * Li_coeffs[0:poly_degree + 1]

            Li_coeffs = new_coeffs
            poly_degree += 1

        term_factor = y_nodes[i] / denominator
        final_coeffs += Li_coeffs * term_factor

    return final_coeffs

def derivative_at_point(coeffs, x_val):
    val = 0.0
    for i in range(1, len(coeffs)):
        exponent = i - 1
        term = i * coeffs[i] * (x_val ** exponent)
        val += term
    return val

def p_alpha(alpha):
    x_nodes = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y_nodes = [0.0, 1.0, 0.0, alpha, 0.0, 1.0]

    coeffs = get_lagrange_poly_coeffs(x_nodes, y_nodes)
    deriv_val = derivative_at_point(coeffs, 7.0)

    return deriv_val

def bisection_root_finder(func, a, b, tol=1e-6):
    fa = func(a)
    fb = func(b)

    if fa * fb >= 0:
        print("Błąd: Funkcja ma ten sam znak na krańcach przedziału!")
        return None

    iters = 0
    while (b - a) / 2.0 > tol:
        iters += 1
        c = (a + b) / 2.0
        fc = func(c)

        if fc == 0:
            return c, iters

        if fc * fa < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2.0, iters

ALPHA_START = -10.0
ALPHA_END = 10.0
TOLERANCE = 1e-6

print(f"Poszukiwanie miejsca zerowego w przedziale [{ALPHA_START}, {ALPHA_END}]")

p_start = p_alpha(ALPHA_START)
p_end = p_alpha(ALPHA_END)
print(f"p({ALPHA_START}) = {p_start:.4f}")
print(f"p({ALPHA_END}) = {p_end:.4f}")

root, iters = bisection_root_finder(p_alpha, ALPHA_START, ALPHA_END, TOLERANCE)
print(f"Znalezione miejsce zerowe alpha: {root:.8f}")
print(f"Wartość p(alpha) w tym punkcie:  {p_alpha(root):.2e}")
print(f"Liczba iteracji: {iters}")
final_alpha = root
x_nodes = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
y_nodes = [0.0, 1.0, 0.0, final_alpha, 0.0, 1.0]

final_coeffs = get_lagrange_poly_coeffs(x_nodes, y_nodes)

print("Współczynniki wielomianu (od wyrazu wolnego do a5):")
print(final_coeffs)
