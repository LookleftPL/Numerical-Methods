import cmath
import random


def horner(coeffs, z):
    result = coeffs[0]
    n = len(coeffs)
    # Pętla od drugiego współczynnika do końca
    for i in range(1, n):
        result = result * z + coeffs[i]
    return result

def get_derivative_coeffs(coeffs):
    n = len(coeffs) - 1  # Stopień wielomianu
    deriv_coeffs = []

    for i in range(n):
        power = n - i
        new_coeff = coeffs[i] * power
        deriv_coeffs.append(new_coeff)

    return deriv_coeffs

def evaluate_laguerre_data(coeffs, z):
    p_val = horner(coeffs, z)

    d1_coeffs = get_derivative_coeffs(coeffs)
    dp_val = horner(d1_coeffs, z)

    d2_coeffs = get_derivative_coeffs(d1_coeffs)
    ddp_val = horner(d2_coeffs, z)

    return p_val, dp_val, ddp_val


def laguerre_step(coeffs, z, n):
    p, dp, ddp = evaluate_laguerre_data(coeffs, z)

    # Zabezpieczenie, jeśli trafiliśmy w pierwiastek
    if abs(p) < 1e-15:
        return z

    term1 = (n - 1) * dp ** 2
    term2 = n * p * ddp
    inside_sqrt = (n - 1) * (term1 - term2)

    sqrt_val = cmath.sqrt(inside_sqrt)

    denom1 = dp + sqrt_val
    denom2 = dp - sqrt_val

    # Wybieramy mianownik o większym module
    if abs(denom1) > abs(denom2):
        denominator = denom1
    else:
        denominator = denom2

    if abs(denominator) < 1e-15:
        return z + 0.001

    offset = (n * p) / denominator
    return z - offset


def find_root(coeffs, guess):
    z = guess
    degree = len(coeffs) - 1

    for i in range(50):
        z_next = laguerre_step(coeffs, z, degree)

        # Warunek stopu
        if abs(z_next - z) < 1e-9:
            return z_next
        z = z_next

    return z


def deflation(coeffs, root):
    n = len(coeffs) - 1
    new_coeffs = []

    # Pierwszy współczynnik jest taki sam
    current = coeffs[0]
    new_coeffs.append(current)

    # Liczymy kolejne współczynniki
    for i in range(1, n):
        current = coeffs[i] + root * current
        new_coeffs.append(current)

    return new_coeffs


def solve_quadratic_eq(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]

    delta = cmath.sqrt(b ** 2 - 4 * a * c)
    z1 = (-b - delta) / (2 * a)
    z2 = (-b + delta) / (2 * a)
    return z1, z2

orig_coeffs = [1, 1j, -1, -1j, 1]
curr_coeffs = [1, 1j, -1, -1j, 1]
roots = []


while len(curr_coeffs) > 3:
    deg = len(curr_coeffs) - 1
    guess = complex(random.random(), random.random())
    z_approx = find_root(curr_coeffs, guess)
    z_polished = find_root(orig_coeffs, z_approx)
    roots.append(z_polished)
    curr_coeffs = deflation(curr_coeffs, z_polished)

z3, z4 = solve_quadratic_eq(curr_coeffs)

z3 = find_root(orig_coeffs, z3)
z4 = find_root(orig_coeffs, z4)

roots.append(z3)
roots.append(z4)

print("WYNIKI KONCOWE:")
for i in range(len(roots)):
    r = roots[i]
    print(f"z{i + 1} = {r.real:.8f} + {r.imag:.8f}j")

print("SPRAWDZENIE (Wartosc wielomianu w punktach):")
for r in roots:
    val = horner(orig_coeffs, r)
    print(f"P(z) = {abs(val):.2e}")