import math

def tent_map(x, r):
    if x < 0.5:
        return 2 * r * x
    else:
        return 2 * r * (1 - x)

def analytical_lyapunov(r):
    return math.log(2 * r)

def numerical_lyapunov(r, x0, iterations):
    x = x0
    log_deriv_sum = 0.0
    for i in range(1000):
        x = tent_map(x, r)

    for i in range(iterations):
        deriv_abs = 2 * r
        if deriv_abs == 0:
            return -float('inf')
        log_deriv_sum += math.log(deriv_abs)
        x = tent_map(x, r)

    return log_deriv_sum / iterations

r_values = [3 / 4, 7 / 8, 15 / 16, 31 / 32]
iterations = 100000
x_start = 0.123456
print(f"{'r':<10} | {'Numeryczny':<12} | {'Analityczny':<12} | {'Błąd':<10}")
print("-" * 52)

for r in r_values:
    lyap_num = numerical_lyapunov(r, x_start, iterations)
    lyap_an = analytical_lyapunov(r)
    error = abs(lyap_num - lyap_an)
    print(f"{r:<10.4f} | {lyap_num:<12.6f} | {lyap_an:<12.6f} | {error:<10.2e}")