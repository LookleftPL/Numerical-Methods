import math

import numpy as np
import matplotlib.pyplot as plt
d = 3
vessels = [-7/8,-5/8,-3/8,-1/8,1/8,3/8,5/8,7/8]
def calculate_values_in_vessels(vessels):
    values = []
    for i in range(len(vessels)):
        current = 1/(1 + 5 * (vessels[i])**2)
        values.append(current)
    return values

def Newton_symbol(upper,lower):
    return math.factorial(upper)/(math.factorial(lower) * math.factorial(upper-lower))

def calculate_weights(vessels,d):
    n = len(vessels)
    n_index = len(vessels) - 1
    weights = []
    for k in range(0, len(vessels)):  # k idzie od 0 do 7
        sum_of_newtons = 0
        start_i = max(0, k - d)
        end_i = min(n_index - d, k)

        for i in range(start_i, end_i + 1):  # Dodane +1
            sum_of_newtons += Newton_symbol(d, k - i)
        sum_of_newtons = sum_of_newtons * (-1) ** (k - d)
        weights.append(sum_of_newtons)
    return weights


def floater_hormann_interpolate(x_val, vessels, vessels_values, weights):
    # Sprawdź czy x nie jest węzłem (z małą tolerancją dla floatów)
    for k in range(len(vessels)):
        if abs(x_val - vessels[k]) < 1e-10:
            return vessels_values[k]

    numerator = 0.0
    denominator = 0.0

    for k in range(len(vessels)):
        # Główny składnik: w_k / (x - x_k)
        term = weights[k] / (x_val - vessels[k])

        numerator += term * vessels_values[k]
        denominator += term

    return numerator / denominator



vessels_values = calculate_values_in_vessels(vessels)
weights = calculate_weights(vessels_values,d)
print(weights)

x_plot = np.linspace(-1.25, 1.25, 200)
# Obliczamy wartości dla każdego punktu z x_plot
y_rational = [floater_hormann_interpolate(x, vessels, vessels_values, weights) for x in x_plot]

plt.figure(figsize=(8, 6))
plt.plot(x_plot, y_rational, label=f'Interpolacja wymierna (d={d})')
plt.scatter(vessels, vessels_values, color='red', label='Węzły')
plt.title("Algorytm Floatera-Hormanna")
plt.legend()
plt.grid(True)
plt.show()

