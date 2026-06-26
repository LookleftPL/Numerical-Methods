import numpy as np
import matplotlib.pyplot as plt

vessels = [-7/8,-5/8,-3/8,-1/8,1/8,3/8,5/8,7/8]
def calculate_values_in_vessels(vessels):
    values = []
    for i in range(len(vessels)):
        current = 1/(1 + 5 * (vessels[i])**2)
        values.append(current)
    return values

def calculate_L0(vessels):
    LO =[]
    for i in range(0,len(vessels)):
        current_denominator = 1
        current_numerator = 1
        for j in range(0,i):
            current_numerator = current_numerator * (-vessels[j])
            current_denominator = current_denominator*(vessels[i]-vessels[j])
        for k in range(i+1,len(vessels)):
            current_numerator = current_numerator * (-vessels[k])
            current_denominator = current_denominator * (vessels[i] - vessels[k])
        LO.append(current_numerator/current_denominator)
    return LO

def calculate_polynomial_coeffs(LO,vessels,vessels_value):
    polynomial_coefficients = []
    vessels_value_working = vessels_value.copy()
    a0 = 0
    for i in range(0,len(vessels)):
        a0 = a0 + LO[i]*vessels_value_working[i]
    polynomial_coefficients.append(a0)
    for i in range(0,len(vessels)-1):
        a_next = 0
        for j in range(0,len(vessels)):
            vessels_value_working[j] = (vessels_value_working[j] - polynomial_coefficients[i])/ vessels[j]
        for k in range(0, len(vessels)):
            a_next = a_next + LO[k] * vessels_value_working[k]
        polynomial_coefficients.append(a_next)
    result = np.array(polynomial_coefficients)
    return result
def horner_poly(x, coefficients):
    result = 0
    # Iterujemy od najwyższej potęgi (ostatni element listy)
    for c in reversed(coefficients):
        result = result * x + c
    return result
vessels_value = calculate_values_in_vessels(vessels)
LO = calculate_L0(vessels)
coeffs = calculate_polynomial_coeffs(LO,vessels,vessels_value)

x_plot = np.linspace(-1.25, 1.25, 200) # Przedział z zadania
y_plot = [horner_poly(x, coeffs) for x in x_plot]
plt.figure(figsize=(10, 6))

plt.plot(x_plot, y_plot, label='Wielomian interpolacyjny', color='blue')

plt.scatter(vessels, vessels_value, color='red', zorder=5, label='Węzły interpolacji')
plt.title('Wykres wielomianu interpolacyjnego')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

for i in range(0,len(vessels)):
    print("x = " + str(vessels[i]) + "   f(x) = " + str(vessels_value[i]))
print(coeffs)
coeffs_rounded = np.around(coeffs, decimals=4)
print(coeffs_rounded)
