import numpy as np
import matplotlib.pyplot as plt
nums = [1,2,3,4,5,6,7,8]
def calculate_Czebyszew_vessels(nums):
    Czebyszew_vessels = np.array([])
    for i in range(len(nums)):
        current = np.cos((2*nums[i] - 1)*np.pi/16)
        Czebyszew_vessels = np.append(Czebyszew_vessels, current)
    return np.sort(Czebyszew_vessels)

def calculate_values_in_vessels(vessels):
    values = np.array([])
    for i in range(len(vessels)):
        current = 1/(1 + 5 * (vessels[i])**2)
        values = np.append(values, current)
    return values

def construct_matrix_equation(vessels, vessels_value,h):
    n = len(vessels)
    num_internal = n - 2
    A = np.zeros((num_internal,num_internal))
    right_vector = np.zeros(num_internal)

    for i in range(num_internal):

        j = i + 1

        h_left = h[i]
        h_right = h[i+1]

        A[i,i] = (h_left + h_right)/3.0

        if i> 0 :
            A[i,i-1] = h_left/6.0

        if i < num_internal -1:
            A[i,i+1] = h_right/6.0

        term1 = (vessels_value[j+1] - vessels_value[j])/ h_right
        term2 = (vessels_value[j] - vessels_value[j-1])/ h_left

        right_vector[i] = term1 - term2

    return A, right_vector
def lu_tridiag_no_pivot(a, b, c):

    #Faktoryzacja LU macierzy trójdiagonalnej A (bez pivotingu).
    #A ma subdiagonale a (n-1), diag b (n), superdiagonale c (n-1).
    #Zwraca:
    # l -- multiplikatory (dolna przekątna L) długości n-1 (L ma jedynki na diag)
    #  u -- główna przekątna U długości n
    #  c_u -- górna przekątna U długości n-1 (zmodyfikowana z c)
    #Czyli L = I + diag(l, -1), U ma diag u i superdiag c_u.
    n = len(b)
    u = b.astype(float).copy()
    c_u = c.astype(float).copy()
    l = np.empty(n-1, dtype=float)

    l[0] = a[0] / u[0]
    u[1] = u[1] - l[0] * c_u[0]

    # pozostałe
    for k in range(1, n-1):
        l[k] = a[k] / u[k]
        u[k+1] = u[k+1] - l[k] * c_u[k]

    return l, u, c_u
def forward_substitution_l(l, d):
    #Rozwiązuje Ly = d, gdzie L = I + diag(l, -1)
    n = len(d)
    y = np.empty(n, dtype=float)
    y[0] = d[0]
    for i in range(1, n):
        y[i] = d[i] - l[i-1] * y[i-1]
    return y

def back_substitution_u(u, c_u, y):
    #Rozwiązuje Ux = y, U ma diag u i superdiag c_u."""
    n = len(u)
    x = np.empty(n, dtype=float)
    x[-1] = y[-1] / u[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - c_u[i] * x[i+1]) / u[i]
    return x


def spline_value(x_val, vessels, vessels_values, ksi):
    # Znajdź odpowiedni przedział [x_j, x_{j+1}] dla x_val
    # Ponieważ węzły są posortowane, szukamy pierwszego węzła większego od x_val (podejście naiwne)
    for j in range(len(vessels) - 1):
        if vessels[j] <= x_val <= vessels[j + 1]:
            # Parametry pomocnicze ze slajdu 43
            h_j = vessels[j + 1] - vessels[j]
            A = (vessels[j + 1] - x_val) / h_j
            B = (x_val - vessels[j]) / h_j
            C = (1 / 6) * (A ** 3 - A) * (h_j ** 2)
            D = (1 / 6) * (B ** 3 - B) * (h_j ** 2)

            y_val = A * vessels_values[j] + B * vessels_values[j + 1] + C * ksi[j] + D * ksi[j + 1]
            return y_val

    return None  # Jeśli x jest poza zakresem



vessels = calculate_Czebyszew_vessels(nums)
vessels_value = calculate_values_in_vessels(vessels)
h = np.diff(vessels)
A, right_vector = construct_matrix_equation(vessels, vessels_value,h)

a_sub = np.diag(A, k=-1).copy()
b_main = np.diag(A, k=0).copy()
c_super = np.diag(A, k=1).copy()
l, u_diag, c_u = lu_tridiag_no_pivot(a_sub, b_main, c_super)

y = forward_substitution_l(l, right_vector)
ksi_internal = back_substitution_u(u_diag, c_u, y)
ksi_full = np.concatenate(([0], ksi_internal,[0]))
print(A)
print(right_vector)
print(ksi_full)
x_plot = np.linspace(-1.25, 1.25, 200)

y_spline = [spline_value(x, vessels, vessels_value, ksi_full) for x in x_plot]
plt.title('Naturalny splajn kubiczny')
plt.plot(x_plot, y_spline, label='Naturalny splajn kubiczny')
plt.scatter(vessels, vessels_value, color='red', label='Węzły')
plt.legend()
plt.show()
