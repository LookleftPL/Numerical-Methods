import numpy as np
import matplotlib.pyplot as plt

N = 128
main_diag = 4
error_tolerance = 1e-8
def Gauss_Seidel(N, main_diag,error_tolerance):
    counter = 0
    norms = []
    x = np.zeros(N)
    safe_start = 4
    safe_end = N-4
    total_norm = 10000000
    while total_norm > error_tolerance:
        current_error_squared = 0.0

        new_val = (1-x[1] - x[4])/main_diag
        diff = new_val - x[0]
        current_error_squared += diff**2
        x[0] = new_val
        for i in range (1,safe_start):
            sigma = x[i-1] + x[i+1] + x[i+4]
            new_val = (1.0-sigma)/main_diag
            diff = new_val - x[i]
            current_error_squared += diff**2
            x[i] = new_val
        for i in range(safe_start,safe_end):
            sigma = x[i-1]+ x[i-4] + x[i+1] + x[i+4]
            new_val = (1.0-sigma)/main_diag
            diff = new_val - x[i]
            current_error_squared += diff**2
            x[i] = new_val
        for i in range(safe_end,N-1):
            sigma = x[i-4] + x[i-1] + x[i+1]
            new_val = (1.0-sigma)/main_diag
            diff = new_val - x[i]
            current_error_squared += diff**2
            x[i] = new_val
        new_val = (1 - x[N-2] - x[N-5])/main_diag
        diff = new_val - x[N-1]
        current_error_squared += diff**2
        x[N-1] = new_val

        total_norm = np.sqrt(current_error_squared)
        norms.append(total_norm)
        counter+= 1
        if counter > 100:
            break

    return x,np.array(norms),counter

def optimized_multiply(v, N):
    y = np.zeros(N)
    safe_start = 4
    safe_end = N-4

    y[0] = 4*v[0] + v[1] + v[4]

    for i in range(1,safe_start):
        y[i] = v[i-1] + 4*v[i] + v[i+1] + v[i+4]

    for i in range(safe_start, safe_end):
        y[i] = v[i - 4] + v[i - 1] + 4 * v[i] + v[i + 1] + v[i + 4]

    for i in range(safe_end,N-1):
        y[i] = v[i-4] + v[i-1] + 4*v[i] +v[i+1]

    y[N-1] = v[N-5] + v[N-2] + 4*v[N-1]

    return y

def ConjugateGradients(N, main_diag,error_tolerance):
    counter = 0
    norms = []
    b = np.ones(N)
    x = np.zeros(N)
    r = b.copy() # bo na poczatku x = 0
    p = r.copy()
    r_squared_old = np.dot(r,r)
    while np.linalg.norm(r)>error_tolerance and counter<1000:
        Ap = optimized_multiply(p, N)
        alpha = r_squared_old/np.dot(p,Ap)

        step_norm = np.linalg.norm(alpha * p)
        norms.append(step_norm)


        r = r - alpha*Ap
        r_squared_new = np.dot(r,r)
        beta = r_squared_new/ r_squared_old

        x = x + alpha * p
        p =  r + beta * p

        r_squared_old = r_squared_new
        counter += 1

    return x,np.array(norms), counter

wynikGauss = Gauss_Seidel(N, main_diag,error_tolerance)
wynikGradients = ConjugateGradients(N, main_diag,error_tolerance)
print("Zestawienie 10 elementow x")
print("Gauss-Seidel")
print("Ilosc iteracji: " + str(wynikGauss[2]))
print(wynikGauss[0][:10])
print("Metoda gradientow sprzężonych")
print("Ilosc iteracji: " + str(wynikGradients[2]))
print(wynikGradients[0][:10])

gauss_norms = wynikGauss[1]
gradients_norms = wynikGradients[1]
x_values_Gauss = np.arange(1,gauss_norms.size+1)
x_values_Gradient = np.arange(1,gradients_norms.size+1)
plt.semilogy(x_values_Gauss,gauss_norms, color = "navy", label = "Gauss-Seidel")
plt.semilogy(x_values_Gradient,gradients_norms, color = "red", label = "Metoda gradientów sprzężonych")
plt.title("Porównanie tempa zbieżności")
plt.xlabel("k-ta norma")
plt.ylabel("Wartość normy")
plt.legend()

plt.show()




