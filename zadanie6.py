import numpy as np

A = np.array([[2,-1,0,0,1],
              [-1,2,1,0,0],
              [0,1,1,1,0],
              [0,0,1,2,-1],
              [1,0,0,-1,2]])
eigenvalue = 0.38197
np.set_printoptions(suppress=True)
def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n) # L ma jedynki na przekątnej
    U = np.zeros((n, n))

    # Algorytm Doolittle'a
    for i in range(n):
        # Wyznaczanie U (górna)
        for k in range(i, n):
            sum_u = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_u

        # Wyznaczanie L (dolna)
        for k in range(i + 1, n):
            sum_l = sum(L[k][j] * U[j][i] for j in range(i))
            if abs(U[i][i]) < 1e-12: # Zabezpieczenie przed dzieleniem przez 0
                continue
            L[k][i] = (A[k][i] - sum_l) / U[i][i]

    return L, U

def lu_solve(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    # 1. Rozwiązujemy Ly = b (podstawianie w przód)
    for i in range(n):
        sum_Ly = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_Ly) / L[i][i]

    # 2. Rozwiązujemy Ux = y (podstawianie wstecz)
    for i in range(n - 1, -1, -1):
        sum_Ux = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_Ux) / U[i][i]
    return x

def find_vector_from_eigenvalue(eigen_value,A):

    n = A.shape[0]
    y = np.random.rand(n)
    y = y / np.linalg.norm(y)
    tau = eigen_value + 1e-8
    I = np.eye(n)
    M = A - tau * I
    L, U = lu_decomposition(M)

    norm_diff = 10000
    iteration = 0
    while norm_diff > 1e-15 and iteration <1000:
        z = lu_solve(L, U, y)
        z_norm = np.linalg.norm(z)
        new_y = z/z_norm
        diff = np.linalg.norm(new_y - y)
        diff_plus = np.linalg.norm(new_y + y)
        norm_diff = min(diff,diff_plus)
        y = new_y
        iteration += 1
    return y

result = find_vector_from_eigenvalue(eigenvalue,A)
print("Unormowany wektor własny dla lambda = 0,38197")
print(result)
unnormalized = result / result[1]
print("Nieunormowany dla lambda = 0,38197")
print(unnormalized)