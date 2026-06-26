import numpy as np
np.set_printoptions(suppress=True,linewidth=200)

H = np.array([
    [0,  1,   0, -1j],
    [1,  0, -1j,   0],
    [0, 1j,   0,   1],
    [1j, 0,   1,   0]])

def H_to_H_symetric(H):
    A = H.real  # czesc rzeczywista
    B = H.imag  # czesc urojona
    H_sym = np.block([
        [A, -B],
        [B, A]
    ])
    return H_sym
def toTridiagonalMatrix(A,n):
    for i in range(n-2):
        x = A[i+1:,i]
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-15:
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1.0
        sign = 1.0
        if x[0]<0:
            sign = -1.0
        u = x + sign*x_norm *e1
        householder_u = householder(u)
        P = np.eye(n)
        P[i+1:,i+1:] = householder_u
        A = P @ A @ P
    return A

# householder dostaje kolumne,ale w formie plaskiej dlatego reshapeujemy
def householder(u):
    n = u.size
    u = u.reshape(n,1) # jedna kolumna , n wierszy
    I = np.eye(n)  # macierz jednostkowa
    uut = np.dot(u,u.T)
    utu = np.dot(u.T,u)
    P = I - 2*(uut/utu)
    return P

def givens_matrix(n,xi,xj,i,j):
    G = np.eye(n)
    denominator = (xi**2 + xj**2)**(1/2)
    if denominator < 1e-15:
        return G, G.T
    c = xi/denominator
    s = xj/denominator
    G[i,i] = c
    G[j,j] = c
    G[i,j] = -s
    G[j,i] = s
    return G, G.T

def qr_step(A):
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)
    for i in range(n-1):
        xi = R[i,i]
        j = i + 1
        xj = R[j,i]
        G = givens_matrix(n,xi,xj,i,j)
        R = G[1] @ R
        Q = Q @ G[0]
    A_new = R @ Q
    return A_new
def find_eigenvalues(A,tolerance=1e-15):
    A_current = A.copy()
    max_value = 10000
    i = 0
    while max_value > tolerance:
        subdiagonal = np.diag(A_current,-1) # diagonala pod przekatna
        max_value = np.max(np.abs(subdiagonal)) # patrzymy czy są jakies wartosci niezerowe
        A_current = qr_step(A_current)
        if i ==100:
            break
        i+=1

    return A_current

def extract_eigenvalues_from_cages(A):
    A_current = A.copy()
    n = A_current.shape[0]
    eigen_values = []
    i = 0
    almost_zero = 1e-10
    while i < n:
        if i == n-1:
            eigen_values.append(A[i,i])
            break

        xi = A_current[i,i]
        if abs(A_current[i+1,i]) < almost_zero:
            eigen_values.append(A[i,i])
            i+=1
        else:
            right_up = A_current[i,i+1]
            left_down = A_current[i+1,i]
            right_down = A_current[i+1,i+1]
            cage = np.array([[xi,right_up],
                             [left_down,right_down]])
            eigen_values_cage = np.linalg.eigvals(cage)
            eigen_values.extend(eigen_values_cage)
            i+=2
    eigen_values_rounded = np.round(eigen_values,8)
    return eigen_values_rounded
def from_double_to_single_eigenvalues(eigen_values):
    n = len(eigen_values)
    sorted_eigen_values = np.sort(eigen_values)
    unique_eigen_values = sorted_eigen_values[::2]

    return unique_eigen_values

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


def reconstruct_complex_vector(v_real):
    # Dzielimy wektor na pół
    n = v_real.shape[0] // 2
    real_part = v_real[:n]  # Pierwsze 4 to część rzeczywista
    imag_part = v_real[n:]  # Kolejne 4 to część urojona

    # Składamy w liczby zespolone
    v_complex = real_part + 1j * imag_part

    # Normalizujemy wynikowy wektor zespolony (norma euklidesowa zespolona)
    return v_complex / np.linalg.norm(v_complex)





H_sym = H_to_H_symetric(H)
H_sym_tridiagonal = toTridiagonalMatrix(H_sym,8)
matrix_caged = find_eigenvalues(H_sym_tridiagonal)
eigen_values_double = extract_eigenvalues_from_cages(matrix_caged)
eigen_values_single = from_double_to_single_eigenvalues(eigen_values_double)
print(H_sym)
print(eigen_values_single)
print("-" * 30)
print("WYNIKI KOŃCOWE:")

# Iterujemy po znalezionych unikalnych wartościach własnych
for val in eigen_values_single:
    print(f"\nDla wartości własnej lambda = {val}:")

    # 1. Znajdź wektor w R^8
    vec_8d = find_vector_from_eigenvalue(val, H_sym)

    # 2. Zamień na C^4
    vec_4d = reconstruct_complex_vector(vec_8d)

    print("[")
    for element in vec_4d:
        # Formatujemy liczbę: 4 miejsca po przecinku dla części R i Im
        napis = f"{element.real:.4f} + {element.imag:.4f}j"
        # Wypisujemy ją w nawiasach, żeby wyglądało jak kolumna macierzy
        print(f" [{napis}]")
    print("]")


