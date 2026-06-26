import numpy as np

def thomas_solver(a, b, c, d):
    """
    Rozwiązuje trójdiagonalny układ Ax = d
    gdzie:
      a -- dolna przekątna długości n-1 (a[0] odpowiada A[1,0])
      b -- główna przekątna długości n
      c -- górna przekątna długości n-1 (c[-1] odpowiada A[n-2,n-1])
      d -- prawa strona długości n
    Zwraca x długości n.
    """
    n = len(b)
    # kopie, bo będziemy modyfikować
    cprime = np.empty(n-1, dtype=float)
    dprime = np.empty(n, dtype=float)
    x = np.empty(n, dtype=float)

    # forward sweep
    cprime[0] = c[0] / b[0]
    dprime[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cprime[i-1]
        cprime[i] = c[i] / denom
        dprime[i] = (d[i] - a[i-1] * dprime[i-1]) / denom
    denom = b[n-1] - a[n-2] * cprime[n-2]
    dprime[n-1] = (d[n-1] - a[n-2] * dprime[n-2]) / denom

    # back substitution
    x[n-1] = dprime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dprime[i] - cprime[i] * x[i+1]
    return x

# ---- przykładowe dane: macierz z zadania (n=7) ----
n = 7
A = np.array([
    [4,1,0,0,0,0,1],
    [1,4,1,0,0,0,0],
    [0,1,4,1,0,0,0],
    [0,0,1,4,1,0,0],
    [0,0,0,1,4,1,0],
    [0,0,0,0,1,4,1],
    [1,0,0,0,0,1,4]
], dtype=float)
b = np.array([1,2,3,4,5,6,7], dtype=float)

# u i v (jedynki na 1 i n)
u = np.zeros(n, dtype=float)
u[0] = 1.0
u[-1] = 1.0
v = u.copy()

# Tworzymy T = A - u v^T --> będzie trójdiagonalna
T = A - np.outer(u, v)             # np.outer() oblicza iloczyn dwóch wektorów

# Wyciągamy przekątne T do Thomasa
main = np.diag(T).copy()           # np.diag(T) zwraca diagonale
upper = np.diag(T, k=1).copy()     # np.diag(T, k = 1) zwraca górną przekątną nad diagonalą
lower = np.diag(T, k=-1).copy()    # np.diag(T, k = -1) zwraca dolną przekątną pod diagonalą

# Rozwiąż T y = b
y = thomas_solver(lower, main, upper, b)

# Rozwiąż T z = u
z = thomas_solver(lower, main, upper, u)

alpha = 1.0 + float(np.dot(v, z))
beta = float(np.dot(v, y))

x = y - (beta / alpha) * z

print("Rozwiązanie x:", x)

# Dla kontroli: sprawdźmy A x ~ b
print("A @ x:", A.dot(x))
print("b:", b)
