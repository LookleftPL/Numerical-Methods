import numpy as np

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

# u i v we wzorze Sherman–Morrison
u_vec = np.zeros(n, dtype=float)
u_vec[0] = 1.0
u_vec[-1] = 1.0
v_vec = u_vec.copy()

# Tworzymy T = A - u v^T → trójdiagonalna część
T = A - np.outer(u_vec, v_vec)

# wyciągamy przekątne T
a_sub = np.diag(T, k=-1).copy()
b_main = np.diag(T, k=0).copy()
c_super = np.diag(T, k=1).copy()

# Faktoryzacja T = L U
l, u_diag, c_u = lu_tridiag_no_pivot(a_sub, b_main, c_super)

# Rozwiązujemy T y = b
y = forward_substitution_l(l, b)
y = back_substitution_u(u_diag, c_u, y)

# Rozwiązujemy T z = u
z = forward_substitution_l(l, u_vec)
z = back_substitution_u(u_diag, c_u, z)

# Sherman–Morrison
alpha = 1.0 + float(np.dot(v_vec, z))
beta  = float(np.dot(v_vec, y))
x = y - (beta / alpha) * z

print("Rozwiązanie x =")
print(x)
