import numpy as np


def toTridiagonalMatrix(A, n):
    # Pracujemy na kopii, żeby nie zepsuć oryginału
    A_work = A.copy()

    # Pętla po kolumnach od 0 do N-3 (ostatnie dwie kolumny zostają same)
    # Dla macierzy 6x6, k przyjmie wartości: 0, 1, 2, 3
    for k in range(n - 2):
        # 1. Wyciągamy wektor x (fragment kolumny pod przekątną)
        # Zgodnie z (26) i (28) ze slajdów
        x = A_work[k + 1:, k]

        # 2. Obliczamy wektor u (krok brakujący w Twoim kodzie!)
        # u = x + sign(x[0]) * ||x|| * e1
        norm_x = np.linalg.norm(x)

        # Jeśli wektor jest już zerowy, pomijamy krok
        if norm_x == 0:
            continue

        # Tworzymy e1 o takiej samej długości jak x
        e1 = np.zeros_like(x)
        e1[0] = 1.0

        # Wybór znaku (plus lub minus) - najlepiej taki sam jak znak pierwszego elementu x,
        # żeby uniknąć błędów numerycznych (odejmowania bliskich liczb)
        sign = 1.0 if x[0] >= 0 else -1.0
        u = x + sign * norm_x * e1

        # 3. Tworzymy małą macierz Householdera Twoją funkcją
        # Teraz przekazujemy poprawne u!
        H_small = householder(u)

        # 4. Tworzymy dużą macierz P (zanurzenie)
        P = np.eye(n)
        # NAPRAWIONY SLICING: używamy [k+1:, k+1:] a nie [k+1:, k+1]
        P[k + 1:, k + 1:] = H_small

        # 5. Aplikujemy transformację z obu stron: P A P^T
        # Ponieważ P jest symetryczne (P=P^T), można pisać P @ A @ P
        A_work = P @ A_work @ P

    return A_work


# Twoja funkcja pomocnicza (jest OK, pod warunkiem że dostaje dobre u)
def householder(u):
    n = u.size
    u = u.reshape(n, 1)
    I = np.eye(n)
    # Używamy operatora @ dla czytelności i .item() dla skalara
    uut = u @ u.T
    utu = u.T @ u
    P = I - 2 * (uut / utu.item())
    return P


# === TEST ===
np.set_printoptions(precision=4, suppress=True)  # Ładne wyświetlanie

A = np.array([[19 / 12, 13 / 12, 5 / 6, 5 / 6, 13 / 12, -17 / 12],
              [13 / 12, 13 / 12, 5 / 6, 5 / 6, -11 / 12, 13 / 12],
              [5 / 6, 5 / 6, 5 / 6, -1 / 6, 5 / 6, 5 / 6],
              [5 / 6, 5 / 6, -1 / 6, 5 / 6, 5 / 6, 5 / 6],
              [13 / 12, -11 / 12, 5 / 6, 5 / 6, 13 / 12, 13 / 12],
              [-17 / 12, 13 / 12, 5 / 6, 5 / 6, 13 / 12, 19 / 12]])

n = A.shape[0]
A_tridiag = toTridiagonalMatrix(A, n)

print("Macierz trójprzekątna:")
print(A_tridiag)