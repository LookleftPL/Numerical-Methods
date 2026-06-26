# Numerical Methods - Academic Projects

Repository contains a comprehensive collection of numerical analysis algorithms implemented in Python during my university course. The projects focus on efficiency, matrix decompositions, optimization, and solving mathematical problems from scratch without relying on high-level library solvers.

## Implemented Algorithms & Topics

### 1. Interpolation & Approximation
* **Chebyshev Nodes & Polynomial Interpolation** (zadanie1_1.py, zadanie3.py): Implementation of polynomial interpolation using Chebyshev nodes to minimize Runge's phenomenon. Evaluation performed via Horner's scheme.
* **Lagrange Polynomial Coefficients** (zadanie11.py): Analytical derivation of full polynomial coefficients from scratch using the Lagrange interpolation properties.
* **Floater-Hormann Barycentric Rational Interpolation** (zadanie14.py): Implementation of the barycentric rational interpolation algorithm with user-defined blending parameter d to maintain stability.
* **General Polynomial Interpolation** (zadanie9.py, zadanie10.py): Computing polynomial coefficients and plotting interpolation curves for custom data sets.

### 2. Spline Interpolation
* **Natural Cubic Spline (Equidistant Nodes)** (zadanie12.py): Constructing cubic splines over uniformly spaced points by transforming difference quotients into a tridiagonal system solved via LU factorization.
* **Natural Cubic Spline (Non-uniform Chebyshev Nodes)** (zadanie13.py): Advanced cubic spline interpolation built dynamically over custom step sizes (Delta h) derived from sorted Chebyshev distributions.

### 3. Root-Finding & Nonlinear Systems
* **Bisection Root-Finder for Functional Derivatives** (zadanie11.py): Combining Lagrange interpolation properties with a stable Bisection solver to find optimal parameters (alpha) that satisfy specific derivative boundary constraints.
* **Multivariate Newton-Raphson Solver** (zadanie18.py): Finding roots of nonlinear systems of equations using a calculated 2D Jacobian matrix and Cramer's rule for localized linear step estimation.
* **Laguerre's Method with Deflation** (zadanie17.py): Comprehensive polynomial root-finder in the complex plane utilizing high-order derivatives, complex square roots, and synthetic division (deflation) to compute all roots.

### 4. Numerical Optimization & Minimization
* **Golden Section Search & Brent's Method (1D)** (zadanie19.py): A comparative study highlighting optimization efficiency between the linear Golden Section search and the superlinear Brent's algorithm (combining inverse parabolic interpolation with golden section fallback).
* **Polynomial Minimization via Brent's Method** (zadanie20.py): Optimization of a high-degree polynomial (7th order) within constrained bounds, accompanied by convergence evaluation and verification of second-order conditions (f''(x) > 0).
* **Powell's Conjugate Direction Method (Multivariate)** (zadanie21.py, zadanie23.py): Multi-dimensional without-derivative optimization algorithm. Utilizes a custom bounding method (bracket_minimum) and Brent's 1D solver as a line search mechanism. Applied to locate global/local minima via random multistart searches (zadanie21.py) and highly valleys-constrained environments like the 4D Rosenbrock function (zadanie23.py).

### 5. Chaotic Systems & Dynamical Analysis
* **Lyapunov Exponents in Tent Map** (zadanie22.py): Exploration of chaotic behaviors in non-linear maps. Implements numerical estimation of Lyapunov exponents via orbit iteration, stabilized by throwing away initial transient data (warm-up phases) and evaluating convergence against formal analytical solutions (ln(2r)).

### 6. Systems of Linear Equations (Direct & Iterative)
* **Gauss-Seidel & Conjugate Gradients** (zadanie2.py): A comparative study of iterative solvers for large sparse matrices (N=128), complete with convergence rate plotting.
* **Tridiagonal Matrix Solvers (Thomas Algorithm)** (zadanie3.py): Fast O(n) solver for tridiagonal systems combined with the Sherman-Morrison formula for cyclic/modified boundary conditions.
* **LU Decomposition for Tridiagonal Matrices** (zadanie4.py): Crout/Doolittle-based tridiagonal LU factorization without pivoting, combined with forward/backward substitution.

### 7. Eigenvalues & Eigenvectors
* **Power Iteration Method** (zadanie5.py): Iterative determination of the first and second dominant eigenvalues along with their respective normalized and unnormalized eigenvectors.
* **Householder Reflections & QR Algorithm** (zadanie6.py, zadanie7.py): Similarity transformations using Householder matrices combined with Givens rotation matrices to compute all eigenvalues of a matrix, including complex vector reconstruction from symmetric form.
* **Inverse Power Iteration** (zadanie7.py, zadanie8.py): Finding specific eigenvectors corresponding to a given eigenvalue (lambda) using shifted inverse iteration combined with custom LU decomposition.

### 8. Numerical Integration (Quadrature)
* **Romberg Integration with Cutoff Optimization** (zadanie15.py): Computing improper integrals by finding an analytical tail cutoff threshold (A = -ln epsilon), applying variable substitution (x = t^2), and generating a recursive Romberg table using Richardson extrapolation.
* **Dynamic Interval Romberg Integration** (zadanie16.py): Numerical estimation of limits at infinity (lim x->inf F(x)) using dynamic bounds expansion and integration checking via cumulative sum evaluation.

## 🛠️ Technologies & Libraries
* **Python 3.x**
* **NumPy** (for array operations, vector products, and basic linear algebra helpers)
* **Matplotlib** (for plotting functions, interpolation shapes, and algorithm convergence rates)

## 📝 How to Run
To execute any specific assignment and see the mathematical output or plots, run the script directly from your terminal:
```bash
python zadanie21.py
