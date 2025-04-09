import numpy as np
import scipy as sp
from scipy.optimize import minimize
from sympy import symbols, Eq, solve, diff
import sympy as sp
# Parameters
p1, p2, I = 2, 1, 6

# Utility function to maximize (negative because we minimize in scipy)
def utility(x):
    x1, x2 = x
    return -(x1 ** 2 + 2 * x2)

# Budget constraint
def budget_constraint(x):
    x1, x2 = x
    return I - (p1 * x1 + p2 * x2)

# Non-negativity constraints
bounds = [(0, None), (0, None)]  # x1 >= 0, x2 >= 0

# Solve using scipy.optimize
result = minimize(utility, [0, 0], bounds=bounds, constraints={'type': 'ineq', 'fun': budget_constraint})

x1_opt, x2_opt = result.x
max_utility = -result.fun

print(f"Optimal x1: {x1_opt:.2f}")
print(f"Optimal x2: {x2_opt:.2f}")
print(f"Maximum Utility: {max_utility:.2f}")

# ---------------------------------------------------------------------
# Define variables
x, y, lam = symbols('x y lam')

# Objective function
f = x**2 + (y + 9)**2

# Constraint function
g = 400 - 4*x**2 - y**2

# Lagrange multiplier equations
L = f + lam * g

# Partial derivatives
df_dx = diff(L, x)
df_dy = diff(L, y)
df_dlam = diff(L, lam)

# Solve the system of equations
solutions = solve([df_dx, df_dy, df_dlam], (x, y, lam))
print(solutions)

# ----------------------------------------------------------------
# Figure out the extrema points
a, b, lam= symbols('a b lam')

h = (1/2)*a**2 + 2*sp.log(b)
g = a + b - 3
L = h - lam * g
# Partial derivatives
df_da = diff(h, a)
df_db = diff(h, b)
df_dlam = diff(L, lam)
# Solve the system of equations
solutions = solve([df_da, df_db, df_dlam], (a, b, lam))
print(solutions)
# To classify the points, you have to use Hessian matrix, put those extrema into the matrix, tell matrix sign H1, H2...


# ----------------------------------------
x, y, lam = symbols('x y lam')

f = (1/2)*(x**2) + 2 * sp.log(y)
g = x + y - 3
L = f - lam * g

df_dx = diff(L, x)
df_dy = diff(L, y)
df_dlam = diff(L, lam)

solutions = solve([df_dx, df_dy, df_dlam], (x, y, lam))
print(solutions)