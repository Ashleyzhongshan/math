import sympy as sp

# Define the symbol
x = sp.Symbol('x', real=True, positive=True)

##################################
# (a) Derivative of f(x) = (x+1)/(2-x)
##################################
f = (x + 1) / (2 - x)
fprime = sp.diff(f, x)
print("(a) f'(x) =", fprime)

##################################
# (b) Limit of (e^x + x^2)^(1/x) as x->oo
##################################
limit_expr = (sp.exp(x) + x**2)**(1/x)
limit_value = sp.limit(limit_expr, x, sp.oo)
print("(b) limit =", limit_value)

##################################
# (c) Integral from 0 to e^2 of 1/(x + sqrt(x)) dx
##################################
integrand = 1/(x + sp.sqrt(x))
res = sp.integrate(integrand, (x, 0, sp.E**2))
print("(c) integral =", res.simplify())


# Question 4
import numpy as np
from scipy.integrate import quad

# Define the parameters
theta = 1  # You can adjust this value

# Define the integrand
def integrand(x, theta):
    return (x**2 / theta) * np.exp(-x**2 / (2 * theta))

# Perform the integration from 0 to infinity
result, error = quad(integrand, 0, np.inf, args=(theta,))

print(f"E(X) = {result:.4f}")


# Double integral!!!
# Define the inner function to integrate
def inner_integral(x):
    return np.exp(2 * x - x**2)

# Outer integral that depends on y
def outer_integral(y):
    upper_limit = 1 - np.cbrt(y)  # 1 - y^(1/3)
    return quad(inner_integral, 0, upper_limit)[0]

# Perform the outer integral over y from 0 to 1
result, error = quad(outer_integral, 0, 1)

print(result, error)