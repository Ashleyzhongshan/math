import sympy as sp
import numpy as np
from sympy import exp, symbols, integrate, oo

# Define the variables
u, mu, sigma = sp.symbols('u mu sigma')

# Step 1: Define the MGF of N(mu, sigma^2)
MGF = sp.exp(u * mu + (sigma**2 * u**2) / 2)
print("MGF of N(μ, σ²):")
sp.pprint(MGF)

# Step 2: Calculate E[X] by taking the first derivative of MGF
# 1.sp.diff(MGF, u) computes the first derivative of  M_X(u)  with respect to  u .
# 2..subs(u, 0) substitutes  u = 0  into the derivative to evaluate it at that point.
E_X = sp.diff(MGF, u).subs(u, 0)
print("\nE[X] (Expectation):")
sp.pprint(E_X)

# Step 3: Calculate E[X^2] by taking the second derivative of MGF
# 1.sp.diff(MGF, u) computes the first derivative of  M_X(u)  with respect to  u .
# 2..subs(u, 0) substitutes  u = 0  into the derivative to evaluate it at that point.
E_X2 = sp.diff(MGF, u, 2).subs(u, 0)
print("\nE[X^2]:")
sp.pprint(E_X2)

# Step 4: Calculate Variance using V[X] = E[X^2] - (E[X])^2
Var_X = E_X2 - E_X**2
print("\nVariance V[X]:")
sp.pprint(Var_X)

# Step 5: Verification (Substitute sample values)
mu_val = 2  # Example mean
sigma_val = 3  # Example standard deviation

E_X_val = E_X.subs({mu: mu_val, sigma: sigma_val})
Var_X_val = Var_X.subs({mu: mu_val, sigma: sigma_val})

print(f"\nE[X] (numerical value): {E_X_val}")
print(f"V[X] (numerical value): {Var_X_val}")

# -----------------------------------------------------------------------------------------
# Define the variable t
t = sp.Symbol('t')

# Moment Generating Function for a Uniform Distribution
def uniform_mgf(t, a, b):
    """
    MGF for Uniform Distribution on [a, b].
    """
    if t == 0:
        return 1
    else:
        return (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))


# Moment Generating Function for a Normal Distribution
def normal_mgf(t, mean, variance):
    """
    MGF for Normal Distribution with mean and variance.
    """
    return sp.exp(mean * t + (variance * t ** 2) / 2)


# Moment Generating Function for an Exponential Distribution
def exponential_mgf(t, rate):
    """
    MGF for Exponential Distribution with rate (lambda).
    Returns a symbolic expression for MGF.
    """
    # Check if the input is symbolic or numeric
    if isinstance(t, sp.Basic):  # SymPy symbolic variable
        return 1 / (1 - t / rate)
    elif isinstance(t, (int, float)):  # Numeric input
        if t < rate:
            return 1 / (1 - t / rate)
        else:
            return "Undefined for t >= rate"


# Moment Generating Function for a Binomial Distribution
def binomial_mgf(t, n, p):
    """
    MGF for Binomial Distribution with parameters n (trials) and p (success probability).
    """
    return (1 - p + p * sp.exp(t)) ** n


# MGF for Bernoulli Distribution
def bernoulli_mgf(t, p):
    """
    MGF for Bernoulli Distribution with probability p.
    """
    return (1 - p) + p * sp.exp(t)



# MGF for Poisson Distribution
def poisson_mgf(t, lam):
    return np.exp(lam * (np.exp(t) - 1))

# Example usage
if __name__ == "__main__":
    # Uniform distribution on [0, 1]
    a, b = 0, 1
    uniform_mgf_func = uniform_mgf(t, a, b)
    print("Uniform MGF on [0, 1]:", uniform_mgf_func)

# Example usage
    # Normal distribution with mean=0 and variance=1
    normal_mgf_func = normal_mgf(t, mean=0, variance=1)
    print("Normal MGF:", normal_mgf_func)

    # Exponential distribution with rate=2
    exp_mgf_func = exponential_mgf(t, rate=2)
    print("Exponential MGF:", exp_mgf_func)

    # Binomial distribution with n=10 and p=0.5
    binom_mgf_func = binomial_mgf(t, n=10, p=0.5)
    print("Binomial MGF:", binom_mgf_func)

    # Bernoulli distribution with p = 0.7
    p = 0.7
    bernoulli_mgf_func = bernoulli_mgf(t, p)
    print("Bernoulli MGF (symbolic):", bernoulli_mgf_func)

    # Numeric example
    t_val = 1  # Substitute a numeric value for t
    bernoulli_mgf_numeric = bernoulli_mgf_func.subs(t, t_val)
    print(f"Bernoulli MGF (numeric, t={t_val}):", bernoulli_mgf_numeric)

    # Poisson MGF example
    lam = 3
    t_poisson = 0.5
    print(f"Poisson MGF (λ={lam}, t={t_poisson}): {poisson_mgf(t_poisson, lam)}")