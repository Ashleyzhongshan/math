import sympy as sp
from sympy import symbols, Matrix, simplify

# Define the variables
x, lam, t = sp.symbols('x lam t')

# Step 1: Define the PDF of the Exponential Distribution
# PDF: f(x) = a * exp(-a * x) for x >= 0
f = lam * sp.exp(-lam * x)

# Step 2: Compute the CDF
# CDF = Integral of f(x) from 0 to x
cdf = sp.integrate(f, (x, 0, x))

# Step 3: Compute the Mean (Expected Value)
# E[X] = Integral of x * f(x) from 0 to infinity
mean = sp.integrate(x * f, (x, 0, sp.oo))

# Step 4: Compute the Variance
# Variance = Integral of (x - E[X])^2 * f(x) from 0 to infinity
variance = sp.integrate((x - mean.args[0][0])**2 * f, (x, 0, sp.oo))

# Step 5: Compute the Moment Generating Function (MGF)
# MGF = E[e^(bX)] = Integral of e^(b*x) * f(x) from 0 to infinity
MGF = sp.integrate((sp.exp(t * x)) * f, (x, 0, sp.oo))

# Display the Results
print(f"The CDF is:\n{cdf}\n")
print(f"The Mean (E[X]) is:\n{mean.args[0][0]}\n")
print(f"The Variance (V[X]) is:\n{variance.args[0][0]}\n")
print(f"The MGF is:\n{sp.expand(MGF.args[0][0])}\n")


# ---------------------------------
x, theta = sp.symbols('x theta', positive=True)

# Step 1: Define the PDF of the Exponential Distribution
g = (2*x)/theta

# Step 2: Compute the Mean (Expected Value)
# E[X] = Integral of x * f(x) from 0 to theta
mean_2 = sp.integrate(x * g, (x, 0, theta))

# Step 3: Compute the Variance
# Variance = Integral of (x - E[X])^2 * f(x) from 0 to infinity
variance_2 = sp.integrate((x - mean_2)**2 * g, (x, 0, theta))

print(f"The Mean_2 (E[X]) is:\n{mean_2}\n")
print(f"The Variance_2 (V[X]) is:\n{variance_2}\n")

# -------------------------------------
# Define the symbols
theta, x = symbols('theta x', positive=True)

# Define the covariance matrix Σ
Sigma = Matrix([[theta, 1], [1, theta]])

# Extract variances and covariance
var_X = Sigma[0, 0]  # Extract variance of X (top-left element)
var_Y = Sigma[1, 1]  # Extract variance of Y (bottom-right element)
cov_XY = Sigma[0, 1]  # Extract covariance of X and Y (top-right element)

# Compute E(Y|X=x)
E_Y_given_X = cov_XY / var_X * x

# Compute Var(Y|X=x)
Var_Y_given_X = var_Y - (cov_XY**2 / var_X)

# Simplify results
E_Y_given_X = simplify(E_Y_given_X)
Var_Y_given_X = simplify(Var_Y_given_X)

print("f E_Y_given_X, Var_Y_given_X) is:", E_Y_given_X, Var_Y_given_X)