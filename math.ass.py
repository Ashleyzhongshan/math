import numpy as np  # For numerical arrays and matrix operations
import pandas as pd  # For data manipulation and displaying matrices as DataFrames
from scipy.linalg import null_space  # For calculating the null space of a matrix
from sympy import Matrix, symbols, diff, limit, sin, cos, tan, log, pprint  # For symbolic computations
import numdifftools as nd  # For numerical differentiation

# Define vector v and matrix a

v = np.array([1, 2, 3, 4])
print("Step 1 - Vector v:", v)

a = np.array([[1, 2, 3], [-1, 2, 5], [4, 3, 1]])
print("Step 1 - Matrix a:\n", a)

# Convert matrix a to a DataFrame for clearer display
A = pd.DataFrame(a)
print("\nStep 1 - Matrix a as DataFrame:\n", A)

# Step 2: Matrix Product of A and B

# Define another matrix B to multiply with A
B = np.array([[2, 3, 1], [5, -2, 4], [3, 1, -1]])

# Matrix product using pandas DataFrames
product_df = A.dot(B_df)
print("\nStep 2 - Matrix Product (A * B):\n", product_df)

# Matrix product using numpy
product_np = np.matmul(a, B)
print("\nStep 2 - Matrix Product (numpy):\n", product_np)

# Step 3: Matrix Transpose

# Transpose of matrix A
transpose_df = A.T
print("\nStep 3 - Transpose of matrix A (DataFrame):\n", transpose_df)

transpose_np = np.transpose(a)
print("\nStep 3 - Transpose of matrix a (numpy):\n", transpose_np)

# Step 4: Matrix Inverse and Determinant
# Check if A is orthogonal
I = pd.DataFrame(np.identity(2))
if A.T.dot(A).equals(I):
    print("Matrix A is orthogonal")
else:
    print("Matrix A is not orthogonal")

# Check if matrix is symmetric
if np.allclose(A, A.T):  # useful function to compare two matrices--boolean->return true if A = A.T
    print("A is symmetric")
else:
    print("A is not symmetric")

# Check if matrix a is invertible (non-singular) by calculating its determinant

determinant_a = np.linalg.det(a)
print("\nStep 4 - Determinant of matrix a:", determinant_a)

if determinant_a != 0:
    inverse_a = np.linalg.inv(a)
print("\nStep 4 - Inverse of matrix a:\n", inverse_a)
else:
print("\nStep 4 - Matrix a is singular and does not have an inverse.")

# Step 5: Rank of Matrix a

rank_a = np.linalg.matrix_rank(a)
print("\nStep 5 - Rank of matrix a:", rank_a)

# Step 7: Identity Matrix

identity_matrix = np.eye(3)
print("\nStep 7 - Identity Matrix of rank 3:\n", identity_matrix)

# Step 8: Compare Two Matrices

# Comparing A and B using np.allclose for floating-point values

are_equal = np.allclose(a, B)
print("\nStep 8 - Are matrices A and B equal?", are_equal)

# Step 9: Adjugate Matrix

# Using sympy for adjugate matrix

c = Matrix(a)  # Define matrix with sympy
adjugate_c = c.adjugate()
print("\nStep 9 - Adjugate of matrix a (sympy):")
pprint(adjugate_c)

# Step 10: Kronecker Product
kron_product = np.kron(a, B)
print("\nStep 10 - Kronecker Product of matrices a and B:\n", kron_product)

# Step 11: Null Space and Nullity
null_space_a = pd.DataFrame(null_space(a))
nullity = null_space_a.shape[1]
print("\nStep 11 - Null Space of matrix a:\n", null_space_a)
print("Step 11 - Nullity of matrix a:", nullity)

# Step 12: Reduced Row Echelon Form (RREF)
c = np.array([[1, 2, 3], [-1, 2, 5], [4, 3, 1]])
# Convert the numpy array to a sympy Matrix for RREF computation
c_sympy = Matrix(c)
# Compute the RREF of the matrix
rref_c, pivot_columns = c_sympy.rref()
print("\nStep 12 - Reduced Row Echelon Form (RREF) of matrix c:")
pprint(rref_c)

# Step 13: Solving Linear System Ax = b
b_vector = np.array([1, 2, 3])  # Define constant vector for Ax = b
solution = np.linalg.solve(a, b_vector)
print("\nStep 13 - Solution of Ax = b for x:", solution)

# Step 14: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(a)
print("\nStep 14 - Eigenvalues of matrix a:", eigenvalues)
print("Step 14 - Eigenvectors of matrix a:\n", eigenvectors)

# Step 15: Trace of Matrix a
trace_a = np.trace(a)
print("\nStep 15 - Trace of matrix a:", trace_a)

# Step 16: Leading Principal Minors
D1 = a[0, 0]
D2 = np.linalg.det(a[0:2, 0:2])
D3 = np.linalg.det(a[0:3, 0:3])

print("\nStep 16 - Leading Principal Minors of matrix a:")
print("D1:", D1, "D2:", D2, "D3:", D3)

# Step 17: Limits and Differentiation
# Define symbolic variables
x = symbols('x')
expr1 = sin(x) / x

# Calculating limit of expr1 as x approaches 0
limit_at_zero = limit(expr1, x, 0)
print("\nStep 17 - Limit of sin(x)/x as x approaches 0:", limit_at_zero)

# Differentiate expr1 with respect to x
diff_expr1 = diff(expr1, x)
print("\nStep 17 - Derivative of sin(x)/x with respect to x:")
pprint(diff_expr1)


# Step 18: Gradient, Hessian, and Jacobian
def rosen(x):
    return (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 2


# Compute gradient, Jacobian, and Hessian for the function rosen at point [2, 3]

grad = nd.Gradient(rosen)([2, 3])
jac = nd.Jacobian(rosen)([2, 3])
hessian = nd.Hessian(rosen)([2, 3])

print("\nStep 18 - Gradient at [2,3]:", grad)
print("Jacobian at [2,3]:\n", jac)
print("Hessian at [2,3]:\n", hessian)

# Step 19: Polynomial Solving

# Define a polynomial expression and solve for roots

expr_poly = x ** 3 - 6 * x ** 2 + 11 * x - 6  # Example polynomial
roots = solve(expr_poly, x)
print("\nStep 19 - Roots of the polynomial x^3 - 6x^2 + 11x - 6:", roots)

# Define symbolic variables and functions
x, y = sp.symbols('x y')
f = sp.sqrt(1 + 1 / x)
f_prime_x = sp.simplify(sp.diff(f, x))
print("First derivative with respect to x:", f_prime_x)

# Hessian matrix of a function
f = x ** 2 * sp.sin(1 / x)
Hessian = sp.hessian(f, (x, y))
print("\nHessian matrix of f with respect to (x, y):")
sp.pretty_print(Hessian)

# Limits and continuity check
x0 = 2
j = sp.Piecewise(((-1 / 2) * x - 3, x <= x0), (x / (-1 / 2), x >= x0))
left_limit = sp.limit(j, x, x0, dir='-')
right_limit = sp.limit(j, x, x0, dir='+')
value_at_x0 = j.subs(x, x0)
print(f"j(x) is {'continuous' if left_limit == right_limit == value_at_x0 else 'not continuous'} at x = {x0}")

# Checking injective, surjective, and bijective properties
f = x ** 2
domain = sp.Interval(0, sp.oo, left_open=True)
codomain = sp.Interval(0, sp.oo)
is_injective = sp.simplify(sp.Eq(f.subs(x, x), f.subs(x, y))) == sp.false
is_surjective = sp.simplify(sp.solve(sp.Eq(f, y), x))
print(f"Function f(x) is injective: {is_injective}")

# Define continuity over an interval
print("Domain of f(x):", sp.calculus.util.continuous_domain(f, x, sp.S.Reals))


