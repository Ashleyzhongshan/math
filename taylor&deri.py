import sys
sys.modules[__name__].__dict__.clear() #Clears all the name of the current module (including variables).

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import approximate_taylor_polynomial as taylor
from matplotlib import style
style.use('ggplot')

x=sp.symbols('x') # Defines that the function is in variable x
f = sp.sqrt(x*sp.sqrt(x-1)) # The function f
print("The derivative of f is {}".format(sp.diff(f))) # Takes the derivative with respect to x

f = sp.sin(x**2) # The function f
print("The 3-rd derivative of f is {}".format(sp.diff(f,x,3))) # Takes the 3-rd derivative with respect to x

f = x/sp.log(x) # The function f
print("The derivative of f is {}".format(sp.diff(f))) # Takes the derivative with respect to x

g = (1+sp.log(x))**(x**2)
print("The derivative of g is {}".format(sp.diff(g))) # Notice that Python doesn't simplify

#Let's do one we did int class
h = sp.sqrt(x*sp.sqrt(x-1)) # Define the function in variable x
derivative_h = sp.diff(h) # Takes the derivative
print("The derivative of h at x=2 is {}".format( sp.diff(h).subs(x,2).evalf() )) # Computes the derivative at point x=2


# Let's do another one
i = x**2 + 2 # Define the function in variable x
derivative_i=sp.diff(i) # Takes the derivative
print("The derivative of i is {}".format(derivative_i))
print("The derivative of i at x=2 is {}".format(derivative_i.subs(x,2))) # Computes the derivative at point x=2


# Total derivatives: Define the variables
x, y = sp.symbols('x y')
f = x**2 + x * sp.cos(y)

# Calculate the partial derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# Display the results
print("The partial derivative of f with respect to x:")
sp.pprint(df_dx)

print("\nThe partial derivative of f with respect to y:")
sp.pprint(df_dy)

total_derivative = df_dx * sp.symbols('dx') + df_dy * sp.symbols('dy')

print("\nThe total derivative of f:")
sp.pprint(total_derivative)
# dx = partial x/ partial x, dy = partial y / partial x
#############################


x=sp.symbols('x')
def fun1(x): return np.exp(-x**2)*np.log(x)**2 # define the function
integral1,err1 = quad(fun1,0,np.inf) # Compute its integral
print(integral1)

def fun2(x): return -x**2 # define the function
integral2,err2 = quad(fun2,0,2) # compute its integral
print(integral2)


###########################

x = sp.symbols('x')
T1 = sp.series(x**2,x,x0=1,n=3) # Computes the taylor expansion of x**2 around a=1
print("T1 tylor expansion:",T1)

# Another Tylor expansion, the one in the midterm
T2 = sp.series(sp.log(x),x,x0=1, n=5) # Computes the 4th order taylor expansion of log(x) around a=1
print("T2 tylor expansion:", T2)

T3 = sp.series(sp.sin(x**2),x,x0=1, n=4) # Computes the 3th order taylor expansion of log(x) around a=1
print("T3 tylor expansion:", T3)
###########################

x=sp.symbols('x')
f = sp.sin(x)/x # Define the function
T6 = sp.series(f,x).removeO() # Computes the taylor expansion order 6
T8 = sp.series(f,x,n=8).removeO() # Computes the taylor expansion order 8
T10 = sp.series(f,x,n=10).removeO() # Computes the taylor expansion order 10
# Now let's plot the original expression f and its approximations T6, T8 and T10. Note how the accuracy of the approximation depends on the truncation order.


