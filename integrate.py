import sympy as sp
# Define the variable x and y
x, y = sp.symbols('x y')

# 1. Integral of x^2 * ln(x) from 1 to e
integral_1 = sp.integrate(x**2 * sp.ln(x), (x, 1, sp.E))
print("1. ∫ x^2 * ln(x) dx from 1 to e =", integral_1)

# 2. Integral of cos(2x) * e^(3x) with respect to x
integral_2 = sp.integrate(sp.cos(2*x) * sp.exp(3*x), x)
print("2. ∫ cos(2x) * e^(3x) dx =", integral_2)

# 3. Integral of sqrt(ln(x)) from 1 to e and Integral of e^(x^2) from 0 to 1
integral_3a = sp.integrate(sp.sqrt(sp.ln(x)), (x, 1, sp.E))
integral_3b = sp.integrate(sp.exp(x**2), (x, 0, 1))
print("3. ∫ sqrt(ln(x)) dx from 1 to e =", integral_3a.evalf())
print("3. ∫ e^(x^2) dx from 0 to 1 =", integral_3b.evalf())

# 4. Double integral ∫_2^3 ∫_x^(2x) (x + 2y) dy dx
integral_4 = sp.integrate(sp.integrate(x + 2*y, (y, x, 2*x)), (x, 2, 3))
print("4. ∫_2^3 ∫_x^(2x) (x + 2y) dy dx =", integral_4)

# 5. Double integral ∫_0^π ∫_y^π sin(x)/x dx dy
integral_5 = sp.integrate(sp.integrate(sp.sin(x)/x, (x, y, sp.pi)), (y, 0, sp.pi))
print("5. ∫_0^π ∫_y^π (sin(x)/x) dx dy =", integral_5)

# 6. Integral of (sin(sqrt(x)) / sqrt(x)) with respect to x
integral_6 = sp.integrate(sp.sin(sp.sqrt(x)) / sp.sqrt(x), x)
print("6. ∫ (sin(sqrt(x)) / sqrt(x)) dx =", integral_6)

# 7. Integral of e^x * sin(x) with respect to x
integral_7 = sp.integrate(sp.exp(x) * sp.sin(x), x)
print("7. ∫ e^x * sin(x) dx =", integral_7)

# 8. Double integral ∫_0^1 ∫_0^1 (1 - x**2 - y**2) dx dy
integral_8 = sp.integrate(sp.integrate(1-x**2-y**2, (x, 0, 1)), (y, 0, 1))
print("8. ∫_0^1 ∫_0^1 (1 - x**2 - y**2) dx dy =", integral_8)

# 9. Double integral ∫_0^1 ∫_x^sqrt(x) exp(y)/y dy dx
integral_9 = sp.integrate(sp.integrate(sp.exp(y)/y, (y, x, sp.sqrt(x))), (x, 0, 1))
print("9. ∫_0^1 ∫_0^1 (1 - x**2 - y**2) dx dy =", integral_9)

# 10. Integral of x * e^((-x**2)/2) with respect to x
integral_10 = sp.integrate(x * sp.exp((-x**2)/2), (x,0,1))
print("10. x * e^((-x**2)/2) dx =", integral_10)

# 11. Integral of x * e^(5*x) with respect to x
integral_11 = sp.integrate(x * sp.exp(5*x), x)
print("11. x * e^(5*x) dx =", integral_11)

# 12. Double integral ∫_0^2 ∫_x**2^4 (x**3 * sp.cos(y**3) dy dx
integral_12 = sp.integrate(sp.integrate((x**3) * (sp.cos(y**3)), (y, (x**2), 4)), (x, 0, 2))

# Define the integrand
integrand = x**3 * sp.cos(y**3)
# Original integral order:
# Inner integral with y: y bounds from x**2 to 4
# Outer integral with x: x bounds from 0 to 2
# Reversed order: switch the role of x and y
# x bounds from 0 to sqrt(y), y bounds from 0 to 4
reversed_integral = sp.integrate(sp.integrate(integrand, (x, 0, sp.sqrt(y))), (y, 0, 4))
# Simplify the result
simplified_result = sp.simplify(reversed_integral)
print("The result of the integral is:", simplified_result)

# 13 integral 2 * ∫-sp.oo^sp.oo (sp.exp(-x + theta)**2) / ((1 + sp.exp(-x + theta))**4)
# Define variables
x, theta = sp.symbols('x theta')
# Define the integrand
integrand = (sp.exp(-x + theta)**2) / ((1 + sp.exp(-x + theta))**4)

# Define the integral with limits -infinity to infinity
integral_13 = 2 * sp.integrate(integrand, (x, -sp.oo, sp.oo))
print("13.", integral_13)


# 14.
integral_14 = sp.integrate(x * sp.log(x), (x, 5, 10))
print("14. x * log^(x) dx =", integral_14)