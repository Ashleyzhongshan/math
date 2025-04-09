# -*- coding: utf-8 -*-


import sympy as sp
# Section 1: general computations. Graphs. Limits
r = sp.symbols('r')
f=(3*r+5)/(r-3)             #define the function f
p0 = sp.plot(f,ylim = (-50,50), title = 'Plot of the function f')

g = r**2 +1                 #define the function g
p1 = sp.plot(g, xlim = (-10,10), title = 'Plot of the function g')

l1 = sp.limit(f,r,4)        #Calculates the limit of f at point 4
print("limit of f at point 4=", l1)
l2 = sp.limit (g,r, 4)      #Calculates the limit of g at point 4
print("limit of g at point 4=", l2)
lAdd = sp.limit(f + g,r, 4)    #Calculates the limit of f+g at point 4
print("limit of f+g at point 4=", lAdd)
lSub = sp.limit(f - g,r, 4)    #Calculates the limit of f-g at point 4
print("limit of f-g at point 4=", lSub)
lMult = sp.limit(f*g,r, 4)     #Calculates the limit of f*g at point 4
print("limit of f*g at point 4=", lMult)
lDiv = sp.limit (f/g,r, 4)     # Calculates the limit of f/g at point 4
print("limit of f*g at point 4=", lDiv)

# Left limit and right limit
q = sp.symbols('q')
p = (q-3)/abs(q-3) #define the function p
p2 = sp.plot(p, xlim = (-1,5), title = 'Plot of the function p')
leftlim = sp.limit(p,q,3,'-')
print('left limit of p at point 3=', leftlim)
rightlim = sp.limit(p,q,3,'+')
print('right limit of p at point 3=', rightlim)

# 3D graphs
from sympy.plotting import plot3d
x,y = sp.symbols('x,y')
z = x**2-y**2
p3 = plot3d(z, (x,-3,3), (y,-3,2), xlabel = 'x axis', ylabel = 'y axis', title = '3d plot')

# Plotting 3d symbolic curves
from sympy.plotting import plot3d_parametric_line
u = sp.symbols('u')
p4 = plot3d_parametric_line(sp.cos(u), sp.sin(u), u, (u,-5,5), title = '3d parametric line plot')

# Multiple plots on different subregions
# The first two inputs to subplot indicate the number of plots in each row and column. The third input specifies which plot is active.
from sympy.plotting import PlotGrid
PlotGrid(2,2,p0, p1, p3,p4)



