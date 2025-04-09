#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:27:03 2020

@author: oksanabashchenko
"""
# =============================================================================
# Markov Chains
# =============================================================================
import quantecon as qe
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define our Markov chain:
P = np.array([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0, 0.5, 0.5]])
mc = qe.MarkovChain(P, state_values=('good', 'bad', 'disaster'))
# Simulate a path
mc.simulate(ts_length=10, init='good')

# Find all stationary distributions for the given Markov Chain:
print(mc.stationary_distributions)

# =============================================================================
# Optimization
# =============================================================================

from scipy.optimize import minimize

fun = lambda x: (x[0] - 5) ** 2 + (x[1] - 3) ** 2

# Unconstrained optimization: we know that minimun is at (5,3)

x0 = (10, 10)  # Initial guess
res_unconstr = minimize(fun, x0)
print(res_unconstr.x)

# Constrained optimization - example from SciPy documentation

fun_constr_1 = lambda x: x[0] - 3 * x[1] + 4
fun_constr_2 = lambda x: -x[0] - 1 * x[1] + 8.5
fun_constr_3 = lambda x: -x[0] + 2 * x[1]

fun = lambda x: (x[0] - 2) ** 2 + (x[1] - 5.7) ** 2
cons = ({'type': 'ineq', 'fun': fun_constr_1},
        {'type': 'ineq', 'fun': fun_constr_2},
        {'type': 'ineq', 'fun': fun_constr_3},)

bnds = ((0, None), (0, None))

res_constr = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(res_constr.x)
