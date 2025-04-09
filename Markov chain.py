import quantecon as qe
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Define our Markov chain:
P = np.array([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0, 0.5, 0.5]])
mc = qe.MarkovChain(P, state_values=('good', 'bad', 'disaster'))
# Simulate a path
mc.simulate(ts_length = 10, init = 'good')
#Find all stationary distributions for the given Markov Chain:
print(mc.stationary_distributions)
