import numpy as np

A = np.array([[4,5,4],[4,2,1],[8,2,1]])


b1 = np.array([-1,-2,6])
c1 = np.array([0.5,0,0.125])
d1 = np.array([0.25,0,0.75])
a1 = np.array([0,1,-1])


print(A@a1)
print(A@b1)
print(A@c1)
print(A@d1)


# A = C*D*C_-1
C = np.array([[5,2],[2,1]])
C_1 = np.linalg.inv(C)
EV = np.array([[0,0],[0,-1]])
P_EV = np.linalg.matrix_power(EV,98)

print(f"Power 98 is:",C@P_EV@C_1)


# Define the transition matrix P and initial state X0
P = np.array([
    [0.2, 0.6, 0.2],
    [0.3, 0.5, 0.2],
    [0.5, 0.3, 0.2]
])

X0 = np.array([1, 0, 0])

# Compute P^4
P4 = np.linalg.matrix_power(P, 4)

# Compute X4 = X0 * P^4
X4 = X0@P4

# Display results
print("P4 is:", P4)
print("X4 is:", X4)