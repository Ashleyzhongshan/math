#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import scipy
from scipy import linalg

# In[2]:

# Declare matrix A
A = [[-7,-4,-8],[2,3,2],[5,4,6]]
A = pd.DataFrame(np.array(A))
A


# In[3]:


# Find the eigenvalues and put them in a matrix
eigenvalues=np.linalg.eig(A)[0]
D=pd.DataFrame(np.diag(eigenvalues))
D


# In[4]:


# Find the eigenvectors
V=pd.DataFrame(np.linalg.eig(A)[1])
V


# In[5]:


# Declare matrix E
E = pd.DataFrame(np.array([[2,1,3,4],[0,-1,-1,4],[0,0,0,0]]))
E


# In[6]:


print("the rank of E is")
print(np.linalg.matrix_rank(E))
print ('The kernel of the matrix E is ')
nullity = (pd.DataFrame(scipy.linalg.null_space(E)))
nullity


# In[7]:


print("The nullity of matrix E is")
print(len(nullity.columns))


# In[8]:


# Check that Ax = 0, with 4 decimals precision
E.dot(nullity).round(4)


# In[9]:


# Class example of matrix decomposition using the Spectral Theorem
# Declare the matrix
A = [[-7,-4,-8],[2,3,2],[5,4,6]]
A = pd.DataFrame(np.array(A))
A


# In[10]:


# Compute the eigenvalues lambda
eigenvalues=np.linalg.eig(A)[0]
D=pd.DataFrame(np.diag(eigenvalues))
D


# In[11]:


# Find the eigenvectors
C=pd.DataFrame(np.linalg.eig(A)[1])
C


# In[12]:


# Show that A = C*D*C'
print("Matrix =")
matrix = C.dot(D).dot(C.T).round()
matrix = matrix.round(decimals=0)
print(matrix)
# We need to remove the ".0"
print("Matrix =")
matrix=matrix.astype(np.int32)
print(matrix)


# In[13]:


# Test if it's the same matrix
if matrix.equals(A):
    print("we de composed the matrix correctly")
else:
    print("we made an error")

# Declare matrix A
A = [[4,2,-4,],[0,2,0],[2,2,-2]]
A = pd.DataFrame(np.array(A))
print(A)

print(np.linalg.det(A))

# Find the eigenvalues and put them in a matrix
eigenvalues=np.linalg.eig(A)[0]
D=pd.DataFrame(np.diag(eigenvalues))
print(D)
# eigenvector
C=pd.DataFrame(np.linalg.eig(A)[1])
print(C)
print(np.linalg.matrix_rank(C))