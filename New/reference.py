# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:24:06 2023

@author: Zeo
"""

import numpy as np

#The grid is P by P, so there are P^2 possible positions
N = 10
P = 2*N+1

# We have the four coin states North,South,East and West
North = np.array([1,0,0,0])
South = np.array([0,1,0,0])
East = np.array([0,0,1,0])
West = np.array([0,0,0,1])

#The shift operators are the same as the ones of the walk on a line
ShiftPlus = np.roll(np.eye(P), 1, axis=0)
ShiftMinus = np.roll(np.eye(P), -1, axis=0)

#To now shift on the grid, we only shift one coordinate, 
#leaving the other unchanged
ShiftUp = np.kron(np.eye(P), ShiftPlus)
ShiftDown = np.kron(np.eye(P), ShiftMinus)
ShiftRight = np.kron(ShiftPlus, np.eye(P))
ShiftLeft = np.kron(ShiftMinus, np.eye(P))

#Now we want to walk up if the coin shows N, down for S, etc.
S_hat = (np.kron(ShiftUp, np.outer(North, North))
         +np.kron(ShiftDown, np.outer(South, South))
         +np.kron(ShiftRight, np.outer(East, East))
         +np.kron(ShiftLeft, np.outer(West, West)))

#The coin operator for our Algorithm is the Grover operator
G_4 = np.eye(4)-0.5*np.ones((4,4))

#The coin operator of the walk is the Kronecker Product of two Hadamard oparators
#H = 0.5*np.kron(np.array([[1,1],[1,-1]]), np.array([[1,1],[1,-1]]))

C_hat = G_4
#C_hat = H



#And the position identity operator is
PosId = np.kron(np.eye(P), np.eye(P))
 
#Putting this all together, the Walk operator is
U_hat = np.dot(S_hat, np.kron(PosId, C_hat))

#Now we can initialize the state
x0 = np.zeros(P)
x0[N] = 1
y0 = np.zeros(P)
y0[N] = 1


Pos0 = np.kron(x0,y0)

# setting initial coin state to North 
Coin0 = North

# getting initial wavefunction of system 
Psi0 = np.kron(Pos0, Coin0)

#After T steps this state will be
T=10
PsiT = np.dot(np.linalg.matrix_power(U_hat, N), Psi0)


prob = np.zeros((P, P))
# getting probailities using measurement operator 
for i in np.arange(0,P):
    posy = np.zeros(P)
    posy[i] = 1
    for j in np.arange(0,P):
        posx = np.zeros(P)
        posx[j] = 1
        posxy = np.kron(posx, posy)
        M_hat_xy = np.kron(posxy, np.eye(4))
        proj = np.dot(M_hat_xy, PsiT)
        prob[j,i] = (np.dot(proj, np.conjugate(proj))).real

        
#plotting probabilities 

import matplotlib.pyplot as plt

x = np.arange(-N,N+1)
y = np.arange(-N,N+1)

X,Y = np.meshgrid(x,y)

ax = plt.axes(projection='3d')
ax.plot_surface(X,Y, prob,
                cmap='jet')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Probability")


# In[ ]:



