# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:29:09 2023

@author: Zeo
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10;
P = 2 * N + 1;
j = 1j;

#Coin States

C00 = np.array([1,0,0,0]);
C11 = np.array([0,1,0,0]);
C22 = np.array([0,0,1,0]);
C33 = np.array([0,0,0,1]);


# The coin we are using is the grover operator

C_hat = 1/2 * np.array([[1, -1, -1, -1],
                       [-1, 1, -1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, -1, 1]]);
# print(C_hat);

ShiftPlus = np.roll(np.eye(P), 1, axis=0);
ShiftMinus = np.roll(np.eye(P), -1, axis=0);

ShiftUp = np.kron(np.eye(P), ShiftPlus)
ShiftDown = np.kron(np.eye(P), ShiftMinus)
ShiftRight = np.kron(ShiftPlus, np.eye(P))
ShiftLeft = np.kron(ShiftMinus, np.eye(P))

S_hat = (np.kron(ShiftUp, np.outer(C00, C00)) +np.kron(ShiftDown, np.outer(C11, C11)) +np.kron(ShiftRight, np.outer(C22, C22)) +np.kron(ShiftLeft, np.outer(C33, C33)))
# print(S_hat);

posI = np.kron(np.eye(P), np.eye(P));

U_hat= S_hat.dot(np.kron(posI, C_hat));

## array indexing starts from 0, so index N is the central posn
#initaial position

pos_x0 = np.zeros(P);
pos_x0[N] = 1;
pos_y0 = np.zeros(P);
pos_y0[N] = 1;

#iniaial pos
pos0 = np.kron(pos_y0, pos_x0);
# C = (1 / np.sqrt(4) * np.ones(4))
# initail wavefunction psi_0
psi0 = np.kron(pos0, C00); #initaial state of the coin is C00


#final wavefuntion
t = 17;

psiN = np. linalg.matrix_power(U_hat, t).dot(psi0);
print(psiN);
psiN = np.reshape(psiN, (P, P, 4));
print(psiN);
# prob = np.zeros((P,P));
prob =np.sum(psiN * psiN.conjugate(), axis = 2).real;

        
#Plot
x,y = np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1));
ax = plt.axes(projection='3d');
ax.plot_surface(x,y, prob, cmap='jet');
ax.set_xlabel('X');
ax.set_ylabel('Y');
ax.set_zlabel('Probability');
ax.set_title('2D Quantum Walks')
plt.savefig('Images/2D_discrete_quantum_walks_01_coin_state', dpi=1080);
plt.show();

