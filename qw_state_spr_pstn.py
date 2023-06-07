# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:17:13 2023

@author: Rico
"""

import numpy as np
import matplotlib.pyplot as plt

n = 100;  # num of random steps
p = 2 * n + 1;  # num of positions

coin0 = np.array([1, 0]);  # |0>
coin1 = np.array([0, 1]);  # |1>

C00 = np.outer(coin0, coin0);  # |0><0|
C01 = np.outer(coin0, coin1);  # |0><1|
C10 = np.outer(coin1, coin0);  # |1><0|
C11 = np.outer(coin1, coin1);  # |1><1|

C_hat = (C00 + C01 + C10 - C11) / np.sqrt(2.);

ShiftPlus = np.roll(np.eye(p), 1, axis=0);
ShiftMinus = np.roll(np.eye(p), -1, axis=0);
S_hat = np.kron(ShiftPlus, C00) + np.kron(ShiftMinus, C11);

U = S_hat.dot(np.kron(np.eye(p), C_hat));

posn0 = np.zeros(p);

posn0[n] = 1;  # array indexing starts from 0, so index N is the central posn

psi0 = np.kron(posn0, (coin0+coin1*1j)/np.sqrt(2.)); #initial state superposition of zero and one


psiN = np.linalg.matrix_power(U, n).dot(psi0);

prob1 = np.empty(p);
for k in range(p):
    posn = np.zeros(p);
    posn[k] = 1;
    M_hat_k = np.kron(np.outer(posn, posn), np.eye(2));
    proj = M_hat_k.dot(psiN);
    prob1[k] = proj.dot(proj.conjugate()).real;


xval = np.arange(p)
# Begin plotting the graph
fig = plt.figure() # Create an overall figure
ax = fig.add_subplot(111) # Add a 3D plot

# NOTE: Only plots non-zero values
ax.plot(xval[np.where(prob1 != 0)], prob1[np.where(prob1 != 0)], linewidth=1, color='r') # Plot the data
ax.plot(xval[np.where(prob1 != 0)], prob1[np.where(prob1 != 0)], 'o', markersize= 3, color='blue') # Plot the data
loc = range (0, p, int(p / 10)) #Location of ticks
plt.xticks(loc) # Set the x axis ticks
plt.xlim(0, p) # Set the limits of the x axis
ax.set_xticklabels(range (-n, n+1, int(p / 10))) # Set the labels of the x axis
plt.xlabel("Position") # Set x label
plt.ylabel("Probability") # Set y label
ax.set_title('Quantum Walk')

plt.savefig('Images/qw_instate_spr_pstn', dpi=720)
plt.show() # Show the graph
