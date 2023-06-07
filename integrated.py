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

psi0 = np.kron(posn0, coin0);  # initial state |0>

psiN = np.linalg.matrix_power(U, n).dot(psi0);

prob = np.empty(p);
for k in range(p):
    posn = np.zeros(p);
    posn[k] = 1;
    M_hat_k = np.kron(np.outer(posn, posn), np.eye(2));
    proj = M_hat_k.dot(psiN);
    prob[k] = proj.dot(proj.conjugate()).real;


cposF = [];

def probability():
    cposIn = 0;
    #    posF = [posIn];
    for i in range(n):
        step = np.random.choice([-1, 1]);
        cposIn += step;
    #        posF.append(posIn);
    return cposIn;


samples = 10000
for i in range(samples):
    cposF.append(probability());

# Probability Distribution
cprob = np.zeros(p);
position, counts = np.unique(cposF, return_counts=True);
cprob[position] = counts / float(samples)
cprob = np.concatenate([cprob[n:], cprob[:n]])

# figure
fig = plt.figure() # Create an overall figure
ax = fig.add_subplot(111)

cxval= np.arange(-n,n+1);
xval = np.arange(-n,n+1);

"""Classical"""
ax.plot(cxval[np.where(cprob != 0)], cprob[np.where(cprob != 0)], linewidth=1, color='r', label='Classical')
ax.plot(cxval[np.where(cprob != 0)], cprob[np.where(cprob != 0)], 'x', markersize= 3, color='black')

"""Quantum"""
ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], linewidth=1, color='g', label = 'Quantum')
ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], 'o', markersize= 3, color='b')

plt.xlabel("Position"); # Set x label
plt.ylabel("Probability"); # Set y label
ax.set_xlim(-n, n);
plt.savefig('Images/integrated.png', dpi=720,bbox_inches='tight')
plt.legend(loc='upper left');
plt.tight_layout();
plt.show();