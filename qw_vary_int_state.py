# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:52:03 2023

@author: Rico
"""

import numpy as np
import matplotlib.pyplot as plt

n = 100;  # num of random steps
p = 2 * n + 1;  # num of positions

j= 1j;

for theta in range(0, int(2*180)):
    for phi in range(0, int(4*180)):
        coin0 = np.array([np.cos(np.radians(theta)), 0]);  # |0>
        coin1 = np.array([0, np.exp(j * phi) * np.sin(np.radians(theta))])  # |1>
    
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
        
        
        xval = np.arange(-p/2,p/2)
        # Begin plotting the graph
        fig = plt.figure() # Create an overall figure
        ax = fig.add_subplot(111) # Add a 3D plot
        
        # NOTE: Only plots non-zero values
        ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], linewidth=1, color='r') # Plot the data
        ax.plot(xval[np.where(prob != 0)], prob[np.where(prob != 0)], 'o', markersize= 3, color='blue') # Plot the data
        
        print(xval, prob);
        plt.xlabel("Position"); # Set x label
        plt.ylabel("Probability"); # Set y label
        ax.set_title('Quantum Walk');
        
        # plt.savefig('Images/qw_varying_instate', dpi=720);
        plt.show() # Show the graph
