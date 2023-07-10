# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:57:20 2023

@author: Zeo
"""

import numpy as np
import matplotlib.pyplot as plt


def classical_walk(samples):
    posN = np.zeros(P);
    posN[N]= 1;
    for i in range (N):
        posN = (np.roll(posN, 1)+ np.roll(posN, -1)) /2;
    return posN;
N = 100;
P = 2 * N + 1;
samples = 1000;
prob = classical_walk(N);
# Plot the probability distribution
plt.bar(np.arange(-N, N+1), prob);
plt.xlabel('Position')
plt.ylabel('Probability')
plt.title('Classical Random Walk Probability Distribution (Time)')
plt.savefig('Images\classical_walk_time', dpi=1080)
plt.show();
