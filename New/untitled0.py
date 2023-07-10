# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:40:37 2023

@author: Zeo
"""

import numpy as np

import matplotlib.pyplot as plt


N = 100;
P = 2* N + 1;

def coin_toss ():
    pos = 0;
    posF = [];
    for i in range(100):
        coin_state= np.random.choice([-1, 1]);
        pos += coin_state;
        posF.append(pos);
    return posF;
    
posF= coin_toss();

#Count the occurrences of each position
position_counts = [];
for position in posF:
    if position in position_counts:
        position_counts[position] += 1
    else:
        position_counts[position] = 1

#Calculate the probability for each position
position_probabilities = {}
total_tosses = len(posF)
for position, count in position_counts.items():
    probability = count / total_tosses
    position_probabilities[position] = probability

# Create lists for positions and probabilities
positions = list(position_probabilities.keys())
probabilities = list(position_probabilities.values())

# Plot the positions and probabilities
plt.bar(positions, probabilities)
plt.xlabel('Position')
plt.ylabel('Probability')
plt.title('Probability Distribution of Positions')
plt.show()