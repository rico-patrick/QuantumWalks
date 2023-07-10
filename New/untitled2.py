# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:47:18 2023

@author: ricoe
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 1000

# Initialize the position
position = 0

# Generate 1000 random walks
for i in range(n_steps):
    # Generate a random number between -1 and 1
    step = np.random.choice([-1, 1])

    # Update the position
    position += step

# Calculate the probability distribution
prob = np.zeros(2 * n_steps + 1)
position_values = np.arange(-n_steps, n_steps + 1)
for i, position_value in enumerate(position_values):
    prob[position_value] = np.count_nonzero(position == position_value) / n_steps

# Plot the probability distribution
plt.plot(position_values, prob)
plt.xlabel("Position")
plt.ylabel("Probability")
plt.show()
