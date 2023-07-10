# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:57:43 2023

@author: Zeo
"""

import random
import matplotlib.pyplot as plt

def random_walk(n_steps):
    position = 0  # Starting position
    positions = [position]  # List to store positions over time
    
    for _ in range(n_steps):
        # Generate a random number to decide the direction
        direction = random.choice([-1, 1])
        position += direction  # Update the position
        positions.append(position)  # Store the current position
        
    return positions

# Simulate a random walk with 10 steps
n_steps = 10
positions = random_walk(n_steps)

time = list(range(n_steps + 1))

# Plot the graph
plt.plot(time, positions, marker='o')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Classical Random Walk')
plt.grid(True)
plt.show()
