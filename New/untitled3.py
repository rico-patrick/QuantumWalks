# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:02:15 2023

@author: ekb22193
"""

import random

def time_evolution_classical_walk(n_steps):
  """
  Performs a time evolution classical walk of n steps.

  Args:
    n_steps: The number of steps to take.

  Returns:
    A list of the positions of the walker at each step.
  """

  positions = []
  position = 0
  for _ in range(n_steps):
    step = random.choice([-1, 1])
    position += step
    positions.append(position)

  return positions


if __name__ == "__main__":
  n_steps = 10
  positions = time_evolution_classical_walk(n_steps)
  print(positions)
import matplotlib.pyplot as plt

def plot_time_evolution_classical_walk(positions):
  """
  Plots the graph of the time evolution of the classical walk.

  Args:
    positions: A list of the positions of the walker at each step.
  """

  plt.plot(range(len(positions)), positions)
  plt.xlabel("Step")
  plt.ylabel("Position")
  plt.show()


if __name__ == "__main__":
  n_steps = 100
  positions = time_evolution_classical_walk(n_steps)
  plot_time_evolution_classical_walk(positions)
