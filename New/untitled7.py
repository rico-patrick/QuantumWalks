# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:05:47 2023

@author: ekb22193
"""

import numpy as np
import matplotlib.pyplot as plt

n = 100;  # num of random steps
p =4  # num of positions

coin0 = np.array([1, 0]);  # |0>
coin1 = np.array([0, 1]);  # |1>

print(np.eye(p))
ShiftPlus = np.roll(np.eye(p), 1, axis=0);
print(ShiftPlus)

North = np.array([1,0,0,0])
South = np.array([0,1,0,0])
East = np.array([0,0,1,0])
West = np.array([0,0,0,1])
print(np.outer(South, South));