
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:24:06 2018

@author: Paul
"""

from KMaze_Continuous import KMaze_Continuous
import numpy as np


N   = 10  # 10 points
K   = 2   # Two dimensions
Dim = 2   # 

env = KMaze_Continuous(N,K,Dim)

done      = False
userInput = False
PossibleActions = [-1,1]

env.render()
while not done:
    if userInput:
        action = np.zeros(K)
        for i in range(K):
            action[i] = float(input('Action {} : '.format(i+1)))
    else:
        action = env.sample()
        
    observation, reward, done, info = env.step(action)
    print(reward)
    env.render()
