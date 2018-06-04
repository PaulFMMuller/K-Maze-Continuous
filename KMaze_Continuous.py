#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:18:35 2018

@author: paul
"""

import matplotlib.pyplot as plt
import numpy as np

class KMaze_Continuous:
    
    def __init__(self,N,K,chosenDim,initPos=None,random_seed=None,minReward=1/1000,maxReward=1):
        if (not initPos is None) and (not (len(initPos) == K)):
            raise Exception('Error : initial position array and dimensional number mismatch : K = {} ; len(initPos) = {}'.format(\
                            K,len(initPos)))
        if N <= 0:
            raise Exception('Error : Empty environment space.')
                
        np.random.seed(random_seed)
        if initPos is None:
            self.agentPos = np.random.uniform(low=0,high=N,size=K)
        else:
            self.agentPos = np.array(initPos)
            
        self.N = N
        self.K = K
        self.chosenDim = chosenDim
        self.minReward = minReward
        self.maxReward = maxReward
        self.startingPosition = self.agentPos
        
        
    def render(self):
        # Only one dimension : we plot a line.
        if   self.K == 1:
            xPlot = [0,self.N]
            yPlot = [0,0]
            xFin  = [self.agentPos[0]]
            yFin  = [0]
            plt.figure()
            plt.plot(xPlot,yPlot)
            plt.scatter(xFin,yFin)
            plt.show()
        # Two dimensions : we plot a grid.
        elif self.K == 2:
            xPlot = [0,0,self.N,self.N,0]
            yPlot = [0,self.N,self.N,0,0]
            xPos  = self.agentPos[0]
            yPos  = self.agentPos[1]
            
            plt.figure()
            plt.plot(xPlot,yPlot)
            plt.scatter(xPos,yPos)
            plt.show()
        else:
            outputString = 'Position : {}'.format(self.agentPos)
            print(outputString)
            #outputString = 'Sorry, no way to visualize dim >= 3 environments yet ! ;)'
        
        
    # Sample a random action from the environment.
    def sample(self):
        return np.random.uniform(low=-1,high=1,size=self.K)
    
    
    # Resets the environment without changing the start position.
    def reset(self):
        self.agentPos = self.startingPos
        return self.agentPos
    
    
    def step(self,action):
        if np.any(np.abs(action) > 1):
            Warning('Warning : New action out of action space. Clipping its values.')
            action[action > 1] = 1
            action[action <-1] =-1

        self.agentPos += action
        observation = self.agentPos
            
        reward,done,info = self.evaluateState(self.agentPos,self.N,self.chosenDim,self.maxReward,self.minReward)
        
        return observation, reward, done, info
        
    
    def evaluateState(self,agentPos,N,goal,maxReward,minReward):
        reward = 0                              # 0 by default
        info   = [None]
        done   = False
        if np.any(agentPos < 0) or np.any(agentPos >= N):
            done = True
            dim = goal // 2
            pos = goal - 2*dim
            if (agentPos[dim] <= 0 and pos == 0) or (agentPos[dim] >= N and pos == 1):
                reward = maxReward
            else:
                reward = minReward
        return reward,done,info
    
    
    # Used at the end of training to find the goal reached.
    def getHERGoal(self):
        i = 0
        while self.agentPos[i] >= 0 and self.agentPos[i] < self.N:
            i = i+1
        K = 2*i
        if self.agentPos[i] >= self.N:
            K = K+1
        return K
    
    
    def seq2SeqNR(self,actionSequence):
        K = 2*self.K-1
        NewSeqs = []
        for g in range(K):
            CurrSeq = []
            for i in range(len(actionSequence)-1):    # Not optimized at all. The "terminal" state must be included within the sequence.
                state,action = actionSequence[i]
                newState,newAc = actionSequence[i+1]    # 1-step
                reward,done,info = self.evaluateState(newState,self.N,g,self.maxReward,self.minReward)
                CurrSeq.append((state,action,g,reward,newState))
            # But terminal states shouldn't be used for q learning.
            NewSeqs.append(CurrSeq)
        return NewSeqs
        
    
        
        
        
        
        
        
        