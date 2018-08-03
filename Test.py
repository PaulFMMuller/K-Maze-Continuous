# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:53:16 2018

@author: Paul
"""

import tensorflow as tf
from DDPG import DDPG
import numpy as np

with tf.Session() as session:
    TestNet = DDPG(3,1,session)
    
    states  = np.random.normal(size=(10,3)).astype('float32')
    goals   = np.random.normal(size=(10,1)).astype('float32')
    actions = np.random.normal(size=(10,3)).astype('float32')
    rewards     = np.random.normal(size=(10,1)).astype('float32')
    nextStates  = np.random.normal(size=(10,3)).astype('float32')

    TestNet.fit(states,actions,goals,rewards,nextStates,session,epochs=10)
    
    if False:
        print(TestNet.predictAction(states,goals,session))
        print(TestNet.predictQValue(states,goals,actions,session))
    
        print(TestNet.predictAction(states,goals,session,True))
        print(TestNet.predictQValue(states,goals,actions,session,True))

    
    
    