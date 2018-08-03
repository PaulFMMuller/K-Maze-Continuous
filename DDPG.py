# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:36:17 2018

@author: Paul
"""

import tensorflow as tf
import numpy as np



def computeJacobian(y,x,lenY):
    outputs = tf.split(y,lenY,axis=1)
    results = [tf.map_fn(lambda u: tf.reshape(tf.gradients(u,x)[0],[-1]),comp) for comp in outputs]
    return tf.stack(results,axis=2)

    


class DDPG:
    def __init__(self,stateShape,goalShape,session,gamma=0.99,learning_rate=0.001,learning_rate_mu=0.001,targetRate=0.01,batch_size=32,sensitivity=1e-4):
        self.gamma         = gamma
        self.learning_rate = learning_rate
        self.targetRate    = targetRate
        self.batch_size    = batch_size
        self.sensitivity   = sensitivity
        self.learning_rate_mu = learning_rate_mu
        self.stateShape    = stateShape
        self.goalShape     = goalShape
        with tf.variable_scope('nn-1',reuse=tf.AUTO_REUSE) as scope:
            self.scope = scope
            self.optimizer    = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)      
            self.mu_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_mu) 
            self.initializeNeuralNetworks(stateShape,goalShape,session)
            self.initializeFit()            
            session.run(tf.global_variables_initializer())

    
    
    def initializeNeuralNetworks(self,stateShape,goalShape,session):
        self.states  = tf.placeholder(tf.float32,shape=[None,stateShape])
        self.goals   = tf.placeholder(tf.float32,shape=[None,goalShape])
        self.actions = tf.placeholder(tf.float32,shape=[None,stateShape])

        mu_input = tf.concat([self.states,self.goals],axis=1)
        q_input  = tf.concat([self.states,self.actions,self.goals],axis=1)

        # Variable Initialization
        self.mu_bias1   = tf.get_variable("mu_bias1",[stateShape+goalShape])        
        self.mu_weight1 = tf.get_variable("mu_weight1",[stateShape+goalShape,stateShape+goalShape])
        
        self.mu_bias2   = tf.get_variable("mu_bias2",[stateShape+goalShape])        
        self.mu_weight2 = tf.get_variable("mu_weight2",[stateShape+goalShape,stateShape+goalShape])
        
        self.mu_bias3   = tf.get_variable("mu_bias3",[stateShape])        
        self.mu_weight3 = tf.get_variable("mu_weight3",[stateShape+goalShape,stateShape])
        
        self.mu_bias1_tar = tf.get_variable("mu_bias1_tar",[stateShape+goalShape]); self.mu_bias1_tar = tf.assign(self.mu_bias1_tar,self.mu_bias1)
        self.mu_weight1_tar = tf.get_variable("mu_weight1_tar",[stateShape+goalShape,stateShape+goalShape]); self.mu_weight1_tar = tf.assign(self.mu_weight1_tar,self.mu_weight1)
        
        self.mu_bias2_tar = tf.get_variable("mu_bias2_tar",[stateShape+goalShape]); self.mu_bias2_tar = tf.assign(self.mu_bias2_tar,self.mu_bias2)
        self.mu_weight2_tar = tf.get_variable("mu_weight2_tar",[stateShape+goalShape,stateShape+goalShape]); self.mu_weight2_tar = tf.assign(self.mu_weight2_tar,self.mu_weight2)
        
        self.mu_bias3_tar = tf.get_variable("mu_bias3_tar",[stateShape]); self.mu_bias3_tar = tf.assign(self.mu_bias3_tar,self.mu_bias3)
        self.mu_weight3_tar = tf.get_variable("mu_weight3_tar",[stateShape+goalShape,stateShape]); self.mu_weight3_tar = tf.assign(self.mu_weight3_tar,self.mu_weight3)

        
        # Use of Variables
        h_mu1 = tf.nn.relu_layer(mu_input, self.mu_weight1, self.mu_bias1)
        h_mu2 = tf.nn.relu_layer(h_mu1, self.mu_weight2, self.mu_bias2)
        self.mu_output = tf.nn.tanh(tf.matmul(h_mu2,self.mu_weight3) + self.mu_bias3)

        h_mu1_tar = tf.nn.relu_layer(mu_input, self.mu_weight1_tar, self.mu_bias1_tar)
        h_mu2_tar = tf.nn.relu_layer(h_mu1_tar, self.mu_weight2_tar, self.mu_bias2_tar)
        self.mu_output_tar = tf.nn.tanh(tf.matmul(h_mu2_tar,self.mu_weight3_tar) + self.mu_bias3_tar)


        
        # Variable Initialization
        self.q_bias1   = tf.get_variable("q_bias1",[2*stateShape+goalShape])        
        self.q_weight1 = tf.get_variable("q_weight1",[2*stateShape+goalShape,2*stateShape+goalShape])
        
        self.q_bias2   = tf.get_variable("q_bias2",[2*stateShape+goalShape])        
        self.q_weight2 = tf.get_variable("q_weight2",[2*stateShape+goalShape,2*stateShape+goalShape])
        
        self.q_bias3   = tf.get_variable("q_bias3",[1])        
        self.q_weight3 = tf.get_variable("q_weight3",[2*stateShape+goalShape,1])
        
        self.q_bias1_tar = tf.get_variable("q_bias1_tar",[2*stateShape+goalShape]); self.q_bias1_tar = tf.assign(self.q_bias1_tar,self.q_bias1)
        self.q_weight1_tar = tf.get_variable("q_weight1_tar",[2*stateShape+goalShape,2*stateShape+goalShape]); self.q_weight1_tar = tf.assign(self.q_weight1_tar,self.q_weight1)
        
        self.q_bias2_tar = tf.get_variable("q_bias2_tar",[2*stateShape+goalShape]); self.q_bias2_tar = tf.assign(self.q_bias2_tar,self.q_bias2)
        self.q_weight2_tar = tf.get_variable("q_weight2_tar",[2*stateShape+goalShape,2*stateShape+goalShape]); self.q_weight2_tar = tf.assign(self.q_weight2_tar,self.q_weight2)
        
        self.q_bias3_tar = tf.get_variable("q_bias3_tar",[1]); self.q_bias3_tar = tf.assign(self.q_bias3_tar,self.q_bias3)
        self.q_weight3_tar = tf.get_variable("q_weight3_tar",[2*stateShape+goalShape,1]); self.q_weight3_tar = tf.assign(self.q_weight3_tar,self.q_weight3)

        # Use of Variables
        h_q1 = tf.nn.relu_layer(q_input, self.q_weight1, self.q_bias1)
        h_q2 = tf.nn.relu_layer(h_q1, self.q_weight2, self.q_bias2)
        self.q_output = tf.nn.tanh(tf.matmul(h_q2,self.q_weight3) + self.q_bias3)

        h_q1_tar = tf.nn.relu_layer(q_input, self.q_weight1_tar, self.q_bias1_tar)
        h_q2_tar = tf.nn.relu_layer(h_q1_tar, self.q_weight2_tar, self.q_bias2_tar)
        self.q_output_tar = tf.nn.tanh(tf.matmul(h_q2_tar,self.q_weight3_tar) + self.q_bias3_tar)

    
    
    
    def predictAction(self,state,goal,session,useTarget=False):
        #with self.scope:
        if useTarget:
            result = session.run(self.mu_output_tar,feed_dict={self.states:state,self.goals:goal})    
        else:
            result = session.run(self.mu_output,feed_dict={self.states:state,self.goals:goal})
        return result
            
        
    
    def predictQValue(self,state,action,goal,session,useTarget=False,evalTensor=True):
        #with self.scope:
        if useTarget:
            if evalTensor:
                result = session.run(self.q_output_tar,feed_dict={self.states:state,self.actions:action,self.goals:goal})
            else:
                result = self.q_output_tar
        else:
            if evalTensor:
                result = session.run(self.q_output,feed_dict={self.states:state,self.actions:action,self.goals:goal})
            else:
                result = self.q_output
        return result                  
                              
                              
    
    
    def getLossGradientNorm(self):
        grad_q_a = tf.stack(tf.gradients(self.q_output,self.actions),2)     # Works because only q_i depends on action_i
        grad_m_mu_bias1     = computeJacobian(self.mu_output,self.mu_bias1,  self.stateShape)
        grad_m_mu_weight1   = computeJacobian(self.mu_output,self.mu_weight1,self.stateShape)
        grad_m_mu_bias2     = computeJacobian(self.mu_output,self.mu_bias2,  self.stateShape)
        grad_m_mu_weight2   = computeJacobian(self.mu_output,self.mu_weight2,self.stateShape)
        grad_m_mu_bias3     = computeJacobian(self.mu_output,self.mu_bias3,  self.stateShape)
        grad_m_mu_weight3   = computeJacobian(self.mu_output,self.mu_weight3,self.stateShape)
        
        
        multFun     = lambda x: tf.matmul(x[0],x[1],transpose_b=False)
        
        scal_bias1   = tf.map_fn(multFun,(grad_m_mu_bias1,grad_q_a),dtype=tf.float32)
        scal_weight1 = tf.map_fn(multFun,(grad_m_mu_weight1,grad_q_a),dtype=tf.float32)
        scal_bias2   = tf.map_fn(multFun,(grad_m_mu_bias2,grad_q_a),dtype=tf.float32)
        scal_weight2 = tf.map_fn(multFun,(grad_m_mu_weight2,grad_q_a),dtype=tf.float32)
        scal_bias3   = tf.map_fn(multFun,(grad_m_mu_bias3,grad_q_a),dtype=tf.float32)
        scal_weight3 = tf.map_fn(multFun,(grad_m_mu_weight3,grad_q_a),dtype=tf.float32)
    
        return tf.reduce_mean(tf.norm(scal_bias1,axis=(1,2))**2+tf.norm(scal_weight1,axis=(1,2))**2+tf.norm(scal_bias2,axis=(1,2))**2+\
                tf.norm(scal_weight2,axis=(1,2))**2+tf.norm(scal_bias3,axis=(1,2))**2+tf.norm(scal_weight3,axis=(1,2))**2)
        
        
    def initializeFit(self):
        self.tensor_targets   = tf.placeholder(tf.float32,shape=[None,1])          
        predictedQValues = self.predictQValue(None,None,None,None,evalTensor=False)
                    
        loss = tf.reduce_mean((predictedQValues-self.tensor_targets)**2)
        self.updateQNetwork = self.optimizer.minimize(loss,var_list=[self.q_bias1,self.q_weight1,self.q_bias2,\
                                                                                               self.q_weight2,self.q_bias3,self.q_weight3])
        
        self.gradientNorm = self.getLossGradientNorm()
        self.updateMuNetwork = self.mu_optimizer.minimize(self.gradientNorm,var_list=[self.mu_bias1,self.mu_weight1,self.mu_bias2,\
                                                                                               self.mu_weight2,self.mu_bias3,self.mu_weight3])   
    
    
        self.update = tf.assign(self.q_bias1_tar,(1-self.targetRate)*self.q_bias1_tar   + self.targetRate * self.q_bias1)
        self.update = tf.assign(self.q_weight1_tar,(1-self.targetRate)*self.q_weight1_tar + self.targetRate * self.q_weight1)
        
        self.update = tf.assign(self.q_bias2_tar,(1-self.targetRate)*self.q_bias2_tar   + self.targetRate * self.q_bias2)
        self.update = tf.assign(self.q_weight2_tar,(1-self.targetRate)*self.q_weight2_tar + self.targetRate * self.q_weight2)
                
        self.update = tf.assign(self.q_bias3_tar,(1-self.targetRate)*self.q_bias3_tar   + self.targetRate * self.q_bias3)
        self.update = tf.assign(self.q_weight3_tar,(1-self.targetRate)*self.q_weight3_tar + self.targetRate * self.q_weight3)
        
        self.update = tf.assign(self.mu_bias1_tar,(1-self.targetRate)*self.mu_bias1_tar   + self.targetRate * self.mu_bias1)
        self.update = tf.assign(self.mu_weight1_tar,(1-self.targetRate)*self.mu_weight1_tar + self.targetRate * self.mu_weight1)
                
        self.update = tf.assign(self.mu_bias2_tar,(1-self.targetRate)*self.mu_bias2_tar   + self.targetRate * self.mu_bias2)
        self.update = tf.assign(self.mu_weight2_tar,(1-self.targetRate)*self.mu_weight2_tar + self.targetRate * self.mu_weight2)
                
        self.update = tf.assign(self.mu_bias3_tar,(1-self.targetRate)*self.mu_bias3_tar   + self.targetRate * self.mu_bias3)
        self.update = tf.assign(self.mu_weight3_tar,(1-self.targetRate)*self.mu_weight3_tar + self.targetRate * self.mu_weight3)


    
        # On fait converger le gradient vers 0 ==> On trouve le minimum local de la policy !

        
    
    def fit(self,states,actions,goals,rewards,nextStates,session,epochs=1):
        with tf.variable_scope(self.scope):
            for epoch in range(epochs):
                # Estimate argmax_a(Q(s_(t+1),a))
                currentActions   = self.predictAction(states,goals,session,useTarget=False)
                actionsNextState = self.predictAction(nextStates,goals,session,useTarget=True)
                newQValue        = self.predictQValue(nextStates,actionsNextState,goals,session,useTarget=True)
                targets          = rewards.reshape(-1,1) + self.gamma * newQValue.reshape(-1,1)
                
                session.run(self.gradientNorm,feed_dict={self.states:states,self.actions:actions,self.goals:goals})
                session.run(self.updateQNetwork,feed_dict={self.states:states,self.actions:actions,self.goals:goals,self.tensor_targets:targets})
                session.run(self.updateMuNetwork,feed_dict={self.states:states,self.actions:currentActions,self.goals:goals})
                
                #♦ Update the guys !
    
    
    
    
"""    def fit(self,states,actions,goals,rewards,nextStates,session,epochs=1):
        with tf.variable_scope(self.scope):
            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learningRate)      
            batchIndexes = list(range(self.batch_size,states.shape[0],self.batch_size))

            for epoch in range(epochs):
                permutations = np.random.permutation(states.shape[0])
                shuffledStates     = states[permutations];
                shuffledActions    = actions[permutations];
                shuffledGoals      = goals[permutations];
                
                # Estimate argmax_a(Q(s_(t+1),a))
                actionsNextState = self.predictAction(nextStates,goals,session,useTarget=True)
                newQValue        = self.predictQValue(nextStates,actionsNextState.reshape(-1),goals,session,useTarget=True)
                targets          = rewards.reshape(-1,1) + self.gamma * newQValue.reshape(-1,1); targets = targets[permutations]
                
                batchStates     = np.split(shuffledStates,batchIndexes,axis=0)
                batchActions    = np.split(shuffledActions,batchIndexes,axis=0)
                batchGoals      = np.split(shuffledGoals,batchIndexes,axis=0)
                batchTargets    = np.split(targets,batchIndexes,axis=0)
        
                for i in range(len(batchStates)):
                    statesUsed      = batchStates[i]
                    actionsUsed     = batchActions[i]
                    goalsUsed       = batchGoals[i]
                    targetsUsed     = batchTargets[i]
                    if len(statesUsed) < 1:
                        continue
                    
                    # Get predicted value for current state,action pair.
                    predictedQValues = self.predictQValue(statesUsed,actionsUsed,goalsUsed,session,evalTensor=False)
                    
                    # Compute target
                    targetsTensor = tf.convert_to_tensor(targetsUsed.reshape(-1,1))
                    loss = tf.reduce_mean((predictedQValues-targetsTensor)**2)
                    opt = optimizer.minimize(loss)
                    session.run(opt,feed_dict={self.states:statesUsed,self.actions:actionsUsed,self.goals:goalsUsed})
                
                print(session.run(loss,feed_dict={self.states:states,self.actions:actions,self.goals:goals}))"""
        
        
        
        
        
        
        
        
        
        
        
        