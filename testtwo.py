# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:46:18 2018

@author: Paul
"""

import tensorflow as tf
import numpy as np


x_input = tf.placeholder(tf.float32)
a = tf.get_variable('a',shape=[1])
b = tf.get_variable('b',shape=[1])
c = tf.get_variable('c',shape=[1])

y_tf = a*x_input**2+b*x_input+c

y_input = tf.placeholder(tf.float32)
Loss = tf.reduce_mean((y_tf-y_input)**2)

operation = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(Loss)


x = np.linspace(-3,3,1000)
y = 3*x**2-5*x+1#+0.05*np.random.normal(0,1,1000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.Print(a,[a])))
    for i in range(1000):
        sess.run(operation,feed_dict={x_input:x,y_input:y})
    print(sess.run(tf.Print(a,[a])))
    print(sess.run(tf.Print(b,[b])))
    print(sess.run(tf.Print(c,[c])))


    print(sess.run(tf.Print(y_tf,[y_tf]),feed_dict={x_input:x,y_input:y}))



