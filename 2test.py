# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:43:52 2018

@author: Paul
"""

import tensorflow as tf

a = tf.constant('Hi')
b = tf.constant('Tianshu !')

k = tf.Print(a,[a])
k = a+b

with tf.Session() as sess:
    sess.run(k)

