#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:41:43 2020

@author: zhuowang
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def GenerateData(batchsize = 100):
    train_X=np.linspace(-1,1,batchsize)
    train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3
    yield train_X, train_Y
#    
Xinput = tf.placeholder("float",(None))
Yinput = tf.placeholder("float",(None))
    
training_epochs = 20
with tf.Session() as sess:
   for epoch in range(training_epochs):
       for x,y in GenerateData():
           xv,yv = sess.run([Xinput, Yinput], feed_dict={Xinput:x, Yinput:y})
    
print(epoch,"| x.shape:",np.shape(xv),"| x[:3]:",xv[:3])
print(epoch,"| y.shape:",np.shape(yv),"| y[:3]:",yv[:3])
    
train_data = list(GenerateData())[0]
plt.plot(train_data[0],train_data[1],'ro',label='Original data')
plt.legend()
plt.show()