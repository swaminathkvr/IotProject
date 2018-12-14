# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:26:55 2018

@author: Swaminath
"""

import numpy as np

# define a neural network class according to our model
class Neural_Network(object):
    def __self__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3 # number of neurons in hidden layer
        self.outputLayerSize = 1

        # shape - 2,3
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) # creates a random number matrix which is normally distributed
        self.W1 = np.abs(self.W1)
        # shape - 3,1        
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) # creates a random number matrix which is normally distributed
        self.W2 = np.abs(self.W2)

        # shape - 3,
        self.B1 = np.random.randn(self.hiddenLayerSize) # creates a random number matrix
        self.B1 = np.abs(self.B1)
        # shape - 1,
        self.B2 = np.random.randn(self.outputLayerSize) # creates a random number matrix
        self.B2 = np.abs(self.B2)
        
        # z = wX + b
    def forward(self, X):
        # shape - 3,3
        self.Z1 = np.dot(X, self.W1) + self.B1
        # shape - 3,3
        self.A1 = sigmoid(self.Z1)
        # shape - 3,1
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        # shape - 3,1
        yhat = sigmoid(self.Z2)
        return yhat

    def backward(self, cost, m, X):
        
        self.Z2 = cost
        self.W2 = (1/m)*(np.dot(self.A1.T, self.Z2))
        self.B2 = (1/m)*(np.sum(self.Z2, axis = 1, keepdims = True))
        
        self.Z1 = np.dot(self.W2.T, self.Z2)*self.A1
        self.W1 = (1/m)*(np.dot(X.T, self.Z1))
        self.B1 = (1/m)*(np.sum(self.Z1, axis = 1, keepdims = True))
        
        return self.W1, self.B1, self.W2, self.B2

# Activation function    
def sigmoid(z):
    return 1/(1+(np.exp(-z)))

X = np.array(([3,5],[5,1],[10,2]), dtype = float)
Y = np.array(([75],[82],[93]), dtype = float)

X = X/np.amax(X, axis = 0) # axis = 0 - row, axis = 1 - column. We are normalising here to bring it in same scale.
Y = Y/100 # 100 since it is the maximum marks

X.shape # shape = 3,2
Y.shape # shape = 3,1

m = Y.shape[0] # number of examples
m

NN = Neural_Network()
NN.__self__()    

# before backward propogation
NN.W1
NN.B1

NN.W2
NN.B2

yhat = NN.forward(X)
yhat

cost = Y - yhat
cost
cost.shape

(new_W1, new_B1, new_W2, new_B2) = NN.backward(cost, m, X)

# After backward propogation
new_W1 
new_B1

new_W2
new_B2

#
#
#NN.W1
#Out[272]: 
#array([[ 0.68159452, -0.80340966, -0.68954978],
#       [-0.4555325 ,  0.01747916, -0.35399391]])
#
#NN.B1
#Out[273]: array([ 0.62523145, -1.60205766, -1.10438334])
#
#NN.W2
#Out[274]: 
#array([[-1.37495129],
#       [-0.6436184 ],
#       [-2.22340315]])
#
#NN.B2
#Out[275]: array([ 0.05216508])