# -*- coding: utf-8 -*-
"""losses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Llm0-_g4Hs2Y5Sfi5JeTEmrxxpS76Y5b
"""

import numpy as np

# stores all loss errors to be passed to the network

class Loss():
  # base class for losses

    def calculate(self, targets, output):
#         print('loss.calculate fired...')
        
        loss = self.forward(targets, output)
        mean_loss = np.mean(loss)
        return mean_loss
    
    def regularization_loss(self, Layer):
        """
        Returns total loss due to Regularization (both L1 & L2)
        """
        
        # initialize regularization loss to 0
        L1_loss = 0
        L2_loss = 0
        
        # if layers L1 parameter > 0, calculate layers L1 loss
        if Layer.L1_regularizer > 0:
            L1_loss += Layer.L1_regularizer * np.sum(np.abs(Layer.weights))
            L1_loss += Layer.L1_regularizer * np.sum(np.abs(Layer.bias))
            
        if Layer.L2_regularizer > 0:
            L2_loss += Layer.L2_regularizer * np.sum(Layer.weights**2)
            L2_loss += Layer.L2_regularizer * np.sum(Layer.bias**2)
            
        return L1_loss, L2_loss


class Mean_Squared_Error(Loss):
    
    def __init__(self):
        self.name = 'MSE'

    def forward(self, targets, output):
        output = np.power(output - targets, 2)
        return output

    def backward(self, targets, predictions):
        self.input_gradient = 2*(predictions - targets)/targets.size
        return self.input_gradient


class Sparse_Categorical_Cross_Entropy(Loss):
    
    def __init__(self):
        self.name = 'Sparse CXE'
    
    def forward(self, targets, output):
        targets = targets.flatten()
        cce = -np.log(output[range(len(output)), targets])      
        return cce
    
    def backward(self, targets, output):
        targets = targets.flatten()
        samples = len(output)
        labels = len(output[0])
        targets = np.eye(labels)[targets]
        self.input_gradient = -targets / output
        self.input_gradient = self.input_gradient / samples
        return self.input_gradient

