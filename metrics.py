#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as np


class Metric():
    # base class for metrics
    
    def calculate(self, targets, output):
        pass

class Accuracy(Metric):
    """
    Used to measure the accuracy of logistic regression models
    """
    
    def __init__(self):
        self.name = 'Accuracy'

    def calculate(self, targets, output):
        return np.mean(np.argmax(output, axis=1) == targets.flatten())
    
class Precision_Accuracy(Metric):
    """
    Used to measure the accuracy of a regression model
    """
    
    def __init__(self, strictness = 250):
        self.name = 'Precision_Accuracy'
        self.strictness = strictness
        
    def calculate(self, targets, output):
        return np.mean(np.absolute(output - targets) < (np.std(targets)/self.strictness))