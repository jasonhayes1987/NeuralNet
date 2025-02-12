
# imports
import numpy as np
import cupy as cp


class Metric():
    # base class for metrics
    def __init__(self, device='CPU'):
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp
    
    def calculate(self, targets, output):
        pass

class Accuracy(Metric):
    """
    Used to measure the accuracy of logistic regression models
    """
    
    def __init__(self, device='CPU'):
        super().__init__(device)
        self.name = 'Accuracy'

    def calculate(self, targets, output):
        return self._xp.mean(self._xp.argmax(output, axis=1) == targets.flatten())
    
class Precision_Accuracy(Metric):
    """
    Used to measure the accuracy of a regression model
    """
    
    def __init__(self, strictness = 250, device='CPU'):
        super().__init__(device)
        self.name = 'Precision_Accuracy'
        self.strictness = strictness
        
    def calculate(self, targets, output):
        return self._xp.mean(self._xp.absolute(output - targets) < (self._xp.std(targets)/self.strictness))