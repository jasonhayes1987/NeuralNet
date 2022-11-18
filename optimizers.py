import numpy as np

class Optimizer():
    # base class for optimizers
    
    def pre_update_params():
        pass
    
    def update_params():
        pass
    
    def post_update_params():
        pass
 

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimization
    """
    
    def __init__(self, learning_rate=.01, decay=0, momentum=0):
        self.name = 'SGD'
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        
    def pre_update_params(self, epoch):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+(self.decay*epoch)))
    
    def update_params(self, Layer):
        """
        Returns amount to update weights and bias of layer by
        
        Returns: weight_update, bias_update
        """
        # if momentum is being used
        if self.momentum:
            # check if layer doesn't have momentum gradients attributes initialized for weights and bias and if not create them
            if not hasattr(Layer, 'weights_momentum'):
                # create weights_momentum and bias_momentum attributes for layer
                Layer.weight_momentum = np.zeros_like(Layer.weights)
                Layer.bias_momentum = np.zeros_like(Layer.bias)
            # return weight and bias updates using momentum
            return (self.momentum * Layer.weight_momentum) - (Layer.weights - (self.current_learning_rate * Layer.weight_gradient)), (self.momentum * Layer.bias_momentum) - (Layer.bias - (self.current_learning_rate * Layer.bias_gradient))
        # if momentum not being used
        else:
            # return updates to weight and bias using standard SGD (without momentum)
            return Layer.weights - (self.current_learning_rate * Layer.weight_gradient), Layer.bias - (self.current_learning_rate * Layer.bias_gradient)

        
class Adagrad(Optimizer):
    """
    Adaptive Gradient Optimization
    """
    
    def __init__(self, learning_rate=0.1, decay=0, epsilon=1e-7):
        self.name = 'Adagrad'
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        
    def pre_update_params(self, epoch):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+(self.decay*epoch)))
            
    def update_params(self, Layer):
        # initialize cache attribute if hasn't already been
        if not hasattr(Layer, 'weight_cache'):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.bias)
        
        # update cache values    
        Layer.weight_cache += Layer.weight_gradient**2
        Layer.bias_cache += Layer.bias_gradient**2
        
        # return updates to weight and bias parameters
        return Layer.weights - (self.current_learning_rate * Layer.weight_gradient / (np.sqrt(Layer.weight_cache) + self.epsilon)), Layer.bias - (self.current_learning_rate * Layer.bias_gradient / (np.sqrt(Layer.bias_cache) + self.epsilon))
    

class RMSprop(Optimizer):
    """
    Root Mean Square Propogation Optimization
    """
    
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, cache_decay=0.999):
        self.name = 'RMSprop'
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache_decay = cache_decay
        
    def pre_update_params(self, epoch):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+(self.decay*epoch)))
            
    def update_params(self, Layer):
        # initialize cache attribute if hasn't already been
        if not hasattr(Layer, 'weight_cache'):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.bias)
            
        # update cache values
        Layer.weight_cache = (self.cache_decay * Layer.weight_cache) + ((1 - self.cache_decay) * Layer.weight_gradient**2)
        Layer.bias_cache = (self.cache_decay * Layer.bias_cache) + ((1 - self.cache_decay) * Layer.bias_gradient**2)
        
        # return updates to weight and bias parameters
        return Layer.weights - (self.current_learning_rate * Layer.weight_gradient / (np.sqrt(Layer.weight_cache) + self.epsilon)), Layer.bias - (self.current_learning_rate * Layer.bias_gradient / (np.sqrt(Layer.bias_cache) + self.epsilon)) 
    

class Adam(Optimizer):
    """
    Adaptive Momentum Optimizer
    """
    
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, momentum=0.9, cache_decay=0.999):
        self.name = 'Adam'
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.momentum = momentum
        self.cache_decay = cache_decay
        self.iteration = 0
        
    def pre_update_params(self, epoch):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+(self.decay*epoch)))
            
    def update_params(self, Layer):
        # initialize cache and momentum attributes if haven't been
        if not hasattr(Layer, 'weight_cache'):
            Layer.weight_momentum = np.zeros_like(Layer.weights)
            Layer.bias_momentum = np.zeros_like(Layer.bias)
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.bias)
            
        # update momentums
        Layer.weight_momentum = (self.momentum * Layer.weight_momentum) + ((1-self.momentum) * Layer.weight_gradient)
        Layer.bias_momentum = (self.momentum * Layer.bias_momentum) + ((1-self.momentum) * Layer.bias_gradient)
        
        # correct momentum for bias
        corrected_weight_momentum = Layer.weight_momentum / (1 - self.momentum**(self.iteration + 1))
        corrected_bias_momentum = Layer.bias_momentum / (1 - self.momentum**(self.iteration + 1))
        
        # update caches
        Layer.weight_cache = (self.cache_decay * Layer.weight_cache) + ((1 - self.cache_decay) * Layer.weight_gradient**2)
        Layer.bias_cache = (self.cache_decay * Layer.bias_cache) + ((1 - self.cache_decay) * Layer.bias_gradient**2)
        
        # correct caches for bias
        corrected_weight_cache = Layer.weight_cache / (1 - self.cache_decay**(self.iteration + 1))
        corrected_bias_cache = Layer.bias_cache / (1 - self.cache_decay**(self.iteration + 1))
        
        # return updated weight and bias parameters
        return Layer.weights - self.current_learning_rate * corrected_weight_momentum / (np.sqrt(corrected_weight_cache) + self.epsilon), Layer.bias - self.learning_rate * corrected_bias_momentum / (np.sqrt(corrected_bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration += 1