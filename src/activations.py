
import numpy as np
import cupy as cp
from cupyx.scipy.special import erf as cp_erf
from scipy.stats import norm
from scipy.special import erf
from src.layer import Activation, Layer

# Stores all the activation functions that can be passed to the activation layer

class Tanh(Activation):
    """
    Tanh activation function.

    Attributes:
        name (str): Name of the activation function.
        input_dims: Dimensions of the input.
        device (str): Device type ('CPU' or 'GPU').
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize Tanh activation.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Tanh'
        super().__init__(self.name, input_dims, device)
    
    def compute(self):
        """
        Compute the hyperbolic tangent of the input.

        Returns:
            Output after applying tanh activation.
        """
        output = self._xp.tanh(self.input)
        return output

    def derivative(self):
        """
        Compute the derivative of the hyperbolic tangent of the input.

        Returns:
            Derivative of tanh activation.
        """
        output = 1 - self._xp.power(self._xp.tanh(self.input),2)
        return output
        
class Softmax(Layer):
    """
    Softmax activation layer.

    Attributes:
        name (str): Name of the layer.
        input_dims: Dimensions of the input.
        device (str): Device type ('CPU' or 'GPU').
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize Softmax activation layer.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Softmax'
        super().__init__(device)
        self._input_dims = input_dims
        self._output_dims = input_dims
    
    def forward(self, x, is_training=True):
        """
        Perform the forward pass using softmax activation.

        Args:
            x: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Softmax probabilities.
        """
        self.input = x
        x_max = self.input.max(axis=1, keepdims=True)
        exp = self._xp.exp(self.input - x_max)
        self.output = exp / self._xp.sum(exp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, optimizer=None):
        """
        Perform the backward pass for softmax activation.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.input_gradient = self._xp.empty_like(output_gradient)
        for i, (output, output_grad) in enumerate(zip(self.output, output_gradient)):
            output = output.reshape(-1,1)
            jacobian = self._xp.diagflat(output) - self._xp.matmul(output, output.T)
            input_gradient = self._xp.dot(jacobian, output_grad)
            self.input_gradient[i] = input_gradient
        return self.input_gradient
    
class Relu(Activation):
    """
    ReLU activation function.
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize ReLU activation.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Relu'
        super().__init__(self.name, input_dims, device)
    
    def compute(self):
        """
        Compute the ReLU activation.

        Returns:
            Output after applying ReLU activation.
        """
        output = self._xp.maximum(self.input, self._xp.zeros_like(self.input))
        return output
        
    def derivative(self):
        """
        Compute the derivative of the ReLU activation.

        Returns:
            Derivative of ReLU activation.
        """
        output = (self.input > 0).astype(self._xp.float32)
        return output

class Elu(Activation):
    """
    ELU activation function.
    """
    def __init__(self, alpha=1, input_dims=None, device='CPU'):
        """
        Initialize ELU activation.

        Args:
            alpha (float): Alpha parameter for ELU.
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Elu'
        super().__init__(self.name, input_dims, device)
        self.alpha = alpha
    
    def compute(self):
        """
        Compute the ELU activation.

        Returns:
            Output after applying ELU activation.
        """
        return self._xp.where(self.input > 0, 
                              self.input, 
                              self.alpha * (self._xp.exp(self.input) - 1))
    
    def derivative(self):
        """
        Compute the derivative of the ELU activation.

        Returns:
            Derivative of ELU activation.
        """
        return self._xp.where(self.input > 0, 
                              1, 
                              self.alpha * self._xp.exp(self.input))
    
class Gelu(Activation):
    """
    GELU activation function.
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize GELU activation.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Gelu'
        super().__init__(self.name, input_dims, device)
    
    def compute(self):
        """
        Compute the GELU activation.

        Returns:
            Output after applying GELU activation.
        """
        if self._device == 'GPU':
            return self.input * (0.5 * (1.0 + cp_erf(self.input / self._xp.sqrt(2.0))))
        else:
            return self.input * (0.5 * (1.0 + erf(self.input / self._xp.sqrt(2.0))))
    
    def derivative(self):
        """
        Compute the derivative of the GELU activation.

        Returns:
            Derivative of GELU activation.
        """
        pdf = 1.0 / self._xp.sqrt(2 * self._xp.pi) * self._xp.exp(-0.5 * self.input**2)
        if self._device == 'GPU':
            erf_term = cp_erf(self.input / self._xp.sqrt(2.0))
        else:
            erf_term = erf(self.input / self._xp.sqrt(2.0))

        return ((1.0 + erf_term) / 2.0) + self.input * pdf

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize Sigmoid activation.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        self.name = 'Sigmoid'
        super().__init__(self.name, input_dims, device)

    def compute(self):
        """
        Compute the Sigmoid activation.

        Returns:
            Output after applying Sigmoid activation.
        """
        output = 1/(1+self._xp.exp(-self.input))
        return output
    
    def derivative(self):
        """
        Compute the derivative of the Sigmoid activation.

        Returns:
            Derivative of Sigmoid activation.
        """
        output = self.output * (1 - self.output)
        return output

