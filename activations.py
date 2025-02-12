
import numpy as np
import cupy as cp
from cupyx.scipy.special import erf as cp_erf
from scipy.stats import norm
from scipy.special import erf
from layer import Activation, Layer

# Stores all the activation functions that can be passed to the activation layer

class Tanh(Activation):

    def __init__(self, input_dims=None, device='CPU'):
        self.name = 'Tanh'
        super().__init__(self, self.name, input_dims, device)
    
    def compute(self):
    # computes and returns the hyperbolic tangent of the input
#         print('Tanh compute fired...')
#         print(f'---input shape: {x.shape}')
        output = self._xp.tanh(self.input)
#         print(f'---output shape: {x.shape}')
#         print('')
        return output

    def derivative(self):
    # computes and returns the derivative of the hyperbolic tangent of the input
#         print('Tanh derivative fired...')
#         print(f'---input shape: {x.shape}')
        output = 1 - self._xp.power(self._xp.tanh(self.input),2)
#         print(f'---output shape: {output.shape}')
#         print('')
        return output
        
class Softmax(Layer):
    
    def __init__(self, input_dims=None, device='CPU'):
        self.name = 'Softmax'
        super().__init__(device)
        self._input_dims = input_dims
        self._output_dims = input_dims
    
    def forward(self, x, is_training=True):
        # print(f'Softmax input:{x}')
        # print(f'---input shape: {x.shape}')
        self.input = x
        x_max = self.input.max(axis=1, keepdims=True)
        exp = self._xp.exp(self.input - x_max)
        self.output = exp / self._xp.sum(exp, axis=1, keepdims=True)
        # print(f'softmax output: {self.output}')
        # print('')
        return self.output
    
    def backward(self, output_gradient, optimizer):
#         print('Softmax.backward fired...')
#         print(f'---output gradient shape: {output_gradient.shape}')
        self.input_gradient = self._xp.empty_like(output_gradient)
        for i, (output, output_grad) in enumerate(zip(self.output, output_gradient)):
            output = output.reshape(-1,1)
            jacobian = self._xp.diagflat(output) - self._xp.matmul(output, output.T)
            input_gradient = self._xp.dot(jacobian, output_grad)
            self.input_gradient[i] = input_gradient
#         print(f'---input gradient shape: {self.input_gradient.shape}')
#         print(self.input_gradient)
        return self.input_gradient
    
class Relu(Activation):

    def __init__(self, input_dims=None, device='CPU'):
        self.name = 'Relu'
        super().__init__(self.name, input_dims, device)
    
    def compute(self):
        output = self._xp.maximum(self.input, self._xp.zeros_like(self.input))
#         print('---relu activation')
#         print(output)
#         print('')
        return output
        
    def derivative(self):
        output = (self.input > 0).astype(self._xp.float32)
#         print('relu derivative')
#         print(output)
        return output

class Elu(Activation):

    def __init__(self, alpha=1, input_dims=None, device='CPU'):
        self.name = 'Elu'
        super().__init__(self.name, input_dims, device)
        self.alpha = alpha
    
    def compute(self):
        # shape = self.input.shape
        # x = self.input.flatten()
        # output = [i if i>0 else self.alpha*(self._xp.exp(i)-1) for i in x]
        # return self._xp.array(output).reshape(shape)
        return self._xp.where(self.input > 0, 
                              self.input, 
                              self.alpha * (self._xp.exp(self.input) - 1))
    
    def derivative(self):
        # shape = self.input.shape
        # x = self.input.flatten()
        # output = [1 if i>0 else self.alpha*self._xp.exp(i) for i in x]
        # return self._xp.array(output).reshape(shape)
        return self._xp.where(self.input > 0, 
                              1, 
                              self.alpha * self._xp.exp(self.input))
    
class Gelu(Activation):

    def __init__(self, input_dims=None, device='CPU'):
        self.name = 'Gelu'
        super().__init__(self.name, input_dims, device)
    
    def compute(self):
        # shape = self.input.shape
        # x = self.input.flatten()
        # output = x * (0.5 * (1.0 + erf(x / np.sqrt(2.0))))
        # return self._xp.array(output).reshape(shape)
        if self._device == 'GPU':
            return self.input * (0.5 * (1.0 + cp_erf(self.input / self._xp.sqrt(2.0))))
        else:
            return self.input * (0.5 * (1.0 + erf(self.input / self._xp.sqrt(2.0))))
    
    def derivative(self):
        # shape = self.input.shape
        # x = self.input.flatten()
        # output = ((1.0 + erf(x / np.sqrt(2.0))) / 2.0) + np.multiply(x, norm.pdf(x))
        # return self._xp.array(output).reshape(shape)

        pdf = 1.0 / self._xp.sqrt(2 * self._xp.pi) * self._xp.exp(-0.5 * self.input**2)
        if self._device == 'GPU':
            erf_term = cp_erf(self.input / self._xp.sqrt(2.0))
        else:
            erf_term = erf(self.input / self._xp.sqrt(2.0))

        return ((1.0 + erf_term) / 2.0) + self.input * pdf

class Sigmoid(Activation):

    def __init__(self, input_dims=None, device='CPU'):
        self.name = 'Sigmoid'
        super().__init__(self.name, input_dims, device)
    def compute(self):
        output = 1/(1+self._xp.exp(-self.input))
        return output
    
    def derivative(self):
        output = self.output * (1 - self.output)
        return output

