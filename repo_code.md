# Project Codebase


## File: .devcontainer/devcontainer.json
```json
{
    "name": "NeuralNet CUDA",
    "build": {
      "context": "..",
      "dockerfile": "Dockerfile",
      "args": {
        "USERNAME": "vscode",
        "BUILDKIT_INLINE_CACHE": "0",
        "CUDA_VERSION": "12.6.2",
        "CLANG_VERSION": ""
      }
    },
    "runArgs": [
      "--gpus", "all",
      "--shm-size", "2g",
      "--ipc", "host"
    ],
    "mounts": [
      "source=E:/Documents/Programming/Projects/NeuralNet,target=/workspace/NeuralNet,type=bind,consistency=cached"
    ],
    "postCreateCommand": [
      "apt-get update"
    ],
    "customizations": {
      "vscode": {
        "settings": {
          "python.pythonPath": "/opt/conda/envs/neuralnet-env/bin/python",
          "editor.tabSize": 4,
          "editor.insertSpaces": true,
          "editor.codeActionsOnSave": {
            "source.fixAll": true
          },
          "editor.minimap.enabled": true,
          "editor.renderWhitespace": "all",
          "editor.quickSuggestions": {
            "strings": true
          },
          "editor.suggest.localityBonus": true,
          "editor.suggestSelection": "first"
        },
        "extensions": [
          "streetsidesoftware.code-spell-checker",
          "ms-python.python",
          "ms-toolsai.jupyter",
          "ms-vscode.cpptools",
          "eamodio.gitlens",
          "njpwerner.autodocstring",
          "visualstudioexptteam.vscodeintellicode",
          "ms-python.pylint"
        ]
      }
    }
  }
  
```



## File: .pytest_cache/README.md
```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```



## File: activations.py
```py

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


```



## File: generate_markdown.sh
```sh
#!/bin/bash

REPO_PATH="."  # Change this if your repo is in another directory

echo "# Project Codebase" > repo_code.md

for f in $(find $REPO_PATH -type f -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.sh"); do
    echo -e "\n\n## File: ${f#$REPO_PATH/}\n\`\`\`${f##*.}" >> repo_code.md
    cat "$f" >> repo_code.md
    echo -e "\n\`\`\`\n" >> repo_code.md
done

echo "Markdown file generated: repo_code.md"

```



## File: layer.py
```py
# -*- coding: utf-8 -*-
"""layer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N3Cyi7maf8zlqt5IT1dZ5d3lwVIshFlP
"""

import numpy as np
from scipy import signal
import cupy as cp
from cupyx.scipy import signal as c_signal
from utils import im2col, col2im

# base layer class
class Layer:
    def __init__(self, device='CPU'):
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp
        self.input = None
        self.output = None
        self._input_dims = None
        self._output_dims = None

    def forward(self, input, is_training=True):
      # computes output 'Y' of layer given input 'X'
      raise NotImplementedError("Must override 'forward' by instantiating child of layer class (Activation, Dense, Convolutional, etc...")

    def backward(self, output_gradient, learning_rate):
      # computes derivative of input 'X' of layer given output error 'dE/dY'
      raise NotImplementedError("Must override 'backward' by instantiating child of layer class (Activation, Dense, Convolutional, etc...")

# class for Activation layers
class Activation(Layer):
    def __init__(self, activation, input_dims=None, device='CPU'):
        super().__init__(device)
        self.activation = activation
        self.name = 'Activation'
        self._input_dims = input_dims
        self._output_dims = input_dims

    def compute(self):
        raise NotImplementedError("Must override 'compute' by instantiating child of Activation class (Relu, Tanh, Sigmoid, etc...")

    def derivative(self):
        raise NotImplementedError("Must override 'derivative' by instantiating child of Activation class (Relu, Tanh, Sigmoid, etc...")

    def forward(self, input, is_training=True):
        self.input = input
        self.output = self.compute()
        return self.output

    def backward(self, output_gradient, optimizer):
        output = self._xp.multiply(output_gradient, self.derivative())
        return output

class Dense(Layer):
    def __init__(self, output_size, L1_regularizer=0, L2_regularizer=0, input_dims=None, device='CPU', init='Default'):
        super().__init__(device)
        self.name = 'Dense'
        self._input_dims = input_dims
        self._output_dims = output_size
        
        if init == 'Xavier':
            fan_in = input_dims
            fan_out = output_size
            limit = self._xp.sqrt(6.0 / (fan_in + fan_out))
            self.weights = self._xp.random.uniform(-limit, limit, (self._input_dims, self._output_dims))
        elif init == 'He':
            fan_in = input_dims
            self.weights = self._xp.random.randn(self._input_dims, self._output_dims) * self._xp.sqrt(2.0 / fan_in)
        else:
            self.weights = self._xp.random.randn(self._input_dims, self._output_dims)
            
        self.bias = self._xp.random.randn(output_size)
        self.L1_regularizer = L1_regularizer
        self.L2_regularizer = L2_regularizer
    

    def forward(self, input, is_training=True):
        self.input = input
        self.output = self._xp.dot(self.input, self.weights) + self.bias
        return self.output


    def backward(self, output_gradient, optimizer):
        # calculates gradients of weights 'dE/dW', and input 'dE/dX' with respect to error
        self.output_gradient = output_gradient
        self.weight_gradient = self._xp.dot(self.input.T, output_gradient)
        self.bias_gradient = self._xp.sum(self.output_gradient, axis=0)
    
        # add in gradients of regularization wrt weights and bias if regularizers > 0
        # for L1 Regularization
        if self.L1_regularizer >= 0:
            # update weight gradient
            L1_weight_gradient = self._xp.ones_like(self.weights)
            L1_weight_gradient[self.weights<0] = -1
            self.weight_gradient += self.L1_regularizer * L1_weight_gradient

            # update bias gradient
            L1_bias_gradient = self._xp.ones_like(self.bias)
            L1_bias_gradient[self.bias<0] = -1
            self.bias_gradient += self.L1_regularizer * L1_bias_gradient

        # for L2 Regularization
        if self.L2_regularizer >= 0:
            # update weight gradient
            self.weight_gradient += 2 * (self.L2_regularizer * self.weights)

            # update bias gradient
            self.bias_gradient += 2 * (self.L2_regularizer * self.bias)

        input_gradient = self._xp.dot(output_gradient, self.weights.T)

        # update weights and bias
        self.weights, self.bias = optimizer.update_params(self)

        # return input_gradient to be used as output_gradient of previous layer
        return input_gradient


    def get_parameters(self):
        return self.weights, self.bias
    
    def set_parameters(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    @staticmethod
    def set_input_size(previous_layer):
        """
        Sets the input size of the layer using previous layers output size
        """
        # check to make sure previous layer shape is 2D
        if len(previous_layer.output.shape == 2):
            return previous_layer.output.shape[1] # return second dimension of previous layer output
        else:
            # Throw ERROR that dense required 2D input data
            print(f'Dense layer requires 2D data as input; passed {len(previous_layer.output.shape)}D')
        
class Dropout(Layer):
    
    def __init__(self, dropout_rate, input_dims=None, output_size=None, device='CPU'):
        super().__init__(device)
        self.name = 'Dropout'
        self.keep_rate = 1 - dropout_rate
        self._input_dims = input_dims
        self._output_dims = output_size
        
    def forward(self, input, is_training=True):
        self.input = input
        
        # skip dropout and just output previous layers output if model is being evaluated
        if not is_training:
            self.output = self.input.copy()
            return self.output
        
        self.mask = self._xp.random.binomial(1, self.keep_rate, input.shape) / self.keep_rate
        self.output = input * self.mask
        return self.output
    
    def backward(self, output_gradient, optimizer):
        self.output_gradient = output_gradient
        return self.output_gradient * self.mask
    
class Convolutional(Layer):
    def __init__(self, kernel_size, depth, mode, input_dims=None, device='CPU', init='default'):
        super().__init__(device)
        self.name = 'Convolutional'
        _input_depth, _input_height, _input_width = input_dims
        self._depth = depth
        self._input_dims = input_dims
        self._input_depth = _input_depth
        self._kernels_shape = (self._depth, self._input_depth, kernel_size, kernel_size)
        
        # Compute fan-in and fan-out for the convolution
        fan_in = _input_depth * kernel_size * kernel_size
        fan_out = depth * kernel_size * kernel_size
        
        if init == 'Xavier':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights = self._xp.random.uniform(-limit, limit, self._kernels_shape)
        elif init == 'He':
            self.weights = self._xp.random.randn(*self._kernels_shape) * np.sqrt(2.0 / fan_in)
        else:
            self.weights = self._xp.random.randn(*self._kernels_shape)
        
        self.mode = mode
        if self.mode == 'valid':
            self._output_dims = (self._depth, _input_height - kernel_size + 1, _input_width - kernel_size + 1)
        elif self.mode == 'same':
            self._output_dims = (self._depth, _input_height, _input_width)
            
        self.bias = self._xp.random.randn(self._depth)

    def compute_asymmetric_padding(self):
        kernel_size = self._kernels_shape[2]
        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return (pad_before, pad_after)

    def forward(self, input, is_training=True):
        self.input = input
        if self.mode == 'same':
            pad_tuple = self.compute_asymmetric_padding()
            self.pad = (pad_tuple, pad_tuple)  # For both height and width
        else:
            self.pad = 0
        self.output = self.conv_forward_im2col(self.input, self.weights, self.bias, stride=1, pad=self.pad)
        return self.output

    def backward(self, output_gradient, optimizer):
        self.output_gradient = output_gradient
        self.weight_gradient, input_gradient = self.conv_backward_im2col(
            self.output_gradient, self.input, self.weights, stride=1, pad=self.pad
        )
        # Compute bias gradient by summing over the batch and spatial dimensions.
        self.bias_gradient = self._xp.sum(self.output_gradient, axis=(0, 2, 3))

        self.weights, self.bias = optimizer.update_params(self)
        return input_gradient

    def conv_forward_im2col(self, x, W, b, stride=1, pad=0):
        N, C, H, W_in = x.shape
        F, _, filter_h, filter_w = W.shape
        if isinstance(pad, int):
            total_pad_h = total_pad_w = 2 * pad
        else:
            pad_h, pad_w = pad
            total_pad_h = sum(pad_h)
            total_pad_w = sum(pad_w)
        out_h = (H + total_pad_h - filter_h) // stride + 1
        out_w = (W_in + total_pad_w - filter_w) // stride + 1
        col = im2col(x, filter_h, filter_w, stride, pad, xp=self._xp)
        W_col = W.reshape(F, -1)
        out = self._xp.dot(col, W_col.T) + b.reshape(1, -1)
        out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
        return out

    def conv_backward_im2col(self, dout, x, W, stride=1, pad=0):
        N, F, out_h, out_w = dout.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        _, C, H, W_in = x.shape
        filter_h, filter_w = W.shape[2], W.shape[3]
        col = im2col(x, filter_h, filter_w, stride, pad, xp=self._xp)
        dW_col = self._xp.dot(dout_reshaped.T, col)
        dW = dW_col.reshape(W.shape)
        W_col = W.reshape(F, -1)
        dcol = self._xp.dot(dout_reshaped, W_col)
        dx = col2im(dcol, x.shape, filter_h, filter_w, stride, pad, xp=self._xp)
        return dW, dx
    
    def get_parameters(self):
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        if weights.shape == self.weights.shape:
            self.weights = weights
        else:
            print("weights don't match kernel shape")

        if bias.shape == self.bias.shape:
            self.bias = bias
        else:
            print("bias doesn't match bias shape")
    
class Flatten(Layer):
    """
    Class used for flattening the input to an NxD dimension going forward, and restoring data
    to it's previous shape when going backward (back-propogating)
    """
    
    def __init__(self, input_dims=None, device='CPU'):
        super().__init__(device)
        self.name = 'Flatten'
        self._input_dims = input_dims
        self._output_dims = 1
        for i in self._input_dims:
            self._output_dims*=i
        
    def forward(self, input, is_training=True):
        self.input_shape = input.shape
        self.output_shape = (self.input_shape[0], self._output_dims)
        return self._xp.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, optimizer):
        return self._xp.reshape(output_gradient, self.input_shape)
    
class Pool(Layer):
    
    def __init__(self, pool_size=2, stride=2, method='max', input_dims=None, device='CPU'):
        super().__init__(device)
        self.name = 'Pool'
        self.pool_size = pool_size
        self.stride = stride
        self.method = method
        self._input_dims = input_dims
        _out_height = 1 + (self._input_dims[1] - self.pool_size) // self.stride
        _out_width = 1 + (self._input_dims[2] - self.pool_size) // self.stride
        self._output_dims = (self._input_dims[0], _out_height, _out_width)
        
    def forward(self, input, is_training=True):
        self.input = input
        # Recompute output dimensions from the actual input shape
        N, C, H, W = self.input.shape
        out_height = 1 + (H - self.pool_size) // self.stride
        out_width  = 1 + (W - self.pool_size) // self.stride
        _out_shape = (N, C, out_height, out_width)
        self.output = self._xp.zeros(_out_shape)
        
        for height in range(out_height):
            for width in range(out_width):
                height_start = height * self.stride
                height_end = height_start + self.pool_size
                width_start = width * self.stride
                width_end = width_start + self.pool_size
                
                if self.method == 'max':
                    self.output[:, :, height, width] = self._xp.max(input[:, :, height_start:height_end, width_start:width_end], axis=(2,3))
                elif self.method == 'average':
                    self.output[:, :, height, width] = self._xp.mean(input[:, :, height_start:height_end, width_start:width_end], axis=(2,3))
        
        return self.output
    
    def backward(self, output_gradient, optimizer):
        self.output_gradient = output_gradient
        N, C, H, W = self.input.shape
        out_height = self.output.shape[2]
        out_width  = self.output.shape[3]
        input_gradient = self._xp.zeros(self.input.shape)
        
        for n in range(N):
            for d in range(C):
                for height in range(out_height):
                    for width in range(out_width):
                        height_start = height * self.stride
                        height_end = height_start + self.pool_size
                        width_start = width * self.stride
                        width_end = width_start + self.pool_size
                        
                        # Extract the pooling region
                        region = self.input[n, d, height_start:height_end, width_start:width_end]
                        # Compute the maximum value in the region
                        max_val = self._xp.max(region)
                        # Get the indices where the region equals the maximum
                        height_idx, width_idx = self._xp.where(region == max_val)
                        
                        if height_idx.size == 0 or width_idx.size == 0:
                            # If this window is empty for some reason, skip it
                            continue
                        
                        # Use the first index (this is common in max pooling backward)
                        input_gradient[n, d, height_start:height_end, width_start:width_end][height_idx[0], width_idx[0]] = self.output_gradient[n, d, height, width]
        
        return input_gradient


class BatchNormalization(Layer):
    def __init__(self, momentum=0.9, epsilon=0.001, input_dims=None, device='CPU'):
        super().__init__(device)
        self._input_dims = input_dims
        self._output_dims = input_dims
        self.num_features = input_dims[0] if isinstance(input_dims, tuple) else input_dims
        self.momentum = momentum
        self.epsilon = epsilon
        # Initialize gamma (scale) as weight and beta (shift) as bias
        # (Initialized with weight and bias names for standardized names
        # to work with optimizer class)
        self.weights = self._xp.ones(self.num_features)
        self.bias = self._xp.zeros(self.num_features)
        # Running mean and variance for inference
        self.running_mean = self._xp.zeros(self.num_features)
        self.running_var = self._xp.ones(self.num_features)
        self.batch_mean = None
        self.batch_var = None
        self.name = 'BatchNorm'

    def forward(self, x, is_training=True):
        # x is assumed to be shape (N, C, H, W) for conv layers,
        # or (N, D) for dense layers.
        if x.ndim == 4:
            # For convolutional layers, compute mean and variance per channel over N, H, W
            if is_training:
                mean = self._xp.mean(x, axis=(0, 2, 3))
                var = self._xp.var(x, axis=(0, 2, 3))
                # Save mean and var as attributes to be used in backwards pass
                self.batch_mean = mean
                self.batch_var = var
                # Update running stats
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            else:
                mean = self.running_mean
                var = self.running_var
            # Reshape for broadcasting: (1, C, 1, 1)
            mean = mean.reshape((1, -1, 1, 1))
            var = var.reshape((1, -1, 1, 1))
            gamma = self.weights.reshape((1, -1, 1, 1))
            beta = self.bias.reshape((1, -1, 1, 1))
        else:
            # For dense layers (shape: (N, D))
            if is_training:
                mean = self._xp.mean(x, axis=0)
                var = self._xp.var(x, axis=0)
                # Save mean and var as attributes to be used in backwards pass
                self.batch_mean = mean
                self.batch_var = var
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            else:
                mean = self.running_mean
                var = self.running_var
            gamma = self.weights
            beta = self.bias

        # Normalize and then scale and shift
        self.normalized = (x - mean) / self._xp.sqrt(var + self.epsilon)
        self.output = gamma * self.normalized + beta
        return self.output

    def backward(self, output_gradient, optimizer):
        self.output_gradient = output_gradient
        if self.output_gradient.ndim == 2:
            # Dense case: x shape (N, D)
            N, D = self.output_gradient.shape
            dgamma = self._xp.sum(self.output_gradient * self.normalized, axis=0)
            dbeta = self._xp.sum(self.output_gradient, axis=0)
            dnormalized = self.output_gradient * self.weights  # (N, D)
            std_inv = 1.0 / self._xp.sqrt(self.batch_var + self.epsilon)
            input_gradient = (1.0 / N) * std_inv * (N * dnormalized - self._xp.sum(dnormalized, axis=0)
                                        - self.normalized * self._xp.sum(dnormalized * self.normalized, axis=0))
        else:
            # Convolutional case: x shape (N, C, H, W)
            N, C, H, W = self.output_gradient.shape
            dgamma = self._xp.sum(self.output_gradient * self.normalized, axis=(0, 2, 3))
            dbeta = self._xp.sum(self.output_gradient, axis=(0, 2, 3))
            dnormalized = self.output_gradient * self.weights.reshape((1, C, 1, 1))
            std_inv = 1.0 / self._xp.sqrt(self.batch_var + self.epsilon)
            std_inv = std_inv.reshape((1, C, 1, 1))
            # Total number of elements per channel
            M = N * H * W
            input_gradient = (1.0 / M) * std_inv * (M * dnormalized - self._xp.sum(dnormalized, axis=(0,2,3), keepdims=True)
                                        - self.normalized * self._xp.sum(dnormalized * self.normalized, axis=(0,2,3), keepdims=True))
        
        self.weight_gradient = dgamma
        self.bias_gradient = dbeta
        # update weights and bias using optimizer
        self.weights, self.bias = optimizer.update_params(self)
        return input_gradient


```



## File: losses.py
```py

import numpy as np
import cupy as cp

# stores all loss errors to be passed to the network

class Loss():
    # base class for losses
    def __init__(self, device='CPU'):
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp

    def calculate(self, targets, output):        
        loss = self.forward(targets, output)
        mean_loss = self._xp.mean(loss)
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
            L1_loss += Layer.L1_regularizer * self._xp.sum(self._xp.abs(Layer.weights))
            L1_loss += Layer.L1_regularizer * self._xp.sum(self._xp.abs(Layer.bias))
            
        if Layer.L2_regularizer > 0:
            L2_loss += Layer.L2_regularizer * self._xp.sum(Layer.weights**2)
            L2_loss += Layer.L2_regularizer * self._xp.sum(Layer.bias**2)
            
        return L1_loss, L2_loss


class Mean_Squared_Error(Loss):
    
    def __init__(self, device='CPU'):
        super().__init__(device)
        self.name = 'MSE'

    def forward(self, targets, output):
        output = self._xp.power(output - targets, 2)
        return output

    def backward(self, targets, predictions):
        self.input_gradient = 2*(predictions - targets)/targets.size
        return self.input_gradient


class Sparse_Categorical_Cross_Entropy(Loss):
    
    def __init__(self, device='CPU'):
        super().__init__(device)
        self.name = 'Sparse CXE'
    
    def forward(self, targets, output):
        targets = targets.flatten().astype(int)
        # Compute cross entropy loss for each sample:
        indices = self._xp.arange(len(output))
        cce = -self._xp.log(output[indices, targets] + 1e-8)
        return cce
    
    # def backward(self, targets, output):
    #     targets = targets.flatten()
    #     samples = len(output)
    #     labels = len(output[0])
    #     one_hot = self._xp.eye(labels)[targets]
    #     self.input_gradient = (output - one_hot) / samples
    #     return self.input_gradient

    # def backward(self, targets, output):
    #     targets = targets.flatten()
    #     samples = output.shape[0]
    #     grad = self._xp.zeros_like(output)
    #     #DEBUG
    #     print(f'targets:{targets}')
    #     print(f'output:{output}')
    #     print(f'grad:{grad}')
    #     for i in range(samples):
    #         print(f'grad[i]:{grad[i]}')
    #         print(f'target[i]:{targets[i]}')
    #         grad[i, targets[i]] = -1.0 / (output[i, targets[i]] + 1e-8)
    #     grad = grad / samples
    #     self.input_gradient = grad
    #     return grad

    def backward(self, targets, output):
        targets = targets.flatten().astype(int)  # ensure indices are ints
        samples = output.shape[0]
        grad = self._xp.zeros_like(output)
        indices = self._xp.arange(samples)
        grad[indices, targets] = -1.0 / (output[indices, targets] + 1e-8)
        grad = grad / samples
        self.input_gradient = grad
        return grad


```



## File: metrics.py
```py

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
```



## File: mnist.py
```py
## IMPORTS ##
from network import Neural_Network
from losses import Sparse_Categorical_Cross_Entropy
from optimizers import Adam
from metrics import Accuracy
from utils import get_confusion_matrix, plot_confusion_matrix

from tensorflow.keras.datasets import mnist
import time

print("Starting MNIST training script...")

## Construct Network ##
print("Building network...")
classifier = Neural_Network()

device = 'GPU'
print(f"Setting device to: {device}")
loss = Sparse_Categorical_Cross_Entropy(device=device)
optimizer = Adam(learning_rate=0.001, device=device)
metrics = [Accuracy(device=device), loss]

classifier.compile_model(loss=loss, optimizer=optimizer, metrics=metrics, device=device)
print("Network compiled successfully.")

# ARCHITECTURE
print("Building network architecture...")

# Block 1
print("Adding Block 1 (Convolutional, BatchNorm, Activation, Pooling)...")
classifier.add_Convolutional(3, 32, 'same', 'He', (1, 28, 28))
classifier.add_BatchNorm()
classifier.add_Activation('relu')
classifier.add_Convolutional(3, 32, 'same', 'He')
classifier.add_BatchNorm()
classifier.add_Activation('relu')
classifier.add_Pool(2, 2)

# Uncomment additional blocks as needed
# Block 2
# print("Adding Block 2...")
# classifier.add_Convolutional(3, 64, 'same', 'He')
# classifier.add_BatchNorm()
# classifier.add_Activation('relu')
# classifier.add_Convolutional(3, 64, 'same', 'He')
# classifier.add_BatchNorm()
# classifier.add_Activation('relu')
# classifier.add_Pool(2, 2)

# Block 3
# print("Adding Block 3...")
# classifier.add_Convolutional(3, 128, 'same', 'He')
# classifier.add_BatchNorm()
# classifier.add_Activation('relu')
# classifier.add_Convolutional(3, 128, 'same', 'He')
# classifier.add_BatchNorm()
# classifier.add_Activation('relu')
# classifier.add_Pool(2, 2)

# Dense head
print("Adding Dense head...")
classifier.add_Flatten()
classifier.add_Dense(512, init='Xavier')
classifier.add_BatchNorm()
classifier.add_Activation('relu')
classifier.add_Dense(10, init='Xavier')
classifier.add_BatchNorm()
classifier.add_Activation('softmax')

print("Network architecture built successfully.")
print("Network layers:")
for i, layer in enumerate(classifier.layers):
    print(f"  Layer {i}: {layer.name}")

## IMPORT AND FORMAT DATA ##
print("Downloading MNIST dataset...")
start_time = time.time()
(x_train, y_train), (x_val, y_val) = mnist.load_data()
print(f"MNIST dataset downloaded in {time.time() - start_time:.2f} seconds.")

print("Normalizing data...")
x_train = x_train / 255.0
x_val = x_val / 255.0

# convert data to cupy arrays if device = GPU
if device == 'GPU':
    x_train = classifier._xp.asarray(x_train)
    y_train = classifier._xp.asarray(y_train)
    x_val = classifier._xp.asarray(x_val)
    y_val = classifier._xp.asarray(y_val)

print("Adding channel dimension...")
x_train = classifier._xp.expand_dims(x_train, axis=1)
x_val = classifier._xp.expand_dims(x_val, axis=1)

# Package data
data = [(x_train, y_train), (x_val, y_val)]
print("Data preprocessing complete.")
print(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")

## RUN TRAINING ##
print("Starting training...")
# classifier.train(data, epochs=1, batch_size=16)
print("Training complete.")

cm = get_confusion_matrix((x_val,y_val), classifier, image_size=(100,100))
plot_confusion_matrix(cm, y_val[0])
```



## File: network.py
```py

import numpy as np
import cupy as cp
from sklearn.utils import shuffle
import layer as Layer
import losses as Losses
import metrics as Metrics
import optimizers as Optimizers
import activations as Activations
# import utils as Utils
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from tabulate import tabulate

# class that defines and builds a neural network

class Neural_Network():

    def __init__(self):
        self.layers = []

        
    def add_Dense(self, output_size, L1_regularization=0, L2_regularization=0, init='Default', input_dims=None):
        """
        Adds a Dense layer to the network
        **Only requires input_size param if first layer in network**
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_size param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Dense(output_size=output_size, L1_regularizer=L1_regularization, L2_regularizer=L2_regularization, input_dims=input_dims, device=self._device, init=init))
                
        else:
            self.layers.append(Layer.Dense(output_size=output_size, L1_regularizer=L1_regularization, L2_regularizer=L2_regularization, input_dims=self.layers[-1]._output_dims, device=self._device, init=init))


    def add_Convolutional(self, kernel_size, depth, mode, init='Default', input_dims=None):
        """
        Adds a Convolutional layer to the network
        **Only requires input_dims param if first layer in network**
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Convolutional(kernel_size=kernel_size, depth=depth, mode=mode, input_dims=input_dims, device=self._device, init=init))
        else:
            self.layers.append(Layer.Convolutional(kernel_size=kernel_size, depth=depth, mode=mode, input_dims=self.layers[-1]._output_dims, device=self._device, init=init))
    
    
    def add_Flatten(self, input_dims=None):
        """
        Adds a Flatten layer to the network
        **Only requires input_dims param if first layer in network**
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Flatten(input_dims=input_dims, device=self._device))

        else:
            self.layers.append(Layer.Flatten(input_dims=self.layers[-1]._output_dims, device=self._device))
        
    
    def add_Dropout(self, dropout_rate, input_dims=None, output_size=None):
        """
        Adds a Dropout layer to the network
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Dropout(dropout_rate=dropout_rate, input_dims=input_dims, output_size = input_dims, device=self._device))

        else:
            self.layers.append(Layer.Dropout(dropout_rate=dropout_rate, input_dims=self.layers[-1]._output_dims, output_size = self.layers[-1]._output_dims, device=self._device))
        
        
    def add_Pool(self, pool_size=2, stride=2, method='max', input_dims=None):
        """
        Adds a Pooling layer to the network
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Pool(pool_size=pool_size, stride=stride, method=method, input_dims=input_dims, device=self._device))
                
        else:
            self.layers.append(Layer.Pool(pool_size=pool_size, stride=stride, method=method, input_dims=self.layers[-1]._output_dims, device=self._device))

    
    def add_BatchNorm(self, momentum=0.9, epsilon=1e-5, input_dims=None):
        """
        Adds a Pooling layer to the network
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.BatchNormalization(momentum=momentum, epsilon=epsilon, input_dims=input_dims, device=self._device))
                
        else:
            self.layers.append(Layer.BatchNormalization(momentum=momentum, epsilon=epsilon, input_dims=self.layers[-1]._output_dims, device=self._device))
        
        
    def add_Activation(self, activation, input_dims=None):
        """
        Adds an Activation layer to the network
        **Only requires input_dims param if first layer in network**
        INPUT:
        activation: {string} from list of activations (below)
        
        Activations:
        -tanh
        -relu
        -selu
        -gelu
        -elu
        -sigmoid
        -softmax
        """
        if activation == 'tanh':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Tanh(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Tanh(input_dims=self.layers[-1]._output_dims, device=self._device))
                
        elif activation == 'softmax':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Softmax(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Softmax(input_dims=self.layers[-1]._output_dims, device=self._device))
                
        elif activation == 'relu':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Relu(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Relu(input_dims=self.layers[-1]._output_dims, device=self._device))
                
        elif activation == 'elu':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Elu(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Elu(input_dims=self.layers[-1]._output_dims, device=self._device))
                
        elif activation == 'gelu':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Gelu(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Gelu(input_dims=self.layers[-1]._output_dims, device=self._device))
                
        elif activation == 'sigmoid':
            if len(self.layers) == 0:
                if input_dims == None:
                    #ERROR
                    print('input_dims param must be given because layer is first in network')
                else:
                    self.layers.append(Activations.Sigmoid(input_dims=input_dims, device=self._device))
                    
            else:
                self.layers.append(Activations.Sigmoid(input_dims=self.layers[-1]._output_dims, device=self._device))


    def compile_model(self, *, loss=None, optimizer=None, metrics=None, device='CPU'):
        """
        Parameters:
        loss: type(Loss); loss function to use to train network
        metrics: [type(Metric/Loss), type(Metric/Loss),...]; metrics to keep track of per epoch
        device: 'CPU'/'GPU';  sets the device the network will use for computation
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if metrics is not None:
            self.metrics = metrics
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp

       
    def train(self, data, epochs, batch_size, visualize=False):  
        """
        trains the network on the x_train data to the y_train data using x_val and y_val as validation sets

        Parameters:
        data: [(x_train, y_train),(x_val, y_val)]; array of tuples of x and y data for train and validation sets
        """

        ## Convert to cupy arrays if gpu==True
        for i in range(len(data)):
            x, y = data[i]
            if type(x) == np.ndarray:
                x = cp.asarray(x)
                data[i] = (x, data[i][1])
            if type(y) == np.ndarray:
                y = cp.asarray(y)
                data[i] = (data[i][0], y)

        self.data = data
        
        # set all layers names to be unique (self.name+index)
        for name in set([layer.name for layer in self.layers if (isinstance(layer, Layer.Layer)) and (hasattr(layer, 'name'))]):
            for index, layer in enumerate([layer for layer in self.layers if (hasattr(layer, 'name')) and (layer.name == name)]):
                layer.name = name + str(index)

        # initialize dict that will keep track of metrics per epoch
        self.metric_data = {}
        for data_type in ['Train','Validation']:    
            for metric in self.metrics:
                if isinstance(metric, Losses.Loss):
                    self.metric_data[f'{data_type} {metric.name}'] = []
                    for layer in self.layers:
                        if hasattr(layer, 'L1_regularizer'):
                            if layer.L1_regularizer > 0:
                                self.metric_data[f'{data_type} L1 Regularization'] = []
                                
                            if layer.L2_regularizer > 0:
                                self.metric_data[f'{data_type} L2 Regularization'] = []

                            if (layer.L1_regularizer > 0) or (layer.L2_regularizer > 0):
                                self.metric_data[f'{data_type} Total Loss'] = []
                                
                elif isinstance(metric, Optimizers.Optimizer):
                    self.metric_data['Learning Rate'] = []
                    
                elif  isinstance(metric, Metrics.Metric):
                    self.metric_data[f'{data_type} {metric.name}'] = []
                
                elif metric == Layer.Dense:
                    for layer in [layer for layer in self.layers if isinstance(layer, Layer.Dense)]:
                        self.metric_data[layer.name] = []
                        
                elif metric == Layer.Convolutional:
                    for layer in [layer for layer in self.layers if isinstance(layer, Layer.Convolutional)]:
                        self.metric_data[layer.name] = []

        # compute number of batches
        num_batches = np.ceil(self.data[0][0].shape[0] / batch_size).astype(np.int32)
        
        # if visualize:
        #     # check to see if figure already exists and if not, create it
        #     if len(plt.get_fignums()) == 0:
        #         # get all dense layers from network
        #         dense = [l for l in self.layers if isinstance(l, Layer.Dense)]
        #         #DEBUG
        #         print(f'network dense layers:{dense}')
        #         fig = plt.figure(tight_layout=True)
        #         grid = fig.add_gridspec(1,3)
        #         wb_grid = grid[0,0:2].subgridspec(nrows=int(np.ceil(len(self.layers)/2)), ncols=2)
        #         metric_grid = grid[0,2].subgridspec(3,1)

        #         for index, l in enumerate(self.layers):
        #             #DEBUG
        #             print(f'layer {index} weights shape:{l.weights.shape}')
        #             subgrid = wb_grid[int(np.floor(index/2)),index%2].subgridspec(3,1)
        #             plot1 = fig.add_subplot(subgrid[0:2,0])
        #             plot1.axes.yaxis.set_ticks(np.arange(l.weights.shape[0]))
        #             plot1.axes.xaxis.set_visible(False)
        #             plot1.grid(False)
        #             plot1.set_title(f'Layer {index}')

        #             plot2 = fig.add_subplot(subgrid[2,0], sharex=plot1)
        #             plot2.axes.yaxis.set_visible(False)
        #             plot2.axes.xaxis.set_ticks(np.arange(l.weights.shape[1]))
        #             plot2.grid(False)      

        #         plot_loss = fig.add_subplot(metric_grid[0,0])
        #         plot_loss.axes.xaxis.set_visible(False)
        #         plot_loss.set_title('Loss')
        #         plot_loss.legend()

        #         plot_accuracy = fig.add_subplot(metric_grid[1,0])
        #         plot_accuracy.axes.xaxis.set_visible(False)
        #         plot_accuracy.set_title('Accuracy')
        #         plot_accuracy.legend()

        #         plot_learning_rate = fig.add_subplot(metric_grid[2,0])
        #         plot_learning_rate.set_title('Learning Rate')

        #     # update graph animation with new data
        #     ani = animation.FuncAnimation(fig, Utils.animate,
        #                                   fargs=(dense, self.metric_data, plot1, plot2, plot_loss, plot_accuracy,
        #                                          plot_learning_rate),
        #                                   interval=200)

        for e in range(epochs):
            # shuffle the training data
            x_train, y_train = shuffle(self.data[0][0], self.data[0][1])          

            for b in range(num_batches):
                x_batch = x_train[b * batch_size:(b+1) * batch_size]
                y_batch = y_train[b * batch_size:(b+1) * batch_size]
                
                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # calculate dE/dY
                gradient = self.loss.backward(y_batch, output)

                # run optimizer pre_update_params
                self.optimizer.pre_update_params(e)

                # back-propogate error over network
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.optimizer)

                # run optimizer post_update_params
                self.optimizer.post_update_params()

                #collect metric data for batch
                for metric in self.metrics:
                    if metric == Layer.Dense:
                        for layer in [layer for layer in self.layers if isinstance(layer, Layer.Dense)]:
                             self.metric_data[layer.name].append([layer.weights, layer.bias])
                            
                    if metric == Layer.Convolutional:
                        for layer in [layer for layer in self.layers if isinstance(layer, Layer.Convolutional)]:
                            self.metric_data[layer.name].append([layer.kernels, layer.biases])
                            
                    elif isinstance(metric, Optimizers.Optimizer):
                        self.metric_data['Learning Rate'].append(metric.current_learning_rate)
                
                for index,(x, y) in enumerate(self.data):
                    output = self.predict(x)
                    if index == 0:
                        data = 'Train'
                    else:
                        data = 'Validation'
                    for metric in self.metrics:
                        if isinstance(metric, Losses.Loss):
                            # collect regularization error across layers
                            total_L1_loss = 0
                            total_L2_loss = 0
                            for l in self.layers:
                                if hasattr(l, 'L1_regularizer'):
                                    L1_loss, L2_loss = metric.regularization_loss(l)
                                    total_L1_loss += L1_loss
                                    total_L2_loss += L2_loss
                            loss = metric.calculate(y, output)
                            total_loss = loss + total_L1_loss + total_L2_loss
                            # append data to end of respective metric_data array
                            self.metric_data[f'{data} {metric.name}'].append(loss)
                            if total_L1_loss > 0:
                                self.metric_data[f'{data} L1 Regularization'].append(total_L1_loss)
                            if total_L2_loss > 0:
                                self.metric_data[f'{data} L2 Regularization'].append(total_L2_loss)
                            if total_loss > loss:
                                self.metric_data[f'{data} Total Loss'].append(total_loss)

                        elif isinstance(metric, Metrics.Metric):
                            m = metric.calculate(y, output)
                            self.metric_data[f'{data} {metric.name}'].append(m)\

                # print update
                # Create a list to hold table rows
                table_rows = []

                # For each data type, build a dictionary of metric names and their values
                for data_type in ['Train', 'Validation']:
                    row = {"Data": data_type}
                    for metric in self.metrics:
                        if (metric is not Layer.Dense) and (metric is not Layer.Convolutional):
                            if isinstance(metric, Losses.Loss):
                                row[metric.name] = f"{float(self.metric_data[f'{data_type} {metric.name}'][-1]):+.4f}"
                                if 'L1 Regularization' in self.metric_data:
                                    row["L1 Reg."] = f"{float(self.metric_data[f'{data_type} L1 Regularization'][-1]):+.4f}"
                                if 'L2 Regularization' in self.metric_data:
                                    row["L2 Reg."] = f"{float(self.metric_data[f'{data_type} L2 Regularization'][-1]):+.4f}"
                                if 'Total Loss' in self.metric_data:
                                    row["Total Loss"] = f"{float(self.metric_data[f'{data_type} Total Loss'][-1]):+.4f}"
                            elif isinstance(metric, Optimizers.Optimizer):
                                row["Learning Rate"] = f"{float(self.metric_data['Learning Rate'][-1]):.4f}"
                            elif isinstance(metric, Metrics.Metric):
                                row[metric.name] = f"{float(self.metric_data[f'{data_type} {metric.name}'][-1]):+.4f}"
                    table_rows.append(row)

                print(f"Epoch {e:>2} Batch {b:>2}")
                print(tabulate(table_rows, headers="keys", tablefmt="pretty"))                   

                    
    def evaluate(self, data):
        """
        Passes data through network and returns metrics (self.metrics)
        
        Parameters:
        data: [(x_test, y_test), (x_test, y_test), ...]
        """
        metrics = []
        for x, y in data:
            output = self.predict(x)
            for metric in self.metrics:
                if isinstance(metric, Metrics.Metric) or isinstance(metric, Losses.Loss):
                    m = metric.calculate(y, output)
                    metrics.append(m)
        return metrics
 

    def predict(self, x, batch_size=32):
        outputs = []
        N = x.shape[0]
        for i in range(0, N, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = x_batch
            for layer in self.layers:
                y_batch = layer.forward(y_batch, is_training=False)
            outputs.append(y_batch)
        return self._xp.concatenate(outputs, axis=0)

    
    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Layer.Dense):
                parameters.append(layer.get_parameters())
        return parameters
     
        
    def set_parameters(self, parameters):
        """
        sets the weight and bias parameters of a network
        """
        for parameter, layer in zip(parameters, [l for l in self.layers if isinstance(l, Layer.Dense)]):
            layer.set_parameters(*parameter)
    
    
    def save_parameters(self, path):
        """
        Saves parameters to the specified path
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    
    def load_parameters(self, path):
        """
        Loads parameters into the network
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
  

    def save(self, path):
        """
        Saves the model
        """
        # create a copy of the model to save
        model = copy.deepcopy(self)
        
        # clear loss gradient data
        self.loss.__dict__.pop('input_gradient', None)
        
        # clear layer input/output/gradient data
        for layer in self.layers:
            for attribute in ['input', 'output', 'input_gradient', 'output_gradient', 'weight_gradient', 'bias_gradient']:
                layer.__dict__.pop(attribute, None)
            
        # clear network.data (train and test data)
        self.data.clear()
        
        # save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
     
    
    @staticmethod
    def load(path):
        """
        Loads model from path
        """   
        with open(path, 'rb') as f:
            return pickle.load(f)
```



## File: optimizers.py
```py
import numpy as np
import cupy as cp

class Optimizer():
    # base class for optimizers
    def __init__(self, device='CPU'):
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp
    
    def pre_update_params():
        raise NotImplementedError
    
    def update_params():
        raise NotImplementedError
    
    def post_update_params():
        raise NotImplementedError
 

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimization
    """
    
    def __init__(self, learning_rate=.01, decay=0, momentum=0, device='CPU'):
        super().__init__(device)
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
                Layer.weight_momentum = self._xp.zeros_like(Layer.weights)
                Layer.bias_momentum = self._xp.zeros_like(Layer.bias)
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
    
    def __init__(self, learning_rate=0.1, decay=0, epsilon=1e-7, device='CPU'):
        super().__init__(device)
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
            Layer.weight_cache = self._xp.zeros_like(Layer.weights)
            Layer.bias_cache = self._xp.zeros_like(Layer.bias)
        
        # update cache values    
        Layer.weight_cache += Layer.weight_gradient**2
        Layer.bias_cache += Layer.bias_gradient**2
        
        # return updates to weight and bias parameters
        return Layer.weights - (self.current_learning_rate * Layer.weight_gradient / (self._xp.sqrt(Layer.weight_cache) + self.epsilon)), Layer.bias - (self.current_learning_rate * Layer.bias_gradient / (self._xp.sqrt(Layer.bias_cache) + self.epsilon))
    

class RMSprop(Optimizer):
    """
    Root Mean Square Propogation Optimization
    """
    
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, cache_decay=0.999, device='CPU'):
        super().__init__(device)
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
            Layer.weight_cache = self._xp.zeros_like(Layer.weights)
            Layer.bias_cache = self._xp.zeros_like(Layer.bias)
            
        # update cache values
        Layer.weight_cache = (self.cache_decay * Layer.weight_cache) + ((1 - self.cache_decay) * Layer.weight_gradient**2)
        Layer.bias_cache = (self.cache_decay * Layer.bias_cache) + ((1 - self.cache_decay) * Layer.bias_gradient**2)
        
        # return updates to weight and bias parameters
        return Layer.weights - (self.current_learning_rate * Layer.weight_gradient / (self._xp.sqrt(Layer.weight_cache) + self.epsilon)), Layer.bias - (self.current_learning_rate * Layer.bias_gradient / (self._xp.sqrt(Layer.bias_cache) + self.epsilon)) 
    

class Adam(Optimizer):
    """
    Adaptive Momentum Optimizer
    """
    
    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, momentum=0.9, cache_decay=0.999, device='CPU'):
        super().__init__(device)
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
            Layer.weight_momentum = self._xp.zeros_like(Layer.weights)
            Layer.bias_momentum = self._xp.zeros_like(Layer.bias)
            Layer.weight_cache = self._xp.zeros_like(Layer.weights)
            Layer.bias_cache = self._xp.zeros_like(Layer.bias)
            
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
        return Layer.weights - self.current_learning_rate * corrected_weight_momentum / (self._xp.sqrt(corrected_weight_cache) + self.epsilon), Layer.bias - self.current_learning_rate * corrected_bias_momentum / (self._xp.sqrt(corrected_bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration += 1
```



## File: README.md
```md
# NeuralNet
 My from scratch neural network 

```



## File: repo_code.md
```md

```



## File: test.py
```py

import network as Network
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import backend
import tensorflow as tf
import unittest

class UnetTest(unittest.TestCase):

    def setUp(self):
        super(UnetTest, self).setUp()

    # def tearDown(self):
    #     pass

    # def test_All(self):
    #     self.test_Layers()
    #     self.test_Activations()
    #     self.test_Losses()

    # def test_Layers(self):
    #     self.test_Activations()
    #     self.test_Dense()
    #     self.test_Convolutional()
    #     self.test_Pool()
    #     self.test_Flatten()

    # def test_Activations(self):
    #     self.test_relu()
    #     self.test_elu()
    #     self.test_gelu()
    #     self.test_tanh()
    #     self.test_sigmoid()
    #     self.test_softmax()

    # def test_Losses(self):
    #     self.test_MSE()
    #     self.test_Sparse_Categorical_Cross_Entropy()

    def test_relu(self):
        test_arr = np.random.randn(5,5)
        tf_out = np.array(activations.relu(test_arr))
        
        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('relu', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)

        # self.assertEqual(tf_out.all(), net_out.all())
        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('Relu.forward() passed')

        x = tf.Variable(test_arr)
        with tf.GradientTape() as tape:
            tf_out = activations.relu(x)
        tf_dx = tape.gradient(tf_out, x).numpy()

        net_dx = net.layers[0].derivative()

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('Relu.derivative() passed')

    def test_tanh(self):
        test_arr = np.random.randn(5,5)
        tf_out = np.array(activations.tanh(test_arr))

        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('tanh', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)

        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('Tanh.forward() passed')

        x = tf.Variable(test_arr)
        with tf.GradientTape() as tape:
            tf_out = activations.tanh(x)
        tf_dx = tape.gradient(tf_out, x).numpy()

        net_dx = net.layers[0].derivative()

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('Tanh.derivative() passed')

    def test_softmax(self):
        test_arr = np.random.randn(5,5)
        tf_test_arr = tf.Variable(test_arr)
        tf_out = np.array(activations.softmax(tf_test_arr))

        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('softmax', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)
        
        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('softmax.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = activations.softmax(tf_test_arr)
        tf_dx = tape.gradient(tf_out, tf_test_arr).numpy()

        net_dx = net.layers[0].backward(np.ones_like(net_out), optimizer=None)

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=0, atol=1e-10) == None:
            print('softmax.backward() passed')

    def test_elu(self):
        test_arr = np.random.randn(5,5)
        tf_test_arr = tf.Variable(test_arr)
        tf_out = np.array(activations.elu(tf_test_arr))

        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('elu', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)

        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('Elu.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = activations.elu(tf_test_arr)
        tf_dx = tape.gradient(tf_out, tf_test_arr).numpy()

        net_dx = net.layers[0].derivative()

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('Elu.derivative() passed')


    def test_gelu(self):
        test_arr = np.random.randn(5,5)
        tf_test_arr = tf.Variable(test_arr)
        tf_out = np.array(activations.gelu(tf_test_arr))

        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('gelu', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)

        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('Gelu.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = activations.gelu(tf_test_arr)
        tf_dx = tape.gradient(tf_out, tf_test_arr).numpy()

        net_dx = net.layers[0].derivative()

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-04) == None:
            print('Gelu.derivative() passed')


    def test_sigmoid(self):
        test_arr = np.random.randn(5,5)
        tf_test_arr = tf.Variable(test_arr)
        tf_out = np.array(activations.sigmoid(tf_test_arr))

        net = Network.Neural_Network()
        net.compile_model()
        net.add_Activation('sigmoid', input_dims=test_arr.shape)
        net_out = net.layers[0].forward(test_arr)

        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-05) == None:
            print('sigmoid.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = activations.sigmoid(tf_test_arr)
        tf_dx = tape.gradient(tf_out, tf_test_arr).numpy()

        net_dx = net.layers[0].derivative()

        # self.assertEqual(tf_dx.all(), net_dx.all())
        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('sigmoid.derivative() passed')


    def test_Dense(self):
        data = np.float32(np.random.randn(5,5))
        tf_data = tf.Variable(data, dtype=tf.float32)

        tf_model = tf.keras.layers.Dense(5, name="dense_1")


        tf_out = tf_model(tf_data)

        tf_weights = tf_model.weights
        weights = tf_weights[0].numpy()
        bias = tf_weights[1].numpy()

        optimizer = Network.Optimizers.Adam(learning_rate=0.01, decay=1e-2)
        net = Network.Neural_Network()
        net.compile_model(optimizer=optimizer)
        net.add_Dense(5,input_dims=data.shape[1])
        net.layers[0].set_parameters(weights, bias)
        net_out = net.layers[0].forward(data)

        if np.testing.assert_allclose(net_out,tf_out, rtol=1e-05) == None:
            print('Dense.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = tf_model(tf_data)
        tf_dx = tape.gradient(tf_out, tf_data).numpy()

        net_dx = net.layers[0].backward(output_gradient=np.ones_like(net.layers[0].input), optimizer=net.optimizer)

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('Dense.backward() passed')


    def test_Convolutional(self):
        data = np.float32(np.random.randn(2,3,5,5))

        tf_data = tf.Variable(data, dtype=tf.float32)

        tf_model = tf.keras.layers.Conv2D(2,3, data_format='channels_first', input_shape=data.shape[1:], name='Conv_1')
        y = tf_model(tf_data)

        weights = tf.transpose(tf_model.kernel, [3,2,0,1]).numpy()

        optimizer = Network.Optimizers.Adam(learning_rate=0.01, decay=1e-2)
        net = Network.Neural_Network()
        net.compile_model(optimizer=optimizer)
        net.add_Convolutional(3, 2, mode='valid', input_dims=data.shape[1:])
        bias = tf_model.bias.numpy()
        net.layers[0].set_parameters(weights, bias)
        net_out = net.layers[0].forward(data)

        if np.testing.assert_allclose(net_out,y, rtol=0, atol=1e-5) == None:
            print('Convolution.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = tf_model(tf_data)
        tf_dx = tape.gradient(tf_out, tf_data).numpy()

        net_dx = net.layers[0].backward(output_gradient=np.ones_like(net.layers[0].output), optimizer=net.optimizer)

        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-05) == None:
            print('Convolutional.backward() passed')


    def test_Pool(self):
        data = np.float32(np.random.randn(1,2,10,10))

        # Transpose data to use with tensorflow because tensorflow MaxPool2D
        # only works in NCHW format (channels last)
        data_tf = np.transpose(data, (0, 2, 3, 1))
        tf_data = tf.Variable(data_tf, dtype=tf.float32)

        tf_model = tf.keras.layers.MaxPool2D(data_format='channels_last')
        y = tf_model(tf_data)
        # Transpose TF output back to channels first format to compare
        y_channels_first = np.transpose(y.numpy(), (0, 3, 1, 2))

        optimizer = Network.Optimizers.Adam(learning_rate=0.01, decay=1e-2)
        net = Network.Neural_Network()
        net.compile_model(optimizer=optimizer)
        net.add_Pool(input_dims=data.shape[1:])

        net_out = net.layers[0].forward(data)

        if np.testing.assert_allclose(net_out,y_channels_first, rtol=1e-05) == None:
            print('Pool.forward() passed')

        with tf.GradientTape() as tape:
            tf_out = tf_model(tf_data)
        tf_dx = tape.gradient(tf_out, tf_data)
        # Transpose TF dx back to channels first format to compare
        tf_dx_channels_first = np.transpose(tf_dx.numpy(), (0, 3, 1, 2))

        net_dx = net.layers[0].backward(output_gradient=np.ones_like(net.layers[0].output), optimizer=net.optimizer)

        if np.testing.assert_allclose(tf_dx_channels_first, net_dx, rtol=1e-05) == None:
            print('Pool.backward() passed')


    def test_Flatten(self):
        shape = (10,8,5,5)
        data = np.zeros(shape)

        optimizer = Network.Optimizers.Adam(learning_rate=0.01, decay=1e-2)
        net = Network.Neural_Network()
        net.compile_model(optimizer=optimizer)
        net.add_Flatten(input_dims=data.shape[1:])

        net_out = net.layers[0].forward(data)

        y = np.zeros((10,8*5*5))

        if np.testing.assert_allclose(net_out,y, rtol=1e-05) == None:
            print('Flatten.forward() passed')

        net_back = net.layers[0].backward(y, optimizer)

        if np.testing.assert_allclose(net_back,data, rtol=1e-05) == None:
            print('Flatten.bacward() passed')


    def test_BatchNormalization(self):
        # Generate random input
        test_arr = np.random.randn(10, 5)  # 10 samples, 5 features

        # Use TensorFlow to get expected output
        tf_bn_layer = tf.keras.layers.BatchNormalization()
        tf_out = tf_bn_layer(test_arr, training=True).numpy()

        # Initialize your network
        optimizer = Network.Optimizers.Adam(learning_rate=0.01, decay=1e-2)
        net = Network.Neural_Network()
        net.compile_model()
        net.add_BatchNorm(input_dims=test_arr.shape[1])
        net_out = net.layers[-1].forward(test_arr, is_training=True)

        # Compare results
        if np.testing.assert_allclose(tf_out, net_out, rtol=1e-03, atol=1e-03) == None:
            print('BatchNormalization.forward() passed')

        # Compute gradients using TensorFlow
        x = tf.Variable(test_arr, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tf_out = tf_bn_layer(x, training=True)
        tf_dx = tape.gradient(tf_out, x).numpy()

        # Compute gradients using your implementation
        net_dx = net.layers[-1].backward(np.ones_like(net_out), optimizer=optimizer)

        # Compare gradients
        if np.testing.assert_allclose(tf_dx, net_dx, rtol=1e-03, atol=1e-03) == None:
            print('BatchNormalization.backward() passed')


    def test_MSE(self):
        preds = np.random.randn(10)
        true = np.random.randn(10)
        tf_preds = tf.Variable(preds)
        tf_true = tf.Variable(true)

        mse = Network.Losses.Mean_Squared_Error()

        net_loss = mse.calculate(true,preds)

        tf_mse = tf.keras.losses.MeanSquaredError()
        tf_loss = tf_mse(true,preds).numpy()

        if np.testing.assert_allclose(net_loss, tf_loss, rtol=1e-05) == None:
            print('Mean_Squared_Error.calculate() passed')

        net_loss_dx = mse.backward(true,preds)

        with tf.GradientTape() as tape:
            tf_loss = tf_mse(tf_true,tf_preds)
        tf_loss_dx = tape.gradient(tf_loss, tf_preds).numpy()

        if np.testing.assert_allclose(net_loss_dx, tf_loss_dx, rtol=1e-05) == None:
            print('Mean_Squared_Error.backward() passed')


    def test_Sparse_Categorical_Cross_Entropy(self):
        preds = np.random.randn(10,5)
        probs = activations.softmax(tf.Variable(preds)).numpy()
        targets = np.random.randint(0,5,size=10)
        tf_probs = tf.Variable(probs, dtype=tf.float32)
        tf_targets = tf.Variable(targets, dtype=tf.float32)

        scce = Network.Losses.Sparse_Categorical_Cross_Entropy()
        net_loss = scce.calculate(targets,probs)

        tf_scce = tf.keras.losses.SparseCategoricalCrossentropy()
        tf_loss = tf_scce(tf_targets,tf_probs).numpy()

        if np.testing.assert_allclose(net_loss, tf_loss, rtol=1e-05) == None:
            print('Mean_Squared_Error.calculate() passed')

        net_loss_dx = scce.backward(targets,probs)

        with tf.GradientTape() as tape:
            tf_loss = tf_scce(tf_targets,tf_probs)
        tf_loss_dx = tape.gradient(tf_loss, tf_probs)

        if np.testing.assert_allclose(net_loss_dx, tf_loss_dx - 0.1, rtol=0, atol=1e-5) == None:
            print('Mean_Squared_Error.backward() passed')
```



## File: utils.py
```py
# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f_gWhHibpUXe_i2PQWKtQdtPuaBR9bLw
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix
from tensorflow.image import resize
import itertools
from glob import glob
from network import Neural_Network

def im2col(input_data, filter_h, filter_w, stride=1, pad=0, xp=np):
    """
    Rearranges image blocks into columns.
    Input:
      - input_data: shape (N, C, H, W)
      - filter_h, filter_w: filter height and width
      - stride: stride for the convolution
      - pad: amount of zero-padding
      - xp: either np (CPU) or cp (GPU)
    Returns:
      - col: 2D array of shape (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape

    # Determine padding values
    if isinstance(pad, int):
        pad_top = pad
        pad_bottom = pad
        pad_left = pad
        pad_right = pad
    else:
        # Expect pad to be ((pad_top, pad_bottom), (pad_left, pad_right))
        (pad_top, pad_bottom), (pad_left, pad_right) = pad

    # Pad the input
    input_padded = xp.pad(input_data, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    # Compute output dimensions
    out_h = (H + pad_top + pad_bottom - filter_h) // stride + 1
    out_w = (W + pad_left + pad_right - filter_w) // stride + 1

    # Get strides for the padded array
    s0, s1, s2, s3 = input_padded.strides

    # Build shape and strides for as_strided
    shape = (N, C, out_h, out_w, filter_h, filter_w)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    cols = xp.lib.stride_tricks.as_strided(input_padded, shape=shape, strides=strides)

    # Rearrange dimensions so that each patch becomes a row
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * out_h * out_w, -1)
    return cols


def col2im(cols, input_shape, filter_h, filter_w, stride=1, pad=0, xp=np):
    """
    Converts column representation back into image blocks.
    Input:
      - col: 2D array from im2col with shape (N*out_h*out_w, C*filter_h*filter_w)
      - input_shape: shape (N, C, H, W) of the original input data
      - filter_h, filter_w: filter dimensions
      - stride, pad: convolution parameters. If pad is an int, symmetric padding is assumed.
                   If pad is a tuple of tuples, it should be ((pad_top, pad_bottom), (pad_left, pad_right)).
      - xp: either np or cp
    Returns:
      - An array with shape (N, C, H, W)
    """
    N, C, H, W = input_shape

    # Determine padding values
    if isinstance(pad, int):
        pad_top = pad
        pad_bottom = pad
        pad_left = pad
        pad_right = pad
    else:
        (pad_top, pad_bottom), (pad_left, pad_right) = pad

    # Compute output dimensions from im2col
    out_h = (H + pad_top + pad_bottom - filter_h) // stride + 1
    out_w = (W + pad_left + pad_right - filter_w) // stride + 1

    # Reshape cols to (N, out_h, out_w, C, filter_h, filter_w)
    cols_reshaped = cols.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # Prepare an output array with padding
    H_padded = H + pad_top + pad_bottom
    W_padded = W + pad_left + pad_right
    img_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    # Instead of looping over the entire image, we only loop over the filter dimensions.
    for y in range(filter_h):
        y_end = y + stride * out_h
        for x in range(filter_w):
            # Accumulate the values from cols_reshaped into the appropriate region of img_padded.
            img_padded[:, :, y:y_end:stride, x:x + stride * out_w] += cols_reshaped[:, :, y, x, :, :]

    # Remove padding and return the original image dimensions.
    return img_padded[:, :, pad_top:H_padded - pad_bottom, pad_left:W_padded - pad_right]


# store functions used for processing data

def train_test_split(x, y, split=0.2):
    """
    returns x and y data split into training and testing sets according to split percentage passed
    Input:
        x: input data
        y: output data
    Return:
        x_train, y_train, x_test, y_test
    """
    
    split_index = int(np.floor(len(y)*split))
    x_train = x[split_index:]
    y_train = y[split_index:]
    x_test = x[:split_index]
    y_test = y[:split_index]
    
    return x_train, y_train, x_test, y_test

def build_metric_figure(layers):
    fig = plt.figure(tight_layout=True)
    grid = fig.add_gridspec(1,3)
    wb_grid = grid[0,0:2].subgridspec(nrows=int(np.ceil(len(layers)/2)), ncols=2)
    metric_grid = grid[0,2].subgridspec(3,1)

    for index, l in enumerate(layers):
        subgrid = wb_grid[int(np.floor(index/2)),index%2].subgridspec(3,1)
        plot1 = fig.add_subplot(subgrid[0:2,0])
        plot1.axes.yaxis.set_ticks(np.arange(l.weights.shape[0]))
        plot1.axes.xaxis.set_visible(False)
        plot1.grid(False)
        plot1.set_title(f'Layer {index}')

        plot2 = fig.add_subplot(subgrid[2,0], sharex=plot1)
        plot2.axes.yaxis.set_visible(False)
        plot2.axes.xaxis.set_ticks(np.arange(l.weights.shape[1]))
        plot2.grid(False)      

    plot_loss = fig.add_subplot(metric_grid[0,0])
    plot_loss.axes.xaxis.set_visible(False)
    plot_loss.set_title('Loss')
    plot_loss.legend()

    plot_accuracy = fig.add_subplot(metric_grid[1,0])
    plot_accuracy.axes.xaxis.set_visible(False)
    plot_accuracy.set_title('Accuracy')
    plot_accuracy.legend()

    plot_learning_rate = fig.add_subplot(metric_grid[2,0])
    plot_learning_rate.set_title('Learning Rate')
    
    return fig

# def animate(network, metrics, plot1, plot2, plot_loss, plot_accuracy, plot_learning_rate):
#     plt.cla()
    
#     for l in network.layers:
#         plot1.imshow(l.weights, cmap="seismic", aspect='auto')
#         plot2.imshow(np.expand_dims(l.bias, axis=1).T, cmap="seismic", aspect='auto')
        
#     # plot metric data
#     # plot loss    
#     # check if total loss exists in metric data (means network applies regularization)
#     if 'Total Loss' in network.metric_data:
#         plot_loss.plot(metrics['Total Loss'][:,0], label='total train', color='red', lw=1)
#         plot_loss.plot(metrics['Total Loss'][:,1], label='total validation', color='red', lw=1, alpha=0.5)
#         # plot regularizations if exist
#         if 'L2 Regularization' in network.metric_data:
#             plot_loss.plot(metrics['L2 Regularization'][:,0], label='L2 train', color='orange', lw=1)
#             plot_loss.plot(metrics['L2 Regularization'][:,1], label='L2 validation', color='orange', lw=1, alpha=0.5)
#         if 'L1 Regularization' in network.metric_data:
#             plot_loss.plot(metrics['L1 Regularization'][:,0], label='L1 train', color='purple', lw=1)
#             plot_loss.plot(metrics['L1 Regularization'][:,1], label='L1 validation', color='purple', lw=1, alpha=0.5)
#     plot_loss.plot(metrics['Sparse CXE'][:,0], label='loss train', color='blue', lw=1)
#     plot_loss.plot(metrics['Sparse CXE'][:,1], label='loss validation', color='blue', lw=1, alpha=0.5)
    
#     # plot accuracy
#     plot_accuracy.plot(metrics['Accuracy'][:,0], label='train', color='green', linewidth=1)
#     plot_accuracy.plot(metrics['Accuracy'][:,1], label='validation', color='green', alpha=0.3, linewidth=1)
    
#     # plot learning rate
#     plot_learning_rate.plot(metrics['Learning Rate'][:,0], label='train', color='red', linewidth=1)

def get_confusion_matrix(data:tuple, model:Neural_Network, image_size=(100,100)):
    """
    returns a confusion matrix generated from predictions made on images passed through an image data generator

    INPUTS
    data: tuple, tuple of (x,y)
    model: Neural_Network, model used to make predictions on images
    image_size: tuple (int, int), default=(100,100), size to scale images to

    RETURNS
    cm: confusion matrix
    """
    print('Generating Confusion Matrix')
    # Unpack data tuple
    x,y = data
    # Resize input if images (ndim > 2)
    if x.ndim > 2:
        # Format images
        if x.ndim == 3: # Grayscale images
            # Add channel dimension at 2nd dimension (axis 1)
            x = model._xp.expand_dims(x, axis=1)
        elif x.ndim == 4:
            # Transpose to CHW format
            x = x.transpose(0,3,1,2)
        x = resize(x, image_size)
    predictions = model.predict(x)
    predictions = model._xp.argmax(predictions, axis=1)
    targets = model._xp.argmax(y, axis=1)
    cm = confusion_matrix(targets, predictions)
    return cm

def plot_confusion_matrix(confusion_matrix, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, wihtout normalization')
    print(confusion_matrix)

    plt.figure(figsize=(15,15))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if confusion_matrix[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_accuracy_from_confusion_matrix(confusion_matrix):
    return confusion_matrix.trace() / confusion_matrix.sum()
```

