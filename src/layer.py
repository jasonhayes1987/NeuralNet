"""Layer module for neural network API.

This module defines base classes for layers including:
- Layer: Base class for all layers.
- Activation: Base class for activation layers.
- Dense: Fully connected layer.
- Dropout: Dropout layer.
- Convolutional: Convolutional layer.
- Flatten: Flatten layer.
- Pool: Pooling layer.
- BatchNormalization: Batch normalization layer.
"""


import numpy as np
from scipy import signal
import cupy as cp
from cupyx.scipy import signal as c_signal
from utils import im2col, col2im

class Layer:
    """
    Base class for layers.
    
    Attributes:
        _device (str): Device type ('CPU' or 'GPU').
        _xp: Reference to numpy or cupy based on the device.
        input: Input data to the layer.
        output: Output data from the layer.
        _input_dims: Dimensions of the input.
        _output_dims: Dimensions of the output.
    """
    def __init__(self, device='CPU'):
        """
        Initialize a Layer.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp
        self.input = None
        self.output = None
        self._input_dims = None
        self._output_dims = None

    def __getstate__(self):
        """
        Return state for pickling, excluding unpicklable attributes.
        """
        state = self.__dict__.copy()
        if "_xp" in state:
            del state["_xp"]
        return state

    def __setstate__(self, state):
        """
        Restore state from pickled state.
        """
        self.__dict__.update(state)
        if hasattr(self, "_device") and self._device == "GPU":
            self._xp = cp
        else:
            self._xp = np

    def forward(self, input, is_training=True):
      """
        Compute the forward pass of the layer.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Raises:
            NotImplementedError: Must be overridden by child classes.
        """
      raise NotImplementedError("Must override 'forward' by instantiating child of layer class (Activation, Dense, Convolutional, etc...")

    def backward(self, output_gradient, optimizer):
      """
        Compute the backward pass of the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            learning_rate: Learning rate for parameter updates.

        Raises:
            NotImplementedError: Must be overridden by child classes.
        """
      raise NotImplementedError("Must override 'backward' by instantiating child of layer class (Activation, Dense, Convolutional, etc...")

class Activation(Layer):
    """
    Base class for activation layers.
    """
    def __init__(self, activation, input_dims=None, device='CPU'):
        """
        Initialize an activation layer.

        Args:
            activation (str): Name of the activation function.
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.activation = activation
        self.name = 'Activation'
        self._input_dims = input_dims
        self._output_dims = input_dims

    def compute(self):
        """
        Compute the activation.

        Raises:
            NotImplementedError: Must be overridden by child classes.
        """
        raise NotImplementedError("Must override 'compute' by instantiating child of Activation class (Relu, Tanh, Sigmoid, etc...")

    def derivative(self):
        """
        Compute the derivative of the activation.

        Raises:
            NotImplementedError: Must be overridden by child classes.
        """
        raise NotImplementedError("Must override 'derivative' by instantiating child of Activation class (Relu, Tanh, Sigmoid, etc...")

    def forward(self, input, is_training=True):
        """
        Compute the forward pass for activation.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Activated output.
        """
        self.input = input
        self.output = self.compute()
        return self.output

    def backward(self, output_gradient, optimizer):
        """
        Compute the backward pass for activation.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
        output = self._xp.multiply(output_gradient, self.derivative())
        return output

class Dense(Layer):
    """
    Fully connected (Dense) layer.
    """
    def __init__(self, output_size, L1_regularizer=0, L2_regularizer=0, input_dims=None, device='CPU', init='Default'):
        """
        Initialize a Dense layer.

        Args:
            output_size (int): Number of neurons in the layer.
            L1_regularizer (float): L1 regularization strength.
            L2_regularizer (float): L2 regularization strength.
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
            init (str): Weight initialization method.
        """
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
        """
        Compute the forward pass for the Dense layer.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Output of the Dense layer.
        """
        self.input = input
        self.output = self._xp.dot(self.input, self.weights) + self.bias
        return self.output


    def backward(self, output_gradient, optimizer):
        """
        Compute the backward pass for the Dense layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.output_gradient = output_gradient
        self.weight_gradient = self._xp.dot(self.input.T, output_gradient)
        self.bias_gradient = self._xp.sum(self.output_gradient, axis=0)
    
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
        """
        Get the parameters (weights and bias) of the layer.

        Returns:
            Tuple of weights and bias.
        """
        return self.weights, self.bias
    
    def set_parameters(self, weights, bias):
        """
        Set the parameters (weights and bias) of the layer.

        Args:
            weights: New weights.
            bias: New bias.
        """
        self.weights = weights
        self.bias = bias
    
    @staticmethod
    def set_input_size(previous_layer):
        """
        Set the input size of the layer using the previous layer's output size.

        Args:
            previous_layer: Previous layer in the network.

        Returns:
            Input size for the current layer.
        """
        # check to make sure previous layer shape is 2D
        if len(previous_layer.output.shape == 2):
            return previous_layer.output.shape[1] # return second dimension of previous layer output
        else:
            # Throw ERROR that dense required 2D input data
            print(f'Dense layer requires 2D data as input; passed {len(previous_layer.output.shape)}D')
        
class Dropout(Layer):
    """
    Dropout layer.
    """
    def __init__(self, dropout_rate, input_dims=None, output_size=None, device='CPU'):
        """
        Initialize a Dropout layer.

        Args:
            dropout_rate (float): Fraction of the input units to drop.
            input_dims: Dimensions of the input.
            output_size: Output size (usually same as input_dims).
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'Dropout'
        self.keep_rate = 1 - dropout_rate
        self._input_dims = input_dims
        self._output_dims = output_size
        
    def forward(self, input, is_training=True):
        """
        Compute the forward pass for Dropout.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Output after applying dropout.
        """
        self.input = input
        
        if not is_training:
            self.output = self.input.copy()
            return self.output
        
        self.mask = self._xp.random.binomial(1, self.keep_rate, input.shape) / self.keep_rate
        self.output = input * self.mask
        return self.output
    
    def backward(self, output_gradient, optimizer):
        """
        Compute the backward pass for Dropout.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.output_gradient = output_gradient
        return self.output_gradient * self.mask
    
class Convolutional(Layer):
    """
    Convolutional layer.
    """
    def __init__(self, kernel_size, depth, mode, input_dims=None, device='CPU', init='default'):
        """
        Initialize a Convolutional layer.

        Args:
            kernel_size (int): Size of the convolution kernel.
            depth (int): Number of filters.
            mode (str): Convolution mode ('valid' or 'same').
            input_dims: Dimensions of the input (depth, height, width).
            device (str): Device type ('CPU' or 'GPU').
            init (str): Weight initialization method.
        """
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
        """
        Compute asymmetric padding for 'same' convolution.

        Returns:
            Tuple containing padding before and after for height and width.
        """
        kernel_size = self._kernels_shape[2]
        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return (pad_before, pad_after)

    def forward(self, input, is_training=True):
        """
        Compute the forward pass for the Convolutional layer.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Output of the convolution.
        """
        self.input = input
        if self.mode == 'same':
            pad_tuple = self.compute_asymmetric_padding()
            self.pad = (pad_tuple, pad_tuple)  # For both height and width
        else:
            self.pad = 0
        self.output = self.conv_forward_im2col(self.input, self.weights, self.bias, stride=1, pad=self.pad)
        return self.output

    def backward(self, output_gradient, optimizer):
        """
        Compute the backward pass for the Convolutional layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.output_gradient = output_gradient
        self.weight_gradient, input_gradient = self.conv_backward_im2col(
            self.output_gradient, self.input, self.weights, stride=1, pad=self.pad
        )
        self.bias_gradient = self._xp.sum(self.output_gradient, axis=(0, 2, 3))

        self.weights, self.bias = optimizer.update_params(self)
        return input_gradient

    def conv_forward_im2col(self, x, W, b, stride=1, pad=0):
        """
        Perform convolution using im2col for forward pass.

        Args:
            x: Input data.
            W: Weights.
            b: Bias.
            stride (int): Stride of convolution.
            pad: Padding.

        Returns:
            Convolution output.
        """
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

    def conv_backward_im2col(self, output_grad, x, weights, stride=1, pad=0):
        """
        Perform convolution backward pass using im2col.

        Args:
            output_grad: Gradient of the loss with respect to the output.
            x: Input data.
            weights: Weights.
            stride (int): Stride of convolution.
            pad: Padding.

        Returns:
            Tuple of gradients: (weight_gradient, input_gradient).
        """
        N, F, out_h, out_w = output_grad.shape
        dout_reshaped = output_grad.transpose(0, 2, 3, 1).reshape(-1, F)
        _, C, H, W_in = x.shape
        filter_h, filter_w = weights.shape[2], weights.shape[3]
        col = im2col(x, filter_h, filter_w, stride, pad, xp=self._xp)
        dW_col = self._xp.dot(dout_reshaped.T, col)
        dW = dW_col.reshape(weights.shape)
        W_col = weights.reshape(F, -1)
        dcol = self._xp.dot(dout_reshaped, W_col)
        dx = col2im(dcol, x.shape, filter_h, filter_w, stride, pad, xp=self._xp)
        return dW, dx
    
    def get_parameters(self):
        """
        Get the parameters (weights and bias) of the convolutional layer.

        Returns:
            Tuple of weights and bias.
        """
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        """
        Set the parameters (weights and bias) of the convolutional layer.

        Args:
            weights: New weights.
            bias: New bias.
        """
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
    Flatten layer that reshapes multi-dimensional input to 2D.
    """
    def __init__(self, input_dims=None, device='CPU'):
        """
        Initialize a Flatten layer.

        Args:
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'Flatten'
        self._input_dims = input_dims
        self._output_dims = 1
        for i in self._input_dims:
            self._output_dims*=i
        
    def forward(self, input, is_training=True):
        """
        Flatten the input data.

        Args:
            input: Multi-dimensional input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Flattened 2D array.
        """
        self.input_shape = input.shape
        self.output_shape = (self.input_shape[0], self._output_dims)
        return self._xp.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, optimizer):
        """
        Reshape the gradient back to the original input shape.

        Args:
            output_gradient: Gradient from the next layer.
            optimizer: Optimizer object.

        Returns:
            Reshaped gradient matching the input shape.
        """
        return self._xp.reshape(output_gradient, self.input_shape)
    
class Pool(Layer):
    """
    Pooling layer.
    """
    def __init__(self, pool_size=2, stride=2, method='max', input_dims=None, device='CPU'):
        """
        Initialize a Pooling layer.

        Args:
            pool_size (int): Size of the pooling window.
            stride (int): Stride of the pooling operation.
            method (str): Pooling method ('max' or 'average').
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
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
        """
        Perform the forward pass for pooling.

        Args:
            input: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Output after pooling.
        """
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
        """
        Compute the backward pass for pooling.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
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
    """
    Batch normalization layer.
    """
    def __init__(self, momentum=0.9, epsilon=0.001, input_dims=None, device='CPU'):
        """
        Initialize BatchNormalization layer.

        Args:
            momentum (float): Momentum for running mean and variance.
            epsilon (float): Small constant to avoid division by zero.
            input_dims: Dimensions of the input.
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self._input_dims = input_dims
        self._output_dims = input_dims
        self.num_features = input_dims[0] if isinstance(input_dims, tuple) else input_dims
        self.momentum = momentum
        self.epsilon = epsilon
        self.weights = self._xp.ones(self.num_features)
        self.bias = self._xp.zeros(self.num_features)
        self.running_mean = self._xp.zeros(self.num_features)
        self.running_var = self._xp.ones(self.num_features)
        self.batch_mean = None
        self.batch_var = None
        self.name = 'BatchNorm'

    def forward(self, x, is_training=True):
        """
        Perform the forward pass for batch normalization.

        Args:
            x: Input data.
            is_training (bool): Flag indicating training mode.

        Returns:
            Normalized, scaled, and shifted output.
        """
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
        """
        Compute the backward pass for batch normalization.

        Args:
            output_gradient: Gradient of the loss with respect to the output.
            optimizer: Optimizer object.

        Returns:
            Gradient of the loss with respect to the input.
        """
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

