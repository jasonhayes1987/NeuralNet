
import numpy as np
import cupy as cp

# stores all loss errors to be passed to the network

class Loss():
    """
    Base class for loss functions.

    Attributes:
        _device (str): Device type ('CPU' or 'GPU').
        _xp: Reference to numpy or cupy based on the device.
    """
    # base class for losses
    def __init__(self, device='CPU'):
        """
        Initialize a Loss.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp

    def __getstate__(self):
        """
        Get state for pickling.
        """
        state = self.__dict__.copy()
        if "_xp" in state:
            del state["_xp"]
        return state

    def __setstate__(self, state):
        """
        Set state after unpickling.
        """
        self.__dict__.update(state)
        if hasattr(self, "_device") and self._device == "GPU":
            self._xp = cp
        else:
            self._xp = np

    def calculate(self, targets, output):
        """
        Calculate the mean loss.

        Args:
            targets: Target values.
            output: Model output.

        Returns:
            Mean loss.
        """
        loss = self.forward(targets, output)
        mean_loss = self._xp.mean(loss)
        return mean_loss
    
    def regularization_loss(self, Layer):
        """
        Calculate the regularization loss (L1 and L2) for a given layer.

        Args:
            Layer: Layer object with regularization attributes.

        Returns:
            Tuple of (L1_loss, L2_loss).
        """
        # initialize regularization loss to 0
        L1_loss = 0
        L2_loss = 0
        
        if Layer.L1_regularizer > 0:
            L1_loss += Layer.L1_regularizer * self._xp.sum(self._xp.abs(Layer.weights))
            L1_loss += Layer.L1_regularizer * self._xp.sum(self._xp.abs(Layer.bias))
            
        if Layer.L2_regularizer > 0:
            L2_loss += Layer.L2_regularizer * self._xp.sum(Layer.weights**2)
            L2_loss += Layer.L2_regularizer * self._xp.sum(Layer.bias**2)
            
        return L1_loss, L2_loss
class Mean_Squared_Error(Loss):
    """
    Mean Squared Error (MSE) loss.
    """
    def __init__(self, device='CPU'):
        """
        Initialize MSE loss.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'MSE'

    def forward(self, targets, output):
        """
        Compute the MSE loss.

        Args:
            targets: Target values.
            output: Model output.

        Returns:
            Squared error loss.
        """
        if targets.ndim == 1:
            targets = self._xp.expand_dims(targets, axis=1)
        output = self._xp.power(output - targets, 2)
        return output

    def backward(self, targets, output):
        """
        Compute the gradient of the MSE loss.

        Args:
            targets: Target values.
            output: Model output.

        Returns:
            Gradient of the loss with respect to the output.
        """
        if targets.ndim == 1:
            targets = self._xp.expand_dims(targets, axis=1)
        self.input_gradient = 2*(output - targets)/targets.size
        return self.input_gradient
class Sparse_Categorical_Cross_Entropy(Loss):
    """
    Sparse Categorical Cross Entropy loss.
    """
    def __init__(self, device='CPU'):
        """
        Initialize Sparse Categorical Cross Entropy loss.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'Sparse CXE'
    
    def forward(self, targets, output):
        """
        Compute the cross entropy loss.

        Args:
            targets: Target class indices.
            output: Model output probabilities.

        Returns:
            Cross entropy loss for each sample.
        """
        targets = targets.flatten().astype(int)
        indices = self._xp.arange(len(output))
        cce = -self._xp.log(output[indices, targets] + 1e-8)
        return cce

    def backward(self, targets, output):
        """
        Compute the gradient of the cross entropy loss.

        Args:
            targets: Target class indices.
            output: Model output probabilities.

        Returns:
            Gradient of the loss with respect to the output.
        """
        targets = targets.flatten().astype(int)  # ensure indices are ints
        samples = output.shape[0]
        grad = self._xp.zeros_like(output)
        indices = self._xp.arange(samples)
        grad[indices, targets] = -1.0 / (output[indices, targets] + 1e-8)
        grad = grad / samples
        self.input_gradient = grad
        return grad

