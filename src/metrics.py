
# imports
import numpy as np
import cupy as cp


class Metric():
    """
    Base class for metrics.

    Attributes:
        _device (str): Device type ('CPU' or 'GPU').
        _xp: Reference to numpy or cupy based on the device.
    """
    def __init__(self, device='CPU'):
        """
        Initialize a Metric.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        self._device = device
        if self._device == 'CPU':
            self._xp = np
        elif self._device == 'GPU':
            self._xp = cp
    
    def calculate(self, targets, output):
        """
        Calculate the metric value.

        Args:
            targets: Target values.
            output: Model output.

        Returns:
            Metric value.
        """
        pass

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

class Accuracy(Metric):
    """
    Accuracy metric for classification.
    """
    def __init__(self, device='CPU'):
        """
        Initialize Accuracy metric.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'Accuracy'

    def calculate(self, targets, output):
        """
        Calculate accuracy.

        Args:
            targets: True class labels.
            output: Model output probabilities.

        Returns:
            Accuracy value.
        """
        return self._xp.mean(self._xp.argmax(output, axis=1) == targets.flatten())
    
class Precision_Accuracy(Metric):
    """
    Precision Accuracy metric for regression.
    """
    def __init__(self, strictness = 250, device='CPU'):
        """
        Initialize Precision Accuracy metric.

        Args:
            strictness (int): Strictness level for accuracy.
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'Precision_Accuracy'
        self.strictness = strictness
        
    def calculate(self, targets, output):
        """
        Calculate precision accuracy.

        Args:
            targets: True values.
            output: Model predictions.

        Returns:
            Precision accuracy value.
        """
        if targets.ndim == 1:
            targets = self._xp.expand_dims(targets, axis=1)
        return self._xp.mean(self._xp.absolute(output - targets) < (self._xp.std(targets)/self.strictness))
    
class R2(Metric):
    """
    R-squared (R2) metric for regression.
    """
    def __init__(self, device='CPU'):
        """
        Initialize R2 metric.

        Args:
            device (str): Device type ('CPU' or 'GPU').
        """
        super().__init__(device)
        self.name = 'R2'

    def calculate(self, targets, output):
        """
        Calculate R2 score.

        Args:
            targets: True values.
            output: Model predictions.

        Returns:
            R2 score.
        """
        if targets.ndim == 1:
            targets = self._xp.expand_dims(targets, axis=1)
        ss_res = self._xp.sum(self._xp.power(targets-output, 2))
        ss_tot = self._xp.sum(self._xp.power(targets-self._xp.mean(targets), 2))
        return 1-(ss_res/ss_tot)