
import numpy as np
import cupy as cp
from sklearn.utils import shuffle
import src.layer as Layer
import src.losses as Losses
import src.metrics as Metrics
import src.optimizers as Optimizers
import src.activations as Activations
from src.utils import build_metric_figure
import pickle
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from tabulate import tabulate

# class that defines and builds a neural network

class Neural_Network():
    """
    Neural Network class for building and training neural networks.

    Attributes:
        layers (list): List of layers in the network.
    """
    def __init__(self):
        """
        Initialize a Neural Network.
        """
        self.layers = []

    def __getstate__(self):
        """
        Get state for pickling.
        """
        state = self.__dict__.copy()
        if "_xp" in state:
            del state["_xp"]
        if "data" in state:
            del state["data"]
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

    def add_Dense(self, output_size, L1_regularization=0, L2_regularization=0, init='Default', input_dims=None):
        """
        Add a Dense layer to the network.

        Args:
            output_size (int): Number of neurons in the Dense layer.
            L1_regularization (float): L1 regularization strength.
            L2_regularization (float): L2 regularization strength.
            init (str): Weight initialization method.
            input_dims: Dimensions of the input (required for the first layer).
        """
        if len(self.layers) == 0:
            if input_dims == None:
                print('input_size param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Dense(output_size=output_size, L1_regularizer=L1_regularization, L2_regularizer=L2_regularization, input_dims=input_dims, device=self._device, init=init))
                
        else:
            self.layers.append(Layer.Dense(output_size=output_size, L1_regularizer=L1_regularization, L2_regularizer=L2_regularization, input_dims=self.layers[-1]._output_dims, device=self._device, init=init))

    def add_Convolutional(self, kernel_size, depth, mode, init='Default', input_dims=None):
        """
        Add a Convolutional layer to the network.

        Args:
            kernel_size (int): Size of the convolution kernel.
            depth (int): Number of filters.
            mode (str): Convolution mode ('valid' or 'same').
            init (str): Weight initialization method.
            input_dims: Dimensions of the input (required for the first layer).
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
        Add a Flatten layer to the network.

        Args:
            input_dims: Dimensions of the input (required for the first layer).
        """
        if len(self.layers) == 0:
            if input_dims == None:
                #ERROR
                print('input_dims param must be given because layer is first in network')
            else:
                self.layers.append(Layer.Flatten(input_dims=input_dims, device=self._device))

        else:
            self.layers.append(Layer.Flatten(input_dims=self.layers[-1]._output_dims, device=self._device))
        
    def add_Dropout(self, dropout_rate, input_dims=None):
        """
        Add a Dropout layer to the network.

        Args:
            dropout_rate (float): Fraction of the input units to drop.
            input_dims: Dimensions of the input (required for the first layer).
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
        Add a Pooling layer to the network.

        Args:
            pool_size (int): Size of the pooling window.
            stride (int): Stride of the pooling operation.
            method (str): Pooling method ('max' or 'average').
            input_dims: Dimensions of the input (required for the first layer).
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
        Add a Batch Normalization layer to the network.

        Args:
            momentum (float): Momentum for running mean and variance.
            epsilon (float): Small constant to avoid division by zero.
            input_dims: Dimensions of the input (required for the first layer).
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
        Add an Activation layer to the network.

        Args:
            activation (str): Activation function name (e.g., 'tanh', 'relu', 'elu', 'gelu', 'sigmoid', 'softmax').
            input_dims: Dimensions of the input (required for the first layer).
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
        Compile the neural network model.

        Args:
            loss: Loss function.
            optimizer: Optimizer.
            metrics: List of metrics.
            device (str): Device type ('CPU' or 'GPU').
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

    def train(self, data, epochs, batch_size, plot=False, save_dir='model.h5'):  
        """
        Train the neural network.

        Args:
            data: List of tuples [(x_train, y_train), (x_val, y_val)].
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            plot (bool): Flag to plot metrics.
            save_dir (str): Path to save the model.
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
                
                # elif metric == Layer.Dense:
                #     for layer in [layer for layer in self.layers if isinstance(layer, Layer.Dense)]:
                #         self.metric_data[layer.name] = []
                        
                # elif metric == Layer.Convolutional:
                #     for layer in [layer for layer in self.layers if isinstance(layer, Layer.Convolutional)]:
                #         self.metric_data[layer.name] = []

        # compute number of batches
        num_batches = np.ceil(self.data[0][0].shape[0] / batch_size).astype(np.int32)

        # Set best accuracy metric to -inf
        best_m = -self._xp.inf

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
                # for metric in self.metrics:
                #     if metric == Layer.Dense:
                #         for layer in [layer for layer in self.layers if isinstance(layer, Layer.Dense)]:
                #              self.metric_data[layer.name].append([layer.weights, layer.bias])
                            
                    # if metric == Layer.Convolutional:
                    #     for layer in [layer for layer in self.layers if isinstance(layer, Layer.Convolutional)]:
                    #         self.metric_data[layer.name].append([layer.kernels, layer.biases])
                            
                    # elif isinstance(metric, Optimizers.Optimizer):
                    #     self.metric_data['Learning Rate'].append(metric.current_learning_rate)
                
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
                        # if (metric is not Layer.Dense) and (metric is not Layer.Convolutional):
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

                print(f"Epoch {e+1} Batch {b+1}")
                print(tabulate(table_rows, headers="keys", tablefmt="pretty"))

            # Check if current calculated metric 'm' is greater than last and if so, save checkpoint
            if m > best_m:
                best_m = m
                self.save(save_dir)
                print(f'model checkpoint saved to {save_dir}')

        if plot:
            fig = build_metric_figure(self.metric_data, num_batches)
            # display(fig) # For notebook rendering
            plt.show()

    def evaluate(self, data):
        """
        Evaluate the neural network on test data.

        Args:
            data: List of tuples [(x_test, y_test), ...].

        Returns:
            List of metric values.
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
        
        # Create list to store metric data
        metrics = []
        # Loop over test data. Make predictions. Calculate metrics
        for x, y in data:
            output = self.predict(x)
            for metric in self.metrics:
                if isinstance(metric, Metrics.Metric) or isinstance(metric, Losses.Loss):
                    m = metric.calculate(y, output)
                    metrics.append(m)
        return metrics
 
    def predict(self, x, batch_size=32):
        """
        Generate predictions for the input data.

        Args:
            x: Input data.
            batch_size (int): Batch size.

        Returns:
            Concatenated predictions.
        """
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
        """
        Get parameters (weights and bias) from Dense layers.

        Returns:
            List of parameters from Dense layers.
        """
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Layer.Dense):
                parameters.append(layer.get_parameters())
        return parameters
     
    def set_parameters(self, parameters):
        """
        Set the network parameters.

        Args:
            parameters: List of parameters for Dense layers.
        """
        for parameter, layer in zip(parameters, [l for l in self.layers if isinstance(l, Layer.Dense)]):
            layer.set_parameters(*parameter)
    
    
    def save_parameters(self, path):
        """
        Save network parameters to a file.

        Args:
            path (str): File path to save parameters.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    
    def load_parameters(self, path):
        """
        Load network parameters from a file.

        Args:
            path (str): File path to load parameters from.
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        """
        Save the entire model to a file.

        Args:
            path (str): File path to save the model.
        """
        model_copy = copy.deepcopy(self)
        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)

    @staticmethod
    def load(path):
        """
        Load a model from a file.

        Args:
            path (str): File path to load the model from.

        Returns:
            Loaded Neural_Network object.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)