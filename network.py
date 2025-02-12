
import numpy as np
import cupy as cp
from sklearn.utils import shuffle
import layer as Layer
import losses as Losses
import metrics as Metrics
import optimizers as Optimizers
import activations as Activations
import utils as Utils
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
        
        if visualize:
            # check to see if figure already exists and if not, create it
            if len(plt.get_fignums()) == 0:
                # get all dense layers from network
                dense = [l for l in self.layers if isinstance(l, Layer.Dense)]
                fig = plt.figure(tight_layout=True)
                grid = fig.add_gridspec(1,3)
                wb_grid = grid[0,0:2].subgridspec(nrows=int(np.ceil(len(self.layers)/2)), ncols=2)
                metric_grid = grid[0,2].subgridspec(3,1)

                for index, l in enumerate(self.layers):
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

            # update graph animation with new data
            ani = animation.FuncAnimation(fig, Utils.animate, fargs=(dense, self.metric_data,), interval=200)

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