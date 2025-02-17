# NeuralNet: A Neural Network From Scratch API

NeuralNet is a lightweight neural network library built entirely from scratch in Python. It offers a modular design for constructing, training, and evaluating neural networks using a variety of layers, activation functions, loss functions, optimizers, and metrics. The code supports both CPU (using NumPy) and GPU (using CuPy) computations.

## Features

- **Modular Architecture**: Easily build networks using layers such as Dense, Convolutional, Flatten, Dropout, Pooling, and Batch Normalization.
- **Diverse Activations**: Includes Tanh, ReLU, ELU, GELU, Sigmoid, and Softmax.
- **Multiple Loss Functions**: Supports Mean Squared Error and Sparse Categorical Cross Entropy.
- **Optimizers**: Implements SGD, Adagrad, RMSprop, and Adam.
- **Metrics**: Includes Accuracy, Precision Accuracy, and R2 score for evaluating performance.
- **GPU Support**: Utilize CuPy for GPU acceleration.
- **Example Scripts**: Includes example training scripts for datasets like MNIST and California Housing.

## Installation

### Using Conda

A Conda environment file (`environment.yml`) is provided to simplify installation. To create and activate the environment, run:

```bash
conda env create -f environment.yml
conda activate neuralnet-env
```
This will install all required dependencies, including NumPy, SciPy, scikit-learn, Matplotlib, and (optionally) CuPy for GPU support.

### Using pip
Clone the repository:

```bash
Copy
git clone https://github.com/yourusername/NeuralNet.git
cd NeuralNet
```
Install the required packages:
```bash
Copy
pip install -r requirements.txt
```

Using Docker
A Dockerfile is provided to build a containerized version of the project. To build and run the Docker container:

Build the Docker Image:

```bash
Copy
docker build -t neuralnet .
```
Run the Docker Container:
```bash
Copy
docker run --gpus all -it --rm -p 8888:8888 neuralnet
```
Note: The --gpus all flag is used if you want to run the container with GPU support. Adjust the command based on your system's capabilities and your project needs.

Alternatively, if you are using Visual Studio Code with a devcontainer, the .devcontainer/devcontainer.json file is configured to set up your development environment automatically.

Usage
Building a Model
Here is an example of constructing and training a simple neural network using the library:

```python
Copy
from network import Neural_Network
from layer import Dense
from activations import Relu, Softmax
from losses import Sparse_Categorical_Cross_Entropy
from optimizers import Adam
from metrics import Accuracy

# Initialize the neural network
model = Neural_Network()

# Compile the model with loss, optimizer, and metrics
loss = Sparse_Categorical_Cross_Entropy()
optimizer = Adam(learning_rate=0.001)
metrics = [Accuracy()]
model.compile_model(loss=loss, optimizer=optimizer, metrics=metrics, device='CPU')

# Build the network architecture
model.add_Dense(128, input_dims=784)
model.add_Activation('relu')
model.add_Dense(10)
model.add_Activation('softmax')

# Train the model
# Provide data as [(x_train, y_train), (x_val, y_val)]
model.train(data, epochs=10, batch_size=32, plot=True, save_dir='model.h5')
```

Example Scripts
MNIST Example: See models/mnist/mnist.py for an example of training a model on the MNIST dataset.
California Housing Example: See models/calif_housing/calif_housing.py for a regression example using the California Housing dataset.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
This project was built to demonstrate a neural network implementation from scratch in Python.
