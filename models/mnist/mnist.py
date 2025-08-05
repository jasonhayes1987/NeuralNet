## IMPORTS ##
from network import Neural_Network
from layer import Dense
from activations import Softmax
from losses import Sparse_Categorical_Cross_Entropy
from optimizers import Adam
from metrics import Accuracy
from utils import get_confusion_matrix, plot_confusion_matrix, train_test_split

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

# Set aside 20% of training data as test data
x_train, y_train, x_test, y_test = train_test_split(x_train, y_train, split=0.2)

print("Normalizing data...")
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# convert data to cupy arrays if device = GPU
if device == 'GPU':
    x_train = classifier._xp.asarray(x_train)
    y_train = classifier._xp.asarray(y_train)
    x_val = classifier._xp.asarray(x_val)
    y_val = classifier._xp.asarray(y_val)
    x_test = classifier._xp.asarray(x_test)
    y_test = classifier._xp.asarray(y_test)

print("Adding channel dimension...")
x_train = classifier._xp.expand_dims(x_train, axis=1)
x_val = classifier._xp.expand_dims(x_val, axis=1)
x_test = classifier._xp.expand_dims(x_test, axis=1)

# Reduce training and validation sets to 1000 samples for testing
x_train = x_train[:64]
y_train = y_train[:64]
x_val = x_val[:64]
y_val = y_val[:64]

# Package data
data = [(x_train, y_train), (x_val, y_val)]
test_data = [(x_test, y_test)]
print("Data preprocessing complete.")
print(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")

## RUN TRAINING ##
print("Starting training...")
classifier.train(data, epochs=2, batch_size=16, plot=True, save_dir="mnist.h5")
print("Training complete.")

## RUN TESTING ##
print("Starting testing...")
classifier.evaluate(test_data)
cm = get_confusion_matrix((x_test, y_test), classifier)
plot_confusion_matrix(cm, 10)
print("Testing complete.")