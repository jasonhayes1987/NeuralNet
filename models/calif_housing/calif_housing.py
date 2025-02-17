## IMPORTS ##
from network import Neural_Network
from network import cp, np
from layer import Dense
from activations import Softmax
from losses import Mean_Squared_Error, Loss
from optimizers import Adam
from metrics import Precision_Accuracy, R2, Metric
from utils import train_test_split, Normalizer, to_numpy

# from tensorflow.keras.datasets import california_housing
from sklearn.datasets import fetch_california_housing
import time

print("Starting California Housing training script...")

device = 'GPU'
print(f"Setting device to: {device}")

# set library to numpy or cupy
if device == 'GPU':
    xp = cp
else:
    xp = np

## IMPORT AND FORMAT DATA ##
print("Downloading California Housing dataset...")
start_time = time.time()
(x, y) = fetch_california_housing(return_X_y = True)
print(f"California Housing dataset downloaded in {time.time() - start_time:.2f} seconds.")

# Normalize x and y data
x_norm = Normalizer(device=device)
y_norm = Normalizer(device=device)
x = x_norm.normalize(x)
y = y_norm.normalize(y)

# Set aside 20% of data as test data
x_train, y_train, x_test, y_test = train_test_split(x, y, split=0.2)

# Set aside 20% of training data as validation data
x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, split=0.2)

# convert data to cupy arrays if device = GPU
if device == 'GPU':
    x_train = cp.asarray(x_train)
    y_train = cp.asarray(y_train)
    x_val = cp.asarray(x_val)
    y_val = cp.asarray(y_val)
    x_test = cp.asarray(x_test)
    y_test = cp.asarray(y_test)

# Reduce training and validation sets to 1000 samples for testing
# x_train = x_train[:64]
# y_train = y_train[:64]
# x_val = x_val[:64]
# y_val = y_val[:64]

# Package data
data = [(x_train, y_train), (x_val, y_val)]
test_data = [(x_test, y_test)]
print("Data preprocessing complete.")
print(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}, Test data shape: {x_test.shape}")

## Construct Network ##
print("Building network...")
model = Neural_Network()
loss = Mean_Squared_Error(device=device)
optimizer = Adam(learning_rate=0.001, device=device)
metrics = [Precision_Accuracy(strictness=5, device=device), loss, R2(device=device)]

model.compile_model(loss=loss, optimizer=optimizer, metrics=metrics, device=device)
print("Network compiled successfully.")

# ARCHITECTURE
print("Building network architecture...")

model.add_Dense(128, L1_regularization=0, L2_regularization=1e-4, init='Xavier', input_dims=x_train.shape[-1])
model.add_Activation('relu')
model.add_Dense(128, L1_regularization=0, L2_regularization=1e-4, init='Xavier')
model.add_Activation('relu')
model.add_BatchNorm()


model.add_Dense(64, L1_regularization=0, L2_regularization=1e-4, init='Xavier', input_dims=x_train.shape[-1])
model.add_Activation('relu')
model.add_Dense(64, L1_regularization=0, L2_regularization=1e-4, init='Xavier')
model.add_Activation('relu')
model.add_BatchNorm()


model.add_Dropout(0.1)

model.add_Dense(1, L1_regularization=0, L2_regularization=1e-4, init='He')

print("Network architecture built successfully.")
print("Network layers:")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {layer.name}")

## RUN TRAINING ##
print("Starting training...")
model.train(data, epochs=200, batch_size=1024, plot=True, save_dir="/workspaces/NeuralNet/models/calif_housing/calif.h5")
print("Training complete.")

## RUN TESTING ##
print("Starting testing...")
test_metrics = model.evaluate(test_data)
metric_names = []
for metric in model.metrics:
    if isinstance(metric, Metric) or isinstance(metric, Loss):
        metric_names.append(metric.name)
for metric, metric_name in zip(test_metrics, metric_names):
    print(f'{metric_name}:{metric}')
# Get average error of denormalized test data
predictions = model.predict(x_test)
predictions = y_norm.denormalize(predictions)
targets = y_norm.denormalize(y_test)
y_test = to_numpy(y_test)
predictions = to_numpy(predictions)
avg_diff = np.mean(predictions-y_test)
diff_std = np.std(predictions-y_test)
print(f'Prediction AVG error: ${100_000*avg_diff:.2f}')
print(f'Prediction Std: ${100_000*diff_std:.2f}')
print(f'Prediction Std value range: ${100_000*(avg_diff-diff_std):.2f} -> ${100_000*(avg_diff+diff_std):.2f}')
print("Testing complete.")