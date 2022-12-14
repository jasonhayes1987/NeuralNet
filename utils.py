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
import itertools
from glob import glob

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

def animate(i, layers, metrics):
    plt.cla()
    
    for index, l in enumerate(layers):
        plot1.imshow(l.weights, cmap="seismic", aspect='auto')
        plot2.imshow(np.expand_dims(l.bias, axis=1).T, cmap="seismic", aspect='auto')
        
    # plot metric data
    # plot loss    
    # check if total loss exists in metric data (means network applies regularization)
    if 'Total Loss' in network.metric_data:
        plot_loss.plot(metrics['Total Loss'][:,0], label='total train', color='red', lw=1)
        plot_loss.plot(metrics['Total Loss'][:,1], label='total validation', color='red', lw=1, alpha=0.5)
        # plot regularizations if exist
        if 'L2 Regularization' in network.metric_data:
            plot_loss.plot(metrics['L2 Regularization'][:,0], label='L2 train', color='orange', lw=1)
            plot_loss.plot(metrics['L2 Regularization'][:,1], label='L2 validation', color='orange', lw=1, alpha=0.5)
        if 'L1 Regularization' in network.metric_data:
            plot_loss.plot(metrics['L1 Regularization'][:,0], label='L1 train', color='purple', lw=1)
            plot_loss.plot(metrics['L1 Regularization'][:,1], label='L1 validation', color='purple', lw=1, alpha=0.5)
    plot_loss.plot(metrics['Sparse CXE'][:,0], label='loss train', color='blue', lw=1)
    plot_loss.plot(metrics['Sparse CXE'][:,1], label='loss validation', color='blue', lw=1, alpha=0.5)
    
    # plot accuracy
    plot_accuracy.plot(metrics['Accuracy'][:,0], label='train', color='green', linewidth=1)
    plot_accuracy.plot(metrics['Accuracy'][:,1], label='validation', color='green', alpha=0.3, linewidth=1)
    
    # plot learning rate
    plot_learning_rate.plot(metrics['Learning Rate'][:,0], label='train', color='red', linewidth=1)

def get_confusion_matrix_from_generator(generator, data_path, model, num_images=None, image_size=(100,100)):
    """
    returns a confusion matrix generated from predictions made on images passed through an image data generator

    INPUTS
    generator: generator instance
    data_path: string, path to images to be passed through generator
    model: Model, model used to make predictions on images
    num_images: int, default=None, number of images to make predictions on. if None, will default to len(data_path)
    image_size: tuple (int, int), default=(100,100), size to scale images to

    RETURNS
    cm: confusion matrix
    """
    if num_images == None:
        # images = glob(validation_path + '/*/*.jp*g')
        num_images = len(glob(data_path + '/*/*.jp*g'))
    print('Generating Confusion Matrix', num_images)
    predictions = []
    targets= []
    i = 0
    n_images = 0
    for x, y in generator.flow_from_directory(
        data_path,
        target_size = image_size,
        shuffle = False):
        i += 1
        n_images += len(y)
        if i % 50 == 0:
            print(f'{n_images} images processed.')
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= num_images:
            break    
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