"""
TensorFlow 1-layer fully_connected with (non-stochastic) gradient descent.
Credit: Some of this code is taken from Udacity TensorFlow course.
"""

# Standard Library Dependencies
from __future__ import division, print_function
from warnings import warn
from time import time
import os
from operator import itemgetter

# External Dependencies
import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm
import tensorflow as tf

# User Parameters
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
num_steps = 801
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(rootdir, 'data')
if DATASET.lower() == 'usps':
    dataset_dict = loadmat(os.path.join(datadir, 'usps', 'USPS.mat'))
elif DATASET.lower() == 'mnist':
    dataset_dict = loadmat(os.path.join(datadir, 'mnist', 'MNIST.mat'))
elif DATASET.lower() == 'notmnist_small':
    dataset_dict = loadmat(os.path.join(datadir, 'notMNIST', 
                            'notMNIST_small_no_duplicates.mat'))
elif DATASET.lower() == 'notmnist_large':
    dataset_dict = loadmat(os.path.join(datadir, 'notMNIST', 
                            'notMNIST_large_no_duplicates.mat'))
else:
    dataset_dict = loadmat(DATASET)

X = dataset_dict['X']
y = dataset_dict['y'].ravel()
distinct_labels = set(y)

# Flatten images
m, h, w = X.shape
n = w*h
X = X.reshape((m, n))

# shuffle data
from random import shuffle
permutation = range(m)
shuffle(permutation)
X = X[permutation]
y = y[permutation]

# normalize data
# X = X - mean(X);  % hurts accuracy
# X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

m_train = int(0.6*m)
X_train = X[:m_train]
y_train = y[:m_train]

m_valid = int(0.2*m)
X_valid = X[m_train : m_train + m_valid + 1]
y_valid = y[m_train : m_train + m_valid + 1]

m_test = m - m_valid - m_train
X_test = X[m_train + m_valid : len(X) + 1]
y_test = y[m_train + m_valid : len(y) + 1]


### TensorFlow 1-layer fully_connected with (non-stochastic) gradient descent
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data dtype:", X_train.dtype)

graph = tf.Graph()
with graph.as_default():
    # Specify shape of input
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Specify variables
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Make Predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
                        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(
                        tf.matmul(tf_test_dataset, weights) + biases)



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
        print('Loss at step {}: {}'.format((step, l)))
        print('Training accuracy: {:.2%}}'.format(accuracy(
        predictions, train_labels[:train_subset, :])))

        print('Validation accuracy: {:.2%}}'.format(accuracy(
        valid_prediction.eval(), valid_labels)))
    print('Test accuracy: {:.2%}'.format(accuracy(test_prediction.eval(), 
                                        test_labels)))

