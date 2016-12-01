"""
Multinomial Logistic Regression using TensorFlow.
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
DESCENT_STEPS = 1000
NORMALIZE = False
BATCH_SIZE = 128
USE_NONSTOCHASTIC_GD = False  # just for fun, slow...
assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
try: 
    rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:  # in case I'm trying to paste into ipython
    rootdir = os.getcwd()
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

X = dataset_dict['X'].astype('float32')
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

# Convert labels to one-hot format
def onehot(data, ordered_label_set):
    return [[(1 if l == y else 0) for l in ordered_label_set] for y in data]
y = np.array(onehot(y, distinct_labels)).astype('float32')
print(y.shape)

# normalize data
if NORMALIZE:
    # X = X - mean(X)
    # X = X./mean(sum((X-mean(X)).^2))
    raise NotImplementedError

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
print("Data type:", X_train.dtype)

graph = tf.Graph()
with graph.as_default():
    # Specify shape of input
    if USE_NONSTOCHASTIC_GD:
        tf_X_train = tf.constant(X_train)
        tf_y_train = tf.constant(y_train)
    else:
        tf_X_train = tf.placeholder(tf.float32, shape=(BATCH_SIZE, n))
        tf_y_train = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 
                                                    len(distinct_labels)))
    tf_X_valid = tf.constant(X_valid)
    tf_X_test = tf.constant(X_test)

    # Specify variables
    weights = tf.Variable(tf.truncated_normal([n, len(distinct_labels)]))
    biases = tf.Variable(tf.zeros([len(distinct_labels)]))

    # Training
    logits = tf.matmul(tf_X_train, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits, tf_y_train))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Make Predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_X_valid, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_X_test, weights) + biases)

def accuracy(predictions, labels):
    num_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return 100.0 * num_correct / predictions.shape[0]

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(DESCENT_STEPS):
        if USE_NONSTOCHASTIC_GD:
            _, l, predictions = session.run([optimizer, loss, 
                                                train_prediction])
        else:

            offset = (step * BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)
            X_batch = X_train[offset:(offset + BATCH_SIZE), :]
            y_batch = y_train[offset:(offset + BATCH_SIZE), :]
            feed_dict = {tf_X_train : X_batch, tf_y_train : y_batch}
            _, l, predictions = session.run([optimizer, loss, 
                                                train_prediction], 
                                            feed_dict=feed_dict)
        if (step % 100 == 0):
            if USE_NONSTOCHASTIC_GD:
                print('Loss at step {}: {}'.format(step, l))
                print('Training accuracy: {:.2f}%'
                      ''.format(accuracy(predictions, y_train)))
            else:
                print('Batch loss at step {}: {}'.format(step, l))
                print('Batch training accuracy: {:.2f}%'
                      ''.format(accuracy(predictions, y_batch)))

            print('Validation accuracy: {:.2f}%'.format(accuracy(
            valid_prediction.eval(), y_valid)))
    print('Test accuracy: {:.2f}%'
          ''.format(accuracy(test_prediction.eval(), y_test)))
