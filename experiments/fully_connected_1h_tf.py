"""
Fully Connected Single Hidden Layer Network using TensorFlow.
"""

# Standard Library Dependencies
from __future__ import division, print_function, absolute_import

# External Dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  # use when in JUPYTER NOTEBOOK (or risk hang)
# plt.ion()  # allow ipython %run to terminate without closing figure

# Internal Dependencies
from utils import load_data

# User Parameters
DATASET = '/Users/Andy/Google Drive/Development/ML/kaggle/leaf-classification/KAGGLE_LEAF.mat'
# DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
DESCENT_STEPS = 5000
TESTING_DESCENT_STEPS = 1000000
NORMALIZE = False
HIDDEN_LAYER_NODES = 1024
BATCH_SIZE = 594  # set to zero to use non-stochastic optimizer
DROPOUT = .5
REG_COEFFS2TRY = [0.000125892541179]#[10**k for k in np.arange(-4, -2, 0.1)]
LEARNING_RATE = 0.5
BATCH_LIMIT = 0  # Limit number of batches used for training

assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
d = load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE, one_hot=True)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = d

if BATCH_LIMIT:
    limit = BATCH_LIMIT*BATCH_SIZE
    X_train = X_train[:limit, :]
    Y_train = Y_train[:limit]

### TensorFlow 1-layer fully_connected with (non-stochastic) gradient descent
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_train.shape)
print("Data type:", X_train.dtype)

N_in = X_train.shape[1]
N_1 = HIDDEN_LAYER_NODES
N_out = Y_train.shape[1]


graph = tf.Graph()
with graph.as_default():

    # Inputs to be fed in at each step
    X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N_in))
    Y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, N_out))
    reg_coeff = tf.placeholder('float')

    # hidden layer
    weights_1 = tf.Variable(tf.truncated_normal([N_in, N_1]))
    biases_1 = tf.Variable(tf.zeros([N_1]))
    a1 = tf.nn.relu(tf.matmul(X, weights_1) + biases_1)
    
    # output layer
    weights_f = tf.Variable(tf.truncated_normal([N_1, N_out]))
    biases_f = tf.Variable(tf.zeros([N_out]))
    logits = tf.matmul(a1, weights_f) + biases_f
    
    # Add dropout
    a1_d = tf.nn.dropout(a1, DROPOUT)
    logits_d = tf.matmul(a1_d, weights_f) + biases_f

    # Training Method
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_d, Y)
    regul = reg_coeff*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_f))

    loss = tf.reduce_mean(cross_entropy) + regul
    beta = [weights_1, biases_1, weights_f, biases_f]
    optimizer = tf.train.GradientDescentOptimizer(
                    LEARNING_RATE).minimize(loss, var_list=beta)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
        
    # Setup Validation (to be compared after training with Y_valid)
    tf_X_valid = tf.constant(X_valid)
    a1 = tf.nn.relu(tf.matmul(tf_X_valid, weights_1) + biases_1)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(a1), weights_f) + biases_f)

    # Setup Testing (to be compared after training with Y_test)
    X_test = tf.constant(X_test)
    a1 = tf.nn.relu(tf.matmul(X_test, weights_1) + biases_1)
    test_prediction = tf.nn.softmax(
        (tf.matmul(tf.nn.relu(a1), weights_f) + biases_f))


def accuracy(predictions_, labels_):
    num_correct = np.sum(np.argmax(predictions_, 1) == np.argmax(labels_, 1))
    return 100.0 * num_correct / predictions_.shape[0]

try:  # In case of KeyboardInterrupt, show plots
    best_valid_accuracy = 0
    best_reg_coeff = REG_COEFFS2TRY[0]
    validation_scores = []
    for reg in REG_COEFFS2TRY:
        if len(REG_COEFFS2TRY) == 1:
            break
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print('-'*50)
            print('Training with Regularization Coefficient: ', reg)
            for step in range(DESCENT_STEPS):

                # get batch ready for training step
                if BATCH_SIZE:
                    offset = (step*BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
                    X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                    Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
                else:
                    X_batch = X_train
                    Y_batch = Y_train

                # Training Step
                feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: reg}
                fetches = [optimizer, loss, train_prediction]
                _, l, predictions = session.run(fetches, feed_dict)

                if (step % 100) == 0:
                    train_accuracy = accuracy(predictions, Y_batch)
                    print('Step: {} | Loss: {} | Training Accuracy: {:.2f}%'
                          ''.format(step, l, train_accuracy))
            # Validate
            valid_accuracy = accuracy(valid_prediction.eval(), Y_valid)
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_reg_coeff = reg

            print('Validation score: {:.2f}%'.format(valid_accuracy.eval()))
            validation_scores.append(valid_accuracy)

except KeyboardInterrupt as e:
    print('*'*50)
    print('Winner: reg_coeff', best_reg_coeff)
    print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))
    plt.semilogx(REG_COEFFS2TRY, validation_scores)
    plt.grid(True)
    plt.title('regularization coeff vs validation score\n'
              'best coeff: {}  score: {}'
              ''.format(best_reg_coeff, best_valid_accuracy))
    plt.show()

try:  # In case of KeyboardInterrupt, show plots
    X_train = np.vstack((X_train, X_valid))
    Y_train = np.vstack((Y_train, Y_valid))
    test_scores = []
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(TESTING_DESCENT_STEPS):

            # get batch ready for training step
            if BATCH_SIZE:
                offset = (step*BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
                X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
            else:
                X_batch = X_train
                Y_batch = Y_train

            # Training Step
            feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: best_reg_coeff}
            fetches = [optimizer, loss, train_prediction]
            _, l, predictions = session.run(fetches, feed_dict=feed_dict)

            if (step % 1000) == 0:
                test_accuracy = accuracy(test_prediction.eval(), Y_test)
                print('Step: {} | Loss: {} | Test Set Accuracy: {:.2f}%'
                      ''.format(step, l, test_accuracy))
                test_scores.append((step, test_accuracy))

            # # Get Test Score
            # test_accuracy = accuracy(test_prediction.eval(), Y_valid)
        print('*'*50)
        print('Winner: reg_coeff', best_reg_coeff)
        print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))
    # print("~~Test Accuracy: {:.2f}%".format(best_test_accuracy))

except KeyboardInterrupt:
    if len(REG_COEFFS2TRY) > 1:
        plt.semilogx(REG_COEFFS2TRY, validation_scores)
        plt.grid(True)
        plt.title('regularization coeff vs validation score\n'
                  'best coeff: {}  score: {}'
                  ''.format(best_reg_coeff, best_valid_accuracy))
        plt.show()

    plt.plot(range(TESTING_DESCENT_STEPS), test_scores)
    plt.grid(True)
    plt.title('SGD step vs test score')
    plt.show()
