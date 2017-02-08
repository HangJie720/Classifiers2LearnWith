"""
Fully Connected Network using TensorFlow.

Note on running (nec. b/c of utils import):
be in root project folder, ipython2, THEN cd experiments
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
DATASET = 'usps'
VALIDATION_PERCENTAGE = .2
TESTING_PERCENTAGE = .2
DESCENT_STEPS = 500
TESTING_DESCENT_STEPS = 3000
NORMALIZE = False
# HIDDEN_LAYER_SIZES = [1024, 2048]  # <-- not working, will fix soon
HIDDEN_LAYER_SIZES = [1024]  
BATCH_SIZE = 128
DROPOUT = .5
# Use [0] for no regularization
# REGULARIZATION = [10**k for k in np.arange(-4, -2, 0.1)]
REGULARIZATION = [0.2, 0.3]
LEARNING_RATE = 0.5
INITIAL_WEIGHT_STDDEV = 0.1
INITIAL_BIAS = 0.1  # why does Google set to 0.1 on MNIST tutorial?
BATCH_LIMIT = 0  # Limit number of batches used for training

assert 0 < VALIDATION_PERCENTAGE + TESTING_PERCENTAGE < 1

# Load dataset
data = load_data(DATASET, VALIDATION_PERCENTAGE, TESTING_PERCENTAGE,
                 one_hot=True)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = [
    np.array(x, dtype='float32') for x in data]

if BATCH_LIMIT:
    limit = BATCH_LIMIT * BATCH_SIZE
    X_train = X_train[:limit, :]
    Y_train = Y_train[:limit]

print("Data shape:", *[x.shape for x in data])
print("Data dtype:", *[x.dtype for x in data])


def accuracy(predictions, labels):
    is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return 100.0 * tf.reduce_mean(tf.cast(is_correct, "float"))


def transform(input_data, weights_, biases_):
    """Feed `input_data` forward through (ReLU*h->softmax) model defined by
    `weights` and `bias`."""
    activations = input_data
    for layer_weights, layer_biases in zip(weights_, biases_)[:-1]:
        activations = tf.nn.relu(
            tf.matmul(activations, layer_weights) + layer_biases
        )
    return tf.nn.softmax(tf.matmul(activations, weights_[-1]) + biases_[-1])


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=INITIAL_WEIGHT_STDDEV)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(INITIAL_BIAS, shape=shape)
    return tf.Variable(initial)


def attach_summaries(var):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights_ = weight_variable([input_dim, output_dim])
            attach_summaries(weights_)
        with tf.name_scope('biases'):
            biases_ = bias_variable([output_dim])
            attach_summaries(biases_)
        with tf.name_scope('WX_plus_b'):
            preactivations = tf.matmul(input_tensor, weights_) + biases_
            tf.summary.histogram('pre_activations', preactivations)

        if act is None:
            return preactivations, weights_, biases_
        else:
            activations = act(preactivations, name='activation')
            tf.summary.histogram('activations', activations)
            return activations, weights_, biases_


if REGULARIZATION == [0]:
    _regularization_on = False
else:
    _regularization_on = True


N_in = X_train.shape[1]
N_out = Y_train.shape[1]
N = [N_in] + HIDDEN_LAYER_SIZES + [N_out]

graph = tf.Graph()
with graph.as_default():
    # optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    # Inputs to be fed in at each step
    with tf.name_scope('input'):
        X = tf.placeholder(X_train.dtype, shape=(BATCH_SIZE, N_in), name='X')
        Y = tf.placeholder(Y_train.dtype, shape=(BATCH_SIZE, N_out), name='Y')

    with tf.name_scope('regularization'):
        reg_coeff = tf.placeholder('float32', name='regularization_coefficient')

    # hidden layer
    weights = []
    biases = []
    a = X
    h = 0
    for n_in, n_out in zip(N[:-1], N[1:]):
        h += 1
        if h != len(N) - 1:
            a, W, b = nn_layer(a, n_in, n_out, 'hidden_layer_'+str(h))
            if DROPOUT:
                with tf.name_scope('dropout'):  # does this do anything here?
                    a = tf.nn.dropout(a, DROPOUT)
        else:
            logits, W, b = nn_layer(a, n_in, n_out, 'output_layer')
            # a = tf.nn.relu(logits)
        weights.append(W)
        biases.append(b)

    with tf.name_scope('regularization'):
        if _regularization_on:
            reg_term = reg_coeff * tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
        else:
            reg_term = 0

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # with tf.name_scope('train'):
    #     train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
    #         cross_entropy)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)

    with tf.name_scope('training'):
        loss = tf.reduce_mean(cross_entropy) + reg_term
        tf.summary.scalar("loss", loss)
        update_W_and_b = optimizer.minimize(loss)

    train_prediction = transform(X, weights, biases)  # can't use eval for this as X is placeholder
    train_accuracy = accuracy(train_prediction, Y)  # can't use eval for this as X is placeholder

    with tf.name_scope('validation'):
        tr_X_valid = tf.constant(X_valid)
        tr_Y_valid = tf.constant(Y_valid)
        valid_prediction = transform(tr_X_valid, weights, biases)
        valid_accuracy = accuracy(valid_prediction, tr_Y_valid)

    with tf.name_scope('testing'):
        tr_X_test = tf.constant(X_test)
        tr_Y_test = tf.constant(Y_test)
        test_prediction = transform(tr_X_test, weights, biases)
        test_accuracy = accuracy(test_prediction, tr_Y_test)

    full_summary = tf.summary.merge_all()


best_valid_accuracy = 0
best_reg_coeff = REGULARIZATION[0]
validation_scores = []
if len(REGULARIZATION) > 1:
    # try:  # In case of KeyboardInterrupt, show plots
    for reg in REGULARIZATION:
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter('fc_output/train', session.graph)
            test_writer = tf.summary.FileWriter('fc_output/test')

            print('-' * 50)
            print('Training with Regularization Coefficient: ', reg)
            for step in range(DESCENT_STEPS):

                # get batch ready for training step
                if BATCH_SIZE:
                    offset = \
                        (step * BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
                    X_batch = X_train[offset:(offset + BATCH_SIZE), :]
                    Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
                else:
                    X_batch = X_train
                    Y_batch = Y_train

                # Training Step
                feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: reg}
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                if step % 100:
                    fetches = [update_W_and_b, full_summary]
                    _, summary = session.run(fetches, feed_dict=feed_dict)
                else:
                    fetches = [update_W_and_b, full_summary, loss,
                               train_accuracy]
                    _, summary, training_loss_, train_accuracy_ = \
                        session.run(fetches, feed_dict=feed_dict)
                    print('Step: {} | Loss: {} | Training Accuracy: {:.2f}%'
                          ''.format(step, training_loss_, train_accuracy_))
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
            # Validate
            valid_accuracy_ = valid_accuracy.eval()
            if valid_accuracy_ > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy_
                best_reg_coeff = reg

            print('Validation score: {:.2f}%'.format(valid_accuracy_))
            validation_scores.append(valid_accuracy_)

    # except KeyboardInterrupt as e:
    #     print(e)
    # finally:
    print('*' * 50)
    print('Winner: reg_coeff', best_reg_coeff)
    print("~~Validation Accuracy: {:.2f}%".format(best_valid_accuracy))
    plt.semilogx(REGULARIZATION, validation_scores)
    plt.grid(True)
    plt.title('regularization coeff vs validation score\n'
              'best coeff: {}  score: {}'
              ''.format(best_reg_coeff, best_valid_accuracy))

# Train further with best regularization coefficient, including validation data
X_train = np.vstack((X_train, X_valid))
Y_train = np.vstack((Y_train, Y_valid))
test_scores = []
# try:  # In case of KeyboardInterrupt, pickle training progress and show plots
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(TESTING_DESCENT_STEPS):

        # get batch ready for training step
        if BATCH_SIZE:
            offset = (step * BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
            X_batch = X_train[offset:(offset + BATCH_SIZE), :]
            Y_batch = Y_train[offset:(offset + BATCH_SIZE), :]
        else:
            X_batch = X_train
            Y_batch = Y_train

        # Training Step
        feed_dict = {X: X_batch, Y: Y_batch, reg_coeff: best_reg_coeff}
        fetches = [update_W_and_b, loss]
        training_loss = session.run(fetches, feed_dict=feed_dict)[1]

        if (step % 1000) == 0:
            test_accuracy_ = test_accuracy.eval()
            print('Step: {} | Loss: {} | Test Set Accuracy: {:.2f}%'
                  ''.format(step, training_loss, test_accuracy_))
            test_scores.append((step, test_accuracy_))

# except KeyboardInterrupt:
if validation_scores:
    plt.semilogx(REGULARIZATION, validation_scores)
    plt.grid(True)
    plt.title('regularization coeff vs validation score\n'
              'best coeff: {}  score: {}'
              ''.format(best_reg_coeff, best_valid_accuracy))
    plt.show()

plt.plot(*zip(*test_scores))
plt.grid(True)
plt.title('SGD step vs test score')
plt.show()
