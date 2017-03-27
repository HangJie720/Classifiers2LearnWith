""" 
Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 16-layers convolutional network to Oxford's 17 Category Flower
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf

# from utils import split_data, load_data


def split_data(X, Y, testpart, validpart=0, shuffle=True):
    """Split data into training, validation, and test sets.
    Args:
        X: any sliceable iterable
        Y: any sliceable iterable
        validpart: int or float proportion
        testpart: int or float proportion
        shuffle: bool
    """
    import random
    m = len(Y)
    assert (isinstance(testpart, int) and isinstance(validpart, int)) \
        or (0 <= testpart < 1 and 0 <= validpart < 1)


    # shuffle data
    if shuffle:
        permutation = range(m)
        random.shuffle(permutation)
        X = X[permutation]
        Y = Y[permutation]

    if 0 < validpart < 1:
        m_valid = int(validpart * m)
        m_test = int(testpart * m)
        m_train = len(Y) - m_valid - m_test
    else:
        m_valid = validpart
        m_test = testpart
        m_train = m - m_valid - m_test

    X_train = X[:m_train]
    Y_train = Y[:m_train]

    X_valid = X[m_train: m_train + m_valid]
    Y_valid = Y[m_train: m_train + m_valid]

    X_test = X[m_train + m_valid: len(X)]
    Y_test = Y[m_train + m_valid: len(Y)]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


class VGG16:
    def __init__(self):
        self._build_graph()

    def _build_graph(self, resize_by_pad=False):
        # Building 'VGG Network'
        network = input_data(shape=[None, 224, 224, 3])

        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)

        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 17, activation='softmax')

        network = regression(network, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        self.model = tflearn.DNN(network, checkpoint_path='model_vgg',
                            max_checkpoints=1, tensorboard_verbose=0)

    def fit(self, X_inputs, Y_targets, n_epoch=500, validation_set=None,
            show_metric=True, batch_size=32, shuffle=True,
            snapshot_epoch=False, snapshot_step=None, excl_trainops=None,
            run_id=None):
        """ Fit.

        Train model, feeding X_inputs and Y_targets to the network.

        NOTE: When not feeding dicts, data assignations is made by
            input/estimator layers creation order (For example, the second
            input layer created will be feeded by the second value of
            X_inputs list).

        Examples:
            ```python
            model.fit(X, Y) # Single input and output
            model.fit({'input1': X}, {'output1': Y}) # Single input and output
            model.fit([X1, X2], Y) # Mutliple inputs, Single output

            # validate with X_val and [Y1_val, Y2_val]
            model.fit(X, [Y1, Y2], validation_set=(X_val, [Y1_val, Y2_val]))
            # 10% of training data used for validation
            model.fit(X, Y, validation_set=0.1)
            ```

        Arguments:
            X_inputs: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed to train
                model.
            Y_targets: array, `list` of array (if multiple inputs) or `dict`
                (with estimators layer name as keys). Targets (Labels) to
                feed to train model.
            n_epoch: `int`. Number of epoch to run. Default: None.
            validation_set: `tuple`. Represents data used for validation.
                `tuple` holds data and targets (provided as same type as
                X_inputs and Y_targets). Additionally, it also accepts
                `float` (<1) to performs a data split over training data.
            show_metric: `bool`. Display or not accuracy at every step.
            batch_size: `int` or None. If `int`, overrides all network
                estimators 'batch_size' by this value.
            shuffle: `bool` or None. If `bool`, overrides all network
                estimators 'shuffle' by this value.
            snapshot_epoch: `bool`. If True, it will snapshot model at the end
                of every epoch. (Snapshot a model will evaluate this model
                on validation set, as well as create a checkpoint if
                'checkpoint_path' specified).
            snapshot_step: `int` or None. If `int`, it will snapshot model
                every 'snapshot_step' steps.
            excl_trainops: `list` of `TrainOp`. A list of train ops to
                exclude from training process (TrainOps can be retrieve
                through `tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)`).
            run_id: `str`. Give a name for this run. (Useful for Tensorboard).

        """
        # X = self.format_x(X)
        # Y = self.format_y(Y)
        # return self.model.fit(self, X_inputs, Y_targets,
        #                       n_epoch=n_epoch,
        #                       validation_set=validation_set,
        #                       show_metric=show_metric,
        #                       batch_size=batch_size,
        #                       shuffle=shuffle,
        #                       snapshot_epoch=snapshot_epoch,
        #                       snapshot_step=snapshot_step,
        #                       excl_trainops=excl_trainops,
        #                       run_id=run_id)
        return self.model.fit(X, Y, n_epoch=500, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_oxflowers17')

    def predict(self, X):
        """ Predict.

        Model prediction for given input data.

        Arguments:
            X: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed for prediction.

        Returns:
            array or `list` of array. The predicted value.

        """
        return self.model.predict(X)

    def save(self, model_file):
        """ Save.

        Save model weights.

        Arguments:
            model_file: `str`. Model path.

        """
        self.model.save(model_file)

    def load(self, model_file):
        """ Load.

        Restore model weights.

        Arguments:
            model_file: `str`. Model path.

        """
        self.model.load(model_file)

    def get_weights(self, weight_tensor):
        """ Get Weights.

        Get a variable weights.

        Examples:
            ```
            dnn = DNNTrainer(...)
            w = dnn.get_weights(denselayer.W) # get a dense layer weights
            w = dnn.get_weights(convlayer.b) # get a conv layer biases
            ```

        Arguments:
            weight_tensor: `Tensor`. A Variable.

        Returns:
            `np.array`. The provided variable weights.
        """
        return self.model.get_weights(weight_tensor)

    def set_weights(self, tensor, weights):
        """ Set Weights.

        Assign a tensor variable a given value.

        Arguments:
            tensor: `Tensor`. The tensor variable to assign value.
            weights: The value to be assigned.

        """
        self.model.set_weights(tensor, weights)

    def evaluate(self, X, Y, batch_size=128):
        """ Evaluate.

        Evaluate model metric(s) on given samples.

        Arguments:
            X: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed to train
                model.
            Y: array, `list` of array (if multiple inputs) or `dict`
                (with estimators layer name as keys). Targets (Labels) to
                feed to train model. Usually set as the next element of a
                sequence, i.e. for x[0] => y[0] = x[1].
            batch_size: `int`. The batch size. Default: 128.

        Returns:
            The metric(s) score.

        """
        return self.model.evaluate(X, Y, batch_size=batch_size)


if __name__ == '__main__':
    # Data loading and preprocessing
    import tflearn.datasets.oxflower17 as oxflower17
    X, Y = oxflower17.load_data(one_hot=True)
    X_train, Y_train, _, _, X_test, Y_test = \
        split_data(X, Y, .3, 0, shuffle=False)

    vgg = VGG16()
    vgg.fit(X_train, Y_train)

    print(vgg.evaluate(X_test, Y_test))
