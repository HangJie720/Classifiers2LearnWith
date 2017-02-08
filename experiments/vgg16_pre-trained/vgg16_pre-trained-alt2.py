"""A pre-trained implimentation of VGG16 with weights trained on ImageNet.

NOTE: It's not a great idea to use tf.constant to take in large arrays that will
not change, better to use a non-trainable variable.
https://stackoverflow.com/questions/41150741/in-tensorflow-what-is-the-difference-between-a-constant-and-a-non-trainable-var?rq=1
"""

##########################################################################
# Special thanks to
# http://www.cs.toronto.edu/~frossard/post/vgg16/
# for converting the caffe VGG16 pre-trained weights to TensorFlow
# this file is essentially just a restylized version of his vgg16.py
##########################################################################

from __future__ import print_function, absolute_import, division
import os

import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

_debug = False


def pretrained_conv_layer(name, input_tensor, W, b):
    r"""Creates a convolutional layer with
    Args:
        name: A `str`, the name for the operation defined by this function.
        input_tensor: A `Tensor`.
        diameter: An `int`, the width and also height of the filter.
        in_dim: An `int`, the number of input channels.
        out_dim: An `int`, the number of output channels.
    """
    with tf.name_scope(name):
        weights = tf.constant(W)
        biases = tf.constant(b)

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='convolution')

        preactivations = tf.nn.bias_add(conv, biases, name='bias_addition')
        activations = tf.nn.relu(preactivations, name='activation')
    return activations


def pretrained_fc_layer(name, in_tensor, W, b, sigmoid=tf.nn.relu):
    with tf.name_scope(name):
        weights = tf.constant(W)
        biases = tf.constant(b)

        preactivations = tf.nn.bias_add(tf.matmul(in_tensor, weights), biases)
        activations = sigmoid(preactivations, name='activation')
    return activations


class PreTrainedVGG16:
    def __init__(self, weights=None, session=None):
        if weights is not None and session is not None:
            self._tmp_parameters = self.load_weights(weights)
            # self._tmp_parameters = np.load(weights)
        self.input_images = tf.placeholder(tf.float32, (None, 224, 224, 3))
        self.activations = self._build_graph()
        self.output = self.activations['fc8']

    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = reversed(sorted(weights.keys()))
        return [weights[k] for k in keys]

    @staticmethod
    def get_class_names():
        with open('ImageNet_Classes.txt') as names_file:
            return [l.replace('\n', '') for l in names_file]

    def get_output(self, images, auto_resize=True):
        """"Takes in a list of images and returns softmax probabilities."""
        if auto_resize:
            images_ = [imresize(im, (224, 224)) for im in images]
        else:
            images_ = images
        feed_dict = {self.input_images: images_}
        return sess.run(vgg.output, feed_dict)[0]

    def get_activations(self, images, auto_resize=True):
        """"Takes in a list of images and returns the activation dictionary."""
        if auto_resize:
            images_ = [imresize(im, (224, 224)) for im in images]
        else:
            images_ = images
        feed_dict = {self.input_images: images_}
        return sess.run(vgg.activations, feed_dict)[0]

    def _build_graph(self):

        parameters = []  # storage for trainable parameters

        # pooling arguments
        _ksize = [1, 2, 2, 1]
        _strides = [1, 2, 2, 1]

        # center the input images
        with tf.name_scope('preprocess_centering'):
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                               shape=[1, 1, 1, 3], name='img_mean')
            c_images = self.input_images - mean

        # images --> conv1_1 --> conv1_2 --> pool1
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv1_1 = pretrained_conv_layer('conv1_1', c_images, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv1_2 = pretrained_conv_layer('conv1_2', conv1_1, W,  b)
        pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')

        # pool1 --> conv2_1 --> conv2_2 --> pool2
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv2_1 = pretrained_conv_layer('conv2_1', pool1, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv2_2 = pretrained_conv_layer('conv2_2', conv2_1, W,  b)
        pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')

        # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv3_1 = pretrained_conv_layer('conv3_1', pool2, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv3_2 = pretrained_conv_layer('conv3_2', conv3_1, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv3_3 = pretrained_conv_layer('conv3_3', conv3_2, W,  b)
        pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')

        # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv4_1 = pretrained_conv_layer('conv4_1', pool3, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv4_2 = pretrained_conv_layer('conv4_2', conv4_1, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv4_3 = pretrained_conv_layer('conv4_3', conv4_2, W,  b)
        pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')

        # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv5_1 = pretrained_conv_layer('conv5_1', pool4, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv5_2 = pretrained_conv_layer('conv5_2', conv5_1, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        conv5_3 = pretrained_conv_layer('conv5_3', conv5_2, W,  b)
        pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')

        # pool5 --> flatten --> fc1 --> fc2 --> fc3
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        shape = int(np.prod(pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1 = pretrained_fc_layer('fc6', pool5_flat, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        fc2 = pretrained_fc_layer('fc7', fc1, W,  b)
        W = self._tmp_parameters.pop()
        b = self._tmp_parameters.pop()
        fc3 = pretrained_fc_layer('fc8', fc2, W,  b, tf.nn.softmax)

        activations = {
            'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
            'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
            'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3, 'pool3': pool3,
            'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3, 'pool4': pool4,
            'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3, 'pool5': pool5,
            'fc6': fc1, 'fc7': fc2, 'fc8': fc3
        }
        return activations


if __name__ == '__main__':
    # Get input
    imlist = ['testflash.jpg', 'testme.jpg']*50
    input_images = [imread(f, mode='RGB') for f in imlist]

    # Check 'vgg16_weights.npz exists
    if not os.path.isfile('vgg16_weights.npz'):
        raise Exception(
            "The weights I use here were converted from the Caffe Model Zoo "
            "weights by Davi Frossard.  He didn't include a license so I'm "
            "hesistant to re-post them. Please download them from his "
            "website:\nhttp://www.cs.toronto.edu/~frossard/post/vgg16/")

    # Build VGG16
    from time import time
    a = time()
    if _debug:
        sess = tf.InteractiveSession()
    else:
        sess = tf.Session()
    vgg = PreTrainedVGG16('vgg16_weights.npz', sess)
    print(time() - a)
    a = time()

    # Run images through network, return softmax probabilities

    class_probabilities = vgg.get_output(input_images)
    print(time()-a)

    # Get Class Names
    class_names = vgg.get_class_names()

    # Report results
    top5 = (np.argsort(class_probabilities)[::-1])[0:5]
    with open('results.txt', 'w') as f:
        for p in np.argsort(class_probabilities)[::-1]:
            f.write(str(class_probabilities[p]) + ' : ' + class_names[p] + '\n')

    for p in top5:
        print(class_probabilities[p], ' : ', class_names[p])

# Notes:
# 10: pre0
# 15.4604289532
# 22.6550397873
#
# 10:pre2
# 2.13571715355
# 26.8176388741
#
# 10: pre0
# 22.5506989956
# 165.328907013
#
# 50: pre2
# 2.87836098671
# 173.718732834