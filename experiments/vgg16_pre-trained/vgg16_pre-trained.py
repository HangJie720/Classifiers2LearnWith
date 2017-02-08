"""A pre-trained implimentation of VGG16 with weights trained on ImageNet."""

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

_debug = True


def conv_layer(input_tensor, diameter, in_dim, out_dim, name=None):
    r"""Creates a convolutional layer with
    Args:
        input_tensor: A `Tensor`.
        diameter: An `int`, the width and also height of the filter.
        in_dim: An `int`, the number of input channels.
        out_dim: An `int`, the number of output channels.
        name: A `str`, the name for the operation defined by this function.
    """
    with tf.name_scope(name):
        filter_shape = (diameter, diameter, in_dim, out_dim)
        initial_weights = tf.truncated_normal(filter_shape, stddev=0.1)
        weights = tf.Variable(initial_weights, name='weights')

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='convolution')

        initial_biases = tf.constant(1.0, shape=[out_dim], dtype=tf.float32)
        biases = tf.Variable(initial_biases, name='biases')

        preactivations = tf.nn.bias_add(conv, biases, name='bias_addition')
        activations = tf.nn.relu(preactivations, name='activation')
    return activations, weights, biases


def fc_layer(in_tensor, in_dim, out_dim, sigmoid=tf.nn.relu, name=None):
    r"""Creates a fully-connected (ReLU by default) layer with
    Args:
        in_tensor: A `Tensor`.
        in_dim: An `int`, the number of input channels.
        out_dim: An `int`, the number of output channels.
        sigmoid: A `function`, the activation operation, defaults to tf.nn.relu.
        name: A `str`, the name for the operation defined by this function.
    """
    with tf.name_scope(name):
        initial_weights = tf.truncated_normal((in_dim, out_dim), stddev=0.1)
        weights = tf.Variable(initial_weights, name='weights')

        initial_biases = tf.constant(0.0, shape=[out_dim], dtype=tf.float32)
        biases = tf.Variable(initial_biases, name='biases')

        preactivations = tf.nn.bias_add(tf.matmul(in_tensor, weights), biases)
        activations = sigmoid(preactivations, name='activation')
    return activations, weights, biases


class PreTrainedVGG16:
    def __init__(self, weights=None, session=None):
        self.input_images = tf.placeholder(tf.float32, (None, 224, 224, 3))
        self.activations, self.parameters = self._build_graph()
        self.output = self.activations['fc3']
        if weights is not None and session is not None:
            self.load_weights(weights, session)

    def load_weights(self, weight_file, session):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            session.run(self.parameters[i].assign(weights[k]))

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
        print("hi", tf.shape(c_images))
        conv1_1, weights1, biases1 = conv_layer(c_images, 3, 3, 64, 'conv1_1')
        conv1_2, weights2, biases2 = conv_layer(conv1_1, 3, 64, 64, 'conv1_2')
        pool1 = tf.nn.max_pool(conv1_2, _ksize, _strides, 'SAME', name='pool1')
        parameters += [weights1, biases1, weights2, biases2]

        # pool1 --> conv2_1 --> conv2_2 --> pool2
        conv2_1, weights1, biases1 = conv_layer(pool1, 3, 64, 128, 'conv2_1')
        conv2_2, weights2, biases2 = conv_layer(conv2_1, 3, 128, 128, 'conv2_2')
        pool2 = tf.nn.max_pool(conv2_2, _ksize, _strides, 'SAME', name='pool2')
        parameters += [weights1, biases1, weights2, biases2]

        # pool2 --> conv3_1 --> conv3_2 --> conv3_3 --> pool3
        conv3_1, weights1, biases1 = conv_layer(pool2, 3, 128, 256, 'conv3_1')
        conv3_2, weights2, biases2 = conv_layer(conv3_1, 3, 256, 256, 'conv3_2')
        conv3_3, weights3, biases3 = conv_layer(conv3_2, 3, 256, 256, 'conv3_3')
        pool3 = tf.nn.max_pool(conv3_3, _ksize, _strides, 'SAME', name='pool3')
        parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

        # pool3 --> conv4_1 --> conv4_2 --> conv4_3 --> pool4
        conv4_1, weights1, biases1 = conv_layer(pool3, 3, 256, 512, 'conv4_1')
        conv4_2, weights2, biases2 = conv_layer(conv4_1, 3, 512, 512, 'conv4_2')
        conv4_3, weights3, biases3 = conv_layer(conv4_2, 3, 512, 512, 'conv4_3')
        pool4 = tf.nn.max_pool(conv4_3, _ksize, _strides, 'SAME', name='pool4')
        parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

        # pool4 --> conv5_1 --> conv5_2 --> conv5_3 --> pool5
        conv5_1, weights1, biases1 = conv_layer(pool4, 3, 512, 512, 'conv5_1')
        conv5_2, weights2, biases2 = conv_layer(conv5_1, 3, 512, 512, 'conv5_2')
        conv5_3, weights3, biases3 = conv_layer(conv5_2, 3, 512, 512, 'conv5_3')
        pool5 = tf.nn.max_pool(conv5_3, _ksize, _strides, 'SAME', name='pool5')
        parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

        # pool5 --> flatten --> fc1 --> fc2 --> fc3
        shape = int(np.prod(pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1, weights1, biases1 = fc_layer(pool5_flat, shape, 4096, name='fc1')
        fc2, weights2, biases2 = fc_layer(fc1, 4096, 4096, name='fc2')
        fc3, weights3, biases3 = fc_layer(fc2, 4096, 1000, tf.nn.softmax, 'fc3')
        parameters += [weights1, biases1, weights2, biases2, weights3, biases3]

        activations = {
            'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
            'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
            'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3, 'pool3': pool3,
            'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3, 'pool4': pool4,
            'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3, 'pool5': pool5,
            'fc1': fc1, 'fc2': fc2, 'fc3': fc3
        }
        return activations, parameters


if __name__ == '__main__':
    # Get input
    imlist = ['testflash.jpg', 'testme.jpg']
    im_names = [os.path.splitext(os.path.basename(imf))[0] for imf in imlist]
    input_images = [imread(f, mode='RGB') for f in imlist]

    # Check 'vgg16_weights.npz exists
    if not os.path.isfile('vgg16_weights.npz'):
        raise Exception(
            "The weights I use here were converted from the Caffe Model Zoo "
            "weights by Davi Frossard.  He didn't include a license so I'm "
            "hesistant to re-post them. Please download them from his "
            "website:\nhttp://www.cs.toronto.edu/~frossard/post/vgg16/")

    # Build VGG16
    if _debug:
        sess = tf.InteractiveSession()
        tf.summary.FileWriter('TensorBoard', sess.graph)
    else:
        sess = tf.Session()
    vgg = PreTrainedVGG16('vgg16_weights.npz', sess)

    # Run images through network, return softmax probabilities
    class_probabilities = vgg.get_output(input_images)
    print(class_probabilities.shape)

    # Get Class Names
    class_names = vgg.get_class_names()
    
#NOTE: only one file at a time is working... must fix

    # Report results
    # for imf, cps in zip(imlist, class_probabilities_list):
    imf = im_names[0]
    print("Top Five Results for", imf + ':')
    top5 = (np.argsort(class_probabilities)[::-1])[0:5]
    with open(imf + '_results.txt', 'w') as fout:
        for p in np.argsort(class_probabilities)[::-1]:
            fout.write(str(class_probabilities[p]) + ' : ' + class_names[p] + '\n')

    for p in top5:
        print(class_probabilities[p], ' : ', class_names[p])
