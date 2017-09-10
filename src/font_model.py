# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def add_conv(input, phase, filter_height, filter_width, in_channels, out_channels, name):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_channels, out_channels], stddev=0.1))
        biases = tf.Variable(tf.zeros([out_channels]))

        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_relu = tf.nn.relu(pre_activation, name=scope.name)

        if phase is None:
            return conv_relu
        else:
            return tf.contrib.layers.batch_norm(conv_relu, center=True, scale=True, is_training=phase, scope=scope.name)


def add_full(input, input_size, output_size, activation, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[output_size]))

        layer_full = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        ret = layer_full if activation is None else activation(layer_full, name=scope.name)
        return ret


def model_conv(data, label_size, dropout_prob, device):
    with tf.device(device):
        conv1 = add_conv(data, None, 3, 3, 1, 100, 'conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        conv2 = add_conv(pool1, None, 2, 2, 100, 200, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv(pool2, None, 2, 2, 200, 300, 'conv3')
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv4 = add_conv(pool3, None, 2, 2, 300, 400, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p4_shape = pool4.get_shape().as_list()
        dims = p4_shape[1] * p4_shape[2] * p4_shape[3]
        reshape = tf.reshape(pool4, [p4_shape[0], dims])

        local5 = add_full(reshape, dims, 500, tf.nn.relu, 'local5')
        local5_dropout = tf.nn.dropout(local5, dropout_prob)

        local6 = add_full(local5_dropout, 500, label_size, None, 'local6')
        return local6


def model2_conv(data, phase, label_size, dropout_prob, device):
    with tf.device(device):
        conv1 = add_conv(data, phase, 3, 3, 1, 100, 'conv1')
        conv2 = add_conv(conv1, phase, 2, 2, 100, 100, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv(pool2, phase, 3, 3, 100, 200, 'conv3')
        conv4 = add_conv(conv3, phase, 2, 2, 200, 200, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv5 = add_conv(pool4, phase, 3, 3, 200, 300, 'conv5')
        conv6 = add_conv(conv5, phase, 2, 2, 300, 400, 'conv6')
        pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv7 = add_conv(pool6, phase, 3, 3, 400, 500, 'conv7')
        pool7 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p7_shape = pool7.get_shape().as_list()
        dims = p7_shape[1] * p7_shape[2] * p7_shape[3]
        reshape = tf.reshape(pool7, [p7_shape[0], dims])
        dropout_label = tf.nn.dropout(reshape, dropout_prob)
        local8 = add_full(dropout_label, dims, label_size, None, 'local8')
        return local8


class FontModel:
    def __init__(self,
                 batch_size,
                 image_size,
                 image_channel,
                 label_size,
                 device):
        with tf.device(device):
            self.batch_size = batch_size
            self.input_data = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, image_channel),
                                             name='input_datas')
            self.input_label = tf.placeholder(tf.float32, shape=(batch_size, label_size), name='input_labels')
            self.input_loss_scale = tf.placeholder(tf.float32, shape=(batch_size), name='input_loss_scales')
            # self.input_phase = tf.placeholder(tf.bool, name='input_phase')
            self.dropout_prob = tf.placeholder(tf.float32, shape=())

            logits = model2_conv(self.input_data, None, label_size, self.dropout_prob, device)
            self.prediction = tf.nn.softmax(logits)
            self.loss = tf.reduce_mean(
                self.input_loss_scale * tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_label))

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def get_accuracy(self, sess, datas, labels, loss_scales):
        data_count = loss_scales.shape[0]
        positive_count = np.sum(np.maximum(loss_scales, 0))
        negative_count = data_count - positive_count

        positive_match_count = 0.0
        negative_match_count = 0.0

        for end in range(self.batch_size, data_count + 1, self.batch_size):
            batch_data = datas[end - self.batch_size:end]
            batch_label = labels[end - self.batch_size:end]
            batch_loss_scale = loss_scales[end - self.batch_size:end]
            batch_predictions = sess.run(self.prediction,
                                         feed_dict={self.input_data: batch_data, self.dropout_prob: 1.0})
            match_it = np.sum(np.argmax(batch_label, 1) == np.argmax(batch_predictions, 1))
            positive_match_it = np.sum(np.argmax(batch_label, 1) == np.argmax(batch_predictions, 1) * batch_loss_scale)
            positive_match_count += positive_match_it
            negative_match_count += (match_it - positive_match_it)

        positive_p = 0 if positive_count == 0 else 100 * positive_match_count / positive_count
        negative_p = 0 if negative_count == 0 else 100 - 100 * negative_match_count / negative_count
        return positive_p, negative_p

    def step(self, sess, batch_data, batch_label, batch_loss_scale, dropout_prob, only_forward):
        input_feed = {}
        input_feed[self.input_data] = batch_data
        input_feed[self.input_label] = batch_label
        input_feed[self.input_loss_scale] = batch_loss_scale
        input_feed[self.dropout_prob] = dropout_prob
        # input_feed[self.input_phase] = 0 if only_forward else 1

        output_feed = [self.loss, self.prediction]
        if only_forward == False:
            output_feed.append(self.optimizer)

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]
