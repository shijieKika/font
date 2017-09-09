# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def add_conv(input, filter_height, filter_width, in_channels, out_channels, name):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_channels, out_channels], stddev=0.1))
        biases = tf.Variable(tf.zeros([out_channels]))

        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_relu = tf.nn.relu(pre_activation, name=scope.name)
    return conv_relu


def add_full(input, input_size, output_size, activation, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[output_size]))

        layer_full = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        ret = layer_full if activation is None else activation(layer_full, name=scope.name)
        return ret


def model_conv(data, label_size, dropout_prob, device):
    with tf.device(device):
        conv1 = add_conv(data, 3, 3, 1, 100, 'conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        conv2 = add_conv(pool1, 2, 2, 100, 200, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv(pool2, 2, 2, 200, 300, 'conv3')
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv4 = add_conv(pool3, 2, 2, 300, 400, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p4_shape = pool4.get_shape().as_list()
        dims = p4_shape[1] * p4_shape[2] * p4_shape[3]
        reshape = tf.reshape(pool4, [p4_shape[0], dims])

        local5 = add_full(reshape, dims, 500, tf.nn.relu, 'local5')
        local5_dropout = tf.nn.dropout(local5, dropout_prob)

        local6 = add_full(local5_dropout, 500, label_size, None, 'local6')
        return local6


class FontModel:
    def __init__(self,
                 batch_size,
                 image_size,
                 image_channel,
                 label_size,
                 device):
        with tf.device(device):
            self.batch_size = batch_size
            self.input_data = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, image_channel))
            self.input_label = tf.placeholder(tf.float32, shape=(batch_size, label_size))
            self.dropout_prob = tf.placeholder(tf.float32, shape=())

            logits = model_conv(self.input_data, label_size, self.dropout_prob, device)
            self.prediction = tf.nn.softmax(logits)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_label))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def get_accuracy(self, sess, datas, labels):
        data_count = datas.shape[0]
        match_count = 0.0
        for end in range(self.batch_size, data_count + 1, self.batch_size):
            batch_data = datas[end - self.batch_size:end]
            batch_label = labels[end - self.batch_size:end]
            predictions = sess.run(self.prediction, feed_dict={self.input_data: batch_data, self.dropout_prob: 1.0})
            match_count += np.sum(np.argmax(predictions, 1) == np.argmax(batch_label, 1))
        return 100 * match_count / data_count

    def step(self, sess, batch_data, batch_label, dropout_prob, only_forward):
        input_feed = {}
        input_feed[self.input_data] = batch_data
        input_feed[self.input_label] = batch_label
        input_feed[self.dropout_prob] = dropout_prob

        output_feed = [self.loss, self.prediction]
        if only_forward == False:
            output_feed.append(self.optimizer)

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]
