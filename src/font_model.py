# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def add_conv(input, filter_height, filter_width, in_channels, out_channels, name):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_channels, out_channels], stddev=0.1))
        biases = tf.Variable(tf.zeros([out_channels]))

        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(pre_activation, name=scope.name)


def add_conv_bn(input, phase, filter_height, filter_width, in_channels, out_channels, name):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_channels, out_channels], stddev=0.1))

        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        with tf.device('/cpu:0'):
            conv_bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn')
        return tf.nn.relu(conv_bn, name=scope.name)


def add_full(input, input_size, output_size, activation, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[output_size]))

        layer_full = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)
        ret = layer_full if activation is None else activation(layer_full, name=scope.name)
        return ret


def model_conv_base(device, data, label_size, dropout_prob):
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


def model_conv_base_bn(device, data, label_size, dropout_prob, phase):
    with tf.device(device):
        conv1 = add_conv_bn(data, phase, 3, 3, 1, 100, 'conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

        conv2 = add_conv_bn(pool1, phase, 2, 2, 100, 200, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv_bn(pool2, phase, 2, 2, 200, 300, 'conv3')
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv4 = add_conv_bn(pool3, phase, 2, 2, 300, 400, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p4_shape = pool4.get_shape().as_list()
        dims = p4_shape[1] * p4_shape[2] * p4_shape[3]
        reshape = tf.reshape(pool4, [p4_shape[0], dims])

        local5 = add_full(reshape, dims, 500, tf.nn.relu, 'local5')
        local5_dropout = tf.nn.dropout(local5, dropout_prob)

        local6 = add_full(local5_dropout, 500, label_size, None, 'local6')
        return local6


def model_conv_deep(device, data, label_size, dropout_prob):
    with tf.device(device):
        conv1 = add_conv(data, 5, 5, 1, 64, 'conv1')
        conv2 = add_conv(conv1, 3, 3, 64, 128, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv(pool2, 3, 3, 128, 256, 'conv3')
        conv4 = add_conv(conv3, 2, 2, 256, 256, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv5 = add_conv(pool4, 3, 3, 256, 512, 'conv5')
        conv6 = add_conv(conv5, 2, 2, 512, 512, 'conv6')
        pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv7 = add_conv(pool6, 3, 3, 512, 512, 'conv7')
        pool7 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p7_shape = pool7.get_shape().as_list()
        dims = p7_shape[1] * p7_shape[2] * p7_shape[3]
        reshape = tf.reshape(pool7, [p7_shape[0], dims])
        dropout_label = tf.nn.dropout(reshape, dropout_prob)
        local8 = add_full(dropout_label, dims, label_size, None, 'local8')
        return local8


def model_conv_bn(device, data, label_size, phase):
    with tf.device(device):
        conv1 = add_conv_bn(data, phase, 5, 5, 1, 128, 'conv1')
        conv2 = add_conv_bn(conv1, phase, 3, 3, 128, 128, 'conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv3 = add_conv_bn(pool2, phase, 5, 5, 128, 256, 'conv3')
        conv4 = add_conv_bn(conv3, phase, 3, 3, 256, 256, 'conv4')
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv5 = add_conv_bn(pool4, phase, 5, 5, 256, 512, 'conv5')
        conv6 = add_conv_bn(conv5, phase, 3, 3, 512, 512, 'conv6')
        pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv7 = add_conv_bn(pool6, phase, 3, 3, 512, 512, 'conv7')
        pool7 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        p7_shape = pool7.get_shape().as_list()
        dims = p7_shape[1] * p7_shape[2] * p7_shape[3]
        reshape = tf.reshape(pool7, [p7_shape[0], dims])
        local8 = add_full(reshape, dims, label_size, None, 'local8')
        return local8


class FontModel:
    def __init__(self,
                 batch_size,
                 image_size,
                 image_channel,
                 label_size,
                 starter_learning_rate,
                 decay_steps,
                 decay_rate,
                 device):
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.global_variables())
        with tf.device(device):
            self.batch_size = batch_size
            self.input_data = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, image_channel),
                                             name='input_datas')
            self.input_label = tf.placeholder(tf.float32, shape=(batch_size, label_size), name='input_labels')
            self.input_phase = tf.placeholder(tf.bool, name='input_phase')
            self.dropout_prob = tf.placeholder(tf.float32, shape=())

            # logits = model_conv_base(device, self.input_data, label_size, self.dropout_prob)
            # logits = model_conv_base_bn(device, self.input_data, label_size, self.dropout_prob, self.input_phase)
            logits = model_conv_deep(device, self.input_data, label_size, self.dropout_prob)
            # logits = model_conv_bn(device, self.input_data, label_size, self.input_phase)

            self.prediction = tf.nn.softmax(logits)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_label))

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.input_label, 1), tf.argmax(logits, 1)), 'float32'))

            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                            decay_steps, decay_rate, staircase=True)

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                      global_step=self.global_step)

    def build_graph(self, sess, checkpoint_dir):
        if checkpoint_dir is None:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            return
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

    def save(self, sess, checkpoint_dir):
        self.saver.save(sess, checkpoint_dir, global_step=self.global_step)

    def get_accuracy(self, sess, datas, labels):
        data_count = labels.shape[0]
        match_count = 0.0

        for end in range(self.batch_size, data_count + 1, self.batch_size):
            batch_data = datas[end - self.batch_size:end]
            batch_label = labels[end - self.batch_size:end]
            batch_predictions = sess.run(self.prediction,
                                         feed_dict={self.input_data: batch_data, self.dropout_prob: 1.0,
                                                    self.input_phase: False})
            match_it = np.sum(np.argmax(batch_label, 1) == np.argmax(batch_predictions, 1))
            match_count += match_it

        p = 0 if data_count == 0 else 100 * match_count / data_count
        return p

    def step(self, sess, global_step, batch_data, batch_label, dropout_prob, only_forward):
        self.global_step = global_step
        input_feed = {}
        input_feed[self.input_data] = batch_data
        input_feed[self.input_label] = batch_label
        input_feed[self.dropout_prob] = dropout_prob
        input_feed[self.input_phase] = False if only_forward else True

        output_feed = [self.loss, self.accuracy, self.learning_rate, self.prediction]
        if only_forward == False:
            output_feed.append(self.optimizer)

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]
