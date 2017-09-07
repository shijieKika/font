# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def variables_conv(label_size):
    w1 = tf.Variable(tf.truncated_normal([3, 3, 1, 100], stddev=0.1))
    b1 = tf.Variable(tf.zeros([100]))

    w2 = tf.Variable(tf.truncated_normal([2, 2, 100, 200], stddev=0.1))
    b2 = tf.Variable(tf.zeros([200]))

    w3 = tf.Variable(tf.truncated_normal([2, 2, 200, 300], stddev=0.1))
    b3 = tf.Variable(tf.zeros([300]))

    w4 = tf.Variable(tf.truncated_normal([2, 2, 300, 400], stddev=0.1))
    b4 = tf.Variable(tf.zeros([400]))

    w5 = tf.Variable(tf.truncated_normal([4 * 4 * 400, 500], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[500]))

    w6 = tf.Variable(tf.truncated_normal([500, label_size], stddev=0.1))
    b6 = tf.Variable(tf.constant(0.1, shape=[label_size]))

    variables = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6,
                 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6}
    return variables


def model_conv(variables, data, dropout_prob):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer3_conv = tf.nn.conv2d(layer2_pool, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])
    layer3_pool = tf.nn.max_pool(layer3_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer4_conv = tf.nn.conv2d(layer3_pool, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])
    layer4_pool = tf.nn.max_pool(layer4_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    l4_shape = layer4_pool.get_shape().as_list()
    flat_layer = tf.reshape(layer4_pool, [l4_shape[0], l4_shape[1] * l4_shape[2] * l4_shape[3]])

    layer5_full = tf.nn.xw_plus_b(flat_layer, variables['w5'], variables['b5'])
    layer5_relu = tf.nn.relu(layer5_full)

    layer5_drop = tf.nn.dropout(layer5_relu, dropout_prob)

    logits = tf.nn.xw_plus_b(layer5_drop, variables['w6'], variables['b6'])

    return logits


class FontModel:
    def __init__(self,
                 batch_size,
                 image_size,
                 image_channel,
                 label_size):
        self.batch_size = batch_size
        self.input_data = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, image_channel))
        self.input_label = tf.placeholder(tf.float32, shape=(batch_size, label_size))
        self.dropout_prob = tf.placeholder(tf.float32, shape=())
        variables = variables_conv(label_size)

        logits = model_conv(variables, self.input_data, self.dropout_prob)

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
