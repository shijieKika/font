# -*- coding:utf-8 -*-

import tensorflow as tf
from utils.load_image import load_image
import numpy as np
import os

np.set_printoptions(threshold='nan')

def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def variables_conv():
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

    w6 = tf.Variable(tf.truncated_normal([500, 8877], stddev=0.1))
    b6 = tf.Variable(tf.constant(0.1, shape = [8877]))

    variables = {'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4, 'w5':w5, 'w6':w6,
                 'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5, 'b6':b6}
    return variables

def model_conv(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1,1,1,1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1,1,1,1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer3_conv = tf.nn.conv2d(layer2_pool, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])
    layer3_pool = tf.nn.max_pool(layer3_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer4_conv = tf.nn.conv2d(layer3_pool, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])
    layer4_pool = tf.nn.max_pool(layer4_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    flat_layer = flatten_tf_array(layer4_pool)
    layer5_full = tf.nn.xw_plus_b(flat_layer, variables['w5'], variables['b5'])
    layer5_relu = tf.nn.relu(layer5_full)
    layer5_drop = tf.nn.dropout(layer5_relu, 0.5)

    logits = tf.nn.xw_plus_b(layer5_drop, variables['w6'], variables['b6'])

    return logits


batch_size = 64
num_labels = 8877

image_width = 64
image_height = 64
image_depth = 1

learning_rate = 0.01
num_steps = 100000

def main():
    graph = tf.Graph()

    with graph.as_default():
        root_path = "/Users/msj/Code/font/debug_data"
        train_dataset, train_labels = load_image(os.path.join(root_path, "train_data"), "utils/chinese.dict")
        test_dataset, test_labels = load_image(os.path.join(root_path, "test_data"), "utils/chinese.dict")

        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        # tf_test_dataset = tf.constant(test_dataset, tf.float32)

        variables = variables_conv()
        logits = model_conv(tf_train_dataset, variables)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model_conv(test_dataset, variables))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Init with learning rate", learning_rate)
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if step % 100 == 0:
                    # saver.save(sess, '/Users/msj/Code/font/tmp/my-model', global_step=step)

                    train_accuracy = accuracy(predictions, batch_labels)
                    test_accuracy = accuracy(test_prediction.eval(), test_labels)
                    message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(
                        step, l, train_accuracy, test_accuracy)
                    print(message)
                    # print(batch_data.reshape((batch_size, 64 * 64)))
                    print(np.argmax(predictions, 1))
                    print(np.argmax(batch_labels, 1))

if __name__ == '__main__':
    main()
