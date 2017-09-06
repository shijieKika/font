# -*- coding:utf-8 -*-

import os
import time
import argparse
import tensorflow as tf
import numpy as np
from utils.load_image import ImageGallery

np.set_printoptions(threshold='nan')

FLAGS = None

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
    b6 = tf.Variable(tf.constant(0.1, shape=[8877]))

    variables = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6,
                 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6}
    return variables


def model_conv(data, variables, is_train=False):
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

    if is_train:
        layer5_drop = tf.nn.dropout(layer5_relu, 0.5)
    else:
        layer5_drop = layer5_relu

    logits = tf.nn.xw_plus_b(layer5_drop, variables['w6'], variables['b6'])

    return logits


def main():
    graph = tf.Graph()

    with graph.as_default():
        print("Init")
        train_data_factory = ImageGallery(FLAGS.data_dir, FLAGS.chinese_dict_dir)
        valid_data_factory = ImageGallery(FLAGS.valid_dir, FLAGS.chinese_dict_dir)

        train_size = train_data_factory.size()
        batch_size = FLAGS.batch_size
        num_steps = train_size * FLAGS.epoch_num / batch_size
        valid_dataset, valid_labels = valid_data_factory.getBatch(None, None)

        tf_train_dataset = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.image_depth))
        tf_train_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.label_num))

        variables = variables_conv()
        logits = model_conv(tf_train_dataset, variables, True)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        def get_accuracy(sess, dataset, labels):
            steps = dataset.shape[0]
            total_count = 0.0
            for start in range(0, steps, batch_size):
                end = min(start + batch_size, steps)
                data = dataset[start:end]
                label = labels[start:end]
                predictions = sess.run(tf.nn.softmax(model_conv(data, variables)))
                match_count = np.sum(np.argmax(predictions, 1) == np.argmax(label, 1))
                total_count += match_count
            return 100 * total_count / steps

        def get_accuracy1(predictions, labels):
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Run with: ")
            print("\tdata_dir: %s" % FLAGS.data_dir)
            print("\tvalid_dir: %s" % FLAGS.valid_dir)
            print("\tcheckpoint_dir: %s" % FLAGS.checkpoint_dir)
            print("\tchinese_dict_dir: %s" % FLAGS.chinese_dict_dir)
            print("\ttrain_size: %d" % train_size)
            print("\tvalid_size: %d" % valid_data_factory.size())
            print("\tbatch_size: %d" % batch_size)
            print("\tepoch_num: %d" % FLAGS.epoch_num)
            print("\tsteps_per_checkpoint: %d" % FLAGS.steps_per_checkpoint)

            start_time = time.time()
            for step in range(num_steps):
                batch_data, batch_labels = train_data_factory.getBatch(step * batch_size, (step + 1) * batch_size)
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)

                if step % FLAGS.steps_per_checkpoint == 0:
                    # saver.save(sess, FLAGS.checkpoint_dir, global_step=step)

                    batch_accuracy = get_accuracy(sess, batch_data, batch_labels)
                    valid_accuracy = get_accuracy(sess, valid_dataset, valid_labels)

                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    epoch = float(step) * batch_size / train_size
                    avg_time = 1000 * elapsed_time / FLAGS.steps_per_checkpoint

                    message = 'Step %d (epoch %.2f), %.1f ms, MiniBatch loss: %.3f, MiniBatch accuracy: %02.2f %%, Validation accuracy: %02.2f %%' % (
                        step, epoch, avg_time, l, batch_accuracy, valid_accuracy)

                    print(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="need data dir")
    parser.add_argument("--valid_dir", help="need valid dir")
    parser.add_argument("--checkpoint_dir", help="need checkpoint dir")
    parser.add_argument("--chinese_dict_dir", help="need chinese dict dir")
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000)

    # const
    parser.add_argument("--label_num", type=int, default=8877)
    parser.add_argument("--image_width", type=int, default=64)
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--image_depth", type=int, default=1)

    FLAGS = parser.parse_args()
    main()
