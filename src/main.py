# -*- coding:utf-8 -*-

import os
import time
import argparse
import shutil
import numpy as np
import tensorflow as tf
from utils.load_image import ImageGallery
from font_model import FontModel

FLAGS = None


def model_evaluate(sess, model, datas, labels):
    data_count = labels.shape[0]
    step_size = data_count / model.batch_size
    total_loss = 0.0
    total_accuracy = 0.0

    for end in range(model.batch_size, data_count + 1, model.batch_size):
        batch_data = datas[end - model.batch_size:end]
        batch_label = labels[end - model.batch_size:end]
        batch_loss, batch_accuracy, _, _ = model.step(sess, batch_data, batch_label, 1.0, True)
        total_loss += batch_loss
        total_accuracy += batch_accuracy

    return total_loss / step_size, total_accuracy / step_size


def build_graph(sess, saver, path):
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh patameters")
        sess.run(tf.global_variables_initializer())


def train():
    graph = tf.Graph()

    with graph.as_default():
        print("Init")
        device = '/gpu:0' if FLAGS.gpu else '/cpu:0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction if FLAGS.gpu else 0.01

        batch_size = FLAGS.batch_size
        train_data_gallery = ImageGallery(FLAGS.data_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
                                          FLAGS.image_channel, FLAGS.image_edge)

        valid_positive_gallery = ImageGallery(FLAGS.valid_positive_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
                                              FLAGS.image_channel, FLAGS.image_edge)

        valid_negative_gallery = ImageGallery(FLAGS.valid_negative_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
                                              FLAGS.image_channel, FLAGS.image_edge)

        # pre-load valid data
        _ = valid_positive_gallery.get_batch(None, None)
        _ = valid_negative_gallery.get_batch(None, None)

        model = FontModel(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_channel, train_data_gallery.label_size(),
                          FLAGS.starter_learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, device)

        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session(config=config) as sess:
            build_graph(sess, saver, FLAGS.checkpoint_dir)

            print("Run with: %s" % device)
            print("\tbatch_size: %d" % batch_size)
            print("\tepoch_size: %d" % FLAGS.epoch_size)
            print("\ttrain_data_dir: %s, size %d" % (FLAGS.data_dir, train_data_gallery.size()))
            print("\tvalid_positive_dir: %s, size %d" % (FLAGS.valid_positive_dir, valid_positive_gallery.size()))
            print("\tvalid_negative_dir: %s, size %d" % (FLAGS.valid_negative_dir, valid_negative_gallery.size()))
            print("\tcheckpoint_dir: %s, steps_per_checkpoint: %d" % (FLAGS.checkpoint_dir, FLAGS.steps_per_checkpoint))
            print("\tstarter_learning_rate: %.6f, decay_steps: %d, decay_rate: %.6f" % (
                FLAGS.starter_learning_rate, FLAGS.decay_steps, FLAGS.decay_rate))

            step_size = train_data_gallery.size() / batch_size
            for epoch_step in range(FLAGS.epoch_size):
                train_data_gallery.shuffle()
                for step in range(step_size):
                    _, batch_data, batch_label = train_data_gallery.get_batch(step * batch_size,
                                                                              (step + 1) * batch_size)
                    _ = model.step(sess, batch_data, batch_label, FLAGS.dropout_prob,
                                   only_forward=False)
                    global_step = sess.run(model.global_step)

                    if global_step % FLAGS.steps_per_checkpoint == 0:
                        if FLAGS.checkpoint_dir is not None:
                            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'font_model'), global_step=global_step)

                        _, batch_data, batch_label = train_data_gallery.get_batch((step + 1) * batch_size,
                                                                                  (step + 2) * batch_size)
                        batch_loss, batch_accuracy, batch_rate, _ = model.step(sess, batch_data,
                                                                               batch_label, 1.0, True)

                        message = '%s, Step %d, MiniBatch loss: %.6f, MiniBatch positive accuracy: %02.2f %%, MiniBatch learning rate: %.6f' % (
                            time.strftime('%X %x %Z'), global_step, batch_loss, 100 * batch_accuracy, batch_rate)
                        print(message)

                valid_positive_gallery.shuffle()
                valid_negative_gallery.shuffle()
                _, valid_datas, valid_labels = valid_positive_gallery.get_batch(None, None)
                _, valid_negative_datas, valid_negative_labels = valid_negative_gallery.get_batch(None, None)

                positive_valid_loss, positive_valid_accuracy = model_evaluate(sess, model, valid_datas, valid_labels)
                _, negative_valid_accuracy = model_evaluate(sess, model, valid_negative_datas, valid_negative_labels)

                message = '%s, Epoch %d, Validation positive loss %.6f, accuracy %02.2f %%, negative accuracy %02.2f %%' % (
                    time.strftime('%X %x %Z'), epoch_step + 1, positive_valid_loss, 100 * positive_valid_accuracy,
                    100 - 100 * negative_valid_accuracy)
                print(message)


def test():
    print('To test in future')


def inference():
    graph = tf.Graph()
    with graph.as_default():
        print("Init")
        device = '/gpu:0' if FLAGS.gpu else '/cpu:0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction if FLAGS.gpu else 0.01

        batch_size = 1
        data_gallery = ImageGallery(FLAGS.data_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
                                    FLAGS.image_channel, FLAGS.image_edge)

        model = FontModel(batch_size, FLAGS.image_size, FLAGS.image_channel, data_gallery.label_size(),
                          FLAGS.starter_learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, device)
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            build_graph(sess, saver, FLAGS.checkpoint_dir)
            print("Run with: %s" % device)
            print("\tdata_dir: %s, size %d" % (FLAGS.data_dir, data_gallery.size()))
            print("\tcheckpoint_dir: %s" % (FLAGS.checkpoint_dir))

            step_size = data_gallery.size()
            total_accuracy = 0.0
            for step in range(step_size):
                temp_path, temp_data, temp_label = data_gallery.get_batch(step, step + batch_size)
                _, temp_accuracy, _, temp_prediction = model.step(sess, temp_data, temp_label, 1.0, True)
                total_accuracy += temp_accuracy
                is_match = (np.argmax(temp_label) == np.argmax(temp_prediction))
                word_dir, word_name = os.path.split(temp_path[0])
                _, font_name = os.path.split(word_dir)

                dst_dir = os.path.join(FLAGS.right_dir if is_match else FLAGS.wrong_dir, font_name)
                if os.path.isdir(dst_dir) == False:
                    os.mkdir(dst_dir)
                shutil.copy(temp_path[0], dst_dir)
                if is_match == False:
                    os.rename(os.path.join(dst_dir, font_name),
                              os.path.join(dst_dir, '(' + data_gallery.get_word(temp_prediction[0]) + ')' + font_name))

            print("Total accuracy: %02.2f %%" % (total_accuracy / step_size * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')

    parser.add_argument("--data_dir", help="Need data dir")
    parser.add_argument("--chinese_dict_dir", help="Need chinese dict dir")
    parser.add_argument("--checkpoint_dir", help="Need checkpoint dir", default=None)

    # for train
    parser.add_argument("--valid_positive_dir", help="Need positive valid dir", default=None)
    parser.add_argument("--valid_negative_dir", help="Need negative valid dir", default=None)

    # for inference
    parser.add_argument("--right_dir", help="Need right dir", default=None)
    parser.add_argument("--wrong_dir", help="Need wrong dir", default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=25)
    parser.add_argument("--steps_per_checkpoint", type=int, default=100)
    parser.add_argument("--dropout_prob", type=float, default=0.5)

    # const
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channel", type=int, default=1)
    parser.add_argument("--image_edge", type=int, default=2)
    parser.add_argument("--gpu_fraction", type=float, default=0.50)
    parser.add_argument("--starter_learning_rate", type=float, default=0.002)
    parser.add_argument("--decay_steps", type=float, default=500)
    parser.add_argument("--decay_rate", type=float, default=0.96)

    FLAGS = parser.parse_args()
    if FLAGS.train:
        train()
    elif FLAGS.inference:
        inference()
    else:
        test()
