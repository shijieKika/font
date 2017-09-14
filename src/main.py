# -*- coding:utf-8 -*-

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from utils.load_image import ImageGallery
from font_model import FontModel

FLAGS = None


def train():
    graph = tf.Graph()

    with graph.as_default():
        print("Init")
        device = '/gpu:0' if FLAGS.gpu else '/cpu:0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction if FLAGS.gpu else 0.01

        batch_size = FLAGS.batch_size
        train_data_gallery = ImageGallery(FLAGS.train_positive_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
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

        with tf.Session(config=config) as sess:
            model.build_graph(sess, None)
            print("Run with: %s" % device)
            print("\tbatch_size: %d" % batch_size)
            print("\tepoch_size: %d" % FLAGS.epoch_size)
            print("\ttrain_positive_dir: %s, size %d" % (FLAGS.train_positive_dir, train_data_gallery.size()))
            print("\tvalid_positive_dir: %s, size %d" % (FLAGS.valid_positive_dir, valid_positive_gallery.size()))
            print("\tvalid_negative_dir: %s, size %d" % (FLAGS.valid_negative_dir, valid_negative_gallery.size()))
            print("\tcheckpoint_dir: %s, steps_per_checkpoint: %d" % (FLAGS.checkpoint_dir, FLAGS.steps_per_checkpoint))
            print("\tchinese_dict_dir: %s" % FLAGS.chinese_dict_dir)

            global_step = 0
            step_size = train_data_gallery.size() / batch_size
            for epoch_step in range(FLAGS.epoch_size):
                train_data_gallery.shuffle()
                for step in range(step_size):
                    batch_data, batch_label = train_data_gallery.get_batch(step * batch_size,
                                                                           (step + 1) * batch_size)
                    _ = model.step(sess, global_step, batch_data, batch_label, FLAGS.dropout_prob,
                                   only_forward=False)

                    if global_step % FLAGS.steps_per_checkpoint == 0:
                        if FLAGS.checkpoint_dir != 'None':
                            model.save(sess, os.path.join(FLAGS.checkpoint_dir, 'font_model'))

                        batch_loss, batch_accuracy, batch_rate, _ = model.step(sess, global_step, batch_data,
                                                                               batch_label, 1.0, True)

                        message = '%s, Step %d, MiniBatch loss: %.3f, MiniBatch positive accuracy: %02.2f %%, MiniBatch learning rate: %02.6f' % (
                            time.strftime('%X %x %Z'), global_step, batch_loss, 100 * batch_accuracy, batch_rate)
                        print(message)
                    global_step += 1

                valid_positive_gallery.shuffle()
                valid_negative_gallery.shuffle()
                valid_datas, valid_labels = valid_positive_gallery.get_batch(None, None)
                valid_negative_datas, valid_negative_labels = valid_negative_gallery.get_batch(None, None)
                positive_valid_accuracy = model.get_accuracy(sess, valid_datas, valid_labels)
                negative_valid_accuracy = 100 - model.get_accuracy(sess, valid_negative_datas, valid_negative_labels)

                message = '%s, Epoch %d, Validation positive accuracy %02.2f %%, negative accuracy %02.2f %%' % (
                    time.strftime('%X %x %Z'), epoch_step + 1, positive_valid_accuracy, negative_valid_accuracy)
                print(message)


def test():
    print('To test in future')


def inference():
    print('To inference in future')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')

    parser.add_argument("--train_positive_dir", help="Need positive train dir")
    parser.add_argument("--train_negative_dir", help="Need negative train dir", default=None)
    parser.add_argument("--valid_positive_dir", help="Need positive valid dir")
    parser.add_argument("--valid_negative_dir", help="Need negative valid dir")
    parser.add_argument("--checkpoint_dir", default='None', help="Need checkpoint dir")
    parser.add_argument("--chinese_dict_dir", help="Need chinese dict dir")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--steps_per_checkpoint", type=int, default=100)
    parser.add_argument("--dropout_prob", type=float, default=0.5)

    # const
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channel", type=int, default=1)
    parser.add_argument("--image_edge", type=int, default=2)
    parser.add_argument("--gpu_fraction", type=float, default=0.50)
    parser.add_argument("--starter_learning_rate", type=float, default=0.002)
    parser.add_argument("--decay_steps", type=float, default=350)
    parser.add_argument("--decay_rate", type=float, default=0.96)

    FLAGS = parser.parse_args()
    if FLAGS.train:
        train()
    elif FLAGS.inference:
        inference()
    else:
        test()
