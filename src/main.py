# -*- coding:utf-8 -*-

import os
import time
import argparse
import tensorflow as tf
import numpy as np
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
        train_data_factory = ImageGallery(FLAGS.data_dir, FLAGS.chinese_dict_dir, FLAGS.image_size, FLAGS.image_channel,
                                          FLAGS.image_edge)
        valid_data_factory = ImageGallery(FLAGS.valid_dir, FLAGS.chinese_dict_dir, FLAGS.image_size,
                                          FLAGS.image_channel, FLAGS.image_edge)

        # pre-load valid data
        _, _ = valid_data_factory.get_batch(None, None)

        model = FontModel(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_channel, train_data_factory.label_size(),
                          device)

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            print("Run with: %s" % device)
            print("\tdata_dir: %s" % FLAGS.data_dir)
            print("\tvalid_dir: %s" % FLAGS.valid_dir)
            print("\tchinese_dict_dir: %s" % FLAGS.chinese_dict_dir)
            print("\tcheckpoint_dir: %s" % FLAGS.checkpoint_dir)
            print("\ttrain_size: %d" % train_data_factory.size())
            print("\tvalid_size: %d" % valid_data_factory.size())
            print("\tbatch_size: %d" % batch_size)
            print("\tepoch_size: %d" % FLAGS.epoch_size)
            print("\tsteps_per_checkpoint: %d" % FLAGS.steps_per_checkpoint)

            total_step = 0
            step_size = train_data_factory.size() / batch_size
            for epoch_step in range(FLAGS.epoch_size):
                train_data_factory.shuffle()
                for step in range(step_size):
                    batch_data, batch_label = train_data_factory.get_batch(step * batch_size, (step + 1) * batch_size)
                    l, p = model.step(sess, batch_data, batch_label, FLAGS.dropout_prob, only_forward=False)

                    if total_step % FLAGS.steps_per_checkpoint == 0:
                        if FLAGS.checkpoint_dir != 'None':
                            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'font_model'), global_step=total_step)

                        batch_accuracy = model.get_accuracy(sess, batch_data, batch_label)
                        message = '%s, Step %d, MiniBatch loss: %.3f, MiniBatch accuracy: %02.2f %%' % (
                            time.strftime('%X %x %Z'), total_step, l, batch_accuracy)
                        print(message)
                    total_step += 1

                valid_data_factory.shuffle()
                valid_datas, valid_labels = valid_data_factory.get_batch(None, None)
                valid_accuracy = model.get_accuracy(sess, valid_datas, valid_labels)
                message = '%s, Epoch %d, Valid accuracy %02.2f %%' % (
                    time.strftime('%X %x %Z'), epoch_step + 1, valid_accuracy)
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

    parser.add_argument("--data_dir", help="Need data dir")
    parser.add_argument("--valid_dir", help="Need valid dir")
    parser.add_argument("--checkpoint_dir", default='None', help="Need checkpoint dir")
    parser.add_argument("--chinese_dict_dir", help="Need chinese dict dir")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--steps_per_checkpoint", type=int, default=500)
    parser.add_argument("--dropout_prob", type=float, default=0.5)

    # const
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channel", type=int, default=1)
    parser.add_argument("--image_edge", type=int, default=2)
    parser.add_argument("--gpu_fraction", type=float, default=0.95)

    FLAGS = parser.parse_args()
    if FLAGS.train:
        train()
    elif FLAGS.inference:
        inference()
    else:
        test()
