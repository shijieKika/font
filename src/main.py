# -*- coding:utf-8 -*-

import os
import time
import argparse
import tensorflow as tf
import numpy as np
from utils.load_image import ImageGallery
from font_model import FontModel

FLAGS = None


def get_accuracy(labels, predictions, loss_scales):
    data_count = labels.shape[0]
    positive_count = np.sum(np.maximum(loss_scales, 0))
    negative_count = data_count - positive_count

    match_count = np.sum(np.argmax(labels, 1) == np.argmax(predictions, 1))
    positive_match_count = np.sum(np.argmax(labels, 1) == np.argmax(predictions, 1) * loss_scales)
    negative_match_count = match_count - positive_match_count

    positive_p = 0 if positive_count == 0 else 100 * positive_match_count / positive_count
    negative_p = 0 if negative_count == 0 else 100 - 100 * negative_match_count / negative_count

    return positive_p, negative_p


def train():
    graph = tf.Graph()

    with graph.as_default():
        print("Init")
        device = '/gpu:0' if FLAGS.gpu else '/cpu:0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction if FLAGS.gpu else 0.01

        batch_size = FLAGS.batch_size
        train_data_factory = ImageGallery(FLAGS.chinese_dict_dir, FLAGS.image_size,
                                          FLAGS.image_channel,
                                          FLAGS.image_edge)
        train_positive_size = train_data_factory.add_data(FLAGS.train_positive_dir, 1.0)
        train_negative_size = 0 if FLAGS.train_negative_dir is None else train_data_factory.add_data(
            FLAGS.train_negative_dir, -1.0)

        valid_data_factory = ImageGallery(FLAGS.chinese_dict_dir, FLAGS.image_size,
                                          FLAGS.image_channel, FLAGS.image_edge)

        valid_positive_size = valid_data_factory.add_data(FLAGS.valid_positive_dir, 1.0)
        valid_negative_size = valid_data_factory.add_data(FLAGS.valid_negative_dir, -1.0)

        # pre-load valid data
        _, _, _ = valid_data_factory.get_batch(None, None)

        model = FontModel(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_channel, train_data_factory.label_size(),
                          device)

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            print("Run with: %s" % device)
            print("\tbatch_size: %d" % batch_size)
            print("\tepoch_size: %d" % FLAGS.epoch_size)
            print("\ttrain_positive_dir: %s, size %d" % (FLAGS.train_positive_dir, train_positive_size))
            print("\ttrain_negative_dir: %s, size %d" % (FLAGS.train_negative_dir, train_negative_size))
            print("\tvalid_positive_dir: %s, size %d" % (FLAGS.valid_positive_dir, valid_positive_size))
            print("\tvalid_negative_dir: %s, size %d" % (FLAGS.valid_negative_dir, valid_negative_size))
            print("\tcheckpoint_dir: %s, steps_per_checkpoint: %d" % (FLAGS.checkpoint_dir, FLAGS.steps_per_checkpoint))
            print("\tchinese_dict_dir: %s" % FLAGS.chinese_dict_dir)

            total_step = 0
            step_size = train_data_factory.size() / batch_size
            for epoch_step in range(FLAGS.epoch_size):
                train_data_factory.shuffle()
                for step in range(step_size):
                    batch_data, batch_label, batch_loss_scale = train_data_factory.get_batch(step * batch_size,
                                                                                             (step + 1) * batch_size)
                    l, p = model.step(sess, batch_data, batch_label, batch_loss_scale, FLAGS.dropout_prob,
                                      only_forward=False)

                    if total_step % FLAGS.steps_per_checkpoint == 0:
                        if FLAGS.checkpoint_dir != 'None':
                            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'font_model'), global_step=total_step)

                        batch_positive_accuracy, batch_negative_accuracy = get_accuracy(batch_label, p,
                                                                                        batch_loss_scale)
                        message = '%s, Step %d, MiniBatch loss: %.3f, MiniBatch positive accuracy: %02.2f %%, negative accuracy: %02.2f %%' % (
                            time.strftime('%X %x %Z'), total_step, l, batch_positive_accuracy, batch_negative_accuracy)
                        print(message)
                    total_step += 1

                valid_data_factory.shuffle()
                valid_datas, valid_labels, valid_pn_scales = valid_data_factory.get_batch(None, None)
                positive_valid_accuracy, negative_valid_accuracy = model.get_accuracy(sess, valid_datas, valid_labels,
                                                                                      valid_pn_scales)

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
    parser.add_argument("--gpu_fraction", type=float, default=0.95)

    FLAGS = parser.parse_args()
    if FLAGS.train:
        train()
    elif FLAGS.inference:
        inference()
    else:
        test()
