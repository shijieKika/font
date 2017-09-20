# -*- coding:utf-8 -*-

import numpy as np
import os
import re
import random
from PIL import Image
from chinese_dict import ChineseDict

re_file_name = re.compile(r"uni.+png")
re_chinese = re.compile(r"uni.*_(.*)\.png")


def convert_raw_to_array(image_dir, image_size, image_channel, image_edge, bin_process):
    '''
    :param image_dir: the dir of font.png
    :param image_size: the size of the result
    :param image_channel: the size of the result's channel
    :param image_edge: the size of the result's edge
    :param bin_process: whether to binarize
    :return:
    '''
    inner_size = image_size - 2 * image_edge
    raw_image = np.array(Image.open(image_dir))
    x, y = raw_image.shape
    pad_width = [[(y - x) / 2, (y - x) / 2], [0, 0]] if x < y else [[0, 0], [(x - y) / 2, (x - y) / 2]]
    squared_image = np.lib.pad(raw_image, pad_width, 'constant', constant_values=255)

    scaled_image = np.array(Image.fromarray(np.uint8(squared_image)).resize((inner_size, inner_size)))

    surrounded_image = np.lib.pad(scaled_image, image_edge, 'constant', constant_values=255)

    image = (surrounded_image < 127).astype(int) if bin_process else surrounded_image

    return image.reshape((image_size, image_size, image_channel))


class ImageGallery:
    def __init__(self, src_path, chinese_path, image_size, image_channel, image_edge, bin_process=True):
        self.chinese = ChineseDict(chinese_path)
        self.font_path_list = []
        self.path_image_map = {}
        self.image_size = image_size
        self.image_channel = image_channel
        self.image_edge = image_edge
        self.bin_process = bin_process
        if src_path is not None:
            self.add_data(src_path)

    def add_data(self, src_path):
        count = 0
        for font_dir_name in os.listdir(src_path):
            font_dir_path = os.path.join(src_path, font_dir_name)
            if os.path.isdir(font_dir_path):
                for font_file_name in os.listdir(font_dir_path):
                    if re_file_name.match(font_file_name) == None:
                        continue
                    font_file_path = os.path.join(font_dir_path, font_file_name)
                    self.font_path_list.append(font_file_path)
                    count += 1
        random.shuffle(self.font_path_list)
        return count

    def size(self):
        return len(self.font_path_list)

    def label_size(self):
        return self.chinese.size()

    def shuffle(self):
        random.shuffle(self.font_path_list)

    def get_word(self, id):
        return self.chinese.get_word(id)

    def get_batch(self, start, end):
        start = start if start != None else 0
        end = end if end != None else len(self.font_path_list)
        if start > end:
            print("start > end")
            return None, None, None
        if len(self.font_path_list) == 0:
            print("Num of font files is 0")
            return None, None, None

        start_id = start % len(self.font_path_list)
        start_epoch = start / len(self.font_path_list)
        end_id = end % len(self.font_path_list)
        end_epoch = end / len(self.font_path_list)

        if start_epoch == end_epoch:
            batch_path = self.font_path_list[start_id:end_id]
        else:
            epoch_skip_count = end_epoch - start_epoch - 1
            batch_path = self.font_path_list[start_id:]
            for i in range(epoch_skip_count):
                batch_path += self.font_path_list[:]
            batch_path += self.font_path_list[:end_id]

        font_list = []
        label_list = []
        for font_file_path in batch_path:
            if font_file_path in self.path_image_map:
                font_list.append(self.path_image_map[font_file_path][0])
                label_list.append(self.path_image_map[font_file_path][1])
            else:
                image_array = convert_raw_to_array(font_file_path, self.image_size, self.image_channel, self.image_edge,
                                                   self.bin_process)
                chinese_word = re_chinese.findall(font_file_path)[0]
                label = np.zeros(self.chinese.size())
                label[self.chinese.get_index(chinese_word)] = 1

                self.path_image_map[font_file_path] = (image_array, label)
                font_list.append(image_array)
                label_list.append(label)

        return batch_path, np.array(font_list, dtype=np.float32), np.array(label_list, dtype=np.float32)


def main():
    import time
    a = ImageGallery("/Users/msj/Code/font/data/debug_data/positive_train",
                     '/Users/msj/Code/font/font/static/chinese.dict', 64, 1, 2, bin_process=False)
    start = time.time()
    for i in range(1):
        dir, im, la = a.get_batch(i, i + 1)
        print(dir[0])
        print(im.shape, la.shape)
        Image.fromarray(np.uint8(im[0].reshape((64, 64)))).show()
    print(time.time() - start)


if __name__ == '__main__':
    main()
