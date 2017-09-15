# -*- coding:utf-8 -*-

import numpy as np
import os
import re
import random
from PIL import Image
from chinese_dict import ChineseDict

re_file_name = re.compile(r"uni.+png")
re_chinese = re.compile(r"uni.*_(.*)\.png")


class ImageGallery:
    def __init__(self, src_path, chinese_path, image_size, image_channel, image_edge, bin_process=True):
        self.chinese = ChineseDict(chinese_path)
        self.font_path_list = []
        self.path_image_map = {}
        self.image_size = image_size
        self.image_channel = image_channel
        self.inner_size = image_size - 2 * image_edge
        self.lr = np.repeat(255, self.inner_size * image_edge).reshape(self.inner_size, image_edge)
        self.ud = np.repeat(255, image_edge * image_size).reshape(image_edge, image_size)
        self.bin_process = bin_process
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

    def get_batch(self, start, end):
        start = start if start != None else 0
        end = end if end != None else len(self.font_path_list)
        if start > end:
            print("start > end")
            return None, None
        if len(self.font_path_list) == 0:
            print("Num of font files is 0")
            return None, None

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
                image_array = self.convert_raw_to_array(font_file_path, True)
                chinese_word = re_chinese.findall(font_file_path)[0]
                label = np.zeros(self.chinese.size())
                label[self.chinese.getIndex(chinese_word)] = 1

                self.path_image_map[font_file_path] = (image_array, label)
                font_list.append(image_array)
                label_list.append(label)

        return batch_path, np.array(font_list, dtype=np.float32), np.array(label_list, dtype=np.float32)

    def convert_raw_to_array(self, file_path, bin_pro):
        raw_photo = np.array(Image.open(file_path))
        x, y = raw_photo.shape
        if x < y:
            head_count = (y - x) / 2
            head_matrix = np.repeat(255, head_count * y).reshape((head_count, y))
            tail_matrix = np.repeat(255, (y - x - head_count) * y).reshape(y - x - head_count, y)
            full_photo = np.concatenate((head_matrix, raw_photo, tail_matrix), axis=0)
        elif x > y:
            head_count = (x - y) / 2
            head_matrix = np.repeat(255, x * head_count).reshape(x, head_count)
            tail_matrix = np.repeat(255, x * (x - y - head_count)).reshape(x, x - y - head_count)
            full_photo = np.concatenate((head_matrix, raw_photo, tail_matrix), axis=1)
        else:
            full_photo = raw_photo

        result = np.array(Image.fromarray(np.uint8(full_photo)).resize((self.inner_size, self.inner_size)))

        result = np.concatenate((self.lr, result), axis=1)
        result = np.concatenate((result, self.lr), axis=1)
        result = np.concatenate((self.ud, result), axis=0)
        result = np.concatenate((result, self.ud), axis=0)

        if self.bin_process:
            for i in range(self.image_size):
                for j in range(self.image_size):
                    result[i, j] = 0 if result[i, j] > 127 else 1

        return result.reshape((self.image_size, self.image_size, self.image_channel))


def main():
    a = ImageGallery('/Users/msj/Code/font/font/static/chinese.dict', 64, 1, 2, bin_process=False)
    a.add_data("/Users/msj/Code/font/data/debug_data/positive_train", 1.0)
    for i in range(0, 1):
        im, la, pn = a.get_batch(i * 16, (i + 1) * 16)
        print(im.shape, la.shape, pn.shape)
        print(pn)
        Image.fromarray(np.uint8(im[i].reshape((64, 64)))).show()


if __name__ == '__main__':
    main()
