# -*- coding:utf-8 -*-

import numpy as np
import os
import re
import random
import chinese_dict

from PIL import Image

re_file_name = re.compile(r"uni.+png")
re_chinese = re.compile(r"uni.*_(.*)\.png")

lr = np.repeat(255, 60 * 2).reshape(60, 2)
ud = np.repeat(255, 2 * 64).reshape(2, 64)


def convert_raw_to_array(file_path, bin_pro=False):
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

    result = np.array(Image.fromarray(np.uint8(full_photo)).resize((60, 60)))

    result = np.concatenate((lr, result), axis=1)
    result = np.concatenate((result, lr), axis=1)
    result = np.concatenate((ud, result), axis=0)
    result = np.concatenate((result, ud), axis=0)

    if bin_pro:
        for i in range(64):
            for j in range(64):
                result[i, j] = 0 if result[i, j] > 127 else 1

    return result.reshape((64, 64, 1))


class ImageGallery:
    def __init__(self, src_path, chinese_path):
        self.chinese = chinese_dict.ChineseDict(chinese_path)
        self.font_path_list = []
        for font_dir_name in os.listdir(src_path):
            font_dir_path = os.path.join(src_path, font_dir_name)
            if os.path.isdir(font_dir_path):
                for font_file_name in os.listdir(font_dir_path):

                    if re_file_name.match(font_file_name) == None:
                        continue
                    font_file_path = os.path.join(font_dir_path, font_file_name)

                    self.font_path_list.append(font_file_path)
        random.shuffle(self.font_path_list)

    def size(self):
        return len(self.font_path_list)

    def getBatch(self, start, end):
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
            font_list.append(convert_raw_to_array(font_file_path, True))
            chinese_word = re_chinese.findall(font_file_path)[0]
            label = np.zeros(8877)
            label[self.chinese.getIndex(chinese_word)] = 1
            label_list.append(label)
        return np.array(font_list, dtype=np.float32), np.array(label_list, dtype=np.float32)


def main():
    np.set_printoptions(threshold='nan')

    # image_path = "/Users/msj/Code/font/data/training_data/positive_data/AaBuYu/uni5DA0_嶠.png"
    # im = convert_raw_to_array(image_path, True)
    # print(im.reshape((64, 64)))
    # Image.fromarray(np.uint8(im.reshape((64, 64)))).show()

    a = ImageGallery("/Users/msj/Code/font/data/debug_data/positive_train", 'chinese.dict')
    for i in range(0, 100):
        im, la = a.getBatch(i * 16, (i + 1) * 16)
        print(im.shape, la.shape)


if __name__ == '__main__':
    main()
