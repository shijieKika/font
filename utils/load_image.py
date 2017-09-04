# -*- coding:utf-8 -*-

import numpy as np
import os
import re
import chinese_dict

from pylab import *
from PIL import Image

re_file_name = re.compile(r"uni.+png")
re_chinese = re.compile(r"uni.*_(.*)\.png")

lr = np.repeat(255, 60 * 2).reshape(60, 2)
ud = np.repeat(255, 2 * 64).reshape(2, 64)

def convert_raw_to_array(file_path, bin_pro=False):
    raw_photo = array(Image.open(file_path))
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

    result = array(Image.fromarray(uint8(full_photo)).resize((60, 60)))

    result = np.concatenate((lr, result), axis=1)
    result = np.concatenate((result, lr), axis=1)
    result = np.concatenate((ud, result), axis=0)
    result = np.concatenate((result, ud), axis=0)

    if bin_pro:
        for i in range(64):
            for j in range(64):
                result[i, j] = 0 if result[i, j] > 127 else 1

    return result.reshape((64, 64, 1))

def load_image(src_path, chinese_path = "chinese.dict"):
    chinese = chinese_dict.ChineseDict()
    chinese.load(chinese_path)
    font_list = []
    label_list = []
    for font_dir_name in os.listdir(src_path):
        font_dir_path = os.path.join(src_path, font_dir_name)
        if os.path.isdir(font_dir_path):
            for font_file_name in os.listdir(font_dir_path):
                if re_file_name.match(font_file_name) == None:
                    continue
                font_file_path = os.path.join(font_dir_path, font_file_name)
                font_list.append(convert_raw_to_array(font_file_path, True))
                chinese_word = re_chinese.findall(font_file_name)[0]
                label = np.zeros(8877)
                label[chinese.getIndex(chinese_word)] = 1
                label_list.append(label)
    return np.array(font_list, dtype=np.float32),np.array(label_list, dtype=np.float32)

def main():
    np.set_printoptions(threshold='nan')

    image_path = "/Users/msj/Code/font/training_data/positive_data/AaBuYu/uni5DA0_嶠.png"
    im = convert_raw_to_array(image_path, True)
    print(im.reshape((64, 64)))
    Image.fromarray(uint8(im.reshape((64, 64)))).show()

    # a, b = load_image("/Users/msj/Code/font/test_data")
    # print(a.shape)
    # print(b.shape)

    # im = Image.open("/Users/msj/Pictures/mi/Camera/IMG_20160514_105141_HDR.jpg")
    # im.convert('L').show()


    pass

if __name__ == '__main__':
    main()