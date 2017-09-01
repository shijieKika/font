# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import chinese_dict

from pylab import *
from PIL import Image

re_file_name = re.compile(r"uni.+png")
re_chinese = re.compile(r"uni.*_(.*)\.png")

def convert_raw_to_array(file_path):
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

    im = Image.fromarray(uint8(full_photo)).resize((100, 100))
    return array(im)

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
                font_list.append(convert_raw_to_array(font_file_path))
                chinese_word = re_chinese.findall(font_file_name)[0]
                label_list.append(chinese.getIndex(chinese_word))
    return font_list,label_list

def main():
    # ret = convert_raw_to_array("../../training_data/positive_data/AaBuYu", "uni8CAC_責.png")
    # plt.imshow(ret)
    # plt.show()

    # a, b = load_image("/Users/msj/Code/font/test_data")
    # for i, j in zip(b, a):
    #     print(i, j.shape)
    # print(len(b))

if __name__ == '__main__':
    main()