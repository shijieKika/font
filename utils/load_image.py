# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import re

from pylab import *
from PIL import Image

file_name = re.compile(r"uni.*_.*\.png")

def convert_raw_to_array(dir_path, file_path):
    path = os.path.join(dir_path, file_path)
    raw_photo = array(Image.open(path))
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

def traversal_data(root_path):
    i = 0
    list_dirs = os.walk(root_path)
    for root, _, files in list_dirs:
        j = 0
        for f in files:
            if file_name.match(f) == None:
                continue
            i += 1
            j += 1
        print(root, j)
    print(root_path, i)

def main():
    # ret = convert_raw_to_array("../../training_data/positive_data/AaBuYu", "uni8CAC_責.png")
    # plt.imshow(ret)
    # plt.show()

    traversal_data('/Users/msj/Code/font/training_data/positive_data')

if __name__ == '__main__':
    main()