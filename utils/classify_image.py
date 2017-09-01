# -*- coding:utf-8 -*-

import os
import re
import random
import shutil

file_name = re.compile(r"uni.*_.*\.png")

def traversal_data(src_path, train_path, test_path, radio = 0.1):
    for font_dir_name in os.listdir(src_path):
        font_dir_path = os.path.join(src_path, font_dir_name)
        if os.path.isdir(font_dir_path):
            file_set = []
            for font_file_name in os.listdir(font_dir_path):
                if file_name.match(font_file_name) == None:
                    continue
                font_file_path = os.path.join(font_dir_path, font_file_name)
                file_set.append(font_file_path)

            random.shuffle(file_set)
            barriar = len(file_set) * radio

            font_train_path = os.path.join(train_path, font_dir_name)
            font_test_path = os.path.join(test_path, font_dir_name)
            os.makedirs(font_train_path)
            os.makedirs(font_test_path)
            for i, item in enumerate(file_set):
                if i < barriar:
                    shutil.copy(item, font_test_path)
                else:
                    shutil.copy(item, font_train_path)




def main():
    traversal_data('/Users/msj/Code/font/training_data/positive_data', '/Users/msj/Code/font/training_data/positive_train', '/Users/msj/Code/font/training_data/positive_test')
    # l = [i for i in range(10000)]
    # print(l)
    # random.shuffle(l)
    # print(l)
    # random.shuffle(l)
    # print(l)

if __name__ == '__main__':
    main()