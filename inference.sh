#!/bin/bash

python2 -u src/main.py --gpu --inference \
                             --data_dir "/mnt/mashijie/data/train_positive" \
                             --right_dir "/mnt/mashijie/inference/right_data" \
                             --wrong_dir "/mnt/mashijie/inference/wrong_data" \
                             --checkpoint_dir "/mnt/mashijie/checkpoint" \
                             --chinese_dict_dir "/home/mashijie/font/static/chinese.dict"