#!/usr/bin/sh

nohup python2 -u main.py --gpu --train --train_positive_dir "/mnt/mashijie/data/train_positive" --valid_positive_dir "/mnt/mashijie/data/valid_positive" --valid_negative_dir "/mnt/data/training_data/negative_data" --checkpoint_dir "/mnt/mashijie/checkpoint" --chinese_dict_dir "/home/mashijie/font/static/chinese.dict" > /mnt/mashijie/train.log &
