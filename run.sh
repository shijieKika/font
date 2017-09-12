#!/usr/bin/sh

nohup python2 -u main.py --gpu --train --train_positive_dir "/home/mashijie/data/training_data/train_positive" --valid_positive_dir "/home/mashijie/data/training_data/valid_positive" --valid_negative_dir "/home/mashijie/data/training_data/negative_data" --checkpoint_dir "/home/mashijie/checkpoint" --chinese_dict_dir "/home/mashijie/font/static/chinese.dict" > /home/mashijie/log/train.log &
