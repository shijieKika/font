#!/usr/bin/sh

nohup python2 -u main.py --train --batch_size 64 --data_dir "/home/mashijie/data/training_data/positive_train" --valid_dir "/home/mashijie/data/training_data/positive_valid" --checkpoint_dir "/home/mashijie/checkpoint" --chinese_dict_dir "/home/mashijie/font/static/chinese.dict" > /home/mashijie/log/train.log &
