#!/usr/bin/sh

nohup python2 -u train.py --batch_size 64 --steps_per_checkpoint 100 --data_dir "/home/mashijie/data/training_data/positive_train" --valid_dir "/home/mashijie/data/training_data/positive_valid" --checkpoint_dir "/home/mashijie/checkpoint" --chinese_dict_dir "/home/mashijie/font/static/chinese.dict" > /home/mashijie/log/train.log &
