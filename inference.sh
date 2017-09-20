#!/bin/bash

python2 -u src/main.py --inference --gpu --device 0 --gpu_fraction 0.5\
                                   --data_dir "/mnt/mashijie/data/train_positive" \
                                   --right_dir "/mnt/mashijie/inference/right_data" \
                                   --wrong_dir "/mnt/mashijie/inference/wrong_data" \
                                   --checkpoint_dir "/mnt/mashijie/checkpoint" \
                                   --chinese_dict_dir "./static/chinese.dict"