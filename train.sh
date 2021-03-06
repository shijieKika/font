#!/bin/bash

nohup python2 -u src/main.py --training --gpu --device 0 --gpu_fraction 0.5\
                                        --data_dir "/mnt/mashijie/data/train_positive" \
                                        --valid_positive_dir "/mnt/mashijie/data/valid_positive" \
                                        --valid_negative_dir "/mnt/data/training_data/negative_data" \
                                        --checkpoint_dir "/mnt/mashijie/checkpoint" \
                                        --chinese_dict_dir "./static/chinese.dict" \
                                        > /mnt/mashijie/train.log &
