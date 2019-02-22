#!/usr/bin/env bash
# scrpt to launch several commands at one
python train.py --net='dense' --depth=100 --save_file='dense-100_2' --no_epochs 300 -b 64 --epoch_step '[150,225]' --weight_decay 0.0001 --lr_decay_ratio 0.1 --print_freq=950 --fast_train

python prune.py --net='dense' --depth 100 --base_model='dense-100_2' --save_file='dense-100-fisher-2300' --learning_rate 1e-3 --weight_decay 1e-4 --batch_size 64 --no_epochs 2300 --print_freq=950 --save_every=400 --fast_train

