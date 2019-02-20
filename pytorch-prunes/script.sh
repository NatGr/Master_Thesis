#!/usr/bin/env bash
# scrpt to launch several commands at one
python train.py --net='res' --save_file='res-40-2-scratch-fisher-600' --print_freq=900 --depth=40 --width=2.0 --list_channels='res-40-2_fisher_600' --bottle

python train.py --net='res' --save_file='res-40-2-scratch-fisher-1100' --print_freq=900 --depth=40 --width=2.0 --list_channels='res-40-2_fisher_1100' --bottle

python prune.py --net='res' --base_model='res-40-2_2' --save_file='res-40-2_2_fisher' --print_freq=900 --save_every=300

python train.py --net='res' --base_file='res-40-2_2_fischer_300_prunes' --prune_train_prune --mask=1 --save_file='res-40-2_2_retrain_300' --print_freq=900
python prune.py --net='res' --resume_ckpt='res-40-2_2_retrain_300' --save_file='res-40-2_2_retrain-300-to-600' --resume  --print_freq=900 --save_every=10000 --no_epochs=600

python train.py --net='res' --base_file='res-40-2-retrain-300-to-600_599_prunes' --prune_train_prune --mask=1 --save_file='res-40-2_2_retrain_600' --print_freq=900
python prune.py --net='res' --resume_ckpt='res-40-2_2_retrain_600' --save_file='res-40-2_2_retrain-600-to-900' --resume  --print_freq=900 --save_every=10000 --no_epochs=900

python train.py --net='res' --base_file='res-40-2-retrain-600-to-900_899_prunes' --prune_train_prune --mask=1 --save_file='res-40-2_2_retrain_900' --print_freq=900
python prune.py --net='res' --resume_ckpt='res-40-2_2_retrain_900' --save_file='res-40-2_2_retrain-900-to-1100' --resume  --print_freq=900 --save_every=10000 --no_epochs=1100

python train.py --net='res' --base_file='res-40-2_2_retrain-900-to-1100_1099_prunes' --prune_train_prune --mask=1 --save_file='res-40-2_2_retrain_1100' --print_freq=900
python prune.py --net='res' --resume_ckpt='res-40-2_2_retrain_1100' --save_file='res-40-2_2_retrain-1100-to-1300' --resume  --print_freq=900 --save_every=10000 --no_epochs=1300

python train.py --net='res' --base_file='res-40-2_2_retrain-1100-to-1300_1299_prunes' --prune_train_prune --mask=1 --save_file='res-40-2_2_retrain_1300' --print_freq=900
