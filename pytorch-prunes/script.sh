#!/usr/bin/env bash
# scrpt to launch several commands at once

_2="_2"
for file in ../morph_net/pickle/*
do
	name=$(echo $file | cut -d'/' -f 4 | cut -d'.' -f 1)
   	python train.py --workers=2 --net='res' --depth=40 --width=2.0 --save_file="morphnet-$name$_2" --print_freq=950 --save_every=400 --channels_morphnet_file=$file
done

python train.py --workers=2 --list_channels='res-40-2_fisher_750' --bottle --net='res' --save_file='res-40-2-scratch-fisher-750' --print_freq=950 --save_every=4000

python train.py --workers=2 --list_channels='res-40-2_2_fisher_450' --bottle --net='res' --save_file='res-40-2_2-scratch-fisher-450' --print_freq=950 --save_every=4000
python train.py --workers=2 --list_channels='res-40-2_2_fisher_600' --bottle --net='res' --save_file='res-40-2_2-scratch-fisher-600' --print_freq=950 --save_every=4000
python train.py --workers=2 --list_channels='res-40-2_2_fisher_750' --bottle --net='res' --save_file='res-40-2_2-scratch-fisher-750' --print_freq=950 --save_every=4000
python train.py --workers=2 --list_channels='res-40-2_2_fisher_900' --bottle --net='res' --save_file='res-40-2_2-scratch-fisher-900' --print_freq=950 --save_every=4000
python train.py --workers=2 --list_channels='res-40-2_2_fisher_1100' --bottle --net='res' --save_file='res-40-2_2-scratch-fisher-1100' --print_freq=950 --save_every=4000
