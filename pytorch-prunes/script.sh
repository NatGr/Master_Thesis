#!/usr/bin/env bash
# scrpt to launch several commands at once

for file in ../morph_net/pickle/*
do
	name=$(echo $file | cut -d'/' -f 4 | cut -d'.' -f 1)
   	python train.py --net='res' --depth=40 --width=2.0 --save_file="morphnet-$name" --print_freq=950 --save_every=400 --channels_morphnet_file=$file
done



