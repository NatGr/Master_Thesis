#!/usr/bin/env bash
# scrpt to launch several commands at once


file_names=('res-40-2-table_fisher-pf=0.8-150ch' 'res-40-2-table_fisher-pf=0.8-300ch' 'res-40-2-table_fisher-pf=0.8-450ch' 'res-40-2-table_fisher-pf=0.8-600ch' 'res-40-2-table_fisher-pf=0.8-750ch' 'res-40-2-table_fisher-pf=0.8-900ch' 'res-40-2-table_fisher-pf=0.8-1100ch')

for file_name in "${file_names[@]}" ; do
    python train.py --save_file=${file_name} --channels_morphnet_file="../NetAdapt/nbr_channels/${file_name}.pickle" >> out.txt
done