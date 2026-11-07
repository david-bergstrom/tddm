#!/bin/bash
# The paper's TimeGAN rows are the mean over seeds 42, 1 and 2
for seed in 42 1 2; do
  for dataset in geolife-100 cabspotting-100 porto-100 brest-100; do
    python main_timegan.py --data_name $dataset --seq_len 128 --seed $seed
  done
done
