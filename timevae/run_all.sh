#!/bin/bash
# Three runs per dataset, as in the paper: one without --seed and seeds 42 and 1
for dataset in geolife-100 porto-100 cabspotting-100 brest-100; do
  python test_vae.py --dataset $dataset --seq_len 128 2>&1 | tee -a timevae-$dataset-128.log
  for seed in 42 1; do
    python test_vae.py --dataset $dataset --seq_len 128 --seed $seed 2>&1 | tee -a timevae-$dataset-128-seed$seed.log
  done
done
