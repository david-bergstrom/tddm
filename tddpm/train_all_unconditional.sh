#!/bin/bash
# Three runs per dataset, as in the paper: one without --seed and seeds 42 and 1
for dataset in geolife-100 porto-100 cabspotting-100 brest-100; do
  python train_unconditional.py $dataset 128
  for seed in 42 1; do
    python train_unconditional.py $dataset 128 --seed $seed
  done
done
