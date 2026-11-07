#!/bin/bash
# Three runs per dataset, as in the paper: one without --seed (upstream default seed 0) and seeds 42 and 1
ARGS="--nepochs 100 --batch_size 32 --nsamples 128 --withCD True --LSTMG True --LSTMD True --criterion BCE --glr 0.001 --dlr 0.001 --cdlr 0.0001 --Ngroups 2 --real_data_fraction 10.0 --CD_type MLP --gamma 5.0 --noise_len 32"
for dataset in geolife-100 porto-100 cabspotting-100 brest-100; do
  python run.py --dataset $dataset $ARGS 2>&1 | tee -a coscigan-$dataset-128.log
  for seed in 42 1; do
    python run.py --dataset $dataset $ARGS --seed $seed 2>&1 | tee -a coscigan-$dataset-128-seed$seed.log
  done
done
