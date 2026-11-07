#!/bin/bash
# Three runs per dataset, as in the paper: the default seed (12345) and seeds 42 and 1.
# The output directory (--name) and checkpoint folder include the seed, so the runs don't overwrite each other.
for dataset in geolife-100 porto-100 cabspotting-100 brest-100; do
  python main.py --name ${dataset}128 --config_file Config/cmp_${dataset}_128.yaml --gpu 0 --benchmark
  for seed in 42 1; do
    python main.py --name ${dataset}-128-seed${seed} --config_file Config/cmp_${dataset}_128.yaml --gpu 0 --benchmark \
      --seed $seed solver.results_folder ./Checkpoints_${dataset}_seed${seed}
  done
done
