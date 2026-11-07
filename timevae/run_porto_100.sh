#!/bin/bash

python test_vae.py --dataset porto-100 --seq_len 128 2>&1 | tee -a timevae-porto-100-128.log
