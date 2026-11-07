#!/bin/bash

python test_vae.py --dataset geolife-100 --seq_len 128 2>&1 | tee -a timevae-geolife-100-128.log
