#!/bin/bash

python test_vae.py --dataset cabspotting-100 --seq_len 128 2>&1 | tee -a timevae-cabspotting-100-128.log
