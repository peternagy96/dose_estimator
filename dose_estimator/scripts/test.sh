#!/bin/bash

source activate tf
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/20190531-084731/ 160 test_jpg