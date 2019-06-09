#!/bin/bash

source activate tf
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/may31_firstCT-PET/ 160 test_jpg
python /home/peter/Documents/dose_estimator-git/dose_estimator/CycleGAN.py /home/peter/Documents/dose_estimator-git/results/may31_firstCT-PET/saved_models/ 160 test_jpg