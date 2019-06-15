#!/bin/bash

source activate tf
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#python /home/peter/Documents/dose_estimator-git/dose_estimator/CycleGAN.py
nohup python /home/peter/code/dose_estimator/CycleGAN.py  /home/peter/code/results/20190613-145256_CTPET/saved_models/ 60 &> nohup2.out&