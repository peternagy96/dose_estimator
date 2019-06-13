#!/bin/bash

source activate tf
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/jun12_PET/saved_models/ 100 test_jpg
python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/jun12_PET/saved_models/ 200 test_jpg
#python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/jun12_CTPET/saved_models/ 100 mip
#python /home/peter/code/dose_estimator/CycleGAN.py /home/peter/saved_models/jun12_CTPET/saved_models/ 200 mip
#python /home/peter/Documents/dose_estimator-git/dose_estimator/CycleGAN.py /home/peter/Documents/dose_estimator-git/results/jun10_petct-fullfilter1/saved_models/ 200 mip