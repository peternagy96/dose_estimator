# !/usr/bin/env python3
#from train import Trainer
import tensorflow as tf
#from pathlib import Path
import numpy as np
#import pytest
#import cv2

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = str(sys.argv[1])
        load_epoch = str(sys.argv[2])
        if len(sys.argv) == 4:
            mode = str(sys.argv[3])
            GAN = CycleGAN(model_path, load_epoch, mode) 
        else:
            GAN = CycleGAN(model_path, load_epoch)
    else:
        GAN = CycleGAN()
