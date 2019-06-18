# !/usr/bin/env python3
import pandas as pd

from models.gan import cycleGAN
# from trainer import Trainer

if __name__ == '__main__':
    # import settings from file
    jobsPath = r"C:\Users\peter\Documents\Thesis\dose_estimator-git\dose_estimator\settings.csv"
    data = pd.read_csv(jobsPath)

    # iterate through the jobs

    # import data

    # import model
    # gan = cycleGAN(result_name='test', mode_G='basic', mode_D='basic', model_path=None,
    #               lr_D = 3e-4, lr_G = 3e-4, image_shape = (128, 128, 2))

    # load trainer
    # trainer = Trainer()

    # decide on mode

    # run training

    # run testing

    # run test on NIFTI

    # plot model
