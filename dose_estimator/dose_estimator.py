# !/usr/bin/env python3
import os

import pandas as pd
from tensorflow.python.client import device_lib

from models.gan import cycleGAN
from helpers.data_loader import Data
from trainer import Trainer
from tester import Tester

if __name__ == '__main__':
    # import settings from file
    if len(device_lib.list_local_devices()) > 1:
        jobsPath = '/home/peter/jobs.csv'
    else:
        jobsPath = os.path.join(os.getcwd(), 'dose_estimator', 'jobs.csv')
    jobs = pd.read_csv(jobsPath)

    # iterate through the jobs
    for index, settings in jobs.iterrows():
        print(f"Processing job {index+1}: {settings['Name']}")

        if settings['Mode'] == 'train':
            # import data
            mods = settings['Mods'].split(', ')
            data = Data(subfolder=settings['Subfolder'], mods=mods,
                             norm=settings['Norm'], aug=settings['Augment'])
            data.load_data()

            # import model
            image_shape = data.A_train.shape[-3:]
            gan = cycleGAN(dim=settings['Dim'], mode_G=settings['Generator'], 
                           mode_D='basic', model_path=settings['Model Path'],
                           image_shape=image_shape)

            # load trainer
            trainer = Trainer(result_name=settings['Name'], model=gan, 
                              init_epoch=settings['Init Epoch'],
                              epochs=settings['Epochs'],
                              lr_D=settings['D LR'], lr_G=settings['G LR'],
                              batch_size=settings['Batch Size'])

            # load model weights if necessary
            if os.path.exists(gan.model_path):
                gan.load_from_files(settings['Init Epoch'])
                print('Model weights loaded from files')
            else:
                print('Model loaded with init weights')

            # run training
            trainer.train(data=data, model=gan)

        elif settings['Mode'] == 'test_jpg':
            pass
        elif settings['Mode'] == 'test_nifti':
            pass
        elif settings['Mode'] == 'test_mip':
            pass
        elif settings['Mode'] == 'plot':
            pass
