# !/usr/bin/env python3
import os
import sys
import time

from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np

from tester import Tester
from trainer import Trainer
from helpers.data_loader import Data
from models.gan import cycleGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#if len(device_lib.list_local_devices()) > 1:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(seed=12345)


if __name__ == '__main__':
    # import settings from file
    if len(device_lib.list_local_devices()) > 1:
        jobsPath = '/home/peter/code/dose_estimator/jobs.csv'
    else:
        jobsPath = os.path.join(os.getcwd(), 'dose_estimator', 'jobs.csv')
    jobs = pd.read_csv(jobsPath)

    # iterate through the jobs
    for index, settings in jobs.iterrows():
        if pd.isna(settings['Done']):
            print(f"Processing job {index+1}: {settings['Name']}")

            # import data
            mods = settings['Mods'].split(', ')
            data = Data(subfolder=settings['Subfolder'], dim=settings['Dim'], mods=mods,
                        norm=settings['Norm'], aug=settings['Augment'])
            data.load_data()
            sys.exit() # ! REMOVE

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
            print(gan.model_path)
            if os.path.exists(gan.model_path):
                gan.load_from_files(settings['Init Epoch'])
                print('Model weights loaded from files')
            else:
                print('Model loaded with init weights')

            if settings['Mode'] == 'train':
                # run training
                trainer.train(data=data, model=gan)
            elif settings['Mode'] == 'plot':
                pass
            else:
                result_name = settings['Name'] + '_' + time.strftime('%Y%m%d-%H%M%S', time.localtime())
                result_path = os.path.join( os.getcwd(), 'results', result_name)
                tester = Tester(data=data, model=gan, result_path=result_path)
                if settings['Mode'] == 'test_jpg':
                    tester.test_jpg(epoch=epoch, mode="forward",
                                    index=40, pat_num=[32, 5], mods=data.mods)
                elif settings['Mode'] == 'test_nifti':
                    pass
                elif settings['Mode'] == 'test_mip':
                    test_path = f"/home/peter/data/{settings['Subfolder']}"
                    tester.testMIP(test_path=test_path, mod_A=[
                                   'CT', 'PET'], mod_B='dose')