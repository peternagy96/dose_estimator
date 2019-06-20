# !/usr/bin/env python3
from tester import Tester
from trainer import Trainer
from helpers.data_loader import Data
from models.gan import cycleGAN
from tensorflow.python.client import device_lib
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # import settings from file
    if len(device_lib.list_local_devices()) > 1:
        jobsPath = '/home/peter/code/dose_estimator/jobs.csv'
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
        elif settings['Mode'] == 'plot':
                pass
        else:
            tester = Tester(data=data, model=gan, result_path=self.result_path)
            if settings['Mode'] == 'test_jpg':
                tester.test_jpg(epoch=epoch, mode="forward", index=40, pat_num=[32,5], mods=data.mods)
            elif settings['Mode'] == 'test_nifti':
                pass
            elif settings['Mode'] == 'test_mip':
                pass
