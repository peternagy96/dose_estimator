# !/usr/bin/env python3
import pandas as pd

from models.gan import cycleGAN
from helpers.data_loader import load_data
from .trainer import Trainer

if __name__ == '__main__':
    # import settings from file
    jobsPath = r"C:\Users\peter\Documents\Thesis\dose_estimator-git\dose_estimator\jobs.csv"
    jobs = pd.read_csv(jobsPath)

    # iterate through the jobs
    for index, settings in jobs.iterrows():
        print(f"Processing job {index+1}: {settings['Name']}")

        if settings['Mode'] == 'train':
            # import data
            mods = settings['Mods'].split(', ')
            data = load_data(subfolder=settings['Subfolder'], mods=mods,
                             norm=settings['Norm'], aug=settings['Augment'])

            # import model
            image_shape = data["trainA_images"].shape[-3:]
            gan = cycleGAN(result_name=settings['Name'], dim=settings['Dim'],
                           mode_G=settings['Generator'], mode_D='basic',
                           model_path=settings['Model Path'],
                           image_shape=image_shape)

            # load trainer
            trainer = Trainer(model=gan, init_epoch=settings['Init Epoch'],
                              epochs=settings['Epochs'],
                              lr_D=settings['D LR'], lr_G=settings['G LR'],
                              batch_size=settings['Batch Size'])

            # load model weights if necessary
            if gan.model_path != '':
                gan.load_model_from_files(settings['Init Epoch'])

            # run training
            trainer.train(data=data)

        elif settings['Mode'] == 'test_jpg':
            pass
        elif settings['Mode'] == 'test_nifti':
            pass
        elif settings['Mode'] == 'test_mip':
            pass
        elif settings['Mode'] == 'plot':
            pass
