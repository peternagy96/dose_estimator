#!/usr/bin/env python3
import os
import load_data
import tensorflow as tf
import keras.backend as K
import cv2
import SimpleITK as sitk
from PIL import Image
import sys
import csv
import math
import json
import time
import datetime
import random
from collections import OrderedDict
from keras.engine.topology import Network
from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.backend import mean
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from models.gan import cycleGAN
from models.generator import Generator
from models.discriminator import Discriminator
from data_loader.data_loader import dataLoader

from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, width=128, height=128, epochs=200,
                 batch=32, checkpoint=50, model_type=-1, mods = ['CT', 'PET', 'SPECT']):
        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type
        self.LATENT_SPACE_SIZE = latent_size

        self.generator = Generator(
            height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(
            height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator.Generator,
                       discriminator=self.discriminator.Discriminator)

        # LOAD DATASET


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(seed=12345)


class CycleGANTrainer():
    def __init__(self, model_path=None, load_epoch=None, mode='train', lr_D=3e-4, lr_G=3e-4, image_shape=(128, 128, 2),  # orig: lr_G=3e-4
                 date_time_string_addition='', image_folder='MR'):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 8.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 8.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        # Number of generator training iterations in each training loop
        self.generator_iterations = 1
        # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 10
        self.epochs = 200  # choose multiples of 25 since the models are save each 25th epoch
        if load_epoch is not None:
            self.init_epoch = int(load_epoch)
            self.epochs = self.epochs + self.init_epoch
        else:
            self.init_epoch = 1
        self.save_interval = 1
        self.synthetic_pool_size = 25

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = True
        # Identity mapping will be done each time the iteration number is divisable with this number
        self.identity_mapping_modulus = 10

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 0.95  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime(
            '%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_A_train_imgs = None
        nr_B_train_imgs = None
        nr_A_test_imgs = None
        nr_B_test_imgs = None

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data.load_data(
                nr_of_channels=self.batch_size, generator=True, subfolder=image_folder)

            # Only store test images
            nr_A_train_imgs = 0
            nr_B_train_imgs = 0

        data = load_data.load_data(nr_of_channels=self.channels,
                                   batch_size=self.batch_size,
                                   nr_A_train_imgs=nr_A_train_imgs,
                                   nr_B_train_imgs=nr_B_train_imgs,
                                   nr_A_test_imgs=nr_A_test_imgs,
                                   nr_B_test_imgs=nr_B_test_imgs,
                                   subfolder=image_folder)

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.testA_image_names = data["testA_image_names"]
        self.testB_image_names = data["testB_image_names"]
        if not self.use_data_generator:
            print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        K.tensorflow_backend.get_session().run(tf.global_variables_initializer())

        # ======= Load model weights if model path is given ========
        if model_path is not None:
            # load model weights
            self.load_model_from_files(model_path, load_epoch)
            print('Model weights loaded from files')
        else:
            print('Model loaded with init weights')

        # ======= Initialize training ==========
        if mode == 'train':
            sys.stdout.flush()
            #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
            self.train(init_epoch=self.init_epoch, epochs=self.epochs,
                       batch_size=self.batch_size, save_interval=self.save_interval)
        elif mode == 'test':
            test_path = '/home/peter/testdata'
            self.test3D(test_path=test_path, mod_A='PET', mod_B='dose')
        elif mode == 'test_jpg':
            test_path = '/home/peter/test_results/'
            self.test_jpg(test_path)


# ===============================================================================
# Training

    def train(self, init_epoch, epochs, batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
                # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(
                    x=real_images_A, y=ones_A)
                DB_loss_real = self.D_B.train_on_batch(
                    x=real_images_B, y=ones_B)
                DA_loss_synthetic = self.D_A.train_on_batch(
                    x=synthetic_images_A, y=zeros_B)
                DB_loss_synthetic = self.D_B.train_on_batch(
                    x=synthetic_images_B, y=zeros_A)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(
                        DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(
                        DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            # Compare reconstructed images to real images
            target_data = [real_images_A, real_images_B]
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones_A)
                target_data.append(ones_B)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------',
                  loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % 10 == 0:
                # Save temporary images continously
                self.save_tmp_images(
                    real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(init_epoch, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_A = images[0]
                    real_images_B = images[1]
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                        # labels
                        if self.use_multiscale_discriminator:
                            label_shape1 = (len(real_images_A),) + \
                                self.D_A.output_shape[0][1:]
                            label_shape2 = (len(real_images_B),) + \
                                self.D_B.output_shape[0][1:]
                            # label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
                            ones1 = np.ones(
                                shape=label_shape1) * self.REAL_LABEL
                            ones2 = np.ones(
                                shape=label_shape2) * self.REAL_LABEL
                            # ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
                            ones = [ones1, ones2]  # , ones4]
                            zeros1 = ones1 * 0
                            zeros2 = ones2 * 0
                            # zeros4 = ones4 * 0
                            zeros = [zeros1, zeros2]  # , zeros4]
                        else:
                            label_shape_A = (len(real_images_A),) + \
                                self.D_A.output_shape[1:]
                            label_shape_B = (len(real_images_B),) + \
                                self.D_B.output_shape[1:]
                            ones_A = np.ones(
                                shape=label_shape_A) * self.REAL_LABEL
                            ones_B = np.ones(
                                shape=label_shape_B) * self.REAL_LABEL
                            zeros_A = ones_A * 0
                            zeros_B = ones_B * 0

                    # Run all training steps
                    run_training_iteration(
                        loop_index, self.data_generator.__len__())

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.A_train
                B_train = self.B_train
                random_order_A = np.random.randint(
                    len(A_train), size=len(A_train))
                random_order_B = np.random.randint(
                    len(B_train), size=len(B_train))
                epoch_iterations = max(
                    len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                if self.use_supervised_learning:
                    random_order_B = random_order_A
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            #indexes_A = np.random.randint(len(A_train), size=batch_size)
                            #indexes_B = random_order_B[loop_index:loop_index + batch_size]
                            indexes_A = random_order_A[loop_index:]
                            indexes_B = random_order_B[loop_index:]
                        else:
                            indexes_B = np.random.randint(
                                len(B_train), size=batch_size)
                            indexes_A = random_order_A[loop_index:
                                                       loop_index + batch_size]
                    else:
                        indexes_A = random_order_A[loop_index:
                                                   loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                   loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]

                    # labels
                    if self.use_multiscale_discriminator:
                        label_shape1 = (len(real_images_A),) + \
                            self.D_A.output_shape[0][1:]
                        label_shape2 = (len(real_images_B),) + \
                            self.D_B.output_shape[0][1:]
                        # label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
                        ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
                        ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
                        # ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
                        ones = [ones1, ones2]  # , ones4]
                        zeros1 = ones1 * 0
                        zeros2 = ones2 * 0
                        # zeros4 = ones4 * 0
                        zeros = [zeros1, zeros2]  # , zeros4]
                    else:
                        label_shape_A = (len(real_images_A),) + \
                            self.D_A.output_shape[1:]
                        label_shape_B = (len(real_images_B),) + \
                            self.D_B.output_shape[1:]
                        ones_A = np.ones(shape=label_shape_A) * self.REAL_LABEL
                        ones_B = np.ones(shape=label_shape_B) * self.REAL_LABEL
                        zeros_A = ones_A * 0
                        zeros_B = ones_B * 0

                    # Run all training steps
                    run_training_iteration(loop_index, epoch_iterations)

            # ================== within epoch loop end ==========================

            if epoch % save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch',
                      epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 20 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch)
                self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()


# ===============================================================================
# Help functions

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = (
            (epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(
            datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)


# ===============================================================================
# Save and load


    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)
