#!/usr/bin/env python3
import os
import tensorflow as tf
import keras.backend as K
import sys
import csv
import json
import time
import datetime
import math
from collections import OrderedDict
from keras.optimizers import Adam

from random import randint
import numpy as np

from helpers.image_pool import ImagePool
from tester import Tester


class Trainer(object):
    def __init__(self, result_name, model, init_epoch=math.nan, epochs=200, lr_D=3e-4, lr_G=3e-4, batch_size=10, gen_iter=2, adv_training=False):
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        # Number of generator training iterations in each training loop
        self.generator_iterations = gen_iter
        # Number of discriminator training iterations in each training loop
        self.discriminator_iterations = 1
        self.adv_training = adv_training
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = int(batch_size)
        # choose multiples of 25 since the models are save each 25th epoch
        self.epochs = int(epochs)
        if not math.isnan(init_epoch):
            self.init_epoch = int(init_epoch)
            self.epochs = int(self.epochs + self.init_epoch)
        else:
            self.init_epoch = 1
        self.save_interval = 1  # ! CHANGE
        self.synthetic_pool_size = 25

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = True
        # Identity mapping will be done each time the iteration number is divisable with this number
        self.identity_mapping_modulus = 10

        # Tweaks
        self.REAL_LABEL = 0.95  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.result_name = result_name + '_' + \
            time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.result_path = os.path.join(
            os.getcwd(), 'results', self.result_name)

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # compile model
        model.compile(self.opt_G, self.opt_D, self.use_identity_learning)

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        K.tensorflow_backend.get_session().run(tf.global_variables_initializer())

    def train(self, data, model):
        batch_size = self.batch_size
        save_interval = self.save_interval
        epochs = self.epochs
        init_epoch = self.init_epoch

        tester = Tester(data=data, model=model, result_path=self.result_path)

        # ======= Create designated run folder and store meta data ==========
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.writeMetaDataToJSON(model, data)

        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
            # Generate batch of synthetic images
            synthetic_images_B = model.G_A2B.model.predict(real_images_A)
            synthetic_images_A = model.G_B2A.model.predict(real_images_B)

            if model.style_loss:                
                features_B = synthetic_images_B[1:10]
                features_A = synthetic_images_A[1:10]
                synthetic_images_B = synthetic_images_B[0]
                synthetic_images_A = synthetic_images_A[0]

            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = model.D_A.model.train_on_batch(
                    x=real_images_A, y=ones_A)
                DB_loss_real = model.D_B.model.train_on_batch(
                    x=real_images_B, y=ones_B)
                DA_loss_synthetic = model.D_A.model.train_on_batch(
                    x=synthetic_images_A, y=zeros_A)
                DB_loss_synthetic = model.D_B.model.train_on_batch(
                    x=synthetic_images_B, y=zeros_B)
                if model.use_multiscale_discriminator:
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
            
            if model.style_loss:
                target_data = [real_images_A]
                target_data.extend(features_A)
                target_data.append(real_images_B)
                target_data.extend(features_B)
            else:
                target_data = [real_images_A, real_images_B]

            if model.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones_A)
                target_data.append(ones_B)

            for _ in range(self.generator_iterations):
                G_loss = model.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)  # ToDo: add loss here
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                target_B = [real_images_B]
                target_A = [real_images_A]
                if model.style_loss:
                    target_B.extend(features_B)
                    target_A.extend(features_A)
                G_A2B_identity_loss = model.G_A2B.model.train_on_batch(
                    x=real_images_B, y=target_B)  # ToDo: add loss here
                G_B2A_identity_loss = model.G_B2A.model.train_on_batch(
                    x=real_images_A, y=target_A)  # ToDo: add loss here
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(model.D_A.model, decay_D)
                self.update_lr(model.D_B.model, decay_D)
                self.update_lr(model.G_model, decay_G)

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

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate(
                len(data.A_train), len(data.B_train))

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(init_epoch, epochs + 1):
            A_train = data.A_train
            B_train = data.B_train
            random_order_A = np.random.randint(
                len(A_train), size=len(A_train))
            random_order_B = np.random.randint(
                len(B_train), size=len(B_train))
            epoch_iterations = max(
                len(random_order_A), len(random_order_B))
            min_nr_imgs = min(len(random_order_A), len(random_order_B))

            for loop_index in range(0, epoch_iterations, batch_size):
                if loop_index + batch_size >= min_nr_imgs:
                    # If all images soon are used for one domain,
                    # randomly pick from this domain
                    if len(A_train) <= len(B_train):
                        # indexes_A = np.random.randint(len(A_train), size=batch_size)
                        # indexes_B = random_order_B[loop_index:loop_index + batch_size]
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

                if self.adv_training:
                    real_images_A = self.add_noise(real_images_A)
                    real_images_B = self.add_noise(real_images_B)

                # labels
                if model.use_multiscale_discriminator:
                    label_shape1 = (len(real_images_A),) + \
                        model.D_A.model.output_shape[0][1:]
                    label_shape2 = (len(real_images_B),) + \
                        model.D_B.model.output_shape[0][1:]
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
                        model.D_A.model.output_shape[1:]
                    label_shape_B = (len(real_images_B),) + \
                        model.D_B.model.output_shape[1:]
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
                # if data.dim == '2D':
                #    tester.test_jpg(epoch=epoch, mode="forward", index=40, pat_num=[32,5], mods=data.mods)
                # elif data.dim == '3D':
                model.save(self.result_path, model.D_A.model, epoch)
                model.save(self.result_path, model.D_B.model, epoch)
                model.save(self.result_path, model.G_A2B.model, epoch)
                model.save(self.result_path, model.G_B2A.model, epoch)
                tester.testMIP(test_path='/home/peter/data/3d_filtered/',
                               mod_A=data.mods[:-1], mod_B=data.mods[-1], epoch=epoch)
                #pat_num = [int(data.A_train.shape[0]), int()]
                tester.test_jpg(epoch=epoch, mode="forward",
                                index=40, pat_num=[32, 5], mods=data.mods)

            """ if epoch % 20 == 0:
                # self.saveModel(self.G_model)
                model.save(self.result_path, model.D_A.model, epoch)
                model.save(self.result_path, model.D_B.model, epoch)
                model.save(self.result_path, model.G_A2B.model, epoch)
                model.save(self.result_path, model.G_B2A.model, epoch) """

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

    @staticmethod
    def add_noise(array):
        out = array.copy()
        for i in range(array.shape[0]):
            pic = array[i]
            row,col,ch = pic.shape
            mean = 0
            var = 0.00001
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col, ch))
            #gauss = gauss.reshape(row,col)
            noisy = pic + gauss
            out[i] = noisy
        return out


    def get_lr_linear_decay_rate(self, lenA, lenB):
        # Calculate decay rates
        max_nr_images = max(lenA, lenB)

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
        with open('{}/loss_output.csv'.format(self.result_path), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self, model, data_orig):
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': model.img_shape,
            'Image view': data_orig.view,
            'data subfolder': data_orig.subfolder,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'lambda_1': model.lambda_1,
            'lambda_2': model.lambda_2,
            'lambda_d': model.lambda_D,
            'Style loss': model.style_loss,
            'Style weight': model.style_weight,
            'CT loss weight': model.ct_loss_weight,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': model.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': model.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'data normalized': str(data_orig.norm),
            'data augmented': str(data_orig.aug),
            '3D window size': str(data_orig.depth),
            '3D step size': str(data_orig.step_size),
            'data downsampled': str(data_orig.down),
            'resize convolution': str(model.use_resize_convolution),
            'number of A train examples': len(data_orig.A_train),
            'number of B train examples': len(data_orig.B_train),
            'number of A test examples': len(data_orig.A_test),
            'number of B test examples': len(data_orig.B_test),
        })

        with open('{}/meta_data.json'.format(self.result_path), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)
