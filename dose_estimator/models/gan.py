from keras.layers import Input
from keras.models import Model

import os
import json
import time

from .discriminator import Discriminator
from .generator import Generator
from .losses import lse, cycle_loss


class cycleGAN(object):
    def __init__(self, dim='2D', mode_G='basic', mode_D='basic',
                 model_path: str = None, image_shape: tuple = (128, 128, 2)):

        self.model_path = model_path
        self.dim = dim
        self.img_shape = image_shape

        # Hyper parameters
        self.lambda_1 = 8.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 8.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        self.build(dim=dim, mode_G=mode_G, mode_D=mode_D, img_shape=self.img_shape)

    def build(self, dim, mode_G, mode_D, img_shape):
        self.D_A = Discriminator(name='A', dim=dim, mode=mode_D, use_patchgan=True,
                                 img_shape=img_shape)
        self.D_B = Discriminator(name='B', dim=dim, mode=mode_D, use_patchgan=True,
                                 img_shape=img_shape)
        self.G_A2B = Generator(name='A2B', dim=dim, mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=img_shape)
        self.G_B2A = Generator(name='B2A', dim=dim, mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=img_shape)

    def compile(self, opt_G, opt_D, use_identity_learning):
        self.D_A.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_A.loss_weights)
        self.D_B.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_B.loss_weights)

        if use_identity_learning:
            self.G_A2B.model.compile(optimizer=opt_G, loss='MAE')
            self.G_B2A.model.compile(optimizer=opt_G, loss='MAE')

        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B.model(real_A)
        synthetic_A = self.G_B2A.model(real_B)
        dA_guess_synthetic = self.D_A.model_static(synthetic_A)
        dB_guess_synthetic = self.D_B.model_static(synthetic_B)
        reconstructed_A = self.G_B2A.model(synthetic_B)
        reconstructed_B = self.G_A2B.model(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [cycle_loss, cycle_loss,
                          lse, lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(lse)
                # * 1e-3)  # Lower weight to regularize the model
                compile_weights.append(self.lambda_D)
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

    def save(self, path, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join(path, 'saved_models')
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = '{}/saved_models/{}_weights_epoch_{}.hdf5'.format(
            path, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = '{}/saved_models/{}_model_epoch_{}.json'.format(
            path, model.name, epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in {}/saved_models}/'.format(model.name, path))

    def load_from_files(self, epoch):
        path = self.model_path
        epoch = int(epoch)
        self.D_A.model.load_weights(os.path.join(
            path, f"D_A_model_weights_epoch_{epoch}.hdf5"))
        self.D_B.model.load_weights(os.path.join(
            path, f"D_B_model_weights_epoch_{epoch}.hdf5"))
        self.G_A2B.model.load_weights(os.path.join(
            path, f"G_A2B_model_weights_epoch_{epoch}.hdf5"))
        self.G_B2A.model.load_weights(os.path.join(
            path, f"G_B2A_model_weights_epoch_{epoch}.hdf5"))

    def saveSummary(self):
        with open(f"/home/peter/generator_{self.dim}.txt", 'w') as f:
            self.G_A2B.model.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(f"/home/peter/discriminator_{self.dim}.txt", 'w') as f:
            self.D_A.summary(print_fn=lambda x: f.write(x + '\n'))
