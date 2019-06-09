from keras.layers import Input
from keras.models import Model

import os
import json
import time

from .discriminator import Discriminator
from .generator import Generator
from .losses import lse, cycle_loss


class cycleGAN(object):
    def __init__(self, result_name, mode_G='basic', mode_D='basic', model_path: str = None,
                 lr_D: int = 3e-4, lr_G: int = 3e-4, image_shape: tuple = (128, 128, 2)):
        # Used as storage folder name
        self.date_time = time.strftime(
            '%Y%m%d-%H%M%S', time.localtime()) + '_' + result_name
        self.result_path = os.path.join(os.path.split(os.path.dirname(
            os.path.realpath(__file__)))[:-1][0], 'results', self.date_time)

        self.img_shape = image_shape
        self.channels = self.img_shape[-1]

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

        self.basicModel(mode_G=mode_G, mode_D=mode_D)

    def basicModel(self, mode_G, mode_D):
        self.D_A = Discriminator(name='A', mode=mode_D, use_patchgan=True,
                                 img_shape=(128, 128, 2))
        self.D_B = Discriminator(name='A', mode=mode_D, use_patchgan=True,
                                 img_shape=(128, 128, 2))
        self.G_A2B = Generator(name='A2B', mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=(128, 128, 2))
        self.G_B2A = Generator(name='A2B', mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=(128, 128, 2))

    def compileModel(self, opt_G, opt_D):
        self.D_A.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_A.loss_weights)
        self.D_B.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_B.loss_weights)

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

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(
            self.date_time, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(
            self.date_time, model.name, epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def load_model_from_files(self, path, epoch):
        self.D_A.model.load_weights(os.path.join(
            path, f"D_A_model_weights_epoch_{epoch}.hdf5"))
        self.D_B.model.load_weights(os.path.join(
            path, f"D_B_model_weights_epoch_{epoch}.hdf5"))
        self.G_A2B.model.load_weights(os.path.join(
            path, f"G_A2B_model_weights_epoch_{epoch}.hdf5"))
        self.G_B2A.model.load_weights(os.path.join(
            path, f"G_B2A_model_weights_epoch_{epoch}.hdf5"))

    def load_model_and_weights(self, model):
        #path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join(
            'generate_images', 'models', '{}.hdf5'.format(model.name))
        #model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)
