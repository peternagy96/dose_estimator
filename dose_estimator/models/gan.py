from keras.layers import Input
from keras.models import Model

import os
import json
import time

from .discriminator import Discriminator
from .generator import Generator
from .losses import lse, mae, mae_style, cycle_loss, s_loss, gm_loss, null_loss


class cycleGAN(object):
    def __init__(self, dim='2D', mode_G='basic', mode_D='basic',
                 model_path: str = None, image_shape: tuple = (128, 128, 2), ct_loss_weight=0.5, 
                 style_loss=False, tv_loss=False, style_weight=0.001):

        self.model_path = model_path
        self.dim = dim
        self.img_shape = image_shape

        # Hyper parameters
        self.lambda_1 = 8.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 8.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.ct_loss_weight = ct_loss_weight
        self.style_loss = style_loss
        self.tv_loss = tv_loss
        self.style_weight = style_weight

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        if mode_G == 'new':
            mode_D = 'new'

        self.build(dim=dim, mode_G=mode_G, mode_D=mode_D, img_shape=self.img_shape)

    def build(self, dim, mode_G, mode_D, img_shape):
        self.D_A = Discriminator(name='A', dim=dim, mode=mode_D, use_patchgan=True,
                                 img_shape=img_shape)
        self.D_B = Discriminator(name='B', dim=dim, mode=mode_D, use_patchgan=True,
                                 img_shape=img_shape)
        self.G_A2B = Generator(name='A2B', dim=dim, mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=img_shape, style_loss=self.style_loss)
        self.G_B2A = Generator(name='B2A', dim=dim, mode=mode_G, use_resize_convolution=False,
                               use_identity_learning=True, img_shape=img_shape, style_loss=self.style_loss)

    def compile(self, opt_G, opt_D, use_identity_learning, style_loss=False):
        self.D_A.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_A.loss_weights)
        self.D_B.model.compile(optimizer=opt_D,
                               loss=lse,
                               loss_weights=self.D_B.loss_weights)

        
        if use_identity_learning:    
            if self.style_loss:       
                identity_loss = [mae(alpha=self.ct_loss_weight)]
                for _ in range(4, 13):
                    if self.tv_loss:
                        identity_loss.append(null_loss)
                        #identity_loss.append(s_loss)
                    else:
                        identity_loss.append(null_loss)
                        #identity_loss.append(gm_loss)
        else:
            identity_loss = [mae(alpha=self.ct_loss_weight)]
            self.G_A2B.model.compile(optimizer=opt_G, loss=identity_loss)
            self.G_B2A.model.compile(optimizer=opt_G, loss=identity_loss)

        real_A = Input(shape=self.img_shape, name='real_A')
        real_B = Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B.model(real_A)
        synthetic_A = self.G_B2A.model(real_B)
        if self.style_loss:
            dA_guess_synthetic = self.D_A.model_static(synthetic_A[0])
            dB_guess_synthetic = self.D_B.model_static(synthetic_B[0])
            reconstructed_A = self.G_B2A.model(synthetic_B[0])
            reconstructed_B = self.G_A2B.model(synthetic_A[0])
        else:
            dA_guess_synthetic = self.D_A.model_static(synthetic_A)
            dB_guess_synthetic = self.D_B.model_static(synthetic_B)
            reconstructed_A = self.G_B2A.model(synthetic_B)
            reconstructed_B = self.G_A2B.model(synthetic_A)

        if self.style_loss:
                model_outputs = reconstructed_A
                model_outputs.extend(reconstructed_B)
        else:
            model_outputs = [reconstructed_A, reconstructed_B]
        
        #identity_loss = [mae(alpha=self.ct_loss_weight), style_loss(gt_dict=gt_dict, gen_dict=gen_dict)]

        model_inputs = [real_A, real_B]

        if self.style_loss:
            compile_losses = [cycle_loss(alpha=self.ct_loss_weight)]
            compile_weights = [self.lambda_1]
            for _ in range(4, 13):
                if self.tv_loss:
                    compile_losses.append(s_loss)
                else:
                    compile_losses.append(gm_loss)
                compile_weights.append(self.style_weight)
            compile_losses.append(cycle_loss(alpha=self.ct_loss_weight))
            compile_weights.append(self.lambda_2)
            for _ in range(4, 13):
                if self.tv_loss:
                    compile_losses.append(s_loss)
                else:
                    compile_losses.append(gm_loss)
                compile_weights.append(self.style_weight)
            compile_losses.extend([lse, lse])
            compile_weights.extend([self.lambda_D, self.lambda_D])
        else:
            compile_losses = [cycle_loss(alpha=self.ct_loss_weight), cycle_loss(alpha=self.ct_loss_weight),
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

        self.G_model = Model(inputs=model_inputs,
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
        print(f"{model.name} has been saved in {path}/saved_models/")

    def load_from_files(self, epoch):
        path = self.model_path
        epoch = int(epoch)
        self.D_A.model.load_weights(os.path.join(
            path, f"A_weights_epoch_{epoch}.hdf5"))
        self.D_B.model.load_weights(os.path.join(
            path, f"B_weights_epoch_{epoch}.hdf5"))
        self.G_A2B.model.load_weights(os.path.join(
            path, f"A2B_weights_epoch_{epoch}.hdf5"))
        self.G_B2A.model.load_weights(os.path.join(
            path, f"B2A_weights_epoch_{epoch}.hdf5"))

    def saveSummary(self):
        with open(f"/home/peter/generator_{self.dim}.txt", 'w') as f:
            self.G_A2B.model.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(f"/home/peter/discriminator_{self.dim}.txt", 'w') as f:
            self.D_A.summary(print_fn=lambda x: f.write(x + '\n'))
