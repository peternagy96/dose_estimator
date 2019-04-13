#!/usr/bin/env python3
from models.gan import GAN
from models.generator import Generator
from models.discriminator import Discriminator
from data_loader.data_loader import dataLoader

from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, width = 28, height= 28, channels = 1, latent_size=100, epochs =50000,
                 batch=32, checkpoint=50,model_type=-1):
        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type
        self.LATENT_SPACE_SIZE = latent_size

        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

        # LOAD DATASET

    def train(self):
        for e in range(self.EPOCHS):
            # Grab a batch
            count_real_images = int(self.BATCH / 2)
            starting_index = randint(0, (len(self.X_train) - count_real_images))
            real_images_raw = self.X_train[starting_index: (starting_index +
                                                            count_real_images)]
            x_real_images = real_images_raw.reshape(count_real_images, self.W,
                                                    self.H, self.C)
            y_real_labels = np.ones([count_real_images, 1])

            # Grab Generated Images for this training batch
            latent_space_samples = self.sample_latent_space(count_real_images)
            x_generated_images = self.generator.Generator.predict(latent_space_samples)
            y_generated_labels = np.zeros([self.BATCH - count_real_images, 1])

            # Combine to train on the discriminator
            x_batch = np.concatenate([x_real_images, x_generated_images])
            y_batch = np.concatenate([y_real_labels, y_generated_labels])

            # Now, train the discriminator with this batch
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)

            # Generate Noise
            x_latent_space_samples = self.sample_latent_space(self.BATCH)
            y_generated_labels = np.ones([self.BATCH, 1])
            generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss:' + str(discriminator_loss) +
                  '], [Generator:: Loss:' + str(generator_loss)+']')
            if e % self.CHECKPOINT == 0:
                self.plot_checkpoint(e)
        return

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))

    def plot_checkpoint(self, e):
        filename = "/data/sample_" + str(e) + ".png"
        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [self.H, self.W])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return