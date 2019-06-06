from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


class cycleGAN(object):
    def __init__(self, discriminator, generator):
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.Generator = generator
        self.Discriminator = discriminator
        self.Discriminator.trainable = False
        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy',
                               optimizer=self.OPTIMIZER)
        self.gan_model.summary()

    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

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
        self.D_A.load_weights(os.path.join(
            path, f"D_A_model_weights_epoch_{epoch}.hdf5"))
        self.D_B.load_weights(os.path.join(
            path, f"D_B_model_weights_epoch_{epoch}.hdf5"))
        self.G_A2B.load_weights(os.path.join(
            path, f"G_A2B_model_weights_epoch_{epoch}.hdf5"))
        self.G_B2A.load_weights(os.path.join(
            path, f"G_B2A_model_weights_epoch_{epoch}.hdf5"))

    def load_model_and_weights(self, model):
        #path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join(
            'generate_images', 'models', '{}.hdf5'.format(model.name))
        #model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def summary(self):
        return self.gan_model.summary()
