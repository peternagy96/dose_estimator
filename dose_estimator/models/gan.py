from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from discriminator import Discriminator
from generator import Generator


class cycleGAN(object):
    def __init__(self, model_path: str = None, lr_D: int = 3e-4, lr_G: int = 3e-4, image_shape: tuple = (128, 128, 2), ):
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

        basicmodel()

    def basicModel(self):
        self.D_A = Discriminator(
            name='A', use_multiscale_discriminator=False, use_patchgan=True, img_shape=(128, 128, 2))
        self.D_B = Discriminator(
            name='A', use_multiscale_discriminator=False, use_patchgan=True, img_shape=(128, 128, 2))
        self.G_A2B
        self.G_B2A

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
