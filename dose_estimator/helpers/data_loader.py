import os
import numpy as np
from tensorflow.python.client import device_lib
import random
import skimage as sk
from skimage import transform


class Data(object):
    def __init__(self, subfolder='data_corrected', dim='2D', mods=['CT', 'PET', 'SPECT'],
                 norm=True, aug=False):
        self.subfolder = subfolder
        self.dim = dim
        self.mods = mods
        self.norm = norm
        self.aug = aug

    def load_data(self):
        train_images = {}
        test_images = {}
        if len(device_lib.list_local_devices()) > 1:
            folder = os.path.join('/home/peter/data', self.subfolder, 'numpy')
        else:
            folder = os.path.join(os.getcwd(), 'data', self.subfolder, 'numpy')
        train_images['CT'] = np.load(os.path.join(
            folder, 'ct_train.npy')).reshape((-1, 128, 128))
        train_images['PET'] = np.load(os.path.join(
            folder, 'pet_train.npy')).reshape((-1, 128, 128))
        train_images['SPECT'] = np.load(os.path.join(
            folder, 'dose_train.npy')).reshape((-1, 128, 128))
        test_images['CT'] = np.load(os.path.join(
            folder, 'ct_test.npy')).reshape((-1, 128, 128))
        test_images['PET'] = np.load(os.path.join(
            folder, 'pet_test.npy')).reshape((-1, 128, 128))
        test_images['SPECT'] = np.load(os.path.join(
            folder, 'dose_test.npy')).reshape((-1, 128, 128))
        train_file = open(os.path.join(folder, "train.txt"),
                          "r", encoding='utf8')
        train_image_names = train_file.read().splitlines()
        test_file = open(os.path.join(folder, "test.txt"),
                         "r", encoding='utf8')
        test_image_names = test_file.read().splitlines()

        # normalize
        if self.norm == 'Y':
            print("Normalizing data...")
            for key in train_images.items():
                train_images[key[0]] = self.normalize_array(
                    train_images[key[0]])
                test_images[key[0]] = self.normalize_array(test_images[key[0]])

        # TODO: make depth and step hyperparameters in the Excel file
        if self.dim == '3D':
            print("Converting data to the 3D format...")
            for key in train_images.items():
                train_images[key[0]] = train_images[key[0]].reshape((-1,81,128,128))
                test_images[key[0]] = test_images[key[0]].reshape((-1,81,128,128))
                train_images[key[0]] = self.convertTo3D(train_images[key[0]], depth=9, step=3)
                test_images[key[0]] = self.convertTo3D(test_images[key[0]], depth=9, step=3)

        # augment
        if self.aug == 'Y':
            print("Augmenting training data...")
            if self.dim == '2D':
                train_images = self.augment2D(train_images)
            elif self.dim == '3D':
                train_images = self.augment3D(train_images)

        trainA_images = []
        testA_images = []
        trainB_images = []
        testB_images = []
        for mod in self.mods[:-1]:
            trainA_images.append(train_images[mod])
            testA_images.append(test_images[mod])
            trainB_images.append(train_images[self.mods[-1]])
            testB_images.append(test_images[self.mods[-1]])

        self.A_train = np.stack(trainA_images, axis=-1)
        self.A_test = np.stack(testA_images, axis=-1)
        self.B_train = np.stack(trainB_images, axis=-1)
        self.B_test = np.stack(testB_images, axis=-1)
        self.train_image_names = train_image_names
        self.test_image_names = test_image_names
        print('Data has been loaded')

    @staticmethod
    def normalize_array(inp):
        # * If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead normalize between 0 and 1
        array = inp.copy()
        for i in range(array.shape[0]):
            pic = array[i, :, :]
            mi = pic.min()
            ma = pic.max()
            pic = ((2 * (pic - mi)) / (ma - mi)) - 1
            array[i:(i+1), :, :] = pic
        return array

    @staticmethod
    def augment2D(images):
        out = {}
        out.update(images)
        for i in images.keys():
            out[i] = np.empty(images[i].shape)
        for j in range(images[list(images.keys())[0]].shape[0]):
            random_degree = random.uniform(-25, 25)
            for i in images.keys():
                out[i][j] = sk.transform.rotate(
                    images[i][j], random_degree, mode='constant', cval=out[i][j].min())
        for i in images.keys():
            out[i] = np.concatenate((images[i], out[i]), axis=0)
        return out

    # TODO: augment 3D dataset by random rotating each patient's scan
    @staticmethod
    def augment3D(images):
        return images

    @staticmethod
    def convertTo3D(inp, depth=9, step=9):
        numpics = inp.shape[0]
        piclen = inp.shape[1]
        out = []
        for j in range(numpics):
            for i in np.arange(0, piclen, step):
                if i+depth <= piclen:
                    out.append(inp[j, i:i+depth, :, :])
                else:
                    out.append(inp[j, piclen-depth:, :, :])
                    break
        return np.array(out)
