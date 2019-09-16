import os
import numpy as np
from tensorflow.python.client import device_lib
import random
import skimage as sk
from skimage import transform
from scipy.ndimage import zoom


class Data(object):
    def __init__(self, subfolder='data_corrected', dim='2D', mods=['CT', 'PET', 'dose'],
                 view='top', norm=True, aug=False, down=False, crop=True, depth=5, step_size=1):
        self.subfolder = subfolder
        self.dim = dim
        self.view = view
        self.mods = mods
        self.norm = norm
        self.aug = aug
        self.down = down
        self.depth = depth
        self.step_size = step_size
        if self.down:
            self.depth = 39
            self.step_size = 39

        self.crop = crop

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
        train_images['dose'] = np.load(os.path.join(
            folder, 'dose_train.npy')).reshape((-1, 128, 128))
        test_images['CT'] = np.load(os.path.join(
            folder, 'ct_test.npy')).reshape((-1, 128, 128))
        test_images['PET'] = np.load(os.path.join(
            folder, 'pet_test.npy')).reshape((-1, 128, 128))
        test_images['dose'] = np.load(os.path.join(
            folder, 'dose_test.npy')).reshape((-1, 128, 128))
        train_file = open(os.path.join(folder, "train.txt"),
                          "r", encoding='utf8')
        train_image_names = train_file.read().splitlines()
        test_file = open(os.path.join(folder, "test.txt"),
                         "r", encoding='utf8')
        test_image_names = test_file.read().splitlines()

        # high pass filter over PET and dose images
        train_images['PET'] = self.HPF(train_images['PET'], thresh=0.1)
        test_images['PET'] = self.HPF(test_images['PET'], thresh=0.1)
        train_images['dose'] = self.HPF(train_images['dose'], thresh=0.001)
        test_images['dose'] = self.HPF(test_images['dose'], thresh=0.001)

        # filter  black slices
        if self.dim != '3D':
            train_ct = []
            train_pet = []
            train_dose = []
            test_ct = []
            test_pet = []
            test_dose = []
            count_train = 0
            count_test = 0
            for i in range(train_images['PET'].shape[0]):
                if train_images['PET'][i].mean() != 0:
                    train_ct.append(train_images['CT'][i])
                    train_pet.append(train_images['PET'][i])
                    train_dose.append(train_images['dose'][i])
                else:
                    count_train += 1
            for i in range(test_images['dose'].shape[0]):
                if test_images['PET'][i].mean() != 0:
                    test_ct.append(test_images['CT'][i])
                    test_pet.append(test_images['PET'][i])
                    test_dose.append(test_images['dose'][i])
                else:
                    count_test += 1
            train_images['CT'] = np.array(train_ct)
            train_images['PET'] = np.array(train_pet)
            train_images['dose'] = np.array(train_dose)
            test_images['CT'] = np.array(test_ct)
            test_images['PET'] = np.array(test_pet)
            test_images['dose'] = np.array(test_dose)
            print(f"Removed {count_train} black training slices. ")
            print(f"Removed {count_test} black test slices. ")

        # normalize
        per_patient = True  # * when set to false then loss goes to NaN
        self.per_patient = per_patient
        self.step2 = False
        if self.norm == 'Y':
            print("Normalizing data...")
            for key in train_images.items():
                train_images[key[0]] = self.normalize(
                    train_images[key[0]], mod=key[0], per_patient=per_patient, step2=self.step2)
                test_images[key[0]] = self.normalize(
                    test_images[key[0]], mod=key[0], per_patient=per_patient, step2=self.step2)


        if self.view == 'front':
            for key in train_images.items():
                train_images[key[0]] = train_images[key[0]
                                                    ].reshape(-1, 81, 128, 128)
                train_images[key[0]] = np.swapaxes(
                    train_images[key[0]], 1, 2).reshape(-1, 81, 128)
                test_images[key[0]] = test_images[key[0]
                                                  ].reshape(-1, 81, 128, 128)
                test_images[key[0]] = np.swapaxes(
                    test_images[key[0]], 1, 2).reshape(-1, 81, 128)


        # crop the images into a square shape
        if self.crop:
            if self.view == 'front':
                for key in train_images.items():
                    train_images[key[0]] = train_images[key[0]][:, :80, 24:104]
                    test_images[key[0]] = test_images[key[0]][:, :80, 24:104]
            elif self.view == 'top':
                for key in train_images.items():
                    train_images[key[0]] = train_images[key[0]
                                                        ][:, 24:104, 24:104]
                    test_images[key[0]] = test_images[key[0]
                                                      ][:, 24:104, 24:104]

        if self.dim == '3D':
            print("Converting data to the 3D format...")
            for key in train_images.items():
                if self.crop:
                    train_images[key[0]] = train_images[key[0]
                                                        ].reshape((-1, 81, 80, 80))
                    test_images[key[0]] = test_images[key[0]
                                                      ].reshape((-1, 81, 80, 80))
                else:
                    train_images[key[0]] = train_images[key[0]
                                                        ].reshape((-1, 81, 128, 128))
                    test_images[key[0]] = test_images[key[0]
                                                      ].reshape((-1, 81, 128, 128))
                if self.down:
                    train_images[key[0]] = zoom(
                        train_images[key[0]], (1, 0.5, 1, 1))
                    test_images[key[0]] = zoom(
                        test_images[key[0]], (1, 0.5, 1, 1))
                else:
                    train_images[key[0]] = self.convertTo3D(
                        train_images[key[0]], depth=self.depth, step=self.step_size)
                    test_images[key[0]] = self.convertTo3D(
                        test_images[key[0]], depth=self.depth, step=self.step_size)

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
        print(self.A_train.shape)
        print('Data has been loaded')

    @staticmethod
    def HPF(image, thresh=0.2):
        array = image.copy()
        mask = image < thresh
        array[mask] = array.min()
        return array

    @staticmethod
    def normalize(inp, mod, per_patient=True, step2=False):
        # * If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead normalize between 0 and 1
        if mod == 'PET':
            return inp/10
        elif mod == 'CT': # CT images are normalized non-linearly
            if step2:
                inp = (inp - inp.mean()) / (inp.std())
            if per_patient:
                mi = inp.min()
                ma = inp.max()
                array = ((2 * (inp - mi)) / (ma - mi)) - 1
            else:
                array = inp.copy()
                for i in range(array.shape[0]):
                    pic = array[i, :, :]
                    mi = pic.min()
                    ma = pic.max()
                    pic = ((2 * (pic - mi)) / (ma - mi)) - 1
                    array[i:(i+1), :, :] = pic
            return array
        else: # dose images do not need to be normalized
            return inp

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

    @staticmethod
    def augment3D(images):
        out = {}
        out.update(images)
        num_imgs = images[list(images.keys())[0]].shape[0]
        num_slices = images[list(images.keys())[0]].shape[1]
        for i in images.keys():
            out[i] = np.empty(images[i].shape)
        for j in range(num_imgs):
            random_degree = random.uniform(-25, 25)
            for i in images.keys():
                for k in range(num_slices):
                    out[i][j, k] = sk.transform.rotate(
                        images[i][j, k], random_degree, mode='constant', cval=out[i][j, k].min())
        for i in images.keys():
            out[i] = np.concatenate((images[i], out[i]), axis=0)
        return out

    @staticmethod
    def convertTo3D(inp, depth=9, step=9):
        numpics = inp.shape[0]
        piclen = inp.shape[1]
        out = []
        for j in range(numpics):
            for i in np.arange(0, piclen, step):
                if i+depth <= piclen:
                    block = inp[j, i:i+depth, :, :]
                    out.append(block)
                else:
                    block = inp[j, piclen-depth:, :, :]
                    out.append(block)
                    break
        return np.array(out)
