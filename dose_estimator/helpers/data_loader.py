import os
import numpy as np
from tensorflow.python.client import device_lib
import random
import skimage as sk
from skimage import transform


def load_data(subfolder='data_corrected', mods=['CT', 'PET', 'SPECT'],
              norm=True, aug=False):

    # load files
    train_images = {}
    test_images = {}
    if len(device_lib.list_local_devices()) > 1:
        folder = os.path.join('/home/peter/data', subfolder, 'numpy')
    else:
        folder = os.path.join(os.getcwd(), 'data', subfolder, 'numpy')
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
    train_file = open(os.path.join(folder, "train.txt"), "r", encoding='utf8')
    train_image_names = train_file.read().splitlines()
    test_file = open(os.path.join(folder, "test.txt"), "r", encoding='utf8')
    test_image_names = test_file.read().splitlines()

    # normalize
    if norm:
        print("Normalizing data...")
        for key in train_images.items():
            train_images[key[0]] = normalize_array(train_images[key[0]])
            test_images[key[0]] = normalize_array(test_images[key[0]])

    # augment
    if aug:
        print("Augmenting training data...")
        train_images = augment(train_images)

    trainA_images = []
    testA_images = []
    trainB_images = []
    testB_images = []
    for mod in mods[:-1]:
        trainA_images.append(train_images[mod])
        testA_images.append(test_images[mod])
        trainB_images.append(train_images[mods[-1]])
        testB_images.append(test_images[mods[-1]])

    trainA_images = np.stack(trainA_images, axis=-1)
    testA_images = np.stack(testA_images, axis=-1)
    trainB_images = np.stack(trainB_images, axis=-1)
    testB_images = np.stack(testB_images, axis=-1)

    return {"trainA_images": trainA_images, "trainB_images": trainB_images,
            "testA_images": testA_images, "testB_images": testB_images,
            "trainA_image_names": train_image_names,
            "trainB_image_names": train_image_names,
            "testA_image_names": test_image_names,
            "testB_image_names": test_image_names}

  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
  # normalize between 0 and 1


def normalize_array(inp):
    array = inp.copy()
    for i in range(array.shape[0]):
        pic = array[i, :, :]
        mi = pic.min()
        ma = pic.max()
        pic = ((2 * (pic - mi)) / (ma - mi)) - 1
        array[i:(i+1), :, :] = pic
    return array


def augment(images):
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


def convertTo3D(depth: int = 3, step: int = 1):
    return 0
