import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import cv2
import random
#from sklearn.preprocessing import MinMaxScaler
#from skimage.io import imread


def load_data(nr_of_channels=1, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0):
    # load files
    trainA_images_ct = np.load('/home/peter/data/numpy/ct_train.npy') #np.load('/home/peter/Documents/dose_estimator/data/pet_train.npy')
    trainA_images_pet = np.load('/home/peter/data/numpy/pet_train.npy')
    trainB_images = np.load('/home/peter/data/numpy/dose_train.npy') #np.load('/home/peter/Documents/dose_estimator/data/ct_train.npy')
    testA_images_ct = np.load('/home/peter/data/numpy/ct_test.npy') #np.load('/home/peter/Documents/dose_estimator/data/pet_test.npy')
    testA_images_pet = np.load('/home/peter/data/numpy/pet_test.npy')
    testB_images = np.load('/home/peter/data/numpy/dose_test.npy') #np.load('/home/peter/Documents/dose_estimator/data/ct_test.npy')
    train_file = open("/home/peter/data/numpy/train.txt", "r", encoding='utf8')
    trainA_image_names = train_file.read().splitlines()
    trainB_image_names = trainA_image_names
    test_file = open("/home/peter/data/numpy/test.txt", "r", encoding='utf8')
    testA_image_names = test_file.read().splitlines()
    testB_image_names = testA_image_names

    # normalize
    trainA_images_ct = normalize_array(trainA_images_ct, trainA_images_ct.shape[0])
    trainA_images_pet = normalize_array(trainA_images_pet, trainA_images_pet.shape[0])
    trainB_images = normalize_array(trainB_images, trainB_images.shape[0])
    testA_images_ct = normalize_array(testA_images_ct, testA_images_ct.shape[0])
    testA_images_pet = normalize_array(testA_images_pet, testA_images_pet.shape[0])
    testB_images = normalize_array(testB_images, testB_images.shape[0])

    trainA_images_ct = filter_zeros(trainA_images_ct)
    trainA_images_pet = filter_zeros(trainA_images_pet)
    testA_images_ct = filter_zeros(testA_images_ct)
    testA_images_pet = filter_zeros(testA_images_pet)
    trainB_images = filter_zeros(trainB_images)
    testB_images = filter_zeros(testB_images)

    """
    # rescale
    trainA_images = upscale_array(trainA_images)
    trainB_images = upscale_array(trainB_images)
    testA_images = upscale_array(testA_images)
    testB_images = upscale_array(testB_images)
    """

    # add extra axis
    if nr_of_channels == 1:
        trainA_images = trainA_images[:, :, :, np.newaxis]
        trainB_images = trainB_images[:, :, :, np.newaxis]
        testA_images = testA_images[:, :, :, np.newaxis]
        testB_images = testB_images[:, :, :, np.newaxis]
    elif nr_of_channels == 2:
        trainA_images = np.stack((trainA_images_ct, trainA_images_pet), axis=-1)
        trainB_images = np.stack((trainB_images, trainB_images), axis=-1)
        testA_images = np.stack((testA_images_ct, testA_images_pet), axis=-1)
        testB_images = np.stack((testB_images, testB_images), axis=-1)

    # individually transform to 0 mean and std 1
    """trainA_images = convert_to_tf(trainA_images)
    trainB_images = convert_to_tf(trainB_images)
    testA_images = convert_to_tf(testA_images)
    testB_images = convert_to_tf(testB_images)"""


    return {"trainA_images": trainA_images, "trainB_images": trainB_images,
            "testA_images": testA_images, "testB_images": testB_images,
            "trainA_image_names": trainA_image_names,
            "trainB_image_names": trainB_image_names,
            "testA_image_names": testA_image_names,
            "testB_image_names": testB_image_names}

    """trainA_path = os.path.join('data', subfolder, 'trainA')

    trainB_path = os.path.join('data', subfolder, 'trainB')
    testA_path = os.path.join('data', subfolder, 'testA')
    testB_path = os.path.join('data', subfolder, 'testB')

    trainA_image_names = os.listdir(trainA_path)
    if nr_A_train_imgs != None:
        trainA_image_names = trainA_image_names[:nr_A_train_imgs]

    trainB_image_names = os.listdir(trainB_path)
    if nr_B_train_imgs != None:
        trainB_image_names = trainB_image_names[:nr_B_train_imgs]

    testA_image_names = os.listdir(testA_path)
    if nr_A_test_imgs != None:
        testA_image_names = testA_image_names[:nr_A_test_imgs]

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainA_images = create_image_array(trainA_image_names, trainA_path, nr_of_channels)
        trainB_images = create_image_array(trainB_image_names, trainB_path, nr_of_channels)
        testA_images = create_image_array(testA_image_names, testA_path, nr_of_channels)
        testB_images = create_image_array(testB_image_names, testB_path, nr_of_channels)
        """



def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)


def convert_to_tf(array):
    return array


  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
  # normalize between 0 and 1
def normalize_array(inp, img_size=81):
    array = inp.copy()
    for i in range(array.shape[0]):
        pic = array[i,:,:]
        pic -= pic.min()
        pic /= pic.ptp()
        pic = np.nan_to_num(pic)
        #mask = (pic != 0.0)
        #pic[mask] = pic[mask] / pic.max()
        #pic[mask] = ((pic[mask] - pic.min()) / (pic.max() - pic.min()))  # pic / np.linalg.norm(pic) -1 # 
        #pic[mask] = (pic[mask] - pic.mean()) / pic.std()
        #pic = ((pic - pic.min()) / (pic.max() - pic.min()))
        array[i:(i+1),:,:] = pic
    return array

def upscale_array(array):
    out = np.empty((array.shape[0], 200, 200))
    for i in range(array.shape[0]):
        pic = array[i,:,:]
        out[i,:,:] = cv2.resize(pic, dsize=(200, 200))
    return out

def filter_zeros(array):
    bad_idx = []
    for i in range(array.shape[0]):
        if np.count_nonzero(array[i,:,:]) == 0:
            bad_idx.append(i)
    for idx in bad_idx:
        while True:
            rand_idx = random.choice(range(array.shape[0]))
            if rand_idx not in bad_idx: break
        array[idx] = array[rand_idx]
    return array

class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, '', 3)
        real_images_B = create_image_array(batch_B, '', 3)

        return real_images_A, real_images_B  # input_data, target_data


if __name__ == '__main__':
    load_data()
