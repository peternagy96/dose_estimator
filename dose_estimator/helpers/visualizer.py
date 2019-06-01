


def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):

    if len(real.shape) > 3:
        real = real[0]
        synthetic = synthetic[0]
        reconstructed = reconstructed[0]

    synthetic = synthetic.clip(min=0)
    reconstructed = reconstructed.clip(min=0)

    # Append and save
    if real_ is not None:
        if len(real_.shape) > 4:
            real_ = real_[0]
        image = np.hstack((real_[0], real, synthetic, reconstructed))
    else:
        image = np.hstack((real, synthetic, reconstructed))

    if self.channels == 1:
        image = image[:, :, 0]

    #toimage(image, cmin=0, cmax=1).save(path_name)
    rescaled = (255.0 / (image.max() - image.min()) * (image - image.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(path_name)

def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
    directory = os.path.join('images', self.date_time)
    if not os.path.exists(os.path.join(directory, 'A')):
        os.makedirs(os.path.join(directory, 'A'))
        os.makedirs(os.path.join(directory, 'B'))
        os.makedirs(os.path.join(directory, 'Atest'))
        os.makedirs(os.path.join(directory, 'Btest'))

    testString = ''

    real_image_Ab = None
    real_image_Ba = None
    for i in range(num_saved_images + 1):
        if i == num_saved_images:
            real_image_A = self.A_test[0]
            real_image_B = self.B_test[0]
            real_image_A = np.expand_dims(real_image_A, axis=0)
            real_image_B = np.expand_dims(real_image_B, axis=0)
            testString = 'test'
            if self.channels == 1:  # Use the paired data for MR images
                real_image_Ab = self.B_test[0]
                real_image_Ba = self.A_test[0]
                real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
        else:
            #real_image_A = self.A_train[rand_A_idx[i]]
            #real_image_B = self.B_train[rand_B_idx[i]]
            if len(real_image_A.shape) < 4:
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)
            if self.channels == 1:  # Use the paired data for MR images
                real_image_Ab = real_image_B  # self.B_train[rand_A_idx[i]]
                real_image_Ba = real_image_A  # self.A_train[rand_B_idx[i]]
                real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                real_image_Ba = np.expand_dims(real_image_Ba, axis=0)

        synthetic_image_B = self.G_A2B.predict(real_image_A)
        synthetic_image_A = self.G_B2A.predict(real_image_B)
        reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
        reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

        self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                'images/{}/{}/epoch{}_sample{}.png'.format(
                                    self.date_time, 'A' + testString, epoch, i))
        self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                'images/{}/{}/epoch{}_sample{}.png'.format(
                                    self.date_time, 'B' + testString, epoch, i))

def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
    try:
        reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
        reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

        real_images = np.vstack((real_image_A[0], real_image_B[0]))
        synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
        reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

        self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                'images/{}/{}.png'.format(
                                    self.date_time, 'tmp'))
    except: # Ignore if file is open
        pass
