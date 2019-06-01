# Return a generated slice from all train and test images 

def test_jpg(self, path_name: str, orig: str = 'A', index: int = 40, dim3: int = 81):

    # create output folders
    path_name = os.path.join(path_name, self.date_time)
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    if orig == 'A':
        num_train_samples = self.A_train.shape[0]
        num_test_samples = self.A_test.shape[0]
        # process training images
        for idx in np.arange(index, num_train_samples, dim3):
            pred = self.G_A2B.predict(self.A_train[np.newaxis,idx,:,:]).squeeze()
            self.save_basic_plot(self.A_train[idx], pred, self.B_train[idx], f"{path_name}/train_{idx}.png")
        # process test images
        for idx in np.arange(index, num_test_samples, dim3):
            pred = self.G_A2B.predict(self.A_test[np.newaxis,idx,:,:]).squeeze()
            self.save_basic_plot(self.A_test[idx], pred, self.B_test[idx], f"{path_name}/test_{idx}.png")

    elif orig == 'B':
        num_train_samples = self.B_train.shape[0]
        num_test_samples = self.B_test.shape[0]
        # process training images
        for idx in np.arange(index, num_train_samples, dim3):
            pred = self.G_B2A.predict(self.B_train[np.newaxis,idx,:,:]).squeeze()
            self.save_basic_plot(self.B_train[idx], pred, self.A_train[idx], f"{path_name}/train_{idx}.png")
        # process test images
        for idx in np.arange(index, num_test_samples, dim3):
            pred = self.G_B2A.predict(self.B_test[np.newaxis,idx,:,:]).squeeze()
            self.save_basic_plot(self.B_test[idx], pred, self.A_test[idx], f"{path_name}/test_{idx}.png")


def save_basic_plot(self, orig, pred, gt, path_name):
    channels = pred.shape[-1]
    if channels == 1:
        orig = self.rescale(orig.clip(min=0)).squeeze()
        pred = self.rescale(pred.clip(min=0)).squeeze()
        gt = self.rescale(gt.clip(min=0)).squeeze()
        s = gt.shape[0]
    elif channels == 2:
        orig = self.rescale(orig.clip(min=0))
        pred = self.rescale(pred.clip(min=0))
        gt = self.rescale(gt.clip(min=0))
        s = gt.shape[0] * 2

        orig = np.vstack((orig[...,0], orig[...,1]))
        pred = np.vstack((pred[...,0], pred[...,1]))
        gt = np.vstack((gt[...,0], gt[...,1]))



    border = np.ones((s, 20)) * 255
    final_img = np.hstack((orig, border, pred, border, gt))
    footer = np.ones((20, final_img.shape[1])) * 255
    final_img = np.vstack((final_img, footer))

    font = cv2.FONT_HERSHEY_SIMPLEX
    final_img = cv2.putText(final_img,'Original PET',(25,int(s/2)+14), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
    final_img = cv2.putText(final_img,'Generated SPECT',(10+int(s/2)+20,s+14), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
    final_img = cv2.putText(final_img,'Ground Truth SPECT',(10+2*(int(s/2)+20),s+14), font, 0.35, (0,0,0), 1, cv2.LINE_AA)

    im = Image.fromarray(final_img).convert("L")
    im.save(path_name)

def rescale(self, image):
    if image.ndim > 2:
        rescaled = np.empty((image.shape))
        for i in range(image.shape[-1]):
            rescaled[...,i] = (255.0 / (image[...,i].max() - image[...,i].min()) * (image[...,i] - image[...,i].min())).astype(np.uint8)
    else:
        rescaled = (255.0 / (image.max() - image.min()) * (image - image.min())).astype(np.uint8)
    return rescaled

# Test and return 3D NIFTI images ==============================================

    def test3D(self, test_path: str, mod_A: str, mod_B: str, dim3: int = 81):
        # load txt file of test file names
        test_file = open("/home/peter/data/test.txt", "r", encoding='utf8')
        indices = test_file.read().splitlines()

        for idx, i in enumerate(indices):
            print(f"Processing {i}...")

            # load NIFTI files
            nifti_in_A = self.read_nifti(test_path, i, mod_A)
            nifti_in_B = self.read_nifti(test_path, i, mod_B)

             # predict output modality
            pred_A = self.predict_nifti(nifti_in_B, direction='B2A')
            pred_B = self.predict_nifti(nifti_in_A, direction='A2B')

            # copy old NIFTI metadata
            nifti_out_A = sitk.GetImageFromArray(pred_A, isVector=False)
            nifti_out_B = sitk.GetImageFromArray(pred_B, isVector=False)
            for k in nifti_in_A.GetMetaDataKeys():
                nifti_out_A.SetMetaData(k, nifti_in_A.GetMetaData(k))
            for k in nifti_in_B.GetMetaDataKeys():
                nifti_out_B.SetMetaData(k, nifti_in_B.GetMetaData(k))

            # save to new folder
            path_out = '/home/peter/test_results'
            self.write_nifti(nifti_out_A, path_out, i, mod_A)
            self.write_nifti(nifti_out_B, path_out, i, mod_B)



    def read_nifti(self, path: str, idx: str, mod: str):
        if mod == 'dose':
            nifti_in = sitk.ReadImage(os.path.join(path, f"{idx}.nii"))
        else:
            nifti_in = sitk.ReadImage(os.path.join(path, f"{idx}_{mod}.nii"))
        return nifti_in

    def predict_nifti(self, image, direction):
        array = sitk.GetArrayFromImage(image)
        array = self.normalize_array(array)
        pred = np.empty(array.shape)
        if direction == "A2B":
            for i in range(array.shape[0]):
                pred[i] = self.G_A2B.predict(array[np.newaxis, i,:,:,np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                pred[i] = self.hist_match(pred[0], pred[i])
        else:
            for i in range(array.shape[0]):
                pred[i] = self.G_B2A.predict(array[np.newaxis, i,:,:,np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                pred[i] = self.hist_match(pred[0], pred[i])

        return pred

    def write_nifti(self, image, path_out: str, i: str, mod: str):
        if mod == 'dose':
            sitk.WriteImage(image, os.path.join(path_out, f"{i}_pred.nii"), True)
        else:
            sitk.WriteImage(image, os.path.join(path_out, f"{i}_{mod}_pred.nii"), True)

    def normalize_array(self, inp, img_size=81):
        array = inp.copy()
        for i in range(array.shape[0]):
            pic = array[i:(i + 1), :, :]
            mask = (pic != 0.0)
            pic[mask] = ((pic[mask] - pic.min()) / (pic.max() - pic.min()))  # pic / np.linalg.norm(pic) -1
            # pic[mask] = (pic[mask] - pic.mean()) / pic.std()
            array[i:(i + 1), :, :] = pic
        return array

    def hist_match(self, source, template):
            oldshape = source.shape
            source = source.ravel()
            template = template.ravel()
            # get the set of unique pixel values and their corresponding indices and
            # counts
            s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                    return_counts=True)
            t_values, t_counts = np.unique(template, return_counts=True)

            # take the cumsum of the counts and normalize by the number of pixels to
            # get the empirical cumulative distribution functions for the source and
            # template images (maps pixel value --> quantile)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]

            # interpolate linearly to find the pixel values in the template image
            # that correspond most closely to the quantiles in the source image
            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
            return interp_t_values[bin_idx].reshape(oldshape)