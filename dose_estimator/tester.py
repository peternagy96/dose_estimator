import os
import sys

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image


class Tester(object):
    def __init__(self, data, model, result_path):
        self.result_path = result_path
        self.data = data
        self.model = model

    # Return a generated slice from all train and test images
    def test_jpg(self, epoch: int, mode: str = 'forward', index: int = 40, pat_num: list = [32, 5], mods: list = ['CT', 'PET', 'SPECT']):

        # create output folders
        path_name = os.path.join(self.result_path, f"epoch_{epoch}")
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        if mode == 'forward':
            num_train_samples = self.data.A_train.shape[0]
            num_test_samples = self.data.A_test.shape[0]
            # process training images
            for idx in np.arange(index, num_train_samples, pat_num[0]):
                pred = self.model.G_A2B.model.predict(
                    self.data.A_train[np.newaxis, idx, :, :]).squeeze()
                self.save_basic_plot(
                    self.data.A_train[idx], pred, self.data.B_train[idx], f"{path_name}/train_{idx}.png", mods)
            # process test images
            for idx in np.arange(index, num_test_samples, pat_num[1]):
                pred = self.model.G_A2B.model.predict(
                    self.data.A_test[np.newaxis, idx, :, :]).squeeze()
                self.save_basic_plot(
                    self.data.A_test[idx], pred, self.data.B_test[idx], f"{path_name}/test_{idx}.png", mods)

        elif mode == 'backward':
            num_train_samples = self.data.B_train.shape[0]
            num_test_samples = self.data.B_test.shape[0]
            # process training images
            for idx in np.arange(index, num_train_samples, pat_num):
                pred = self.model.G_B2A.model.predict(
                    self.data.B_train[np.newaxis, idx, :, :]).squeeze()
                self.save_basic_plot(self.data.B_train[idx], pred, self.data.A_train[idx],
                                     f"{path_name}/train_{idx}.png", [mods[-1], mods[-1], f"{mods[0]/mods[1]}"])
            # process test images
            for idx in np.arange(index, num_test_samples, pat_num):
                pred = self.model.G_B2A.model.predict(
                    self.data.B_test[np.newaxis, idx, :, :]).squeeze()
                self.save_basic_plot(self.data.B_test[idx], pred, self.data.A_test[idx],
                                     f"{path_name}/test_{idx}.png", [mods[-1], mods[-1], f"{mods[0]/mods[1]}"])

    def save_basic_plot(self, orig, pred, gt, path_name, mods):
        if len(mods) == 2:
            orig = self.rescale(orig).squeeze()
            pred = self.rescale(pred).squeeze()
            gt = self.rescale(gt).squeeze()
            s = gt.shape[0]
            error = np.abs(pred-gt)
        elif len(mods) == 3:
            orig = self.rescale(orig)
            pred = self.rescale(pred)
            gt = self.rescale(gt)
            s = gt.shape[0] * 2

            orig = np.vstack((orig[..., 0], orig[..., 1]))
            pred = np.vstack((pred[..., 0], pred[..., 1]))
            gt = np.vstack((gt[..., 0], gt[..., 1]))
            error = np.abs(pred-gt)

        s2 = gt.shape[0] * 2

        border = np.ones((s, 10)) * 255
        final_img = np.hstack((orig, border, pred, border, gt, border, error))
        footer = np.ones((20, final_img.shape[1])) * 255
        final_img = np.vstack((final_img, footer))

        font = cv2.FONT_HERSHEY_SIMPLEX
        final_img = cv2.putText(
            final_img, f"Input", (40, s2+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(
            final_img, f"Generated {mods[-1]}", (int(s2/2)+20, s2+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f"Ground Truth {mods[-1]}", (2*(
            int(s2/2)+15), s2+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(
            final_img, 'Error Map', (10+3*(int(s2/2)+20), s2+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        im = Image.fromarray(final_img).convert("L")
        im.save(path_name)

    @staticmethod
    def rescale(image):
        if image.ndim > 2:
            rescaled = np.empty((image.shape))
            for i in range(image.shape[-1]):
                mi = image[..., i].min()
                ma = image[..., i].max()
                rescaled[..., i] = (255.0 / (ma - mi) *
                                    (image[..., i] - mi)).astype(np.uint8)
        else:
            mi = image.min()
            ma = image.max()
            rescaled = (255.0 / (ma - mi) * (image - mi)).astype(np.uint8)
        return rescaled

    @staticmethod
    def rescale_mip(image):
        array = image.copy()
        for i in range(array.shape[0]):
            pic = array[i, :, :]
            mi = pic.min()
            ma = pic.max()
            pic = (255.0 / (ma - mi) * (pic - mi))  # .astype(np.uint8)
            array[i:(i+1), :, :] = pic
        return array

    @staticmethod
    def normalize(inp):
        array = inp.copy()
        for i in range(array.shape[0]):
            pic = array[i, :, :]
            mi = pic.min()
            ma = pic.max()
            pic = ((2 * (pic - mi)) / (ma - mi)) - 1
            array[i:(i+1), :, :] = pic
        return array

# Create MIP of prediction and ground truth for all patients from NIFTI

    def testMIP(self, test_path: str, mod_A, mod_B: str, epoch=''):
     # load txt file of test file names
        test_file = open(f"{test_path}/numpy/test.txt", "r", encoding='utf8')
        train_file = open(f"{test_path}/numpy/train.txt", "r", encoding='utf8')
        indices = test_file.read().splitlines()
        testlen = len(indices)
        indices.extend(train_file.read().splitlines())

        if epoch == '':
            if not os.path.exists(os.path.join(os.path.join(self.result_path, 'MIP'))):
                os.makedirs(os.path.join(self.result_path, 'MIP'))

        for idx, i in enumerate(indices):
            print(f"Processing {i}...")

            # load NIFTI files
            if len(mod_A) == 2:
                in1 = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[0])))
                in2 = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[1])))
                nifti_in_A = np.concatenate((in1, in2), axis=1)
            else:
                nifti_in_A = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[0])))

            nifti_in_B = self.normalize(sitk.GetArrayFromImage(
                self.read_nifti(test_path, i, mod_B)))

            # predict output modality
            print("    files loaded")
            pred_B = np.empty(in1.shape)
            if self.model.dim == '2D':
                for j in range(nifti_in_A.shape[0]):
                    pred_B[j] = self.model.G_A2B.model.predict(np.stack((in1[j], in2[j]), axis=2)[
                                                                np.newaxis, :, :, :]).squeeze()[:, :, 0]  # .reshape((256,128))
            elif self.model.dim == '3D':
                depth = self.model.img_shape[0]
                max_depth = nifti_in_B.shape[0]
                for j in range(0, max_depth, depth):
                    if j+depth <= max_depth:
                        pred_B[j:j+depth] = self.model.G_A2B.model.predict(np.stack((in1[j:j+depth], in2[j:j+depth]), axis=3)[
                                                                np.newaxis, :, :, :, :]).squeeze()[:, :, :, 0]  # .reshape((256,128))
                    else:
                        pred_B[max_depth-depth:] = self.model.G_A2B.model.predict(np.stack((in1[max_depth-depth:], in2[max_depth-depth:]), axis=3)[
                                                                np.newaxis, :, :, :, :]).squeeze()[:, :, :, 0]  # .reshape((256,128))
                    
            
                
            # TODO: fix histogram matching
            # pred_B[j] = self.hist_match(pred_B[0], pred_B[j])

            # create MIP for all images
            print("    predictions done")
            mip_ct = np.max(self.rescale_mip(in1), axis=1)
            mip_pet = np.max(self.rescale_mip(in2), axis=1)
            mip_orig = np.max(self.rescale_mip(nifti_in_B), axis=1)
            mip_pred = np.max(self.rescale_mip(pred_B), axis=1)
            error = np.abs(mip_pred - mip_orig)

            # create plot
            s = error.shape[0]
            s2 = error.shape[1] + 10
            border = np.ones((s, 10)) * 255
            final_img = np.hstack(
                (mip_ct, border, mip_pet, border, mip_orig, border, mip_pred, border, error))
            footer = np.ones((20, final_img.shape[1])) * 255
            final_img = np.vstack((final_img, footer))

            font = cv2.FONT_HERSHEY_SIMPLEX
            final_img = cv2.putText(
                final_img, f"CT", (70, s+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"PET", (s2+60, s+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"GT SPECT", (2*s2+40, s+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"Gen SPECT", (3*s2+40, s+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, 'Error Map', (4*s2+40, s+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            if idx+1 > testlen:
                addition = 'train'
            else:
                addition = 'test'
            if epoch != '':
                path_out = f"{self.result_path}/Epoch {epoch}/MIP/{i}_{addition}.png"
            else:
                path_out = f"{self.result_path}/{i}_{addition}.png"
            im = Image.fromarray(final_img).convert("L")
            im.save(path_out)


# Test and return 3D NIFTI images ==============================================


    def test3D(self, test_path: str, mod_A: str, mod_B: str):
        # load txt file of test file names
        test_file = open(f"{test_path}/test.txt", "r", encoding='utf8')
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
            self.write_nifti(nifti_out_A, i, mod_A)
            self.write_nifti(nifti_out_B, i, mod_B)

    def read_nifti(self, path: str, idx: str, mod: str):
        if mod == 'dose':
            nifti_in = sitk.ReadImage(os.path.join(path, f"{idx}.nii.gz"))
        else:
            nifti_in = sitk.ReadImage(os.path.join(
                path, f"{idx}_{mod.lower()}.nii.gz"))
        return nifti_in

    def predict_nifti(self, image, direction):
        array = sitk.GetArrayFromImage(image)
        array = self.normalize_array(array)
        pred = np.empty(array.shape)
        if direction == "A2B":
            for i in range(array.shape[0]):
                pred[i] = self.model.G_A2B.predict(
                    array[np.newaxis, i, :, :, np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                pred[i] = self.hist_match(pred[0], pred[i])
        else:
            for i in range(array.shape[0]):
                pred[i] = self.model.G_B2A.predict(
                    array[np.newaxis, i, :, :, np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                pred[i] = self.hist_match(pred[0], pred[i])

        return pred

    def write_nifti(self, image, i: str, mod: str):
        if mod == 'dose':
            sitk.WriteImage(image, os.path.join(
                self.result_path, 'nifti', f"{i}_pred.nii"), True)
        else:
            sitk.WriteImage(image, os.path.join(
                self.result_path, 'nifti', f"{i}_{mod}_pred.nii"), True)

    @staticmethod
    def normalize_array(inp):
        array = inp.copy()
        for i in range(array.shape[0]):
            pic = array[i:(i + 1), :, :]
            mask = (pic != 0.0)
            # pic / np.linalg.norm(pic) -1
            pic[mask] = ((pic[mask] - pic.min()) / (pic.max() - pic.min()))
            # pic[mask] = (pic[mask] - pic.mean()) / pic.std()
            array[i:(i + 1), :, :] = pic
        return array

    @staticmethod
    def hist_match(source, template):
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
