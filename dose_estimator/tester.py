import os
import sys
import math

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import zoom


class Tester(object):
    def __init__(self, data, model, result_path):
        self.result_path = result_path
        self.data = data
        self.model = model

    # Return a generated slice from all train and test images
    def test_jpg(self, epoch: int = '', mode: str = 'forward', index: int = 40, pat_num: list = [32, 5], mods: list = ['CT', 'PET', 'dose'], depth: int = 5):

        # create output folders
        if epoch == '':
            path_name = self.result_path
        else:
            path_name = os.path.join(self.result_path, f"epoch_{epoch}")
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        if mode == 'forward':
            num_train_samples = self.data.A_train.shape[0]
            num_test_samples = self.data.A_test.shape[0]  
            #if self.model.dim == '3D' and self.model.img_shape[0] != 81:
             #   pat_num = [1,1]
             #   index = 0

            # process training images
            for idx in np.arange(index, num_train_samples, pat_num[0]):
                if self.model.dim == '2D':
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :])[0].squeeze()
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :]).squeeze()
                    self.save_basic_plot(self.data.A_train[idx], pred, self.data.B_train[idx], f"{path_name}/train_{idx}.png", mods)
                elif self.model.dim == '3D' and self.model.img_shape[-2] > 40:
                    pad_l = int((depth-1)/2)
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :,  :, :])[0].squeeze()[pad_l]
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :,  :, :]).squeeze()[pad_l]
                    self.save_basic_plot(self.data.A_train[idx,pad_l], pred, self.data.B_train[idx,pad_l], f"{path_name}/train_{idx}.png", mods)
                else: # full 3D model
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :, :])[0].squeeze()[20]
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :, :]).squeeze()[20]
                    self.save_basic_plot(self.data.A_train[idx,20], pred, self.data.B_train[idx,20], f"{path_name}/train_{idx}.png", mods)


            # process test images
            for idx in np.arange(index, num_test_samples, pat_num[1]):
                if self.model.dim == '2D':
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_test[np.newaxis, idx, :, :])[0].squeeze()
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_test[np.newaxis, idx, :, :]).squeeze()
                    self.save_basic_plot(self.data.A_test[idx], pred, self.data.B_test[idx], f"{path_name}/test_{idx}.png", mods)
                elif self.model.dim == '3D' and self.model.img_shape[-2] > 40:
                    pad_l = int((depth-1)/2)
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_test[np.newaxis, idx, :, :, :])[0].squeeze()[pad_l]
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_test[np.newaxis, idx, :, :, :]).squeeze()[pad_l]
                    self.save_basic_plot(self.data.A_test[idx,pad_l], pred, self.data.B_test[idx,pad_l], f"{path_name}/test_{idx}.png", mods)
                else:
                    if self.model.style_loss:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :, :])[0].squeeze()[20]
                    else:
                        pred = self.model.G_A2B.model.predict(self.data.A_train[np.newaxis, idx, :, :, :]).squeeze()[20]
                    self.save_basic_plot(self.data.A_test[idx,20], pred, self.data.B_test[idx,20], f"{path_name}/test_{idx}.png", mods)
                
        elif mode == 'backward':
            num_train_samples = self.data.B_train.shape[0]
            num_test_samples = self.data.B_test.shape[0]
            # process training images
            for idx in np.arange(index, num_train_samples, pat_num[0]):
                pred = self.model.G_B2A.model.predict(
                    self.data.B_train[np.newaxis, idx, :, :]).squeeze()
                self.save_basic_plot(self.data.B_train[idx], pred, self.data.A_train[idx],
                                     f"{path_name}/train_{idx}.png", [mods[-1], mods[-1], f"{mods[0]/mods[1]}"])
            # process test images
            for idx in np.arange(index, num_test_samples, pat_num[1]):
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

        border = np.ones((s, 10)) * 255
        final_img = np.hstack((orig, border, pred, border, gt, border, error))
        footer = np.ones((20, final_img.shape[1])) * 255
        final_img = np.vstack((final_img, footer))

        font = cv2.FONT_HERSHEY_SIMPLEX
        final_img = cv2.putText(final_img, f"Input", (int(s/2*0.37), s+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f"Gen dose", (int(s/2*1.23), s+14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f"GT dose", (int(s/2*2.47), s+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.putText(final_img, 'Error Map', (int(s/2*3.53), s+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        im = Image.fromarray(final_img).convert("L")
        im.save(path_name)
        #cv2.imwrite(path_name,final_img)

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

    def normalize(self, inp, mod=''):
        array = inp.copy()
        if self.data.norm:
            if self.data.per_patient:
                mi = inp.min()
                ma = inp.max()
                array = ((2 * (inp - mi)) / (ma - mi)) - 1
            else:
                for i in range(array.shape[0]):
                    pic = array[i, :, :]
                    mi = pic.min()
                    ma = pic.max()
                    pic = ((2 * (pic - mi)) / (ma - mi)) - 1
                    array[i:(i+1), :, :] = pic
        else:
            if mod == 'CT':
               array = array/1024.0123291015625
            elif mod == 'PET':
                array = array/10
        return array

# Create MIP of prediction and ground truth for all patients from NIFTI

    def testMIP(self, test_path: str, mod_A, mod_B: str, epoch=''):
        crop = self.data.crop
        view = self.data.view
     # load txt file of test file names
        test_file = open(f"{test_path}/numpy/test.txt", "r", encoding='utf8')
        train_file = open(f"{test_path}/numpy/train.txt", "r", encoding='utf8')
        indices = test_file.read().splitlines()
        testlen = len(indices)
        indices.extend(train_file.read().splitlines())

        # initalize variable used for average error calculations
        count_train = 0
        count_test = 0
        avg_rmse_train = 0
        avg_psnr_train = 0
        avg_rmse_test = 0
        avg_psnr_test = 0

        # used for pic with all test MIP images
        collage_gt = []
        collage_pred = []
        collage_rmse = []
        collage_psnr = []

        if epoch == '':
            if not os.path.exists(os.path.join(os.path.join(self.result_path, 'MIP'))):
                os.makedirs(os.path.join(self.result_path, 'MIP'))
        else:
            if not os.path.exists(os.path.join(os.path.join(self.result_path, f"epoch_{epoch}", 'MIP'))):
                os.makedirs(os.path.join(self.result_path, f"epoch_{epoch}", 'MIP'))

        for idx, i in enumerate(indices):
            print(f"Processing {i}...")

            # load NIFTI files
            if len(mod_A) == 2:
                in1 = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[0])), mod_A[0])
                in2 = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[1])), mod_A[1])
                if self.model.dim == '3D' and self.model.img_shape[0] != 81:
                    pass#in1 = zoom(in1, (0.5, 1, 1))
                    #in2 = zoom(in2, (0.5, 1, 1))
                if crop:
                    #if self.data.view == 'front':
                    #    pred_B = np.empty((128, 80, 80))
                    #elif self.data.view == 'top':
                    pred_B = np.empty((80, 80, 80))
                    in1 = in1[:80,24:104,24:104]
                    in2 = in2[:80,24:104,24:104]
                else:
                    pred_B = np.empty(in1.shape)
                # pad input when using a 3D model
                depth = self.model.img_shape[0]
                if self.model.dim == '3D' and self.model.img_shape[0] != 81:
                    pad_l = int((depth-1)/2)
                    in1_pad = np.pad(in1, ((pad_l, pad_l), (0,0), (0,0)), 'constant', constant_values=(0))
                    in2_pad = np.pad(in2, ((pad_l, pad_l), (0,0), (0,0)), 'constant', constant_values=(0))
                    nifti_in_A = np.concatenate((in1_pad, in2_pad), axis=1)
                else:
                    nifti_in_A = np.concatenate((in1, in2), axis=1)
            else:
                nifti_in_A = self.normalize(sitk.GetArrayFromImage(
                    self.read_nifti(test_path, i, mod_A[0])), mod_A[0])
                if self.model.dim == '3D':
                    pad_l = int((depth-1)/2)
                    nifti_in_A = np.pad(nifti_in_A, ((pad_l, pad_l), (0,0), (0,0)), 'constant', constant_values=(0))
                    


            ref_volume = sitk.GetArrayFromImage(
                self.read_nifti(test_path, i, mod_B))
            nifti_in_B = self.normalize(ref_volume, mod_B)
            if self.model.dim == '3D' and self.model.img_shape[0] < 80:
                pass#nifti_in_B = zoom(nifti_in_B, (0.5, 1, 1))
            """
            if view == 'front':
                if self.model.dim == '3D' and self.model.img_shape[0] != 81:
                    in1_pad = np.swapaxes(in1_pad,0,1)
                    in2_pad = np.swapaxes(in2_pad,0,1)
                else:
                    in1 = np.swapaxes(in1,0,1)
                    in2 = np.swapaxes(in2,0,1)
                nifti_in_B = np.swapaxes(nifti_in_B,0,1)
            """
                           
            if crop:
                nifti_in_B = nifti_in_B[:80,24:104,24:104]

            # predict output modality
            if self.model.dim == '2D':
                for j in range(nifti_in_B.shape[0]):
                    if view == 'front':
                        if self.model.style_loss:
                            pred_B[:,j,:] = self.model.G_A2B.model.predict(np.stack((in1[:,j,:], in2[:,j,:]), axis=2)[
                                                                        np.newaxis, :, :, :])[0].squeeze()[:, :, 0]  # .reshape((256,128))
                        else:
                            pred_B[:,j,:] = self.model.G_A2B.model.predict(np.stack((in1[:,j,:], in2[:,j,:]), axis=2)[
                                                                    np.newaxis, :, :, :]).squeeze()[:, :, 0]  # .reshape((256,128))
                    elif view == 'top':
                        if self.model.style_loss:
                            pred_B[j] = self.model.G_A2B.model.predict(np.stack((in1[j], in2[j]), axis=2)[
                                                                    np.newaxis, :, :, :])[0].squeeze()[:, :, 0]  # .reshape((256,128))
                        else:
                            pred_B[j] = self.model.G_A2B.model.predict(np.stack((in1[j], in2[j]), axis=2)[
                                                                np.newaxis, :, :, :]).squeeze()[:, :, 0]  # .reshape((256,128))
            elif self.model.dim == '3D' and self.model.img_shape[0] < 40:               
                max_depth = nifti_in_B.shape[0]
                for j in range(0, max_depth, 1):
                    if self.model.style_loss:
                        pred_B[j] = self.model.G_A2B.model.predict(np.stack((in1_pad[j:j+depth], in2_pad[j:j+depth]), axis=3)[
                                                                np.newaxis, :, :, :, :])[0].squeeze()[int((depth-1)/2), :, :, 0]  # .reshape((256,128))   
                    else:
                        pred_B[j] = self.model.G_A2B.model.predict(np.stack((in1_pad[j:j+depth], in2_pad[j:j+depth]), axis=3)[
                                                            np.newaxis, :, :, :, :]).squeeze()[int((depth-1)/2), :, :, 0]  # .reshape((256,128))
            else:
                if self.model.style_loss:
                    pred_B = self.model.G_A2B.model.predict(np.stack((in1, in2), axis=3)[
                                                                np.newaxis, :, :, :, :])[0].squeeze()[:, :, :, 0]
                else:
                    pred_B = self.model.G_A2B.model.predict(np.stack((in1, in2), axis=3)[
                                                            np.newaxis, :, :, :, :]).squeeze()[:, :, :, 0]
                
            # TODO: fix histogram matching
            pred_B = self.matchHistVolume(pred_B, nifti_in_B)
            
            """
            if view == 'front':
                in1 = np.swapaxes(in1,0,1)
                in2 = np.swapaxes(in2,0,1)
                pred_B = np.swapaxes(pred_B,0,1)
                nifti_in_B = np.swapaxes(nifti_in_B,0,1)
            """
                

            # create MIP for all images
            mip_ct = (255 - np.max(self.rescale_mip(in1), axis=1))
            mip_pet = (255 - np.max(self.rescale_mip(in2), axis=1))
            mip_orig = (255 - np.max(self.rescale_mip(nifti_in_B), axis=1))
            mip_pred = (255 - np.max(self.rescale_mip(pred_B), axis=1))
            error = np.abs(mip_pred - mip_orig)
            rmse = self.rmse(nifti_in_B, pred_B)
            psnr = self.psnr(nifti_in_B, pred_B)           

            # create plot
            s = error.shape[0]
            s2 = error.shape[1] + 10
            border = np.ones((s, 10)) * 255
            final_img = np.hstack(
                (mip_ct, border, mip_pet, border, mip_orig, border, mip_pred, border, error))
            footer = np.ones((25, final_img.shape[1])) * 255
            final_img = np.vstack((final_img, footer))

            final_img = cv2.cvtColor(final_img.astype(np.float32), cv2.COLOR_GRAY2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            final_img = cv2.putText(
                final_img, f"CT", (int(s2*0.37), s+17), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"PET", (int(s2*1.33), s+17), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"GT SPECT", (int(2.1*s2), s+17), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"Gen SPECT", (int(3.1*s2), s+17), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)#final_img = cv2.putText(
                        #    final_img, 'Error Map', (int(4.16*s2), s+14), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"RMSE: {np.around(rmse,4)}", (int(4*s2), s+8), font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            final_img = cv2.putText(
                final_img, f"PSNR: {np.around(psnr,2)}", (int(4*s2), s+20), font, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            if idx+1 > testlen:
                addition = 'train'
                count_train += 1
                avg_rmse_train += rmse
                avg_psnr_train += psnr
            else:
                addition = 'test'
                count_test += 1
                avg_rmse_test += rmse
                avg_psnr_test += psnr
                collage_gt.append(mip_orig)
                collage_pred.append(mip_pred)
                collage_rmse.append(rmse)
                collage_psnr.append(psnr)
            if epoch != '':
                path_out = f"{self.result_path}/epoch_{epoch}/MIP/{addition}_{i}.png"
            else:
                path_out = f"{self.result_path}/{addition}_{i}.png"
            #im = Image.fromarray(final_img).convert("L")
            #im.save(path_out)
            cv2.imwrite(path_out,final_img)

        # calculate train and test avg error and save them to file
        avg_rmse_train /= count_train
        avg_psnr_train /= count_train
        avg_rmse_test /= count_test
        avg_psnr_test /= count_test
        if epoch == '':
            error_path = f"{self.result_path}/error.txt"
        else:
            error_path = f"{self.result_path}/epoch_{epoch}/MIP/error.txt"
        with open(error_path, 'w') as f:
            f.write(f"Train avg RMSE: {np.around(avg_rmse_train, 4)}\n")
            f.write(f"Train avg PSNR: {np.around(avg_psnr_train, 4)}\n")
            f.write(f"Test avg RMSE: {np.around(avg_rmse_test, 4)}\n")
            f.write(f"Test avg PSNR: {np.around(avg_psnr_test, 4)}\n")

        # create test set collage
        self.createTestCollage(collage_gt, collage_pred, collage_rmse, collage_psnr, self.result_path, epoch)


# Create collage of test MIP images

    @staticmethod
    def createTestCollage(collage_gt, collage_pred, collage_rmse, collage_psnr, result_path, epoch):
        font = cv2.FONT_HERSHEY_SIMPLEX
        border = np.ones((collage_gt[0].shape[0], 10)) * 255
        small_footer = np.ones((25, collage_gt[0].shape[1])) * 255
        img_height = collage_gt[0].shape[0]
        img_width = collage_gt[0].shape[1]       
        for i in range(len(collage_gt)):
            # put the vertical image together
            vertical_img = np.empty(())
            vertical_img = np.vstack((collage_gt[i], small_footer, collage_pred[i], small_footer))
            vertical_img = cv2.putText(vertical_img, f"RMSE: {np.around(collage_rmse[i], 4)}", (1, int(img_height*2.45)), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            vertical_img = cv2.putText(vertical_img, f"PSNR: {np.around(collage_psnr[i], 4)}", (1, int(img_height*2.6)), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            
            # add it to the final image
            border = np.ones((vertical_img.shape[0], 10)) * 255
            if i == 0:
                final_img = vertical_img
            else:
                final_img = np.hstack((final_img, border,vertical_img))
                
        final_img = cv2.putText(
                final_img, f"Ground Truth", (int((final_img.shape[1]/2.5)), int(final_img.shape[0]/2.15)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if epoch != '':
                path_out = f"{result_path}/epoch_{epoch}/MIP/MIP_test.png"
        else:
            path_out = f"{result_path}/MIP_test.png"
        cv2.imwrite(path_out,final_img)


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
                if self.model.style_loss:
                    pred[i] = self.model.G_A2B.predict(
                        array[np.newaxis, i, :, :, np.newaxis])[0].squeeze()
                else:
                    pred[i] = self.model.G_A2B.predict(
                        array[np.newaxis, i, :, :, np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                #pred[i] = self.hist_match(pred[0], pred[i])
        else:
            for i in range(array.shape[0]):
                if self.model.style_loss:
                    pred[i] = self.model.G_B2A.predict(
                        array[np.newaxis, i, :, :, np.newaxis])[0].squeeze()
                else:
                    pred[i] = self.model.G_B2A.predict(
                        array[np.newaxis, i, :, :, np.newaxis]).squeeze()
                #pred[i] = (255.0 / (pred[i].max() - pred[i].min()) * (pred[i] - pred[i].min())).astype(np.uint8)
                #pred[i] = self.hist_match(pred[0], pred[i])
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

    def matchHistVolume(self, syntheticVolume, referenceVolume):
        for kkk in range(np.shape(referenceVolume)[0]):
            matched = self.hist_match(syntheticVolume[kkk, ...], referenceVolume[kkk, ...])
            syntheticVolume[kkk, ...] = matched
        return syntheticVolume

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

    @staticmethod
    def psnr(img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    @staticmethod
    def rmse(img1, img2):
        return np.sqrt(np.mean((img1-img2)**2))
