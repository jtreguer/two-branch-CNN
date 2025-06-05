#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
import os
import rasterio
import scipy.io as io
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from einops import rearrange


class Dataset(data.Dataset):

    # UNUSED
    patch_size = 64
    SNR = 25
    
    def __init__(self, args, sp_matrix, mask=None, isTrain=True):
        super(Dataset, self).__init__()

        self.args = args
        self.sp_matrix = sp_matrix
        self.msi_channels = sp_matrix.shape[1]

        self.isTrain = isTrain

        default_datapath = os.getcwd()
        data_folder = os.path.join(default_datapath, args.data_path_name)

        self.img_path_list = []

        if os.path.exists(data_folder):
            for file in os.listdir(data_folder):
                if file.endswith('.img'):
                    self.img_path_list.append(file)
        else:
            return None

        print(self.img_path_list)

        self.img_list = []
        for file in self.img_path_list:
            file_path = os.path.join(os.getcwd(),args.data_path_name,file)
            with rasterio.open(file_path) as src:
                print("Reading HS data")
                hyperspectral_data = src.read()
                hyperspectral_data = hyperspectral_data[:,args.start_x_pixel:args.start_x_pixel+args.window_size,args.start_y_pixel:args.start_y_pixel+args.window_size]
                if mask is not None:
                    print("Excluding water bands")
                    hyperspectral_data = hyperspectral_data[mask,:,:]

                # Display information about the hyperspectral data
                print('Shape of hyperspectral data:', hyperspectral_data.shape)
                self.img_list.append(rearrange(hyperspectral_data, 'c h w -> h w c'))


        # for i in range(len(self.imgpath_list)):
        #     # loadmat expects Matlab format matrices .mat
        #     self.img_list.append(io.loadmat(self.imgpath_list[i])["img"]) # [0:96,0:96,:]
        #     # self.img_list.append(io.loadmat(self.imgpath_list[i])["HSI"]) # CAVE

        "for single HSI"
        (_, _, self.hsi_channels) = self.img_list[0].shape

        "generate simulated data"
        self.img_patch_list = []
        self.img_lr_list = []
        self.img_msi_list = []

         # Calculate global min/max across all images
        all_mins = []
        all_maxs = []
        for img in self.img_list:
           all_mins.append(np.min(img))
           all_maxs.append(np.max(img))
        self.global_min = min(all_mins)
        self.global_max = max(all_maxs)

        for _, img in enumerate(self.img_list):
            (h, w, c) = img.shape
            s = self.args.scale_factor
            "Ensure that the side length can be divisible"
            r_h, r_w = h % s, w % s
            img_patch = img[
                int(r_h / 2) : h - (r_h - int(r_h / 2)),
                int(r_w / 2) : w - (r_w - int(r_w / 2)),
                :,
            ]
            # Normalization of patch
            # img_patch = (img_patch - np.min(img_patch)) / (np.max(img_patch) - np.min(img_patch))
            img_patch = self.normalize_image(img_patch, self.global_min, self.global_max)
            self.img_patch_list.append(img_patch)
            "LrHSI"
            img_lr = self.generate_LrHSI(img_patch, s)
            
            # img_lr = self.normalize_image(img_lr)
            
            # sigmah = np.sqrt(np.sum(img_lr)**2 / (10 ** (SNR/10)) / (r_h*r_w*c))
            # img_lr_noise = img_lr + sigmah * np.random.normal(size=no.shape(img_lr))
            print(f"img_lr.shape {img_lr.shape}")
            self.img_lr_list.append(img_lr)

            (self.lrhsi_height, self.lrhsi_width, p) = img_lr.shape
            self.spectral_manifold = self.generate_spectral_manifold(
                np.reshape(img_lr, [self.lrhsi_height * self.lrhsi_width, p]), k=15
            )
            "HrMSI"
            img_msi = self.generate_HrMSI(img_patch, self.sp_matrix)
            (self.msi_height, self.msi_width, p) = img_msi.shape
            img_msi = img_msi.transpose(1,2,0)
            self.spatial_manifold_1 = self.generate_spectral_manifold(
                np.reshape(img_msi, [self.msi_width * p, self.msi_height]), k=25
            )
            img_msi = img_msi.transpose(2,0,1)
            img_msi = img_msi.transpose(0,2,1)
            self.spatial_manifold_2 = self.generate_spectral_manifold(
                np.reshape(img_msi, [self.msi_height * p, self.msi_width]), k=25
            )
            img_msi = img_msi.transpose(0,2,1)
            print(f"img_msi.shape {img_msi.shape}")
            # io.savemat(r"D:\\Dataset\\Hyperspectral Image Plot\\Pavia_manifold.mat", 
            #            {"spem":self.spectral_manifold,"spam1":self.spatial_manifold_1,"spam2":self.spatial_manifold_2,
            #             "spem_w":self.spectral_weights,"spam1_w":self.spatial_manifold_1_weights,"spam2_w":self.spatial_manifold_2_weights})
            # sigmam = np.sqrt(np.sum(img_msi)**2 / (10 ** (SNR/10)) / (h*w*p))
            # img_msi_noise = img_lr + sigmam * np.random.normal(size=no.shape(img_msi))
            self.img_msi_list.append(img_msi)
            # io.savemat(r"D:\\Dataset\\MIAE\\MIAE\\data\\pavia\\paviac_data_r80.mat", {'MSI':img_msi,'HSI':img_lr, 'REF':img_patch})
            # io.savemat(r"D:\\Dataset\\MIAE\\MIAE\\data\\pavia\\{}_r80.mat".format("img1"), {'MSI':img_msi,'HSI':img_lr, 'REF':img_patch})
            print("Dataset initialized")
            print(type(img_lr))
            print(f"min max of input images {np.max(img_lr)}, {np.min(img_lr)}, {np.max(img_msi)}, {np.min(img_msi)}, {np.max(img_patch)}")

    def normalize_image(self, img, global_min=None, global_max=None):
      """
      Normalize image to [0,1] range
      Args:
         img: Input image
         global_min: Optional global minimum for consistent normalization
         global_max: Optional global maximum for consistent normalization
      """
      if global_min is None:
         global_min = np.min(img)
      if global_max is None:
         global_max = np.max(img)
      
      normalized = (img - global_min) / (global_max - global_min + 1e-8)  # Add epsilon to avoid division by zero
      return np.clip(normalized, 0, 1)  # Ensure values are in [0,1]



    # UNUSED
    def generate_spatial_manifold(self, clustered_data, msi, sigma_D=900):
        m, n, _ = msi.shape
        # generate the spatial laplacian matrix of MSI
        spatial_weights = np.zeros([m * n, m * n])
        msi_2d = np.reshape(msi, [m*n, _])
        ks = kneighbors_graph(msi_2d, n_neighbors=40, mode="connectivity").toarray()
        for i in range(m):
            for j in range(n):
                pixel_ij = msi[i, j, :]
                idx_manifold_row = i * m + j
                label_ij = clustered_data[i, j]
                idxes = np.where(clustered_data == label_ij)
                for idn in range(idxes[0].shape[0]):
                    ni, nj = idxes[0][idn], idxes[1][idn]
                    idx_manifold_col = ni * m + nj
                    pixel_nij = msi[ni, nj, :]
                    weight = np.exp(-np.sum((pixel_ij - pixel_nij) ** 2) / sigma_D)
                    spatial_weights[idx_manifold_row, idx_manifold_col] = weight
        spatial_diag = np.diag(np.sum(spatial_weights, axis=1)) - spatial_weights
        return spatial_diag

    def generate_spectral_manifold(self, lrhsi_2d, k=15):
        """
        Generate manifold from three modes of a tensor
        K is for the number of neighbours
        """
        _, p = lrhsi_2d.shape
        # generate the spectral Laplacian matrix of Lr HSI
        spectral_weights = np.zeros([p, p])
        ka = kneighbors_graph(lrhsi_2d.T, n_neighbors=k, mode="connectivity").toarray()
        sigma_S = 1000
        for i in range(p):
            idx = np.where(ka[i] == 1)[0]
            i_image = lrhsi_2d[:, i]
            for id in idx:
            # for j in range(p):
                id_image = lrhsi_2d[:, id]
                # id_image = lrhsi_2d[:, j]
                weight = np.exp(-np.sum((i_image - id_image) ** 2) / sigma_S)
                spectral_weights[i, id] = weight
                # spectral_weights[i, j] = weight
        spectral_diag = np.diag(np.sum(spectral_weights, axis=1)) - spectral_weights
        return spectral_diag # , spectral_weights

    def downsamplePSF(self, img, sigma, stride):
        def matlab_style_gauss2D(shape=(3, 3), sigma=1):
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m : m + 1, -n : n + 1]
            h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride, stride), sigma)
        if img.ndim == 3:
            img_w, img_h, img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w, img_h = img.shape
            img = img.reshape((img_w, img_h, 1))
        # from scipy import signal
        from scipy.ndimage.filters import convolve
        out_img = np.zeros((img_w // stride, img_h // stride, img_c))
        for i in range(img_c):
            out = convolve(img[:, :, i], h)
            out_img[:, :, i] = out[::stride, ::stride]
        return out_img

    def generate_LrHSI(self, img, scale_factor):
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        # np.random.seed(10)
        # SNRm = 30
        # w, h, c = np.shape(img_lr)
        # sigmam = np.math.sqrt(np.sum(img_lr) ** 2) / (10 ** (SNRm / 10)) / np.size(img_lr)
        # img_lr = img_lr.reshape(w*h, c)
        # img_lr = img_lr + sigmam * np.random.randn(np.shape(img_lr)[0], np.shape(img_lr)[1])
        # img_lr = img_lr.reshape(h, w, c)
        return img_lr

    def generate_HrMSI(self, img, sp_matrix):
        (h, w, c) = img.shape
        self.msi_channels = sp_matrix.shape[1]
        if sp_matrix.shape[0] == c:
            img_msi = np.dot(img.reshape(w * h, c), sp_matrix).reshape(h, w, sp_matrix.shape[1])
        else:
            raise Exception("The shape of sp matrix does not match the image")
        # np.random.seed(10)
        # SNRm = 35
        # sigmam = np.math.sqrt(np.sum(img_msi) ** 2) / (10 ** (SNRm / 10)) / np.size(img_msi)
        # img_msi = img_msi.reshape(w*h, self.msi_channels)
        # img_msi = img_msi + sigmam * np.random.randn(np.shape(img_msi)[0], np.shape(img_msi)[1])
        # img_msi = img_msi.reshape(h, w, self.msi_channels)
        return img_msi

    def __getitem__(self, index):
        img_patch = self.img_patch_list[index]
        img_lr = self.img_lr_list[index]
        img_msi = self.img_msi_list[index]
        img_name = os.path.basename(self.img_path_list[index]).split(".")[0]
        # print(img_name)
        # print("img_lr transpose",img_lr.transpose(2,0,1).shape)
        # Save as tensors 'c h w'
        img_tensor_lr = torch.from_numpy(img_lr.transpose(2, 0, 1).copy()).float()
        img_tensor_hr = torch.from_numpy(img_patch.transpose(2, 0, 1).copy()).float()
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2, 0, 1).copy()).float()
        # spectral manifold
        img_tensor_sm = torch.from_numpy(self.spectral_manifold.copy()).float()
        # spatial manifold
        img_tensor_spm1 = torch.from_numpy(self.spatial_manifold_1.copy()).float()
        img_tensor_spm2 = torch.from_numpy(self.spatial_manifold_2.copy()).float()
        return {
            "lhsi": img_tensor_lr,
            "hmsi": img_tensor_rgb,
            "hhsi": img_tensor_hr,
            "sm": img_tensor_sm,
            "spm1": img_tensor_spm1,
            "spm2": img_tensor_spm2,
            "name": img_name,
        }
        # "spm1": img_tensor_spm1,
        # "spm2": img_tensor_spm2,

    def __len__(self):
        return len(self.img_path_list)
