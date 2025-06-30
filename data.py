import torch
import numpy as np
import cv2
from cv2 import GaussianBlur
from einops import rearrange
from glob import glob
import matplotlib.pyplot as plt #DEBUG
import argparse
import math
import os
import rasterio
import spectral
from tqdm import tqdm
import xlrd


    

class AvirisDataset(torch.utils.data.Dataset):

    sigma = 0.5

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.subres = int(self.args.patch_size/self.args.scale)
        print(f"Low resolution {self.subres}")
        self.sp_matrix = None
        self.wavelengths = None #spaceholder for image spectral wavelenghts
        # load images
        self.image_path = args.image_path
        self.image_list = self.get_image_list()
        print(len(self.image_list))
        self.image_list = self.image_list[:min(self.args.image_number, len(self.image_list))]
        print(f"Number of tiles {len(self.image_list)}")
        self.training_images_number = int(len(self.image_list)*self.args.training_ratio)
        self.image_ids = []
        self.GT_list = []
        self.HRMSI_list = []
        self.LRHSI_list = []
        self.test_GT_list = []
        self.test_HRMSI_list = []
        self.test_LRHSI_list = []
        if self.args.train_mode:
            self.make_set()
            print("Dataset built")
            # Supprimer les fréquences de la vapeur d'eau ?

            # Make tensor lists
            # In c h w order for the model to digest  
            self.GT_tensor_list = self.make_cuda_tensor(self.GT_list, is_spectrum=True)
            self.LRHSI_tensor_list = self.make_cuda_tensor(self.LRHSI_list, is_spectrum=True)
            self.HRMSI_tensor_list = self.make_cuda_tensor(self.HRMSI_list)
            print("Dataset loaded as tensors")
        pass

    def __getitem__(self, index):
          return self.GT_tensor_list[index], self.LRHSI_tensor_list[index], self.HRMSI_tensor_list[index]
    
    def __len__(self):
        return len(self.GT_list)
  
    def make_cuda_tensor(self, arr_list, is_spectrum=False):
        tensor_list = []
        for arr in arr_list:
            if is_spectrum:
                # continue                     # (c,) → (1,c) and no HWC→CHW swap
                tensor = torch.from_numpy(arr).float().unsqueeze(0)
            else:                                 # (H,W,C) → (C,H,W)
                arr = rearrange(arr,'h w c-> c h w')
                tensor = torch.from_numpy(arr).float()
            tensor_list.append(tensor)
        return tensor_list

    def get_image_list(self):
        hdr_files = []
        for dirpath, _, filenames in os.walk(self.image_path):
            # Check if the directory name starts with 'f111'
            if os.path.basename(dirpath).startswith("f111"):
                for file in filenames:
                    if file.endswith("sc01_ort_img.hdr"):
                        hdr_files.append(os.path.join(dirpath, file))
        return hdr_files

    def get_spectral_response(self):
        xls_path = self.args.srf_path
        if not os.path.exists(xls_path):
            raise Exception("Spectral response path does not exist!")
        data = xlrd.open_workbook(xls_path)
        srf = data.sheets()[0]
        srf_arr = np.array([srf.col_values(i) for i in range(srf.ncols)]).T
        sp_matrix = np.empty((len(self.wavelengths),srf.ncols-1),dtype=np.float32)
        for i in range(1,srf.ncols): # start from 1 to exclude 1st column = wavelengths
            sp_matrix[:,i-1] = np.interp(self.wavelengths, srf_arr[:,0], srf_arr[:,i],left=0, right=0)
        # print(f"sp_matrix.shape {sp_matrix.shape}")
        # print(f"Min and max of sp_matrix {np.min(sp_matrix / sp_matrix.sum(axis=0))}, {np.max(sp_matrix/ sp_matrix.sum(axis=0))}")
        return sp_matrix / sp_matrix.sum(axis=0)

    def make_set(self):
        p = self.args.patch_size
        s = self.args.stride
        centre = self.args.patch_size // 2   
        for image_id, image_header in enumerate(self.image_list):
            image_name = image_header[:-4]
            with rasterio.open(image_name) as src:
                print(f"Reading HS data from {image_name}")
                hyperspectral_data = src.read()
                # Display information about the hyperspectral data
                print('Shape of hyperspectral data:', hyperspectral_data.shape)
                print('Number of bands:', src.count)
                c , h, w = hyperspectral_data.shape
                p_row, p_col, start_row, start_col= self.compute_patch_number(h, w)
                patch_number = p_row * p_col
                print(f"Number of patches {patch_number}")
                hyperspectral_data = rearrange(hyperspectral_data,'c h w -> h w c')
                global_max = np.max(hyperspectral_data)
                global_min = np.min(hyperspectral_data)
                print(f"global min and max {global_min}, {global_max}")
                header_spectral = spectral.open_image(image_header)
                # Access the wavelengths associated with each band
                self.wavelengths = header_spectral.bands.centers
                self.sp_matrix = self.get_spectral_response()


                for i in tqdm(range(patch_number)):
                    j = math.floor(i / p_col)
                    k = i % p_col
                    patch = hyperspectral_data[start_row+j*s:start_row+j*s+p,start_col+k*s:start_col+k*s+p,:]
                    # NORMALIZATION at IMAGE LEVEL
                    patch = self.normalize_image(patch,global_min=global_min, global_max=global_max)
                    self.GT_list.append(patch[centre, centre,:])
                    patch_msi = self.make_msi(patch)
                    self.HRMSI_list.append(patch_msi)
                    patch_hsi = self.make_hsi(patch)
                    patch_hsi = self.upscale_hyperspectral(patch_hsi, method='bicubic')
                    lr_spec  = patch_hsi[centre, centre, :]
                    self.LRHSI_list.append(lr_spec)
                    self.image_ids.append(image_id) # record from which image the sample comes from
        return
    
    def make_test_set(self, test_image_ids):
        for image_id, image_header in enumerate(self.image_list):
            if image_id in test_image_ids:
                image_name = image_header[:-4]
                with rasterio.open(image_name) as src:
                    print(f"Reading HS data from {image_name}")
                    hyperspectral_data = src.read()
                    # Display information about the hyperspectral data
                    print('Shape of hyperspectral data:', hyperspectral_data.shape)
                    print('Number of bands:', src.count)
                    c , h, w = hyperspectral_data.shape
                    hyperspectral_data = rearrange(hyperspectral_data,'c h w -> h w c')
                    global_max = np.max(hyperspectral_data)
                    global_min = np.min(hyperspectral_data)
                    header_spectral = spectral.open_image(image_header)
                    self.wavelengths = header_spectral.bands.centers
                    self.sp_matrix = self.get_spectral_response()
                    hyperspectral_data = self.normalize_image(hyperspectral_data,global_min=global_min, global_max=global_max)
                    self.test_GT_list.append(hyperspectral_data)
                    self.test_HRMSI_list.append(self.make_msi(hyperspectral_data))
                    self.test_LRHSI_list.append(self.make_hsi(hyperspectral_data))
        return self.test_GT_list, self.test_HRMSI_list, self.test_LRHSI_list

    def make_msi(self, img):
        (h, w, c) = img.shape
        self.msi_channels = self.sp_matrix.shape[1]
        if self.sp_matrix.shape[0] == c:
            img_msi = np.dot(img.reshape(w * h, c), self.sp_matrix).reshape(h, w, self.sp_matrix.shape[1])
        else:
            raise Exception("The shape of sp matrix does not match the image")
        return img_msi

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
    
   
    def test_resolutions(self):
        for image_header in self.image_list:
            image_name = image_header[:-4] # ignore .hdr extension to get image name
            with rasterio.open(image_name) as src:
                print(f"Reading HS data from {image_name}")
                hyperspectral_data = src.read()
                # Display information about the hyperspectral data
                print('Shape of hyperspectral data:', hyperspectral_data.shape)
                print('Number of bands:', src.count)
                c , h, w = hyperspectral_data.shape
                print(self.compute_patch_number(h,w))
        return

    def compute_patch_number(self,h,w):
        # Compute the total number of patches extractible from the image
        vertical_number = math.floor((h - self.args.patch_size) / self.args.stride) + 1
        horizontal_number = math.floor((w - self.args.patch_size) / self.args.stride) + 1
        start_row = (h - (vertical_number - 1)* self.args.stride - self.args.patch_size) // 2
        start_col = (w - (horizontal_number - 1)* self.args.stride - self.args.patch_size) // 2
        return vertical_number, horizontal_number, start_row, start_col
    
    def make_hsi(self, img):
        # Gaussian blur, 3x3 kernel, then scale reduction
        blurred_hs = GaussianBlur(img,(3,3),sigmaX=self.sigma, borderType=0)
        blurred_hs_normalized = np.empty(blurred_hs.shape, dtype=blurred_hs.dtype)
        # Normalization by channel - Global normalization?
        for j in range(len(self.wavelengths)):
            blurred_hs_normalized[:,:,j] = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # Downsampling    
        lr_hsi =cv2.resize(blurred_hs_normalized,(self.subres,self.subres),interpolation=cv2.INTER_NEAREST)
        return lr_hsi
    

    
    def upscale_hyperspectral(self, img: np.ndarray, method: str = 'bicubic') -> np.ndarray:
        """
        Upscales a hyperspectral image spatially using bilinear or bicubic interpolation.
        
        Parameters:
            img (np.ndarray): Hyperspectral image of shape (bands, height, width).
            scale (int): Upscaling factor (e.g., 4).
            method (str): Interpolation method: 'bilinear' or 'bicubic'.
            
        Returns:
            np.ndarray: Upscaled hyperspectral image of shape (bands, height*scale, width*scale).
        """
        assert img.ndim == 3, "Input image must have shape (bands, height, width)"
        assert method in ['bilinear', 'bicubic'], "Method must be 'bilinear' or 'bicubic'"
        
        interp = cv2.INTER_LINEAR if method == 'bilinear' else cv2.INTER_CUBIC
        h, w, bands = img.shape
        scale = self.args.scale
        upscaled = np.zeros((h * scale, w * scale, bands), dtype=img.dtype)
        
        for b in range(bands):
            upscaled[:,:,b] = cv2.resize(img[:,:,b], (w * scale, h * scale), interpolation=interp)
        
        return upscaled
        
    def upscale_hsi_np(self, lr_hsi: np.ndarray,
                    scale: int = 4,
                    interp: str = "bicubic") -> np.ndarray:
        """
        Upscale a low-resolution hyperspectral image (H, W, C) with OpenCV.

        Parameters
        ----------
        lr_hsi : np.ndarray
            Input cube, shape (H, W, C), any numeric dtype.
        scale : int, default 4
            Spatial magnification factor (must be ≥1).
        interp : {"nearest", "bilinear", "bicubic", "lanczos"}, default "bicubic"
            Interpolation kernel to use.

        Returns
        -------
        np.ndarray
            Up-scaled cube, shape (H*scale, W*scale, C), same dtype as input.
        """
        if lr_hsi.ndim != 3:
            raise ValueError("Input must have shape (H, W, C)")

        h, w, _ = lr_hsi.shape
        target_size = (w * scale, h * scale)          # cv2 expects (width, height)

        interps = {
            "nearest":  cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic":  cv2.INTER_CUBIC,
            "lanczos":  cv2.INTER_LANCZOS4,
        }
        if interp not in interps:
            raise ValueError(f"`interp` must be one of {list(interps)}")

        # Convert to float32 for accurate interpolation, then cast back
        up = cv2.resize(lr_hsi.astype(np.float32), target_size,
                        interpolation=interps[interp])

        return up.astype(lr_hsi.dtype)


# class NPZDataset(torch.utils.data.Dataset):
#     def __init__(self,data_path,get_graph=False):
#         super(NPZDataset, self).__init__()
#         loaded_data = np.load(data_path)
#         self.LRHSI_list = loaded_data['LRHSI']
#         self.RGB_list = loaded_data['RGB']
#         self.GT_list = loaded_data['GT']

#     def __getitem__(self, index):
#         return torch.from_numpy(self.GT_list[index]).float(), \
#                 torch.from_numpy(self.LRHSI_list[index]).float(),\
#                 torch.from_numpy(self.RGB_list[index]).float()
    
#     def __len__(self):
#         return len(self.GT_list)
    
# class ChikuseiDataset(torch.utils.data.Dataset):

#     sigma = 0.5
#     target_wavelengths = [0.49,0.56,0.665,0.89]
#     sigma_filter = 0.05

#     def __init__(self,full_image: np.array, training_zone: list,  wave_vector: np.array, device: torch.device, scale: int=4,gt_size: int=64):
#         super().__init__()
#         self.device = device
#         self.full_image = full_image.astype(np.float32) # TYPE CONVERSION
#         self.scale = scale
#         self.gt_size = gt_size
#         self.training_zone = training_zone #defined by (x0,y0,x1,y1)
#         self.width = training_zone[2] - training_zone[0]
#         self.height = training_zone[3] - training_zone[1]
#         self.wave_vector = wave_vector  
#         self.channels = full_image.shape[-1]
#         self.subres = int(gt_size/scale)
#         # In h w c order
#         self.GT_list = self.make_gt()
#         self.LRHSI_list = self.make_lr_hs()
#         self.HRMSI_list = self.make_hr_ms()
#         # Normalize GT
#         for gt in self.GT_list:
#             for k in range(len(self.wave_vector)):
#                 gt[:,:,k] = cv2.normalize(gt[:,:,k],None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
#         self.dataset_size = len(self.GT_list)
#         # Make tensor lists
#         # In c h w order for the model to digest  
#         self.GT_tensor_list = self.make_cuda_tensor(self.GT_list)
#         self.LRHSI_tensor_list = self.make_cuda_tensor(self.LRHSI_list)
#         self.HRMSI_tensor_list = self.make_cuda_tensor(self.HRMSI_list)
#         # DEBUG
#       #   lr = self.LRHSI_tensor_list[0].cpu().detach().numpy()[3,:,:]
#       #   hr = self.HRMSI_tensor_list[0].cpu().detach().numpy()[3,:,:]
#       #   print("DEBUG")
#       #   plt.imshow(lr)
#       #   plt.show()
#       #   plt.imshow(hr)
#       #   plt.show()


#     def __getitem__(self, index):
#           return self.GT_tensor_list[index], self.LRHSI_tensor_list[index], self.HRMSI_tensor_list[index]
    
#     def __len__(self):
#         return len(self.GT_list)
  
#     def make_cuda_tensor(self, arr_list):
#         tensor_list = []
#         for arr in arr_list:
#             arr = rearrange(arr,'h w c-> c h w')
#             tensor_list.append(torch.from_numpy(arr).float().to(self.device))
#         return tensor_list

#     def make_lr_hs(self):
#         # Gaussian blur, 3x3 kernel, then scale reduction
#         lr_hs_chikusei = []
#         for i in range(len(self.GT_list)):
#             blurred_hs = GaussianBlur(self.GT_list[i],(3,3),sigmaX=self.sigma, borderType=0)
#             if 1 :
#                 blurred_hs_normalized = np.empty(blurred_hs.shape, dtype=blurred_hs.dtype)
#                 for j in range(len(self.wave_vector)):
#                     blurred_hs_normalized[:,:,j] = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#             if 0:
#                  blurred_hs_normalized = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#             # Downsampling    
#             lr_hs_chikusei.append(cv2.resize(blurred_hs_normalized,(self.subres,self.subres),interpolation=cv2.INTER_NEAREST))
#         return lr_hs_chikusei
        
#     def make_hr_ms(self):
#         ms_list = []
#         for j in range(len(self.GT_list)):
#             ms = np.empty((self.gt_size, self.gt_size,len(self.target_wavelengths)))
#             for i,wl in enumerate(self.target_wavelengths):
#                 filter = self.gaussian_response(self.wave_vector,wl,self.sigma_filter)
#                 filter /= np.max(filter)
#                 ms[:,:,i] = np.max(self.GT_list[j]*filter.reshape(1,1,len(filter)),axis=2)
#                 ms[:,:,i] = cv2.normalize(ms[:,:,i],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#             ms_list.append(ms)
#         return ms_list

#     def make_gt(self):
#         i_range = self.height // self.gt_size
#         j_range = self.width // self.gt_size
#         # print(i_range, j_range)
#         GT_list = []
#         x0 = self.training_zone[0]
#         y0 = self.training_zone[1]
#         for i in range(i_range):
#             for j in range(j_range):
#                 target_zone = self.full_image[y0+i*self.gt_size:y0+(i+1)*self.gt_size,x0+j*self.gt_size:x0+(j+1)*self.gt_size,:]
#                 GT_list.append(target_zone)
#       #   print("First target zone")
#       #   plt.matshow(GT_list[0][:,:,3]) 
#         return GT_list
    
#     def gaussian_response(self, x, mean, sigma):
#         norm = 1/(sigma*np.sqrt(2*np.pi))
#         return norm*np.exp(-0.5*((x-mean)/sigma)**2)
    

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    # parse.add_argument('--image_path', type=str, default='./images/')
    parse.add_argument('--image_path', type=str, default='/mnt/c/data/AVIRIS/')
    parse.add_argument('--image_number', type=int, default=1)
    parse.add_argument('--patch_size', type=int, default=31)
    parse.add_argument('--stride', type=int, default=31)
    parse.add_argument('--training_ratio', type=int, default=0.7)
    parse.add_argument('--training_set', type=bool, default=True)
    parse.add_argument('--scale', type=int, default=2)   
    parse.add_argument('--srf_path', type=str, default='srf/Landsat8_BGRI_SRF.xls')
    parse.add_argument('--train_mode', type=int, default=True)  

    args = args = parse.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_set = AvirisDataset(args,device=device)
    print(training_set.GT_tensor_list[0].shape, training_set.GT_list[0].shape)
    print(training_set.HRMSI_tensor_list[0].shape, training_set.LRHSI_tensor_list[0].shape)
    # print(toto.get_image_list())
    # toto.test_resolutions()


    

