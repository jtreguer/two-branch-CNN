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
import pickle
import psutil
import rasterio
import spectral
from tqdm import tqdm
import xlrd
from skimage.transform import resize
from utils import max_inner_rectangle


class AvirisDataset(torch.utils.data.Dataset):

    sigma = 0.5
    dataset_path = './dataset'

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.subres = int(self.args.patch_size/self.args.scale)
        print(f"Low resolution {self.subres}")
        self.dataset_filename = f"dataset_training_{self.args.stride}"
        self.sp_matrix = None
        self.wavelengths = None #spaceholder for image spectral wavelenghts
        # load images
        self.image_path = args.image_path
        self.image_list = self.get_image_list()
        print(len(self.image_list))
        self.image_list = self.image_list[:min(self.args.image_number, len(self.image_list))]
        print(f"Number of tiles {len(self.image_list)}")
        self.training_images_number = math.ceil(len(self.image_list)*self.args.training_ratio)
        self.image_ids = []
        self.GT_list = []
        self.HRMSI_list = []
        self.LRHSI_list = []
        self.test_GT_list = []
        self.test_HRMSI_list = []
        self.test_LRHSI_list = []
        if self.args.train_mode:
            if self.dataset_file():
                self.load_dataset()
                print("Dataset loaded from file")
            else:
                self.make_set()
                print("Dataset built")
                self.save_dataset()
            self.GT_tensor_list = self.make_cuda_tensor(self.GT_list, is_spectrum=True)
            self.LRHSI_tensor_list = self.make_cuda_tensor(self.LRHSI_list, is_spectrum=True)
            self.HRMSI_tensor_list = self.make_cuda_tensor(self.HRMSI_list)
            print("Dataset loaded as tensors")
        else:
            print("Making test dataset")
            self.make_test_set()
            print("Test dataset built")

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

    def dataset_file(self):
        target_file = os.path.join(self.dataset_path,self.dataset_filename+".pkl")
        return os.path.isfile(target_file)
    
    def save_dataset(self):
        target_file = os.path.join(self.dataset_path,self.dataset_filename+".pkl")
        payload = [self.GT_list, self.LRHSI_list, self.HRMSI_list, self.image_ids]
        with open(target_file,"wb") as f:
            pickle.dump(payload,f)      
    
    def load_dataset(self):
        target_file = os.path.join(self.dataset_path,self.dataset_filename+".pkl")
        with open(target_file,"rb") as f:
            self.GT_list, self.LRHSI_list, self.HRMSI_list, self.image_ids = pickle.load(f)

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
                hyperspectral_data = rearrange(hyperspectral_data,'c h w -> h w c')
                x, y, w, h = max_inner_rectangle(hyperspectral_data,-50) # Inner rectangle excluding -50
                valid_pixels = hyperspectral_data[y:y+h,x:x+w,:]
                print(f"Cropped image shape {valid_pixels.shape}")
                global_max = np.max(valid_pixels)
                global_min = np.min(valid_pixels)
                print(f"global min and max {global_min}, {global_max}")
                c = valid_pixels.shape[-1]
                p_row, p_col, start_row, start_col= self.compute_patch_number(h, w)
                patch_number = p_row * p_col
                print(f"Maximum number of patches {patch_number}")
                header_spectral = spectral.open_image(image_header)
                # Access the wavelengths associated with each band
                self.wavelengths = header_spectral.bands.centers
                self.sp_matrix = self.get_spectral_response()
                patch_count = 0
                for i in tqdm(range(patch_number)):
                    j = math.floor(i / p_col)
                    k = i % p_col
                    patch = valid_pixels[start_row+j*s:start_row+j*s+p,start_col+k*s:start_col+k*s+p,:]
                    if 0:#(patch == -50).any(): # exclude non valid pixels
                        continue
                    else:
                        # NORMALIZATION at IMAGE LEVEL
                        patch = self.normalize_image(patch,global_min=global_min, global_max=global_max)
                        self.GT_list.append(patch[centre, centre,:].astype(np.float16))
                        patch_msi = self.make_msi(patch)
                        self.HRMSI_list.append(patch_msi.astype(np.float16))
                        patch_hsi = self.make_hsi(patch)
                        patch_hsi = self.upscale_hyperspectral(patch_hsi, method='bicubic')
                        lr_spec  = patch_hsi[centre, centre, :]
                        self.LRHSI_list.append(lr_spec.astype(np.float16))
                        self.image_ids.append(image_id) # record from which image the sample comes from
                        patch_count += 1
                    # if i % 100 == 0:
                    #     print(f"RAM Used {psutil.virtual_memory().percent}")
                print(f"Effective number of patches extracted from current last tile {patch_count}")
                print(f"Total number of patches extracted {len(self.GT_list)}")
        return
    
    def make_test_set(self):
        with open('selected_for_training.pkl','rb') as f:
            selected = pickle.load(f)
        test_image_ids = [i for i in range(self.args.image_number) if i not in selected]
        for image_id, image_header in tqdm(enumerate(self.image_list)):
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
                    x, y, w, h = max_inner_rectangle(hyperspectral_data,-50) # Inner rectangle excluding -50
                    valid_pixels = hyperspectral_data[y:y+h,x:x+w,:]
                    print(f"Cropped image shape {valid_pixels.shape}")
                    global_max = np.max(valid_pixels)
                    global_min = np.min(valid_pixels)
                    print(f"global min and max {global_min}, {global_max}")
                    c = valid_pixels.shape[-1]
                    header_spectral = spectral.open_image(image_header)
                    self.wavelengths = header_spectral.bands.centers
                    self.sp_matrix = self.get_spectral_response()
                    valid_pixels = self.normalize_image(valid_pixels,global_min=global_min, global_max=global_max)
                    self.test_GT_list.append(valid_pixels)
                    self.test_HRMSI_list.append(self.make_msi(valid_pixels))
                    self.test_LRHSI_list.append(self.make_hsi(valid_pixels))
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
        method = cv2.INTER_AREA #cv2.INTER_AREA not supported with more than 4 channels, cv2.INTER_LINEAR?
        blurred_hs = GaussianBlur(img,(3,3),sigmaX=self.sigma, borderType=0)
        blurred_hs_normalized = np.empty(blurred_hs.shape, dtype=blurred_hs.dtype)
        # Normalization by channel - Global normalization?
        for j in range(len(self.wavelengths)):
            blurred_hs_normalized[:,:,j] = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # Downsampling   
        if self.args.train_mode:
            if method == cv2.INTER_LINEAR:
                lr_hsi = cv2.resize(blurred_hs_normalized,dsize=(self.subres,self.subres),interpolation=method)
            else: # use no aliasing alternative to cv2.INTER_AREA from skimage
                lr_hsi = resize(blurred_hs_normalized, (self.subres, self.subres, blurred_hs_normalized.shape[2]), \
                        anti_aliasing=True, preserve_range=True).astype(blurred_hs_normalized.dtype)                
        else:
            if method == cv2.INTER_LINEAR:
                lr_hsi = cv2.resize(blurred_hs_normalized,None,fx=1/self.args.scale,fy=1/self.args.scale,interpolation=method)
            else:
                h, w, c = blurred_hs_normalized.shape
                new_h = int(h / self.args.scale)
                new_w = int(w / self.args.scale)
                lr_hsi = resize(blurred_hs_normalized, (new_h, new_w, c), \
                        anti_aliasing=True, preserve_range=True).astype(blurred_hs_normalized.dtype)   
        return lr_hsi
    

    def upscale_hyperspectral(self, img: np.ndarray, method: str = 'bicubic') -> np.ndarray:
        """
        Upscales a hyperspectral image spatially using bilinear or bicubic interpolation.
        
        Parameters:
            img (np.ndarray): Hyperspectral image of shape (height, width, bands).
            scale (int): Upscaling factor (e.g., 4).
            method (str): Interpolation method: 'bilinear' or 'bicubic'.
            
        Returns:
            np.ndarray: Upscaled hyperspectral image of shape (height*scale, width*scale, bands).
        """
        assert img.ndim == 3, "HxWxC"
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
    print(training_set.__getitem__(0)[1])
    print(training_set.__getitem__(12)[1])
    # print(toto.get_image_list())
    # toto.test_resolutions()


    

