import cv2
import bm3d
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from numba import njit
import pywt

def nl_means_filter(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(img))
    denoised_image = denoise_nl_means(img,h=1.*sigma_est,
                                        fast_mode=False,patch_size=5,
                                        patch_distance=3)
    return denoised_image


def bm3d_filter(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    denoised_image = bm3d.bm3d(img,sigma_psd=0.2,stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised_image  

@njit
def sigma_filter(img, window_size=3, sigma=0.03):
    pad = window_size // 2
    h, w = img.shape
    denoised_image = np.zeros_like(img)
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            sum_val = 0.0
            count = 0
            center = img[y, x]
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    ny, nx = y + dy, x + dx
                    val = img[ny, nx]
                    if abs(val - center) <= sigma:
                        sum_val += val
                        count += 1
            if count > 0:
                denoised_image[y, x] = sum_val / count
            else:
                denoised_image[y, x] = center
   
    return denoised_image



def wavelet_filter(img, wavelet_level=3, sigma_est=None):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    if sigma_est is None:
        coeffs2 = pywt.dwt2(img, 'db1')
        LL, (LH, HL, HH) = coeffs2
        sigma_est = np.median(np.abs(HH)) / 0.6745

    coeffs = pywt.wavedec2(img, 'db1', level=wavelet_level)

    threshold = sigma_est * np.sqrt(2 * np.log(img.size))
    new_coeffs = []
    new_coeffs.append(coeffs[0]) 
    for i in range(1, len(coeffs)):
        new_coeffs.append(tuple(pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[i]))
    
    denoised_image = pywt.waverec2(new_coeffs, 'db1')

    return denoised_image

#extract fingerprint function
def extract_fingerprint(img, denoise_filter):
    '''
    This takes an image and filter as input.
    '''
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img_as_float(img)
    denoised = denoise_filter(img)
    return img - denoised  # raw fingerprint (float)


#for robust visualization of the fingerprint
def fingerprint_to_uint8(fp, method="robust", k=3.0):
    med = np.median(fp)
    mad = np.median(np.abs(fp - med)) + 1e-12
    sigma = mad / 0.6745
    T = max(k * sigma, 1e-6)
    fp_c = np.clip(fp - med, -T, T)
    vis = (fp_c + T) / (2 * T)
    return img_as_ubyte(vis)



img = img_as_float(io.imread('/home/chinasa/python_projects/denoising/images/synthetic/plus/cycleGAN/all_rs/reference/001-PLUS-FV3-Laser_PALMAR_001_01_02_01.png',as_gray=True))

fingerprint = extract_fingerprint(img,nl_means_filter)
fingerprint_vis = fingerprint_to_uint8(fingerprint)
io.imsave("/home/chinasa/python_projects/denoising/output/fingerprint.png", fingerprint_vis)

