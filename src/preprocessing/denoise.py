import cv2
import numpy as np

def reduce_noise(img: np.ndarray, config) -> np.ndarray:
    k = config.denoise_kernel
    if config.denoise_method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    elif config.denoise_method == "median":
        return cv2.medianBlur(img, k)
    elif config.denoise_method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    return img
