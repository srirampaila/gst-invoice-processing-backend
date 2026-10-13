import cv2
import numpy as np

def apply_threshold(img: np.ndarray, config) -> np.ndarray:
    if config.threshold_method == "otsu":
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif config.threshold_method == "adaptive":
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            config.adaptive_block_size, config.adaptive_c
        )
    return img
