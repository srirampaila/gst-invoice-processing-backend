import cv2
import numpy as np

def enhance_contrast(img: np.ndarray, config) -> np.ndarray:
    if config.contrast_method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid_size)
        return clahe.apply(img)
    elif config.contrast_method == "hist":
        return cv2.equalizeHist(img)
    return img
