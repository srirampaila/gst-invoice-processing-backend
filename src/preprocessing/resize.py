import cv2
import numpy as np

def upscale_image(img: np.ndarray, config) -> np.ndarray:
    factor = config.upscale_factor
    if factor <= 1.0:
        return img
        
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    if max(new_h, new_w) > config.max_dimension:
        # Scale to max dimension
        scale = config.max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
    if new_h == h and new_w == w:
        return img
        
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
