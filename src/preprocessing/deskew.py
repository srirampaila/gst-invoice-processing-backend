import cv2
import numpy as np

def deskew_image(img: np.ndarray, config) -> np.ndarray:
    if not config.deskew_enabled:
        return img
        
    # Assume image is grayscale or thresholded
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
        
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Limit angle
    if abs(angle) > config.deskew_max_angle:
        return img
        
    if abs(angle) < 0.5:
        return img
        
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
