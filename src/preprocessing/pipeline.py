import time
import cv2
import numpy as np
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.image_loader import load_image, save_image
from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.resize import upscale_image
from src.preprocessing.denoise import reduce_noise
from src.preprocessing.contrast import enhance_contrast
from src.preprocessing.deskew import deskew_image
from src.preprocessing.threshold import apply_threshold

def preprocess_image(input_path: str, output_path: str, config: PreprocessingConfig = None):
    if config is None:
        config = PreprocessingConfig()
        
    start_time = time.time()
    
    # 1. Load
    img = load_image(input_path, config)
    orig_h, orig_w = img.shape[:2]
    
    # 2. Grayscale
    img = convert_to_grayscale(img)
    
    # 3. Upscale
    img = upscale_image(img, config)
    
    # 4. Denoise
    img = reduce_noise(img, config)
    
    # 5. Contrast
    img = enhance_contrast(img, config)
    
    # 6. Thresholding
    # Usually we threshold before deskewing to find text, but let's invert for deskew
    thresh_for_deskew = cv2.bitwise_not(apply_threshold(img, config))
    
    # 7. Deskew
    if config.deskew_enabled:
        # We deskew the original enhanced image, not the thresholded one
        # but we compute angle using the thresholded text mask
        coords = np.column_stack(np.where(thresh_for_deskew > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if 0.5 < abs(angle) <= config.deskew_max_angle:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # 8. Final Thresholding (Optional, but useful for pure OCR text)
    # Tesseract often likes grayscale with good contrast better than pure binary
    # We will save the contrast-enhanced grayscale as final.
    # If binary is strictly required, uncomment the next line:
    # img = apply_threshold(img, config)
    
    # 9. Save
    save_image(img, output_path)
    
    end_time = time.time()
    
    return {
        "input_dimensions": (orig_w, orig_h),
        "output_dimensions": (img.shape[1], img.shape[0]),
        "processing_time": end_time - start_time,
        "success": True
    }
