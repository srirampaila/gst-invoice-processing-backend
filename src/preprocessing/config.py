import cv2
import os

class PreprocessingConfig:
    def __init__(self):
        self.supported_extensions = {".png", ".jpg", ".jpeg"}
        
        # Resize
        self.upscale_factor = 2.0
        self.max_dimension = 4000  # Prevent unbounded memory usage
        
        # Denoise
        self.denoise_method = "gaussian" # gaussian, median, bilateral
        self.denoise_kernel = 3
        
        # Contrast
        self.contrast_method = "clahe" # clahe, hist
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        
        # Threshold
        self.threshold_method = "otsu" # otsu, adaptive
        self.adaptive_block_size = 11
        self.adaptive_c = 2
        
        # Deskew
        self.deskew_enabled = True
        self.deskew_max_angle = 15.0 # Max rotation to correct
        
        # Perspective
        self.perspective_enabled = False # Too unreliable without proper document boundaries
