import cv2
import os
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_image(filepath: str, config):
    path = Path(filepath)
    if not path.exists():
        logger.error("File does not exist")
        raise FileNotFoundError(f"File not found: {filepath}")
        
    if path.suffix.lower() not in config.supported_extensions:
        logger.error("Unsupported extension")
        raise ValueError(f"Unsupported file format: {path.suffix}")
        
    if path.stat().st_size == 0:
        logger.error("Zero-byte file")
        raise ValueError("File is empty (zero bytes)")
        
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Failed to decode image")
        raise ValueError("Corrupt or unreadable image file")
        
    return img

def save_image(img: np.ndarray, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, img)
