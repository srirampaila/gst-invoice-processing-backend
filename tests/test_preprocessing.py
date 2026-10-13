import os
import cv2
import numpy as np
import pytest
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.image_loader import load_image
from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.resize import upscale_image
from src.preprocessing.denoise import reduce_noise
from src.preprocessing.contrast import enhance_contrast
from src.preprocessing.deskew import deskew_image
from src.preprocessing.threshold import apply_threshold
from src.preprocessing.pipeline import preprocess_image

def create_test_image(path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img.fill(255)
    cv2.imwrite(path, img)
    return img

def test_image_loader(tmp_path):
    config = PreprocessingConfig()
    p = str(tmp_path / "test.png")
    create_test_image(p)
    img = load_image(p, config)
    assert img is not None
    assert img.shape == (100, 100, 3)

def test_grayscale():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = convert_to_grayscale(img)
    assert len(gray.shape) == 2

def test_resize():
    config = PreprocessingConfig()
    config.upscale_factor = 2.0
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = upscale_image(img, config)
    assert res.shape == (200, 200, 3)

def test_pipeline(tmp_path):
    config = PreprocessingConfig()
    p_in = str(tmp_path / "in.png")
    p_out = str(tmp_path / "out.png")
    create_test_image(p_in)
    
    res = preprocess_image(p_in, p_out, config)
    assert res["success"]
    assert os.path.exists(p_out)
    
def test_deskew():
    config = PreprocessingConfig()
    img = np.zeros((100, 100), dtype=np.uint8)
    # just test it doesn't crash on empty
    res = deskew_image(img, config)
    assert res.shape == (100, 100)
