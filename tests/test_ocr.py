import pytest
import os
import cv2
import numpy as np
from src.ocr.config import OCRConfig
from src.ocr.service import extract_text
from src.ocr.exceptions import OCREngineNotAvailableError
from src.ocr.engine import TesseractEngine

def create_test_image(path):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img.fill(255)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(path, img)
    return img

def test_ocr_config():
    config = OCRConfig()
    assert config.language == "eng"
    config.language = "deu"
    assert config.language == "deu"

def test_tesseract_availability():
    config = OCRConfig()
    try:
        engine = TesseractEngine(config)
        assert engine is not None
    except OCREngineNotAvailableError:
        pytest.skip("Tesseract not installed, skipping test.")

def test_ocr_extraction(tmp_path):
    config = OCRConfig()
    try:
        TesseractEngine(config)
    except OCREngineNotAvailableError:
        pytest.skip("Tesseract not installed, skipping test.")
        
    p = str(tmp_path / "test_ocr.png")
    create_test_image(p)
    
    res = extract_text(p, config)
    assert res.success
    assert "TEST" in res.raw_text.upper()
    assert res.word_count > 0

def test_ocr_missing_file():
    config = OCRConfig()
    res = extract_text("nonexistent_file.png", config)
    assert not res.success
    assert "not found" in res.error.lower()
