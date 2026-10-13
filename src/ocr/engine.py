import pytesseract
import cv2
import pandas as pd
import numpy as np
from src.ocr.config import OCRConfig
from src.ocr.exceptions import OCREngineNotAvailableError, OCRProcessingError

class TesseractEngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        self._check_availability()
        
    def _check_availability(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise OCREngineNotAvailableError("Tesseract executable is not installed or not in PATH.")
        except Exception as e:
            raise OCREngineNotAvailableError(f"Tesseract check failed: {e}")
            
    def extract_text(self, img: np.ndarray) -> str:
        custom_config = f'--oem {self.config.oem} --psm {self.config.psm}'
        try:
            return pytesseract.image_to_string(img, lang=self.config.language, config=custom_config, timeout=self.config.timeout)
        except Exception as e:
            raise OCRProcessingError(f"Tesseract processing failed: {e}")
            
    def extract_data(self, img: np.ndarray) -> pd.DataFrame:
        custom_config = f'--oem {self.config.oem} --psm {self.config.psm}'
        try:
            return pytesseract.image_to_data(img, lang=self.config.language, config=custom_config, output_type=pytesseract.Output.DATAFRAME, timeout=self.config.timeout)
        except Exception as e:
            raise OCRProcessingError(f"Tesseract data extraction failed: {e}")
