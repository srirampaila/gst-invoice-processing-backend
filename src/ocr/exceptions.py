class OCRError(Exception):
    pass

class OCREngineNotAvailableError(OCRError):
    pass

class OCRProcessingError(OCRError):
    pass
