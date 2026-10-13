class OCRConfig:
    def __init__(self):
        self.language = "eng"
        self.psm = 6 # Assume a single uniform block of text
        self.oem = 3 # Default, based on what is available
        self.timeout = 30 # seconds
        self.confidence_threshold = 0.0 # for reporting
