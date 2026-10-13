import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Verify that core modules can be imported successfully."""
    try:
        import app
        import src
        import src.preprocessing
        import src.ocr
        import src.extraction
        import src.validation
        import src.confidence
        import src.gst
        import src.export
        import src.utils
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_logger_import():
    """Verify the logger utility works."""
    from src.utils.logger import get_logger
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"

def test_environment_vars():
    """Verify dotenv can load placeholder variables."""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / ".env.example")
    assert os.getenv("APP_ENV") == "development"
    assert os.getenv("LOG_LEVEL") == "INFO"
