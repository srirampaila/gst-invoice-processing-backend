import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.dataset_validator import DatasetValidator

def test_dataset_integrity():
    validator = DatasetValidator()
    valid, errors = validator.validate_integrity()
    assert valid, f"Dataset integrity validation failed: {errors}"
