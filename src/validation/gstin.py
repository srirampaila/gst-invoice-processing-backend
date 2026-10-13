"""
Deterministic Indian GSTIN (Goods and Services Tax Identification Number) Validation.
Implements format, state code, PAN structure, and Luhn Mod-36 checksum algorithms.
"""
import re
from typing import Optional, Dict
from src.validation.models import GSTINValidationResult

# Standard Indian State / Union Territory Codes under GST
INDIAN_STATE_CODES: Dict[str, str] = {
    "01": "Jammu and Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Daman and Diu",
    "26": "Dadra and Nagar Haveli",
    "27": "Maharashtra",
    "28": "Andhra Pradesh (Old)",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman and Nicobar Islands",
    "36": "Telangana",
    "37": "Andhra Pradesh (New)",
    "38": "Ladakh",
    "97": "Other Territory",
    "99": "Centre Jurisdiction"
}

# Regex for GSTIN: 2 digits + 5 alpha + 4 digits + 1 alpha + 1 alpha-num + 'Z' + 1 alpha-num
GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")

# Luhn Mod-36 Charset
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {c: i for i, c in enumerate(CHARS)}

def compute_gstin_checksum(gstin_14: str) -> str:
    """
    Computes the 15th check digit for a 14-character GSTIN prefix
    using the official Indian GST Luhn Mod-36 algorithm.
    """
    if len(gstin_14) < 14:
        raise ValueError(f"Expected at least 14 characters, got {len(gstin_14)}")
        
    total = 0
    for idx, char in enumerate(gstin_14[:14].upper()):
        if char not in CHAR_MAP:
            return ""
        val = CHAR_MAP[char]
        factor = 1 if (idx % 2 == 0) else 2
        product = val * factor
        quotient, remainder = divmod(product, 36)
        total += quotient + remainder
        
    remainder = total % 36
    check_code = (36 - remainder) % 36
    return CHARS[check_code]

def get_state_name(state_code: Optional[str]) -> Optional[str]:
    """Resolves 2-digit state code to full name."""
    if not state_code:
        return None
    return INDIAN_STATE_CODES.get(state_code.strip())

def validate_gstin(
    gstin: Optional[str],
    strict_checksum: bool = False
) -> GSTINValidationResult:
    """
    Deterministic validation of an Indian GSTIN string.
    
    Checks:
    1. Null / emptiness
    2. Exactly 15 characters
    3. Structural regex compliance
    4. Valid 2-digit State / UT code
    5. Entity structure and default 'Z' character at position 14
    6. Luhn Mod-36 checksum calculation
    """
    if not gstin or not gstin.strip():
        return GSTINValidationResult(
            gstin=None,
            valid=False,
            errors=["GSTIN is empty or missing"]
        )
        
    clean_gstin = gstin.strip().upper()
    errors = []
    warnings = []
    
    # Check 1: Length
    if len(clean_gstin) != 15:
        errors.append(f"GSTIN must be exactly 15 characters, found {len(clean_gstin)}")
        return GSTINValidationResult(
            gstin=clean_gstin,
            valid=False,
            errors=errors
        )
        
    # Check 2: State Code
    state_code = clean_gstin[:2]
    state_name = INDIAN_STATE_CODES.get(state_code)
    if not state_name:
        errors.append(f"Invalid state code '{state_code}' in GSTIN")
        
    # Check 3: Format Regex
    if not GSTIN_REGEX.match(clean_gstin):
        errors.append(f"GSTIN '{clean_gstin}' does not match standard GST format pattern")
        
    # Check 4: PAN Structure (positions 3 to 12)
    pan = clean_gstin[2:12]
    entity_code = clean_gstin[12]
    checksum_char = clean_gstin[14]
    
    # Check 5: 14th Character must be 'Z'
    if clean_gstin[13] != "Z":
        errors.append(f"14th character of GSTIN must be 'Z', found '{clean_gstin[13]}'")
        
    # Check 6: Checksum Calculation
    calculated_checksum = None
    checksum_valid = False
    try:
        calculated_checksum = compute_gstin_checksum(clean_gstin[:14])
        checksum_valid = (calculated_checksum == checksum_char)
        if not checksum_valid:
            if strict_checksum:
                errors.append(
                    f"GSTIN checksum mismatch: expected '{calculated_checksum}', got '{checksum_char}'"
                )
            else:
                warnings.append(
                    f"GSTIN checksum variance (synthetic/non-strict): expected '{calculated_checksum}', got '{checksum_char}'"
                )
    except Exception as e:
        warnings.append(f"Could not calculate GSTIN checksum: {e}")
        
    is_valid = (len(errors) == 0)
    
    return GSTINValidationResult(
        gstin=clean_gstin,
        valid=is_valid,
        state_code=state_code,
        state_name=state_name,
        pan=pan,
        entity_code=entity_code,
        checksum_char=checksum_char,
        calculated_checksum=calculated_checksum,
        checksum_valid=checksum_valid,
        errors=errors,
        warnings=warnings
    )
