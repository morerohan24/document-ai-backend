import os
import re
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError


DATE_PATTERN = r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"


def _find_tesseract() -> Optional[str]:
    configured_path = os.getenv("TESSERACT_CMD")
    if configured_path:
        return shutil.which(configured_path) or (
            configured_path if Path(configured_path).is_file() else None
        )

    executable = shutil.which("tesseract")
    if executable:
        return executable

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get(
        "ProgramFiles(x86)", r"C:\Program Files (x86)"
    )
    common_locations = (
        Path(program_files) / "Tesseract-OCR" / "tesseract.exe",
        Path(program_files_x86) / "Tesseract-OCR" / "tesseract.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    )
    return next(
        (str(path) for path in common_locations if path.is_file()), None
    )


def read_text(image: np.ndarray) -> str:
    executable = _find_tesseract()
    if executable is None:
        raise RuntimeError(
            "Tesseract OCR engine was not found. Install Tesseract and add "
            "its folder to PATH, or set TESSERACT_CMD to the full path of "
            "tesseract.exe."
        )

    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    except TesseractNotFoundError as error:
        raise RuntimeError(
            f"Tesseract could not start from {executable!r}. Check that the "
            "executable exists and is accessible."
        ) from error


def _value_after_label(lines: list[str], labels: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?:{labels})\s*[.:#\-]*\s*(.*)", re.IGNORECASE
    )
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            value = match.group(1).strip(" .:#-")
            if value:
                return value
            if index + 1 < len(lines):
                following = lines[index + 1].strip()
                if following:
                    return following
    return None


def _unlabeled_name(lines: list[str]) -> Optional[str]:
    metadata_labels = re.compile(
        r"\b(?:BLOOD\s+GROUP|DATE\s+OF\s+BIRTH|DOB|ADDRESS|CONTACT|"
        r"PHONE|MOBILE|UID|ID\s+NO|COURSE|CLASS|DEPARTMENT|VALID|"
        r"EXPIRY|GENDER|EMAIL)\b",
        re.IGNORECASE,
    )
    non_name_words = {
        "ADDRESS", "BLOOD", "CARD", "CLASS", "CONTACT", "COURSE",
        "DATE", "DEPARTMENT", "DOB", "DRIVER", "EMAIL", "EXPIRY",
        "GENDER", "GOVERNMENT", "GROUP", "IDENTITY", "LICENSE",
        "LICENCE", "MOBILE", "NAME", "NUMBER", "PHONE", "PROGRAM",
        "STUDENT", "UID", "UNIVERSITY", "VALID",
    }

    for line in lines:
        candidate = metadata_labels.split(line, maxsplit=1)[0].strip(" .,:;|_~")
        words = candidate.split()
        normalized_words = [re.sub(r"[^A-Za-z]", "", word) for word in words]
        if not 2 <= len(normalized_words) <= 4:
            continue
        if any(len(word) < 2 for word in normalized_words):
            continue
        if any(word.upper() in non_name_words for word in normalized_words):
            continue
        if candidate.isupper() or all(word[:1].isupper() for word in words):
            return " ".join(normalized_words)
    return None


def extract_fields(text: str) -> dict[str, Optional[str]]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    name = _value_after_label(
        lines,
        r"^\s*(?:FULL\s+NAME|CARDHOLDER(?:'S)?\s+NAME|NAME)\b",
    )
    if name:
        name = re.split(
            r"\s+(?:ID|DOCUMENT|LICENSE|LICENCE|DOB|DATE OF BIRTH)\b",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,:-")
        if not re.search(r"[A-Za-z]", name) or len(name) < 2:
            name = None
    if not name:
        name = _unlabeled_name(lines)

    document_number = _value_after_label(
        lines,
        r"(?:DOCUMENT\s*(?:NO|NUMBER)|UID\s*(?:NO|NUMBER)|"
        r"ID\s*(?:NO|NUMBER)|"
        r"(?:DRIVER'?S?\s*)?LICEN[CS]E\s*(?:NO|NUMBER)|DL\s*NO)",
    )
    if document_number:
        document_number = re.split(r"\s+(?:DOB|DATE OF BIRTH|EXPIRY|EXPIRES)\b",
                                   document_number, maxsplit=1,
                                   flags=re.IGNORECASE)[0].strip(" ,:-")
        document_number = document_number or None

    birth_line = _value_after_label(
        lines, r"(?:DATE OF BIRTH|BIRTH DATE|DOB|BORN)"
    )
    expiry_line = _value_after_label(
        lines, r"(?:DATE OF EXPIRY|EXPIRY DATE|EXPIRES|EXPIRATION|VALID THRU)"
    )
    date_of_birth = (
        re.search(DATE_PATTERN, birth_line).group(0)
        if birth_line and re.search(DATE_PATTERN, birth_line)
        else None
    )
    expiry_date = (
        re.search(DATE_PATTERN, expiry_line).group(0)
        if expiry_line and re.search(DATE_PATTERN, expiry_line)
        else None
    )

    return {
        "full_name": name,
        "document_number": document_number,
        "date_of_birth": date_of_birth,
        "expiry_date": expiry_date,
    }
