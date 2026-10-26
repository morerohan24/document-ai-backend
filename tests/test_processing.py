import cv2
import numpy as np
import pytest

import app.image_processing as image_processing
from app.image_processing import InvalidImageError, preprocess_image
from app.ocr import _find_tesseract, extract_fields


def test_extracts_labeled_identity_fields():
    result = extract_fields(
        "FULL NAME: JANE SAMPLE\n"
        "ID NUMBER: AB123456\n"
        "DOB: 04/12/1992\n"
        "EXPIRY DATE: 04/12/2030"
    )

    assert result == {
        "full_name": "JANE SAMPLE",
        "document_number": "AB123456",
        "date_of_birth": "04/12/1992",
        "expiry_date": "04/12/2030",
    }


def test_parses_student_badge_without_mistaking_course_name_for_person():
    result = extract_fields(
        "Dr. Example University Course Name: MCA\n"
        "Government UID No.: AB123456\n"
        "Address: SAMPLE ADDRESS\n"
        "ALEX MORGAN Blood Group: B+"
    )

    assert result["full_name"] == "ALEX MORGAN"
    assert result["document_number"] == "AB123456"


def test_leaves_unrecognized_fields_empty():
    assert extract_fields("STATE IDENTIFICATION CARD") == {
        "full_name": None,
        "document_number": None,
        "date_of_birth": None,
        "expiry_date": None,
    }


def test_preprocesses_image_when_no_card_boundary_is_found():
    image = np.full((240, 360, 3), 225, dtype=np.uint8)
    cv2.putText(image, "SAMPLE ID", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    success, encoded = cv2.imencode(".png", image)
    assert success

    prepared = preprocess_image(encoded.tobytes())

    assert prepared.document_detected is False
    assert prepared.original_width == 360
    assert prepared.original_height == 240
    assert prepared.ocr_image.shape == (800, 1200)


def test_detects_and_rectifies_portrait_orientation_card():
    image = np.full((800, 500, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (115, 100), (385, 700), (20, 20, 20), 8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    prepared = preprocess_image(encoded.tobytes())

    assert prepared.document_detected is True
    assert prepared.document.shape[0] > prepared.document.shape[1]
    assert prepared.ocr_image.shape[1] >= 1000
    assert prepared.ocr_image.shape[1] >= prepared.document.shape[1]


def test_rejects_non_image_upload():
    with pytest.raises(InvalidImageError):
        preprocess_image(b"not an image")


def test_rejects_image_over_pixel_limit(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    monkeypatch.setattr(image_processing, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(InvalidImageError, match="too large"):
        preprocess_image(encoded.tobytes())


def test_finds_tesseract_from_windows_install_location(tmp_path, monkeypatch):
    install_dir = tmp_path / "Tesseract-OCR"
    install_dir.mkdir()
    executable = install_dir / "tesseract.exe"
    executable.write_text("", encoding="utf-8")
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))

    assert _find_tesseract() == str(executable)


def test_honors_tesseract_command_override(tmp_path, monkeypatch):
    executable = tmp_path / "custom-tesseract.exe"
    executable.write_text("", encoding="utf-8")
    monkeypatch.setenv("TESSERACT_CMD", str(executable))

    assert _find_tesseract() == str(executable)


def test_reports_invalid_tesseract_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing.exe"))

    assert _find_tesseract() is None
