from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_PIXELS


class InvalidImageError(ValueError):
    pass


@dataclass
class PreparedImage:
    original_width: int
    original_height: int
    document: np.ndarray
    ocr_image: np.ndarray
    document_detected: bool


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImageError("Upload a JPEG, PNG, or WebP image.")
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("The image is too large to process.")
            source.verify()
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as error:
        raise InvalidImageError("The image is too large to process.") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The uploaded file is not a readable image.") from error

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("The uploaded file is not a readable image.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _find_card(image: np.ndarray) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    sample = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_area = sample.shape[0] * sample.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.08 or area > frame_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        corners = _order_points(polygon.reshape(4, 2).astype(np.float32))
        top_left, top_right, bottom_right, bottom_left = corners
        card_width = max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
        card_height = max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
        ratio = card_width / max(card_height, 1)
        is_landscape_card = 1.15 <= ratio <= 2.05
        is_portrait_card = 0.45 <= ratio <= 0.95
        if not (is_landscape_card or is_portrait_card):
            continue

        output_width = int(card_width / scale)
        output_height = int(card_height / scale)
        if output_width < 160 or output_height < 90:
            continue
        destination = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners / scale, destination)
        return cv2.warpPerspective(image, transform, (output_width, output_height))
    return None


def preprocess_image(image_bytes: bytes) -> PreparedImage:
    image = decode_image(image_bytes)
    original_height, original_width = image.shape[:2]
    card = _find_card(image)
    document = card if card is not None else image
    gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    scale = min(4.0, max(1.0, 1200 / gray.shape[1]))
    if scale > 1:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    denoised = cv2.bilateralFilter(gray, 9, 50, 50)
    ocr_image = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return PreparedImage(
        original_width=original_width,
        original_height=original_height,
        document=document,
        ocr_image=ocr_image,
        document_detected=card is not None,
    )


def save_portrait(document: np.ndarray, output_path: str) -> bool:
    gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    faces = classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35)
    )

    face_detected = len(faces) > 0
    if face_detected:
        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        margin_x = int(width * 0.25)
        margin_y = int(height * 0.25)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(document.shape[1], x + width + margin_x)
        y2 = min(document.shape[0], y + height + margin_y)
        portrait = document[y1:y2, x1:x2]
    else:
        height, width = document.shape[:2]
        x1 = int(width * 0.62)
        portrait = document[int(height * 0.12) : int(height * 0.88), x1:]
        if portrait.size == 0:
            portrait = document

    if not cv2.imwrite(output_path, portrait):
        raise OSError("Could not save the ID portrait thumbnail.")
    return face_detected
