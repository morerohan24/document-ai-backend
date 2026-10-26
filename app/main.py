import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    DATABASE_PATH,
    DATA_DIR,
    FACE_DIR,
    MAX_UPLOAD_BYTES,
    PROJECT_ROOT,
)
from app.database import get_record, initialize_database, save_record
from app.image_processing import InvalidImageError, preprocess_image, save_portrait
from app.ocr import extract_fields, read_text
from app.schemas import ProcessedRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    initialize_database(DATABASE_PATH)
    yield


app = FastAPI(
    title="ID Lens",
    description=(
        "Detect, preprocess, and extract key fields from identity card images. "
        "OCR results are best-effort and should be verified by a human."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/media", StaticFiles(directory=FACE_DIR, check_dir=False), name="media")
app.mount(
    "/assets",
    StaticFiles(directory=PROJECT_ROOT / "frontend", check_dir=False),
    name="assets",
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/process-id", response_model=ProcessedRecord)
def process_id(file: UploadFile = File(...)) -> dict:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        prepared = preprocess_image(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Image preprocessing failed")
        raise HTTPException(
            status_code=500, detail="Image preprocessing failed."
        ) from error

    try:
        raw_text = read_text(prepared.ocr_image)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail="OCR processing failed.") from error

    record_id = uuid.uuid4().hex
    safe_filename = Path(file.filename or "identity-card").name
    thumbnail_name = f"{record_id}.jpg"
    try:
        face_detected = save_portrait(
            prepared.document, str(FACE_DIR / thumbnail_name)
        )
        document_height, document_width = prepared.document.shape[:2]
        image_metadata = {
            "filename": safe_filename,
            "content_type": file.content_type or "application/octet-stream",
            "width": prepared.original_width,
            "height": prepared.original_height,
            "processed_width": document_width,
            "processed_height": document_height,
            "document_detected": prepared.document_detected,
            "face_detected": face_detected,
            "face_thumbnail_url": f"/media/{thumbnail_name}",
        }
        record = save_record(
            DATABASE_PATH,
            record_id=record_id,
            filename=safe_filename,
            content_type=image_metadata["content_type"],
            extracted_data=extract_fields(raw_text),
            raw_text=raw_text,
            image_metadata=image_metadata,
        )
    except Exception as error:
        logger.exception("Could not save processed ID record")
        raise HTTPException(
            status_code=500, detail="Could not save the processed ID record."
        ) from error

    return record


@app.get("/api/v1/records/{record_id}", response_model=ProcessedRecord)
def read_record(record_id: str) -> dict:
    record = get_record(DATABASE_PATH, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return record
