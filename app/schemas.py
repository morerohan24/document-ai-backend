from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ExtractedFields(BaseModel):
    full_name: Optional[str] = None
    document_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    expiry_date: Optional[str] = None


class ImageMetadata(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    processed_width: int
    processed_height: int
    document_detected: bool
    face_detected: bool
    face_thumbnail_url: str


class ProcessedRecord(BaseModel):
    id: str
    extracted_data: ExtractedFields
    raw_text: str
    image_metadata: ImageMetadata
    created_at: datetime
