# ID Lens

A local-first ID card processing service built with FastAPI, OpenCV, Tesseract OCR, and SQLite. It detects and perspective-corrects a card when possible, enhances the image for OCR, extracts labeled identity fields, saves a portrait thumbnail, and persists the result for later retrieval. A responsive browser interface and OpenAPI docs are included.

> OCR is best-effort, not identity verification. Always compare extracted values with the original document. Use only documents you are authorized to process.

## Requirements

- Python 3.9 or newer
- Tesseract OCR installed and available on `PATH`
- The packages in `requirements.txt`

Windows: install the Tesseract OCR engine (for example, the UB Mannheim Windows build) and add its install directory to `PATH`. The app also checks the standard Program Files install locations. If you installed it elsewhere, set `TESSERACT_CMD` to the full executable path, for example:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m uvicorn app.main:app --reload
```

On Debian/Ubuntu, install it with `sudo apt-get install tesseract-ocr`.

## Run locally

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the upload interface and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

## API

### `POST /api/v1/process-id`

Send one multipart form field named `file`. JPEG, PNG, and WebP images up to 10 MB are accepted. Images that cannot be decoded or exceed the pixel limit return `422`; unsupported media types return `415`; missing Tesseract returns `503`.

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/process-id `
  -F "file=@C:\path\to\id-card.jpg"
```

The response includes:

- `id`: persisted record identifier
- `extracted_data`: `full_name`, `document_number`, `date_of_birth`, and `expiry_date` (nullable if OCR could not identify a field)
- `raw_text`: text recognized by OCR for verification
- `image_metadata`: original and processed dimensions, boundary/face detection indicators, and portrait thumbnail URL
- `created_at`: UTC timestamp

### `GET /api/v1/records/{id}`

Retrieve the saved result. Returns `404` when the record does not exist.

### `GET /api/v1/health`

Simple service health endpoint.

## Processing notes

- Card boundary detection uses edge contours and a four-corner perspective transform. If no plausible boundary is found, OCR runs on the full image and `document_detected` is `false`.
- OpenCV's Haar cascade detects a face for the thumbnail. When a face is not confirmed, a right-side portrait-area crop is saved instead; `face_detected` is `false`.
- Field extraction relies on printed labels and numeric date patterns. Unsupported layouts or low-quality images may return `null` fields; the raw OCR text is included so results can be checked.
- The original upload is processed in memory and is not saved. SQLite records and generated portrait crops live under `data/` by default. Set `ID_APP_DATA_DIR` to choose a different data directory.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
