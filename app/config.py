import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("ID_APP_DATA_DIR", PROJECT_ROOT / "data"))
DATABASE_PATH = DATA_DIR / "records.sqlite3"
FACE_DIR = DATA_DIR / "faces"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
