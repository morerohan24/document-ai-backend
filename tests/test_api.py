import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as main_module


def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main_module, "DATABASE_PATH", tmp_path / "records.sqlite3")
    monkeypatch.setattr(main_module, "FACE_DIR", tmp_path / "faces")


def test_process_and_retrieve_record(tmp_path, monkeypatch):
    isolate_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(
        main_module,
        "read_text",
        lambda _: "FULL NAME: JANE SAMPLE\nID NUMBER: AB123456\nDOB: 04/12/1992",
    )
    image = np.full((240, 360, 3), 225, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/process-id",
            files={"file": ("card.png", encoded.tobytes(), "image/png")},
        )
        assert response.status_code == 200
        record = response.json()
        assert record["extracted_data"]["full_name"] == "JANE SAMPLE"
        assert record["extracted_data"]["document_number"] == "AB123456"
        assert record["image_metadata"]["width"] == 360

        fetched = client.get(f"/api/v1/records/{record['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == record
        assert list((tmp_path / "faces").glob("*.jpg"))


def test_rejects_unsupported_media_type(tmp_path, monkeypatch):
    isolate_storage(tmp_path, monkeypatch)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/process-id",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415


def test_rejects_unreadable_image_and_missing_record(tmp_path, monkeypatch):
    isolate_storage(tmp_path, monkeypatch)
    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/process-id",
            files={"file": ("broken.png", b"not an image", "image/png")},
        )
        assert response.status_code == 422
        assert client.get("/api/v1/records/not-a-record").status_code == 404
