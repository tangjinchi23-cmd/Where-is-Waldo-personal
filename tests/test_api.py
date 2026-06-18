"""API 层测试：用 TestClient + monkeypatch service，不打真实 VLM。"""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from service.waldo_service import WaldoCase, IMAGES_DIR, OUTPUTS_DIR


def test_cases_endpoint_maps_paths_to_urls(monkeypatch):
    fake = [WaldoCase(
        name="1",
        image_path=str(IMAGES_DIR / "1.jpg"),
        result_path=str(OUTPUTS_DIR / "1_result.jpg"),
        has_result=True,
    )]
    monkeypatch.setattr("api.main.list_cases", lambda: fake)

    client = TestClient(app)
    r = client.get("/api/cases")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["name"] == "1"
    assert body[0]["image_url"] == "/static/original-images/1.jpg"
    assert body[0]["result_url"] == "/static/outputs/1_result.jpg"
    assert body[0]["has_result"] is True


def test_upload_rejects_bad_extension():
    client = TestClient(app)
    r = client.post("/api/upload", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_detect_404_when_image_missing(monkeypatch):
    monkeypatch.setattr("api.main.resolve_image", lambda name: None)
    client = TestClient(app)
    r = client.get("/api/detect", params={"name": "nope"})
    assert r.status_code == 404


def test_detect_streams_sse_frames(monkeypatch):
    monkeypatch.setattr("api.main.resolve_image", lambda name: Path("fake.jpg"))

    def fake_run(path):
        yield {"stage": "segment", "patches": 3}
        yield {"stage": "done", "found": False, "verify_ran": False, "bbox": None, "result_path": None}

    monkeypatch.setattr("api.main.run_detection", fake_run)

    client = TestClient(app)
    with client.stream("GET", "/api/detect", params={"name": "foo"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert '"stage": "segment"' in text
    assert '"stage": "done"' in text
    assert text.count("data:") == 2
