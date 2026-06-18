"""service.waldo_service 契约冒烟测试。

架构占位阶段：只验证 service 的边界契约可用（前端依赖的形状稳定），
不细抠功能逻辑——后续真实现时再补功能测试。
"""

from service.waldo_service import WaldoCase, list_cases, get_case


def _make_image(path):
    """写一个占位图片文件（service 只看扩展名/存在性，无需有效图像数据）。"""
    path.write_bytes(b"fake-image-bytes")


def test_list_cases_returns_waldocases_with_result_pairing(tmp_path):
    """契约：list_cases 返回 WaldoCase 列表，并正确标记有无结果。"""
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "1.jpg")
    _make_image(images / "2.jpg")
    _make_image(outputs / "1_result.jpg")  # 只有 1 有结果

    cases = list_cases(images_dir=images, outputs_dir=outputs)

    assert all(isinstance(c, WaldoCase) for c in cases)
    assert [c.name for c in cases] == ["1", "2"]
    assert cases[0].has_result is True
    assert cases[0].result_path == str(outputs / "1_result.jpg")
    assert cases[1].has_result is False
    assert cases[1].result_path is None


def test_list_cases_empty_when_images_dir_missing(tmp_path):
    """边界：图片目录不存在时返回空列表，不抛异常。"""
    assert list_cases(images_dir=tmp_path / "nope", outputs_dir=tmp_path) == []


def test_get_case_hit_and_miss(tmp_path):
    """契约：get_case 命中返回 WaldoCase，未命中返回 None。"""
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "1.jpg")

    assert get_case("1", images_dir=images, outputs_dir=outputs).name == "1"
    assert get_case("zzz", images_dir=images, outputs_dir=outputs) is None


# ── resolve_image ────────────────────────────────────────────────
from service.waldo_service import resolve_image, run_detection


def test_resolve_image_prefers_original_then_uploads(tmp_path):
    images = tmp_path / "original-images"
    uploads = tmp_path / "uploads"
    images.mkdir()
    uploads.mkdir()
    (images / "1.jpg").write_bytes(b"x")
    (uploads / "2.png").write_bytes(b"x")

    assert resolve_image("1", images_dir=images, uploads_dir=uploads) == images / "1.jpg"
    assert resolve_image("2", images_dir=images, uploads_dir=uploads) == uploads / "2.png"
    assert resolve_image("zzz", images_dir=images, uploads_dir=uploads) is None


# ── run_detection ────────────────────────────────────────────────
class _FakeGraph:
    def __init__(self, updates):
        self._updates = updates

    def stream(self, state):
        yield from self._updates


def _patch_graph(monkeypatch, updates):
    monkeypatch.setattr("service.waldo_service.initial_state", lambda p: {"original_image_path": p})
    monkeypatch.setattr("service.waldo_service.build_graph", lambda: _FakeGraph(updates))


def test_run_detection_full_pipeline_with_verify(tmp_path, monkeypatch):
    updates = [
        {"segment": {"candidates": [{}, {}, {}]}},
        {"detect": {"candidates": [
            {"crop_path": "a", "confidence": 0.9, "has_waldo": True, "orig_bbox": [1, 2, 3, 4]},
            {"crop_path": "b", "confidence": 0.8, "has_waldo": True, "orig_bbox": [5, 6, 7, 8]},
        ]}},
        {"verify": {"candidates": [
            {"verified": True, "verify_crop_path": "va", "verify_looks_waldo": True, "orig_bbox": [1, 2, 3, 4]},
            {"verified": False, "verify_crop_path": "vb", "verify_looks_waldo": False, "orig_bbox": [5, 6, 7, 8]},
        ], "verified_result": [1, 2, 3, 4]}},
        {"visualize": {}},
    ]
    _patch_graph(monkeypatch, updates)

    events = list(run_detection("foo.jpg"))
    assert [e["stage"] for e in events] == ["segment", "detect", "verify", "done"]
    assert events[0]["patches"] == 3
    assert events[1]["count"] == 2
    assert events[2]["ran"] is True and events[2]["choice"] == 0
    assert events[-1]["found"] is True
    assert events[-1]["verify_ran"] is True
    assert events[-1]["bbox"] == [1, 2, 3, 4]


def test_run_detection_skips_verify_single_candidate(tmp_path, monkeypatch):
    updates = [
        {"segment": {"candidates": [{}]}},
        {"detect": {"candidates": [
            {"crop_path": "a", "confidence": 0.9, "has_waldo": True, "orig_bbox": [1, 2, 3, 4]},
        ]}},
        {"visualize": {}},
    ]
    _patch_graph(monkeypatch, updates)

    events = list(run_detection("foo.jpg"))
    assert [e["stage"] for e in events] == ["segment", "detect", "verify", "done"]
    assert events[2]["ran"] is False
    assert events[-1]["found"] is True
    assert events[-1]["verify_ran"] is False
    assert events[-1]["bbox"] == [1, 2, 3, 4]


def test_run_detection_emits_error_event(monkeypatch):
    class _BoomGraph:
        def stream(self, state):
            raise RuntimeError("no api key")

    monkeypatch.setattr("service.waldo_service.initial_state", lambda p: {"original_image_path": p})
    monkeypatch.setattr("service.waldo_service.build_graph", lambda: _BoomGraph())

    events = list(run_detection("foo.jpg"))
    assert events[-1]["stage"] == "error"
    assert "no api key" in events[-1]["message"]
