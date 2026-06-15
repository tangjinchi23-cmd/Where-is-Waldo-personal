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
