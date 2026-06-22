"""核心逻辑测试：流水线编排与路由（monkeypatch 各节点，不调 VLM）。"""

import agent.pipeline as pipeline
from agent.pipeline import stream_pipeline, run_pipeline


def _patch_nodes(monkeypatch, n_candidates):
    """把各节点替换成纯桩，detect 产出 n_candidates 个候选。"""
    monkeypatch.setattr(pipeline, "segment_node",
                        lambda s: {"candidates": [{}] * n_candidates})
    monkeypatch.setattr(pipeline, "detect_node",
                        lambda s: {"candidates": [{"has_waldo": True, "orig_bbox": [1, 2, 3, 4]}] * n_candidates})
    monkeypatch.setattr(pipeline, "verify_node",
                        lambda s: {"candidates": s["candidates"], "verified_result": [1, 2, 3, 4]})
    monkeypatch.setattr(pipeline, "visualize_node", lambda s: {})


def test_multi_candidate_runs_verify(monkeypatch):
    _patch_nodes(monkeypatch, n_candidates=3)
    nodes = [node for node, _ in stream_pipeline("x.jpg")]
    assert nodes == ["segment", "detect", "verify", "visualize"]


def test_single_candidate_skips_verify(monkeypatch):
    _patch_nodes(monkeypatch, n_candidates=1)
    nodes = [node for node, _ in stream_pipeline("x.jpg")]
    assert nodes == ["segment", "detect", "visualize"]


def test_empty_candidates_skips_verify(monkeypatch):
    _patch_nodes(monkeypatch, n_candidates=0)
    nodes = [node for node, _ in stream_pipeline("x.jpg")]
    assert nodes == ["segment", "detect", "visualize"]


def test_run_pipeline_returns_final_state(monkeypatch):
    _patch_nodes(monkeypatch, n_candidates=3)
    state = run_pipeline("x.jpg")
    assert state["verified_result"] == [1, 2, 3, 4]
    assert state["original_image_path"] == "x.jpg"
