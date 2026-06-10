"""Tests for the analyze node.

Unit tests (no API):
  - _parse_grid_dims: all code paths including the Bug 1 empty-string case
  - Grid generation: correct region count, full coverage, positive dimensions
  - Config assertion: ANALYZE_MAX_TOKENS >= 128 (currently FAILS at 64)

Integration tests (real GPT-4o API, needs OPENAI_API_KEY):
  - analyze_node returns valid focus_regions on a real image
  - Raw VLM response is non-empty (catches the Bug 1 empty-response case)

Run:
    pytest tests/test_analyze_node.py -v            # unit tests (integration auto-skipped)
    pytest tests/test_analyze_node.py -v -s -m integration   # with API calls + printed output
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

from agent.nodes.analyze import _parse_grid_dims, ANALYZE_MAX_TOKENS, analyze_node

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_TEST_IMAGE = os.path.join(_PROJECT_ROOT, "original-images", "1.jpg")


# ================================================================
# Config sanity check  ← Bug 1: this FAILS with the current value of 64
# ================================================================

class TestConfig:
    def test_analyze_max_tokens_is_sufficient(self):
        """ANALYZE_MAX_TOKENS must cover reasoning tokens + output for the gpt-5.5 reasoning model.

        Bug 1 root cause (confirmed 2026-06-10): gpt-5.5 spends max_completion_tokens on
        internal reasoning BEFORE emitting visible output. Measured reasoning usage reached
        ~232 tokens while the JSON output is only ~20. A budget of 128 was fully consumed by
        reasoning (finish_reason='length', content=''), forcing a heuristic fallback.
        512 is the practical floor; we use 1024 for headroom.
        """
        assert ANALYZE_MAX_TOKENS >= 512, (
            f"ANALYZE_MAX_TOKENS={ANALYZE_MAX_TOKENS} is too low for a reasoning model. "
            "gpt-5.5 burns ~230 reasoning tokens before output; a low cap yields empty content. "
            "Raise to >= 512 (1024 recommended) in agent/nodes/analyze.py."
        )


# ================================================================
# _parse_grid_dims — unit tests (no API)
# ================================================================

class TestParseGridDims:

    # --- happy path ---

    def test_plain_json(self):
        rows, cols = _parse_grid_dims('{"rows": 4, "cols": 6}', 2, 2)
        assert rows == 4 and cols == 6

    def test_markdown_json_block(self):
        text = '```json\n{"rows": 3, "cols": 5}\n```'
        rows, cols = _parse_grid_dims(text, 2, 2)
        assert rows == 3 and cols == 5

    def test_markdown_block_no_lang(self):
        text = '```\n{"rows": 2, "cols": 4}\n```'
        rows, cols = _parse_grid_dims(text, 1, 1)
        assert rows == 2 and cols == 4

    def test_extra_fields_ignored(self):
        text = '{"rows": 5, "cols": 7, "reason": "dense crowd"}'
        rows, cols = _parse_grid_dims(text, 2, 2)
        assert rows == 5 and cols == 7

    def test_float_values_coerced_to_int(self):
        rows, cols = _parse_grid_dims('{"rows": 3.9, "cols": 4.1}', 2, 2)
        assert rows == 3 and cols == 4

    # --- clamp to min 1 ---

    def test_zero_clamped_to_one(self):
        rows, cols = _parse_grid_dims('{"rows": 0, "cols": 0}', 2, 2)
        assert rows >= 1 and cols >= 1

    def test_negative_clamped_to_one(self):
        rows, cols = _parse_grid_dims('{"rows": -3, "cols": -1}', 2, 2)
        assert rows >= 1 and cols >= 1

    # --- regex fallback (two bare integers) ---

    def test_prose_with_two_numbers(self):
        text = "I'd recommend splitting into 4 rows and 6 columns."
        rows, cols = _parse_grid_dims(text, 2, 2)
        assert rows == 4 and cols == 6

    def test_only_one_number_uses_fallback(self):
        text = "Use 5 rows please."
        rows, cols = _parse_grid_dims(text, 3, 4)
        assert rows == 3 and cols == 4

    # --- Bug 1 scenarios: empty / whitespace / garbage ---

    def test_empty_string_uses_fallback(self):
        """Bug 1 root case: GPT-4o returns '' when ANALYZE_MAX_TOKENS is too low."""
        rows, cols = _parse_grid_dims("", 3, 5)
        assert rows == 3 and cols == 5

    def test_whitespace_only_uses_fallback(self):
        rows, cols = _parse_grid_dims("   \n  \t  ", 4, 6)
        assert rows == 4 and cols == 6

    def test_complete_garbage_uses_fallback(self):
        rows, cols = _parse_grid_dims("???!!!", 2, 3)
        assert rows == 2 and cols == 3

    def test_truncated_json_uses_fallback(self):
        """Simulates a response cut off mid-JSON by a low token limit."""
        rows, cols = _parse_grid_dims('{"rows": 4, "col', 2, 3)
        assert rows == 2 and cols == 3

    def test_fallback_values_are_positive(self):
        """Fallback must always return usable grid dimensions."""
        for bad_input in ["", "nope", '{"rows": -1}']:
            r, c = _parse_grid_dims(bad_input, 2, 4)
            assert r >= 1 and c >= 1, f"Non-positive fallback for input: {bad_input!r}"

    # --- missing fields fall back to their per-axis fallback ---

    def test_missing_cols_uses_fallback_cols(self):
        rows, cols = _parse_grid_dims('{"rows": 4}', 2, 6)
        assert rows == 4 and cols == 6

    def test_missing_rows_uses_fallback_rows(self):
        rows, cols = _parse_grid_dims('{"cols": 5}', 3, 2)
        assert rows == 3 and cols == 5

    def test_empty_json_object_uses_both_fallbacks(self):
        rows, cols = _parse_grid_dims("{}", 3, 5)
        assert rows == 3 and cols == 5


# ================================================================
# Grid generation — unit tests with mocked VLM (no API)
# ================================================================

class TestGridGeneration:
    """Test that analyze_node builds correct focus_regions for any rows×cols."""

    @pytest.fixture(autouse=True)
    def skip_if_no_image(self):
        if not os.path.exists(_TEST_IMAGE):
            pytest.skip(f"Test image not found: {_TEST_IMAGE}")

    def _run_with_mock_response(self, mock_response: str):
        """Run analyze_node with a fake VLM response."""
        from llm.vlm_client import get_vlm_client

        class FakeVLM:
            def call(self, image_path, prompt, max_tokens=None):
                return mock_response

        with patch("agent.nodes.analyze.get_vlm_client", return_value=FakeVLM()):
            from agent.state import initial_state
            state = initial_state(_TEST_IMAGE)
            return analyze_node(state)

    def test_region_count_equals_rows_times_cols(self):
        result = self._run_with_mock_response('{"rows": 3, "cols": 4}')
        assert len(result["focus_regions"]) == 12

    def test_region_count_for_2x2(self):
        result = self._run_with_mock_response('{"rows": 2, "cols": 2}')
        assert len(result["focus_regions"]) == 4

    def test_all_regions_have_positive_size(self):
        result = self._run_with_mock_response('{"rows": 4, "cols": 5}')
        for r in result["focus_regions"]:
            x, y, w, h = r
            assert w > 0 and h > 0, f"Zero-size region: {r}"

    def test_regions_are_non_negative(self):
        result = self._run_with_mock_response('{"rows": 3, "cols": 3}')
        for r in result["focus_regions"]:
            x, y, w, h = r
            assert x >= 0 and y >= 0

    def test_regions_cover_full_image_width(self):
        from PIL import Image
        img_w, _ = Image.open(_TEST_IMAGE).size
        result = self._run_with_mock_response('{"rows": 2, "cols": 4}')
        regions = result["focus_regions"]
        total_w = sum(r[2] for r in regions[:4])  # first row
        assert total_w == img_w

    def test_regions_cover_full_image_height(self):
        from PIL import Image
        _, img_h = Image.open(_TEST_IMAGE).size
        result = self._run_with_mock_response('{"rows": 3, "cols": 2}')
        regions = result["focus_regions"]
        # first column: every other region (stride = cols=2)
        total_h = sum(regions[i][3] for i in range(0, 6, 2))
        assert total_h == img_h

    def test_grid_rows_cols_returned_correctly(self):
        result = self._run_with_mock_response('{"rows": 5, "cols": 3}')
        assert result["grid_rows"] == 5
        assert result["grid_cols"] == 3

    def test_empty_vlm_response_falls_back_to_heuristic(self):
        """Bug 1: empty response must not crash; regions still cover the full image."""
        result = self._run_with_mock_response("")
        assert len(result["focus_regions"]) > 0
        assert result["grid_rows"] >= 1
        assert result["grid_cols"] >= 1



# ================================================================
# Integration tests — real GPT-4o API (needs OPENAI_API_KEY)
# ================================================================

@pytest.mark.integration
class TestAnalyzeNodeIntegration:
    """Live API tests. Run with: pytest -m integration -v -s"""

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        if not os.path.exists(_TEST_IMAGE):
            pytest.skip(f"Test image not found: {_TEST_IMAGE}")

    def test_analyze_returns_focus_regions(self):
        from agent.state import initial_state
        state = initial_state(_TEST_IMAGE)
        result = analyze_node(state)
        assert len(result["focus_regions"]) > 0

    def test_analyze_grid_dims_are_positive(self):
        from agent.state import initial_state
        state = initial_state(_TEST_IMAGE)
        result = analyze_node(state)
        assert result["grid_rows"] >= 1
        assert result["grid_cols"] >= 1

    def test_analyze_vlm_response_is_not_empty(self, capsys):
        """Bug 1 catch: if GPT-4o returns empty, we'll see the fallback warning in stdout.

        This test captures stdout and fails if the fallback warning appears,
        which indicates ANALYZE_MAX_TOKENS is still too low.
        """
        from agent.state import initial_state
        state = initial_state(_TEST_IMAGE)
        analyze_node(state)
        captured = capsys.readouterr()
        assert "Failed to parse grid dims" not in captured.out, (
            "analyze_node fell back to heuristic — VLM returned empty/unparseable response.\n"
            "Likely cause: ANALYZE_MAX_TOKENS too low for gpt-5.5 reasoning tokens "
            "(reasoning eats the budget before output). Raise to >= 512.\n"
            f"Full stdout:\n{captured.out}"
        )

    def test_analyze_region_count_matches_grid(self):
        from agent.state import initial_state
        state = initial_state(_TEST_IMAGE)
        result = analyze_node(state)
        expected = result["grid_rows"] * result["grid_cols"]
        assert len(result["focus_regions"]) == expected
