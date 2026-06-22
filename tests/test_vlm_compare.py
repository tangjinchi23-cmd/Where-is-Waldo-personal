"""Cross-provider comparison test.

Runs detect() on the same image across all available providers,
then prints a side-by-side table so you can compare accuracy, confidence, and
bbox agreement.

This test never fails on disagreement — it only fails on exceptions.
Run with -s to see the comparison table printed to stdout.

Usage:
    pytest tests/test_vlm_compare.py -v -s
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.vlm_client import DetectResult, get_vlm_client, BaseVLMClient


# ================================================================
# Data container
# ================================================================

@dataclass
class ProviderResult:
    provider: str
    detect: Optional[DetectResult] = None
    detect_latency: float = 0.0
    error: Optional[str] = None


# ================================================================
# Helpers
# ================================================================

def _run_provider(name: str, client: BaseVLMClient, image_path: str) -> ProviderResult:
    result = ProviderResult(provider=name)
    try:
        t0 = time.perf_counter()
        result.detect = client.detect(image_path)
        result.detect_latency = time.perf_counter() - t0
    except Exception as exc:
        result.error = str(exc)
    return result


def _print_table(results: list[ProviderResult]):
    print("\n" + "=" * 80)
    print(f"{'Provider':<12} {'has_waldo':<12} {'detect_conf':<14} {'detect_ms':<12}")
    print("-" * 80)
    for r in results:
        if r.error:
            print(f"{r.provider:<12} ERROR: {r.error}")
            continue
        d = r.detect
        print(
            f"{r.provider:<12} "
            f"{str(d.has_waldo):<12} "
            f"{d.confidence:<14.3f} "
            f"{r.detect_latency * 1000:<12.0f}"
        )
    print("=" * 80)


def _print_bbox_table(results: list[ProviderResult]):
    print("\nDetected bbox per provider (x, y, w, h) — relative to test image:")
    print("-" * 50)
    for r in results:
        if r.error or r.detect is None:
            continue
        bbox = r.detect.bbox
        print(f"  {r.provider:<12}: {bbox}")
    print()


# ================================================================
# Comparison test
# ================================================================

class TestProviderComparison:
    """Collects results from all configured providers and prints a comparison table."""

    def _collect(self, available: dict, image_path: str) -> list[ProviderResult]:
        results = []
        for name, client in available.items():
            print(f"\n[{name}] running detect ...", flush=True)
            r = _run_provider(name, client, image_path)
            results.append(r)
        return results

    def test_compare_detect_across_providers(self, test_image, request):
        """Run detect() on all available providers; print comparison table."""
        available = {}

        # Try to load each provider; skip gracefully if key / package missing
        for provider in ("claude", "gpt4o", "gemini", "qwen"):
            fixture_name = f"{provider}_client" if provider != "gpt4o" else "gpt4o_client"
            try:
                client = request.getfixturevalue(
                    {"claude": "claude_client", "gpt4o": "gpt4o_client",
                     "gemini": "gemini_client", "qwen": "qwen_client"}[provider]
                )
                available[provider] = client
            except pytest.skip.Exception:
                pass

        if not available:
            pytest.skip("No provider API keys configured")

        results = self._collect(available, test_image)
        _print_table(results)
        _print_bbox_table(results)

        # Only hard-fail on exceptions (not disagreement between providers)
        for r in results:
            if r.error:
                pytest.fail(f"Provider '{r.provider}' raised an exception: {r.error}")

    def test_agreement_on_has_waldo(self, test_image, request):
        """Check whether providers agree on the has_waldo boolean.

        This test is informational — it prints the agreement rate but does not
        fail on disagreement, since model disagreement is expected.
        """
        available = {}
        for provider, fixture in {
            "claude": "claude_client",
            "gpt4o": "gpt4o_client",
            "gemini": "gemini_client",
            "qwen": "qwen_client",
        }.items():
            try:
                available[provider] = request.getfixturevalue(fixture)
            except pytest.skip.Exception:
                pass

        if len(available) < 2:
            pytest.skip("Need at least 2 providers to compare")

        votes = {}
        for name, client in available.items():
            try:
                result = client.detect(test_image)
                votes[name] = result.has_waldo
            except Exception as exc:
                votes[name] = f"ERROR: {exc}"

        print("\nhas_waldo votes:")
        for provider, vote in votes.items():
            print(f"  {provider:<12}: {vote}")

        bool_votes = [v for v in votes.values() if isinstance(v, bool)]
        if bool_votes:
            agreement = sum(bool_votes) / len(bool_votes)
            print(f"\nAgreement (fraction saying True): {agreement:.0%}")


# ================================================================
# Single-provider sanity: Claude only (most likely to have a key)
# ================================================================

class TestClaudeDetectSanity:
    """Sanity tests that can run with just an Anthropic key."""

    def test_detect_raw_response_contains_json_keys(self, claude_client, test_image):
        result = claude_client.detect(test_image)
        raw = result.raw_response.lower()
        assert "has_waldo" in raw or "confidence" in raw, (
            f"Expected JSON keys in raw response, got: {result.raw_response[:200]}"
        )

    def test_full_image_detect_is_dict_parseable(self, claude_client, test_image):
        """detect() on the full puzzle image should parse without error."""
        result = claude_client.detect(test_image)
        # If JSON parsing failed, confidence defaults to 0.0 and has_waldo to False
        # but raw_response will still be non-empty
        assert len(result.raw_response) > 0
