"""Shared fixtures for VLM API tests.

Run with:
    pytest tests/                        # all tests
    pytest tests/test_vlm_unit.py        # unit tests only (no API calls)
    pytest tests/test_vlm_integration.py # integration tests (requires API keys)
    pytest tests/test_vlm_compare.py     # cross-provider comparison
"""

import os
import sys

import pytest

# Add project root so `from llm.vlm_client import ...` works
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_TEST_IMAGE = os.path.join(_PROJECT_ROOT, "original-images", "1.jpg")


# ---------- image fixture ----------

@pytest.fixture(scope="session")
def test_image():
    """Path to a real Waldo puzzle image for integration / comparison tests."""
    path = os.path.abspath(_TEST_IMAGE)
    if not os.path.exists(path):
        pytest.skip(f"Test image not found: {path}")
    return path


# ---------- per-provider client fixtures ----------

@pytest.fixture(scope="session")
def claude_client():
    """ClaudeVLMClient — skipped if ANTHROPIC_API_KEY not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from llm.vlm_client import ClaudeVLMClient
    return ClaudeVLMClient()


@pytest.fixture(scope="session")
def gpt4o_client():
    """GPT4oVLMClient — skipped if OPENAI_API_KEY not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    try:
        from llm.vlm_client import GPT4oVLMClient
        return GPT4oVLMClient()
    except ImportError:
        pytest.skip("openai package not installed")


@pytest.fixture(scope="session")
def gemini_client():
    """GeminiVLMClient — skipped if GOOGLE_API_KEY not set or package missing."""
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        from llm.vlm_client import GeminiVLMClient
        return GeminiVLMClient()
    except ImportError:
        pytest.skip("google-generativeai package not installed")


@pytest.fixture(scope="session")
def qwen_client():
    """QwenVLMClient — skipped if DASHSCOPE_API_KEY not set."""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY not set")
    try:
        from llm.vlm_client import QwenVLMClient
        return QwenVLMClient()
    except ImportError:
        pytest.skip("openai package not installed (needed for Qwen)")
