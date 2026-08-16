"""
tests/conftest.py — shared fixtures.

Two rules for this suite:
  1. No network. yfinance responses are frozen in tests/fixtures/*.json.
     The handful of tests that do hit the API are marked `network` and are
     excluded by the default addopts in pyproject.toml.
  2. No LLM. Ever. Nothing here constructs a real client.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


def load_info(ticker: str) -> dict:
    """Load a frozen yfinance `.info` response captured 2026-08-17."""
    return json.loads((FIXTURES / f"{ticker.lower()}_info.json").read_text(encoding="utf-8"))


@pytest.fixture
def aapl_info() -> dict:
    """Apple: the D1/D2 case — huge balance sheet, D/E of 78.445 percent."""
    return load_info("AAPL")


@pytest.fixture
def ko_info() -> dict:
    """Coca-Cola: the D3 case — dividendYield of 2.42 must not become 242%."""
    return load_info("KO")


@pytest.fixture
def nvda_info() -> dict:
    """NVIDIA: least-levered megacap, D/E of 6.555 percent = ratio 0.066."""
    return load_info("NVDA")
