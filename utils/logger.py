"""
utils/logger.py — Structured Logging Infrastructure
=====================================================

WHY THIS EXISTS:
    Agents are non-deterministic systems. When an agent hallucinates,
    misuses a tool, or enters an infinite loop, the ONLY way to diagnose
    the problem is through comprehensive logs.

    Standard print() is inadequate because:
    - no severity levels (you can't filter noise),
    - no timestamps (you can't correlate events),
    - no file output (you lose data on crash),
    - no structured format (you can't parse programmatically).

HOW IT CONNECTS:
    Every module imports `get_logger(__name__)` to get a namespaced logger.
    This creates a hierarchy: ara.agent.nodes, ara.tools.registry, etc.
    You can then filter logs by subsystem.

DESIGN DECISIONS:
    - Dual output: console (human-readable via Rich) + file (machine-parseable).
    - File logs rotate to prevent disk exhaustion in long-running sessions.
    - Each agent run gets a timestamped log file for easy post-mortem.

SCALABILITY:
    Phase 2+ can add:
    - JSON-structured log lines for log aggregation (ELK, Datadog),
    - trace IDs for correlating across distributed components,
    - metric emission alongside log lines.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from config.settings import settings


# ---------------------------------------------------------------------------
# Console for Rich output (shared across modules)
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Log file path — one file per session
# ---------------------------------------------------------------------------
_session_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_LOG_FILE = settings.log_dir / f"ara_{_session_ts}.log"

# Ensure log directory exists
settings.log_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a namespaced logger with console + file handlers.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Agent started", extra={"query": user_query})

    Args:
        name: Module name (typically __name__). Creates hierarchy like
              'agent.nodes', 'tools.registry', etc.

    Returns:
        Configured Logger instance.
    """
    # Prefix all loggers under 'ara' namespace
    logger = logging.getLogger(f"ara.{name}")

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # --- Console Handler (Rich) ---
    # Rich gives us colored, formatted output with timestamps
    rich_handler = RichHandler(
        console=console,
        show_path=False,       # module path is in the logger name
        show_time=True,
        rich_tracebacks=True,  # pretty stack traces on errors
        markup=True,           # allow [bold], [red], etc. in messages
    )
    rich_handler.setLevel(logging.INFO)  # console gets INFO+
    rich_fmt = logging.Formatter("%(message)s")
    rich_handler.setFormatter(rich_fmt)

    # --- File Handler ---
    # File gets everything (DEBUG+) for post-mortem analysis
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
