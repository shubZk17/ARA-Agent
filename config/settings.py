"""
config/settings.py — Centralized Configuration Management
==========================================================

WHY THIS EXISTS:
    Every production system needs a single source of truth for configuration.
    Scattering os.getenv() calls across files creates:
    - hidden dependencies on environment variables,
    - inconsistent defaults,
    - impossible-to-debug configuration errors.

    This module loads .env once, validates required values, and exposes
    a typed Settings object that every other module imports.

HOW IT CONNECTS:
    - agent/ imports Settings to know which LLM to use and iteration limits.
    - tools/ imports Settings for API keys if needed.
    - utils/logger.py imports Settings for log level.
    - main.py imports Settings to validate config at startup.

SCALABILITY:
    As we add Phase 2+ features (vector DBs, external APIs, caching),
    new config values go HERE — not sprinkled across files.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (two levels up from config/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings.

    frozen=True prevents accidental mutation after initialization,
    which is a common source of subtle bugs in long-running agents.
    """

    # --- LLM Provider ---
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # --- Agent Behaviour ---
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10"))
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # --- Paths ---
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    log_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")

    # --- Phase 2: Vector Store ---
    vector_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "chroma")
    )
    vector_collection_name: str = field(
        default_factory=lambda: os.getenv("VECTOR_COLLECTION", "ara_financial_docs")
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "chroma"
    )

    # --- Phase 2: Embeddings ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # --- Phase 2: Retrieval ---
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5"))
    )
    retrieval_min_similarity: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.3"))
    )

    # --- Phase 2: Episodic Memory ---
    episodic_memory_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "episodic"
    )

    # --- Phase 3: Report Generation ---
    report_output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "reports"
    )

    # --- Phase 4: Observability ---
    telemetry_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "telemetry"
    )
    evaluation_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "evaluations"
    )
    failure_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "failures"
    )

    # --- Phase 4: Checkpointing ---
    checkpoint_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "checkpoints"
    )
    enable_checkpoints: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CHECKPOINTS", "true").lower() == "true"
    )

    # --- Phase 4: Retry / Fallback ---
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: float(os.getenv("RETRY_BACKOFF", "1.0"))
    )
    enable_fallback: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
    )

    # --- Phase 4: Evaluation ---
    enable_evaluation: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    )
    enable_failure_injection: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FAILURE_INJECTION", "false").lower() == "true"
    )

    # --- Phase 4: API Server ---
    api_host: str = field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    api_port: int = field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )

    def validate(self) -> list[str]:
        """
        Returns a list of configuration errors (empty = valid).

        We return errors instead of raising immediately so callers can
        display ALL issues at once rather than fix-one-crash-fix-another.
        """
        errors: list[str] = []

        if self.llm_provider not in ("claude", "openai", "groq"):
            errors.append(
                f"LLM_PROVIDER must be 'claude', 'openai', or 'groq', got '{self.llm_provider}'"
            )

        if self.llm_provider == "claude" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")

        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        if self.llm_provider == "groq" and not self.groq_api_key:
            errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")

        if self.max_iterations < 1:
            errors.append("MAX_ITERATIONS must be >= 1")

        # Phase 2 validation (non-blocking — Phase 2 is optional)
        if self.embedding_api_key:
            pass  # Embeddings will work
        else:
            pass  # Will use fallback embeddings — acceptable for dev

        return errors

    @property
    def active_model(self) -> str:
        """Return the model name for the currently selected provider."""
        if self.llm_provider == "claude":
            return self.anthropic_model
        if self.llm_provider == "groq":
            return self.groq_model
        return self.openai_model


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
