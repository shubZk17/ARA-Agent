"""
agent/retry_handler.py — Retry Logic, Exponential Backoff & LLM Fallback
==========================================================================

WHY THIS EXISTS:
    In production, API calls FAIL. LLM providers go down. Rate limits
    hit. Network timeouts happen. Without retry/fallback logic:
    - A single Groq timeout kills the entire analysis.
    - A rate limit at iteration 5 wastes all prior computation.
    - Users see cryptic errors instead of graceful degradation.

    This module provides:
    1. RETRY with exponential backoff — transient errors heal themselves.
    2. FALLBACK chains — Groq → OpenAI → Anthropic.
    3. DEGRADED modes — continue with partial capability.

WHAT PROBLEM IT SOLVES:
    Answers "What happens when the LLM goes down?" with:
    "We retry 3 times, then switch to a backup provider."

HOW IT INTEGRATES:
    - Wraps LLM calls in agent/nodes.py.
    - Records retry/fallback events to observability/collector.py.
    - Preserves execution state (no restart needed).
    - Configured via config/settings.py.

PRODUCTION TRADEOFFS:
    - Retry = higher latency on transient failures.
    - Fallback = potentially different model quality.
    - Max retries prevent infinite loops.
    - Backoff prevents overwhelming recovering services.

SCALABILITY:
    - Add circuit breaker pattern (stop calling dead service).
    - Add health checks (preemptive provider switch).
    - Add cost-based routing (cheapest available provider).
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple = (
        Exception,  # Broad — we'll narrow in production
    )


@dataclass
class FallbackProvider:
    """A fallback LLM provider configuration."""
    name: str
    provider: str  # "groq", "openai", "claude"
    model: str
    api_key: str
    priority: int = 0  # Lower = preferred


@dataclass
class RetryResult:
    """Result of a retry/fallback operation."""
    success: bool
    result: Any = None
    attempts: int = 0
    provider_used: str = ""
    model_used: str = ""
    total_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    fallback_used: bool = False


class RetryHandler:
    """
    Handles retry logic and LLM provider fallback for agent execution.

    The retry strategy:
    1. Try the primary provider up to max_retries times.
    2. On each failure, wait with exponential backoff.
    3. If all retries fail, try the next fallback provider.
    4. If all providers fail, return a degraded result.

    Usage:
        handler = RetryHandler(
            primary_provider="groq",
            fallback_chain=["openai", "claude"],
            config=RetryConfig(max_retries=3),
        )

        result = handler.execute_with_retry(
            func=call_llm,
            args=(messages,),
        )
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        collector: Any | None = None,  # TelemetryCollector
    ) -> None:
        self._config = config or RetryConfig()
        self._collector = collector
        self._fallback_providers: list[FallbackProvider] = []

    def add_fallback_provider(self, provider: FallbackProvider) -> None:
        """Add a fallback provider to the chain."""
        self._fallback_providers.append(provider)
        self._fallback_providers.sort(key=lambda p: p.priority)

    def configure_fallback_chain(self) -> None:
        """
        Auto-configure fallback chain from environment.

        Reads API keys from settings and builds the chain.
        Only adds providers that have valid API keys.
        """
        try:
            from config.settings import settings

            providers = [
                FallbackProvider(
                    name="groq_fallback",
                    provider="groq",
                    model=settings.groq_model,
                    api_key=settings.groq_api_key,
                    priority=1,
                ),
            ]

            if settings.openai_api_key:
                providers.append(FallbackProvider(
                    name="openai_fallback",
                    provider="openai",
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                    priority=2,
                ))

            if settings.anthropic_api_key:
                providers.append(FallbackProvider(
                    name="anthropic_fallback",
                    provider="claude",
                    model=settings.anthropic_model,
                    api_key=settings.anthropic_api_key,
                    priority=3,
                ))

            self._fallback_providers = [
                p for p in providers if p.api_key
            ]

            logger.info(
                f"Fallback chain configured: "
                f"{' → '.join(p.name for p in self._fallback_providers)}"
            )

        except Exception as e:
            logger.warning(f"Failed to configure fallback chain: {e}")

    def execute_with_retry(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        operation_name: str = "operation",
    ) -> RetryResult:
        """
        Execute a function with retry and backoff.

        Args:
            func: The function to call.
            args: Positional arguments.
            kwargs: Keyword arguments.
            operation_name: Name for logging.

        Returns:
            RetryResult with success status and result.
        """
        kwargs = kwargs or {}
        result = RetryResult()
        start_time = time.time()

        for attempt in range(1, self._config.max_retries + 1):
            try:
                logger.debug(
                    f"Attempt {attempt}/{self._config.max_retries} for {operation_name}"
                )
                result.result = func(*args, **kwargs)
                result.success = True
                result.attempts = attempt
                result.total_time_ms = (time.time() - start_time) * 1000

                if attempt > 1:
                    logger.info(f"{operation_name} succeeded on attempt {attempt}")

                # Record successful retry
                if self._collector and attempt > 1:
                    from observability.collector import EventType
                    self._collector.record_event(
                        EventType.RETRY,
                        operation_name,
                        duration_ms=result.total_time_ms,
                        success=True,
                        metadata={"attempt": attempt},
                    )

                return result

            except self._config.retryable_exceptions as e:
                error_msg = f"{operation_name} attempt {attempt} failed: {type(e).__name__}: {str(e)}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                result.attempts = attempt

                # Record retry event
                if self._collector:
                    from observability.collector import EventType
                    self._collector.record_event(
                        EventType.RETRY,
                        operation_name,
                        success=False,
                        error=str(e),
                        metadata={"attempt": attempt},
                    )

                # Don't wait after the last attempt
                if attempt < self._config.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        # All retries exhausted
        result.total_time_ms = (time.time() - start_time) * 1000
        result.success = False
        logger.warning(
            f"{operation_name} failed after {self._config.max_retries} attempts"
        )

        return result

    def execute_with_fallback(
        self,
        create_llm_func: Callable,
        call_func: Callable,
        messages: list,
        operation_name: str = "llm_call",
    ) -> RetryResult:
        """
        Execute an LLM call with retry AND provider fallback.

        This is the full resilience pipeline:
        1. Try primary provider with retries.
        2. If primary fails, try each fallback provider.
        3. Each fallback gets its own retry cycle.

        Args:
            create_llm_func: Function(provider, model, api_key) → LLM client.
            call_func: Function(llm, messages) → response.
            messages: Messages to send.
            operation_name: Name for logging.

        Returns:
            RetryResult with the first successful response.
        """
        result = RetryResult()
        start_time = time.time()
        all_errors = []

        for provider in self._fallback_providers:
            try:
                logger.info(
                    f"Trying {provider.name} ({provider.model}) "
                    f"for {operation_name}"
                )

                # Create LLM for this provider
                llm = create_llm_func(
                    provider.provider,
                    provider.model,
                    provider.api_key,
                )

                # Try with retries
                retry_result = self.execute_with_retry(
                    func=call_func,
                    args=(llm, messages),
                    operation_name=f"{operation_name}[{provider.name}]",
                )

                if retry_result.success:
                    result.success = True
                    result.result = retry_result.result
                    result.attempts = retry_result.attempts
                    result.provider_used = provider.provider
                    result.model_used = provider.model
                    result.total_time_ms = (time.time() - start_time) * 1000
                    result.errors = all_errors
                    result.fallback_used = provider != self._fallback_providers[0]

                    if result.fallback_used:
                        logger.info(
                            f"Fallback successful: using {provider.name}"
                        )
                        if self._collector:
                            from observability.collector import EventType
                            self._collector.record_event(
                                EventType.FALLBACK,
                                f"fallback_to_{provider.name}",
                                success=True,
                                metadata={
                                    "provider": provider.provider,
                                    "model": provider.model,
                                },
                            )

                    return result

                all_errors.extend(retry_result.errors)

            except Exception as e:
                error_msg = f"Provider {provider.name} unavailable: {e}"
                logger.warning(error_msg)
                all_errors.append(error_msg)

        # All providers failed
        result.success = False
        result.errors = all_errors
        result.total_time_ms = (time.time() - start_time) * 1000

        logger.error(f"All LLM providers failed for {operation_name}")
        return result

    def _compute_delay(self, attempt: int) -> float:
        """Compute delay with exponential backoff and optional jitter."""
        delay = self._config.initial_delay_seconds * (
            self._config.backoff_multiplier ** (attempt - 1)
        )
        delay = min(delay, self._config.max_delay_seconds)

        if self._config.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)


def create_llm_for_provider(
    provider: str,
    model: str,
    api_key: str,
):
    """
    Factory function to create an LLM client for any supported provider.

    Used by the fallback system to dynamically create LLM clients
    for different providers.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
        )
    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
