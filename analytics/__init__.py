"""
analytics/ — pure numerical primitives over price series.

Deliberately a LEAF package: it imports nothing from tools/, analysis/ or
knowledge/, so both sides can use it without a circular import. tools/
computes a price summary with it at the trust boundary; analysis/ grades that
summary. Nothing here knows what a stock is — it is arithmetic on a Series.
"""

from analytics.indicators import (  # noqa: F401
    atr,
    ema,
    macd,
    max_drawdown,
    realized_volatility,
    returns_table,
    rsi,
    sma,
)
