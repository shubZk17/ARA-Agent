# ARA-1 — Pitch & Architecture Walkthrough

> A 5-minute read. Every number below traces to a named test or a logged run — the "how do you know" pointer is in each section.

---

## 1. The problem

Manual equity research is slow and, worse, **unfalsifiable**. An analyst note says "we like NVDA long-term" with no entry price, no holding period, and nothing that would prove it wrong. Six months later you cannot tell whether the call was good or lucky. There is no dated, gradeable verdict — so there is no feedback loop, and confidence never gets calibrated against reality.

---

## 2. What ARA-1 does differently — two engines, not one

Most "AI analyst" tools are an LLM prompt with a nice UI. ARA-1 splits the job:

- **The LLM decides what evidence to gather.** It runs a ReAct loop over real tools (price, fundamentals, technicals, SEC filings, news) and stops when it has enough.
- **Deterministic rules decide what the evidence means.** A 6-stage, no-LLM synthesis pipeline (`analysis/`) grades the evidence against declarative threshold tables and produces the BUY/HOLD/SELL verdict, a health score, a risk level, and a confidence number.

The LLM never picks the verdict. Re-run the same state through synthesis and you get byte-identical output.

*How you know:* `ARCHITECTURE.md` §1 (the graph vs. synthesis split); synthesis is invoked as a post-processing step at `main.py` after `graph.invoke()` returns, not as a graph node.

---

## 3. The honesty mechanism — the centerpiece

The confidence score is engineered to be *hard to inflate*:

1. **It abstains.** Below `MIN_METRICS_FOR_RECOMMENDATION = 8` gathered metrics (`analysis/engine.py`), the system returns `INSUFFICIENT_DATA` instead of a verdict. A thin run says "I don't know" — on a real news-only run that path fired and confidence landed at **40%**, not the ~90% every run used to produce.
2. **It penalizes single-sourcing.** `data_completeness` is multiplied by a source-diversity factor, so four tools reading one endpoint no longer score "comprehensive."
3. **It caps on degraded infra.** If embeddings fall back to the hash implementation, confidence is capped at **50%** and logged at ERROR (defect D6).
4. **It recalibrates from graded outcomes.** Phase 8 re-fetches the price at each recommendation's review date, grades hit rate + Brier score, and feeds `(measured_hit_rate − 0.5)` back into the confidence constants once ≥10 outcomes exist (`analysis/confidence_calibrator._empirical_reliability_baseline`). *Honest status: the mechanism is built and tested but not yet active — no logged recommendation has reached its review date yet.*

*How you know:* `tests/test_confidence.py`, `tests/test_empirical_calibration.py` (4 tests), and the Phase 5 gate below.

---

## 4. Evidence it actually works

- **Same ticker, two horizons, two verdicts — pinned as a test.** Strong fundamentals + a broken downtrend produce `short_term` **SELL** and `long_term` **HOLD**, each with its own invalidation clauses and review window. This is `tests/test_horizons.py::test_gate_two_horizons_reach_materially_different_verdicts`, not a one-off — the horizon re-weights synthesis, it isn't a cosmetic filter.
- **Two real defects found by execution, each fixed with a test.** **D2:** yfinance returns `debtToEquity` as a percent; the code graded it as a ratio → fired "High Leverage" → drove NVDA (one of the least-levered megacaps) to risk **CRITICAL (1.00)**. Fixed 2026-08-16; D/E now reads **0.07** on the NVDA report. **D12:** the risk aggregator was `mean_severity / 0.6`, so *any two* MEDIUM risks scored 0.833 and tripped the CRITICAL threshold — found only while verifying the D2 fix. Both are regression-pinned (`tests/test_units.py`, `tests/test_risk_scoring.py`).
- **Confidence stopped being a constant.** Before Phase 5, every successful run reported ~0.90. After: NVDA **78%**, AAPL **80%**, thin run **40%** — verified live.
- **186 tests**, ~6s, no network, no LLM (81 at Phase 5 → 131 at Phase 6 → 186 today).
- **A real run, today (2026-09-03):** NVDA long-term, 6 tools, 1 cross-source conflict detected, 0 errors, 98% on the internal evaluation. Verdict: BUY, confidence 81% (High), health 0.80, risk HIGH, with a numeric invalidation condition — *"a sustained (>1 month) close below the 200-day SMA (196.29), or revenue growth falling below 95.9% YoY (currently 105.9%), or net margin below 58.7% (currently 63.7%); review by 2026-12-02."* That is a call Phase 8 can grade without hindsight.

---

## 5. What's next

- **Wire the 3rd source family into synthesis (Phase 7.4).** All three families now exist (`yfinance`, `sec_edgar`, `sec_edgar_ownership`), but `analysis/financial_engine.py` still only ingests yfinance-shaped data — the conflict-driven reroute is unbuilt.
- **Close the Phase 8 gate.** Run `python -m validation.score` once the logged recommendations mature past their review dates, so empirical recalibration goes live.
- **Cut the ~250s analysis time.** Measured: ~93% is LLM calls, and per-iteration cost grows because `reasoning_node` resends the full observation history every loop. The fix is to digest superseded observations in the *prompt* while keeping the full trace in state for episodic memory.
