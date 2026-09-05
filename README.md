---
app_port: 7860
colorFrom: indigo
colorTo: purple
emoji: 📈
license: mit
pinned: false
sdk: docker
title: ARA-1 --- Autonomous Research Agent
---

# ARA-1 --- Autonomous Research Agent

An evidence-driven equity research agent that uses an LLM to gather
financial evidence through tools and a deterministic analysis pipeline
to turn that evidence into investment theses.

> **Core principle:** the LLM decides **what evidence to gather**.
> Deterministic analysis decides **what the evidence means**.

## Overview

ARA-1 performs structured research on publicly traded companies. A
typical run:

1.  Accepts a company ticker or research question.
2.  Uses a LangGraph ReAct loop to decide which tools to call.
3.  Collects market, company, financial, news, and SEC data.
4.  Scores evidence reliability and surfaces conflicts.
5.  Stores and retrieves useful evidence through ChromaDB and episodic
    memory.
6.  Runs a deterministic seven-stage analysis pipeline.
7.  Produces a horizon-specific investment thesis and report.
8.  Records recommendations so they can later be evaluated against
    outcomes.

ARA-1 is intentionally **not a chatbot wrapper**. The LLM is responsible
for evidence gathering and tool selection; it does not directly generate
the final BUY/HOLD/SELL decision.

------------------------------------------------------------------------

## Architecture

``` text
                         User Query
                             |
                             v
                    +----------------+
                    |    main.py     |
                    |  / api/server  |
                    +-------+--------+
                            |
                            v
                 +----------------------+
                 |       agent/         |
                 |   LangGraph ReAct   |
                 |                      |
                 | reasoning -> tool   |
                 |      ^          |    |
                 |      |          v    |
                 |      +------ output  |
                 +----------+-----------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
       +-------------+             +-------------+
       |   tools/    |             | knowledge/  |
       |             |             |             |
       | yfinance    |             | retrieval   |
       | SEC EDGAR   |             | memory      |
       | news/data   |             | reliability |
       +------+------+             +------+------+
              |                           |
              +-------------+-------------+
                            |
                            v
                 +----------------------+
                 |      analysis/       |
                 |                      |
                 | financial            |
                 | technical            |
                 | sentiment            |
                 | misalignment         |
                 | risk                 |
                 | confidence           |
                 | thesis               |
                 +----------+-----------+
                            |
                 +----------+----------+
                 |          |          |
                 v          v          v
              Report   Recommendation  Evaluation
              MD/PDF      JSONL        / scoring
```

The graph is only the **ReAct evidence-gathering loop**. Once the graph
finishes, `analysis/engine.py` receives the completed `AgentState` and
runs the investment synthesis separately.

### Package boundaries

  -----------------------------------------------------------------------
  Package                             Responsibility
  ----------------------------------- -----------------------------------
  `agent/`                            ReAct loop, prompts, state,
                                      parsing, routing

  `tools/`                            External financial data
                                      acquisition; tools never call the
                                      LLM

  `knowledge/`                        Ingestion, semantic retrieval,
                                      episodic memory, reliability and
                                      conflict handling

  `analysis/`                         Deterministic financial analysis,
                                      thesis generation, reports,
                                      evaluation, backtesting and
                                      recommendation scoring

  `api/`                              FastAPI backend and API endpoints

  `webui/`                            Lightweight HTML/CSS/JavaScript
                                      frontend served by FastAPI

  `config/`                           Settings, investment horizons and
                                      logging

  `tests/`                            Offline test suite
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## How a Run Works

``` text
User Query
    |
    v
Initial AgentState
    |
    v
+-----------------------------+
| LangGraph ReAct Loop        |
|                             |
| Reason -> Tool -> Observe   |
|          ^          |       |
|          +----------+       |
+-------------+---------------+
              |
              | evidence satisfied
              v
       Completed AgentState
              |
              v
+-----------------------------+
| Deterministic Synthesis     |
|                             |
| Financial                   |
|     ↓                       |
| Technical                   |
|     ↓                       |
| Sentiment                   |
|     ↓                       |
| Misalignment                |
|     ↓                       |
| Risk                        |
|     ↓                       |
| Confidence                 |
|     ↓                       |
| Investment Thesis           |
+-------------+---------------+
              |
       +------+------+
       |             |
       v             v
   MD / PDF      JSONL record
```

### 1. Evidence gathering

The LLM receives the research question, previous context, available
tools, and retrieved evidence. It decides which tool is useful next.

The current tool layer includes:

-   Stock price
-   Financial metrics
-   Company information
-   News
-   Price history
-   Market context
-   SEC filings
-   SEC insider activity

### 2. Evidence governance

Tool outputs are treated as **evidence candidates, not unquestionable
truth**.

The knowledge layer:

-   Cleans and chunks tool output.
-   Generates embeddings.
-   Stores evidence in ChromaDB.
-   Retrieves semantically relevant evidence.
-   Combines similarity with reliability.
-   Accounts for source staleness.
-   Detects conflicting evidence.

The agent also records evidence confidence and conflict reports in its
state.

### 3. Deterministic synthesis

After the ReAct loop completes, the finished state passes through:

``` text
Financial Analysis
       ↓
Technical Analysis
       ↓
Sentiment Analysis
       ↓
Misalignment Detection
       ↓
Risk Analysis
       ↓
Confidence Calibration
       ↓
Investment Thesis
```

The synthesis layer does not make another LLM call. It converts
structured evidence into an `InvestmentOutlook` and horizon-specific
`HorizonRecommendation`.

Recommendations carry explicit conditions such as:

-   investment horizon
-   holding period
-   entry condition
-   review date
-   price at recommendation
-   confidence
-   invalidation condition

The invalidation condition makes the thesis **falsifiable** rather than
simply descriptive.

------------------------------------------------------------------------

## Investment Horizons

ARA-1 supports horizon-specific reasoning through `config/horizons.py`.

A horizon profile controls:

-   evidence weighting
-   analysis thresholds
-   required tools
-   review period
-   prompt guidance

The same company can therefore receive different short-, medium-, and
long-term assessments because the decision criteria change with the
investment horizon.

------------------------------------------------------------------------

## Data Sources

The tool layer currently uses:

-   **yfinance** for market and company data
-   **SEC EDGAR / companyfacts** for filed financial information
-   News data available through the financial data layer

The architecture is designed so additional data sources can be added as
independent tools without changing the core ReAct loop.

------------------------------------------------------------------------

## Project Structure

``` text
ARA-Agent/
│
├── agent/                 # ReAct agent and LangGraph control flow
├── tools/                 # External financial data tools
├── knowledge/             # Retrieval, memory and evidence governance
├── analysis/              # Deterministic investment analysis
├── api/                   # FastAPI backend
├── webui/                 # Static frontend
├── config/                # Settings, horizons and logging
├── tests/                 # Offline test suite
│
├── main.py                # CLI entry point and application wiring
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── Dockerfile             # Container deployment
└── .env.example           # Environment configuration template
```

Runtime data such as the vector database, episodic memory,
recommendations, evaluations and logs are kept outside the core source
structure.

------------------------------------------------------------------------

## Getting Started

### Prerequisites

-   Python 3.12+
-   Git
-   An LLM API key supported by the current configuration

### Installation

``` bash
git clone https://github.com/shubZk17/ARA-Agent.git
cd ARA-Agent

python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

### Environment

Copy the example environment file:

``` bash
cp .env.example .env
```

Then configure the required LLM credentials and application settings in
`.env`.

### Run from the CLI

Interactive mode:

``` bash
python main.py
```

With a research question:

``` bash
python main.py "Analyze NVDA as a long-term investment"
```

Other examples:

``` bash
python main.py "What are the financial risks of investing in AAPL?"
python main.py "Compare the financial performance of NVDA and AMD"
```

------------------------------------------------------------------------

## Web API

FastAPI is the only backend. The same application also serves the static
web UI, so there is no separate web gateway.

The primary API surface includes:

``` text
GET  /
GET  /health
GET  /config
POST /analyze
POST /evaluate
```

The application can be run with the project's configured ASGI server or
through Docker.

------------------------------------------------------------------------

## Adding a Tool

Tools are intentionally isolated from the agent's reasoning code.

To add a new financial data source:

1.  Create a module in `tools/`.
2.  Implement the `BaseTool` contract.
3.  Define its name, description and parameters.
4.  Implement data fetching and rendering.
5.  Register the tool in the tool registry.

Conceptually:

``` text
tools/
└── new_data_source.py
        |
        v
    BaseTool
        |
        v
  ToolRegistry
        |
        v
   ReAct agent
```

A tool should fetch and format evidence. It should **not** decide what
that evidence means and should **never call the LLM**.

------------------------------------------------------------------------

## Evaluation and Outcome Tracking

ARA-1 distinguishes between **process evaluation** and **investment
outcome evaluation**.

### Process evaluation

`analysis/evaluation/` contains run-quality metrics and checks such as
tool usage, execution behaviour and hallucination-related signals.

### Outcome evaluation

Recommendations are stored in an append-only JSONL record. Once a
recommendation reaches its review date, `analysis/outcome_scorer.py` can
evaluate the realized outcome.

Additional analysis includes:

-   recommendation hit rate
-   Brier-style scoring
-   technical backtesting
-   empirical confidence calibration

This creates a path from:

``` text
Recommendation
      ↓
Realized Outcome
      ↓
Outcome Score
      ↓
Calibration
```

------------------------------------------------------------------------

## Testing

The test suite is designed to run primarily without network access or
live LLM calls.

Run:

``` bash
pytest
```

The project also uses compile-time checks and targeted tests for state
contracts, graph routing, horizons, tools, analysis and observability.

------------------------------------------------------------------------

## Design Principles

### Evidence is not truth

Every external result is treated as evidence with reliability and
freshness characteristics.

### The LLM gathers; deterministic code synthesizes

The LLM handles tool selection and evidence gathering. The investment
thesis is produced by deterministic analysis code.

### Structured data is preferred

Tools maintain canonical structured payloads for downstream analysis.
Human-readable strings are derived from those payloads rather than
parsed back into numbers.

### Recommendations should be falsifiable

A recommendation includes an explicit invalidation condition and review
date.

### Keep the agent bounded

The ReAct loop has bounded iterations and bounded parsing retries. The
agent should gather enough evidence, not run indefinitely.

### Keep components replaceable

Tools, horizons and analysis dimensions can be extended without
rewriting the core graph.

------------------------------------------------------------------------

## Development Guide

If you are learning the codebase, read it in this order:

1.  `main.py` --- application wiring
2.  `agent/state.py` --- the `AgentState` contract
3.  `agent/graph.py` --- the ReAct control flow
4.  `agent/nodes.py` --- reasoning and tool execution
5.  `tools/stock_price.py` --- simplest example of a tool
6.  `knowledge/` --- retrieval and evidence handling
7.  `analysis/engine.py` --- deterministic synthesis pipeline

The most important distinction to remember is:

``` text
                 LLM
                  |
           "What should I
            investigate?"
                  |
                  v
          Evidence / Tools
                  |
                  v
          Completed State
                  |
                  v
        Deterministic Analysis
                  |
          "What does the
           evidence imply?"
                  |
                  v
          Investment Thesis
```

------------------------------------------------------------------------

## License

MIT License.

This project is intended for educational and research purposes. It is
not financial advice.
