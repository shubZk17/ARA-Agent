---
title: ARA-1 Autonomous Research Agent
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

<p align="center">
  <img src="docs/banner.png" alt="ARA-1 Banner" width="100%"/>
</p>

<h1 align="center">ARA-1 — Autonomous Research Agent</h1>


<p align="center">
  <b>A retrieval-aware autonomous financial intelligence system built with LangGraph and the ReAct framework.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-ReAct-FF6F00?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-4A154B?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Claude-10A37F?style=for-the-badge" alt="LLM"/>
</p>

---

## 📌 What is ARA-1?

**ARA-1** is an autonomous AI agent that performs end-to-end financial research on publicly traded companies. Give it a ticker symbol or a research question, and it will:

- 🔍 **Dynamically select tools** to gather real-time stock data, financial metrics, company profiles, and news
- 🧠 **Reason step-by-step** using the ReAct (Reasoning + Acting) framework
- 📚 **Store & retrieve knowledge** from a persistent vector memory (ChromaDB)
- 📊 **Score source reliability** using a 3-tier evidence governance system
- ⚡ **Detect conflicting information** and surface it transparently
- 🗂️ **Learn from past analyses** through episodic memory

ARA-1 is **not** a chatbot wrapper. It is a fully autonomous agent that decides *what to do*, *when to do it*, and *when to stop* — all without human intervention.

---

## 🏗️ Architecture

ARA-1 is built on a **layered modular architecture** with clean separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                     CLI / main.py                       │  ← Entry Point
├─────────────────────────────────────────────────────────┤
│              LangGraph StateGraph (agent/)              │  ← Orchestration
│         reasoning_node → tool_node → output_node        │
├──────────────────┬──────────────────┬───────────────────┤
│   Tool Layer     │  Retrieval Layer │  Memory Layer     │
│   (tools/)       │  (retrieval/)    │  (memory/)        │
│                  │  (ingestion/)    │                   │
│  • stock_price   │  • vector_store  │  • short_term     │
│  • company_info  │  • embeddings    │  • long_term      │
│  • fin. metrics  │  • retriever     │  • episodic       │
│  • news          │  • chunker       │                   │
├──────────────────┴──────────────────┴───────────────────┤
│              Evidence Governance (reliability/)          │
│         scorer · tiers · conflict_resolver               │
├─────────────────────────────────────────────────────────┤
│   parsers/  │  prompts/  │  config/  │  utils/logger    │  ← Support
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works — End-to-End Workflow

```mermaid
flowchart TD
    A["🧑 User Query"] --> B["Load Episodic Memory<br/>(prior run context)"]
    B --> C["Build Initial State"]
    C --> D["LangGraph: reasoning_node"]

    D --> E{"LLM Decision"}
    E -->|Use a Tool| F["tool_node<br/>Execute Tool"]
    F --> G["Auto-Ingest Output<br/>→ Clean → Chunk → Embed → Store"]
    G --> D

    E -->|Final Answer| H["output_node"]

    D --> I["Retrieve Evidence<br/>from Vector Memory"]
    I --> J["Score Reliability<br/>+ Detect Conflicts"]
    J --> D

    H --> K["Save Episode<br/>to Episodic Memory"]
    K --> L["Save Final Analysis<br/>to Vector Memory"]
    L --> M["🖥️ Display Results"]

    style A fill:#1a1a2e,color:#e94560
    style D fill:#0f3460,color:#16213e,color:#fff
    style G fill:#533483,color:#fff
    style H fill:#0f3460,color:#fff
    style M fill:#1a1a2e,color:#e94560
```

### Step-by-Step Breakdown

| Step | What Happens |
|------|-------------|
| **1. Query** | User provides a research question (e.g., *"Analyze NVDA stock"*) |
| **2. Episodic Recall** | Agent checks if it has analyzed this ticker before and loads prior experience |
| **3. Reasoning Loop** | LLM uses ReAct framework: **Thought** → **Action** → **Observation** → repeat |
| **4. Tool Execution** | Agent dynamically selects and calls tools (stock price, metrics, news, etc.) |
| **5. Auto-Ingestion** | Every tool output is automatically cleaned, chunked, embedded, and stored in ChromaDB |
| **6. Evidence Retrieval** | At each reasoning step, relevant evidence is semantically retrieved from vector memory |
| **7. Reliability Scoring** | Retrieved evidence is scored by source tier (Tier 1–3) with staleness decay |
| **8. Conflict Detection** | Contradictory evidence is flagged (numeric, sentiment, or temporal conflicts) |
| **9. Final Synthesis** | Agent produces a grounded analysis citing evidence with confidence scores |
| **10. Memory Update** | Episode saved for future learning; final analysis stored in vector memory |

---

## 🚀 Local Setup & Usage

### Prerequisites

- **Python 3.12+**
- **Git**
- At least one LLM API key: **Groq** (free), **OpenAI**, or **Anthropic**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ara-agent.git
cd ara-agent
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example config and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your preferred editor:

```env
# ============================================================
# ARA-1 Environment Configuration
# ============================================================

# --- LLM Provider (pick one) ---
# Option A: Groq (FREE — recommended for getting started)
GROQ_API_KEY=your-groq-api-key-here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile

# Option B: OpenAI
# OPENAI_API_KEY=your-openai-api-key-here
# LLM_PROVIDER=openai
# OPENAI_MODEL=gpt-4o

# Option C: Anthropic
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# LLM_PROVIDER=claude
# ANTHROPIC_MODEL=claude-sonnet-4-20250514

# --- Agent Settings ---
MAX_ITERATIONS=10
LOG_LEVEL=INFO

# --- Phase 2: Embeddings (optional, enhances retrieval quality) ---
# OPENAI_API_KEY=your-openai-api-key-here
```

> **💡 Tip:** Get a free Groq API key at [console.groq.com](https://console.groq.com). No credit card required.

### 5. Run the Agent

**Interactive mode** (prompts for a query):

```bash
python main.py
```

**With a query argument:**

```bash
python main.py "Analyze Tesla stock as a long-term investment"
```

**More example queries:**

```bash
python main.py "What are the financial risks of investing in AAPL?"
python main.py "Compare NVDA and AMD financial performance"
python main.py "Provide a comprehensive analysis of Microsoft (MSFT)"
```

---

## 📂 Project Structure

```
ARA-1/
│
├── agent/                    # 🧠 Core Agent Logic
│   ├── state.py              #    TypedDict state + Phase 2 memory fields
│   ├── graph.py              #    LangGraph StateGraph definition
│   └── nodes.py              #    ReAct nodes: reasoning, tool, output
│
├── tools/                    # 🔧 Financial Data Tools
│   ├── base.py               #    Abstract BaseTool interface
│   ├── registry.py           #    Dynamic tool registry
│   ├── stock_price.py        #    Real-time stock price (yfinance)
│   ├── company_info.py       #    Company profile & overview
│   ├── financial_metrics.py  #    P/E, EPS, margins, ratios
│   └── news.py               #    Recent financial news
│
├── retrieval/                # 🔍 Semantic Retrieval (Phase 2)
│   ├── schemas.py            #    Document, Chunk, Evidence data models
│   ├── vector_store.py       #    ChromaDB abstraction layer
│   ├── embeddings.py         #    OpenAI embedding pipeline + fallback
│   └── retriever.py          #    Semantic search orchestrator
│
├── ingestion/                # 📥 Document Ingestion Pipeline (Phase 2)
│   ├── pipeline.py           #    Clean → Chunk → Embed → Store
│   ├── chunker.py            #    Recursive text splitting with overlap
│   ├── cleaners.py           #    HTML/Unicode/financial text cleaning
│   └── loaders.py            #    Source-type document loaders
│
├── memory/                   # 🗂️ Memory Layer (Phase 2)
│   ├── base.py               #    Abstract memory interface
│   ├── short_term.py         #    In-session context memory
│   ├── long_term.py          #    Persistent vector-backed memory
│   └── episodic.py           #    Run experience storage (JSON)
│
├── reliability/              # 🛡️ Evidence Governance (Phase 2)
│   ├── tiers.py              #    Source reliability tier definitions
│   ├── scorer.py             #    Reliability scoring + staleness decay
│   └── conflict_resolver.py  #    Contradiction detection engine
│
├── parsers/
│   └── react_parser.py       # 🔄 LLM response → structured JSON parser
│
├── prompts/
│   └── system.py             # 📝 Dynamic system prompt templates
│
├── config/
│   └── settings.py           # ⚙️ Centralized configuration management
│
├── utils/
│   └── logger.py             # 📋 Rich console + file logging
│
├── data/                     # 💾 Persistent Storage
│   ├── chroma/               #    ChromaDB vector database files
│   └── episodic/             #    Episodic memory JSON files
│
├── logs/                     # 📄 Session log files
├── docs/                     # 📖 Documentation assets
│
├── main.py                   # 🚀 Entry point & system assembly
├── requirements.txt          # 📦 Python dependencies
├── .env.example              # 🔑 Environment template
└── .gitignore                # 🚫 Git exclusions
```

---

## 🧪 Example Output

```
┌─────────────────────────────────────────────────────────────┐
│  ARA-1 - Autonomous Research Agent                          │
│  Phase 2: Retrieval-Aware Financial Intelligence            │
└─────────────────────────────────────────────────────────────┘
[OK] Configuration valid
[OK] LLM Provider: groq (llama-3.3-70b-versatile)
[OK] Phase 2 systems initialized:
     Vector Store: chroma (4 existing docs)
     Embeddings: text-embedding-3-small
     Episodic Memory: 1 prior episodes

Starting analysis...
Query: Analyze NVDA stock performance
Max iterations: 10

  === Iteration 1/10 ===
  Thought: I need to gather NVDA's current stock price...
  >> Action: get_stock_price({'ticker': 'NVDA'})
  Ingested 1 chunk (total in store: 5)

  === Iteration 2/10 ===
  Retrieved 5 evidence items in 6.3ms
  Thought: Now I need financial metrics...
  >> Action: get_financial_metrics({'ticker': 'NVDA'})
  Ingested 1 chunk (total in store: 6)

  ...

┌──────────────────── [OK] Analysis Complete ─────────────────────┐
│                                                                  │
│  NVIDIA Corporation (NVDA) is a technology company operating     │
│  in the semiconductors industry. Current price: USD 224.65,     │
│  trailing P/E: 45.83, revenue growth: 73.20%, market cap:       │
│  $5.46T. Strong financial position with 55.60% profit margin.   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Tool Usage Summary:
┌───┬───────────────────────┬────────────────────┬────────┐
│ # │ Tool                  │ Input              │ Status │
├───┼───────────────────────┼────────────────────┼────────┤
│ 1 │ get_stock_price       │ {'ticker': 'NVDA'} │ OK     │
│ 2 │ get_financial_metrics │ {'ticker': 'NVDA'} │ OK     │
│ 3 │ get_company_info      │ {'ticker': 'NVDA'} │ OK     │
└───┴───────────────────────┴────────────────────┴────────┘

Memory Operations:
  📦 ingested:get_stock_price:NVDA:1chunks
  📦 ingested:get_financial_metrics:NVDA:1chunks
  📦 ingested:get_company_info:NVDA:1chunks
```

---

## 🔌 Adding Custom Tools

ARA-1's tool system is fully extensible. To add a new tool:

**1. Create a new file** in `tools/`:

```python
# tools/my_custom_tool.py
from tools.base import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "Description of what this tool does"

    @property
    def parameters(self) -> dict:
        return {
            "param1": {"type": "string", "description": "What this param is", "required": True}
        }

    def execute(self, **kwargs) -> ToolResult:
        param1 = kwargs.get("param1", "")
        # Your logic here
        result = f"Result for {param1}"
        return ToolResult(success=True, data=result)
```

**2. Register it** in `main.py`:

```python
from tools.my_custom_tool import MyCustomTool

def create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    # ... existing tools ...
    registry.register(MyCustomTool())  # ← Add this line
    return registry
```

That's it. The agent will automatically discover and use your tool when relevant.

---

## 🔑 Supported LLM Providers

| Provider | Model | Free? | Configuration |
|----------|-------|-------|---------------|
| **Groq** | `llama-3.3-70b-versatile` | ✅ Yes | `LLM_PROVIDER=groq` |
| **OpenAI** | `gpt-4o` | ❌ Paid | `LLM_PROVIDER=openai` |
| **Anthropic** | `claude-sonnet-4-20250514` | ❌ Paid | `LLM_PROVIDER=claude` |

> **Recommendation:** Start with **Groq** — it's free, fast, and the `llama-3.3-70b-versatile` model works excellently with ARA-1's ReAct prompts.

---

## 🛡️ Source Reliability Tiers

ARA-1 doesn't treat all information equally. Every piece of evidence is scored:

| Tier | Score Range | Sources | Examples |
|------|------------|---------|----------|
| **Tier 1** | 0.85 – 1.0 | Official/Primary | SEC filings, exchange data, tool API outputs |
| **Tier 2** | 0.60 – 0.84 | Established Media | Reuters, Bloomberg, Yahoo Finance, WSJ |
| **Tier 3** | 0.30 – 0.59 | Secondary/Informal | Seeking Alpha, Reddit, blogs, social media |

Scores also decay over time — a stock price from last week is less reliable than one from today.

---

## 📊 Phase Progression

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | ReAct loop, tool execution, structured parsing, error handling |
| **Phase 2** | ✅ Complete | Vector memory, semantic retrieval, evidence governance, episodic learning |
| **Phase 3** | 🔮 Planned | Multi-agent collaboration, async tools, streaming UI, human-in-the-loop |

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph (StateGraph) |
| **Reasoning** | ReAct (Reason + Act) |
| **LLM Interface** | LangChain Core |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Financial Data** | yfinance |
| **Data Validation** | Pydantic v2 |
| **CLI / Display** | Rich |
| **Logging** | Python logging + Rich |
| **Config** | python-dotenv |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-tool`)
3. Commit your changes (`git commit -m 'Add SEC filing tool'`)
4. Push to the branch (`git push origin feature/new-tool`)
5. Open a Pull Request

---

## 📜 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with 🧠 by the ARA-1 Team</b><br/>
  <i>Autonomous reasoning. Grounded evidence. Financial intelligence.</i>
</p>
