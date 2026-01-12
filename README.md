<div align="center">

# Synthetic Data Generator AI

**Generate realistic synthetic datasets using natural language prompts powered by LLMs**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/familorujov/synthetic-data-generator-ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

*Describe your data in plain English and get structured datasets instantly*

[Quick Start](#quick-start) • [Features](#features) • [Architecture](#architecture) • [Documentation](#documentation)

</div>

---

## What is This?

A Streamlit-based tool that generates realistic tabular data using LLMs. Instead of writing scripts or using random generators, just describe what you need:

```
"Customer data with name, email, age 25-60, and US city"
```

And get a clean pandas DataFrame ready to use.

## Features

| Feature | Description |
|---------|-------------|
| **Natural Language** | Describe your data in plain English |
| **Multi-Provider** | Ollama, OpenAI, Gemini, Anthropic, Groq |
| **Smart Generation** | Add columns incrementally to existing data |
| **Full Control** | Remove rows/columns, toggle index, export CSV |
| **Docker Ready** | One command deployment |

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull and run
docker pull familorujov/synthetic-data-generator-ai:v1.0
docker run -p 8501:8501 familorujov/synthetic-data-generator-ai:v1.0

# Or use docker compose
docker compose up
```

Open http://localhost:8501

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/FamilOrujov/synthetic-data-generator-ai.git
cd synthetic-data-generator-ai
```

**Using uv (recommended):**
```bash
uv sync
uv run streamlit run app.py
```

**Using pip and venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Using pip directly:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                            │
│                          (app.py)                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DataGenerator                              │
│                  (src/data_generator.py)                        │
│  • Builds prompts from user instructions                        │
│  • Parses JSON responses into DataFrames                        │
│  • Normalizes array lengths for consistency                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Client Layer                           │
│                       (src/llm.py)                              │
├─────────────┬─────────────┬─────────────┬───────────┬───────────┤
│   Ollama    │   OpenAI    │   Gemini    │ Anthropic │   Groq    │
│   (local)   │   (cloud)   │   (cloud)   │  (cloud)  │  (cloud)  │
└─────────────┴─────────────┴─────────────┴───────────┴───────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **UI Layer** | `app.py` | Streamlit interface, user interactions, session state |
| **Generator** | `src/data_generator.py` | Core logic: prompt building, response parsing, DataFrame creation |
| **LLM Clients** | `src/llm.py` | Provider-specific API integrations with unified interface |
| **Utilities** | `src/utils.py` | JSON extraction, array normalization, CSV export |

## Documentation

### Usage Guide

1. **Select Provider** — Choose Ollama (local), OpenAI, Gemini, Anthropic, or Groq
2. **Configure** — Enter API key (if using cloud provider) and click "Apply Settings"
3. **Describe Data** — Enter natural language instructions
4. **Generate** — Click Generate and get your dataset
5. **Export** — Download as CSV

### Example Prompts

```
Customer database with first_name, last_name, email, age 25-65, city

Product catalog: name, price ($10-500), category, stock quantity, rating 1-5

Employee records: name, department, job_title, salary, years_employed
```

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Temperature | Higher = more creative, lower = more consistent | 0.7 |
| Max Tokens | Maximum response length | 800 |
| Timeout | Request timeout in seconds | 300 |

### Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Ollama** | llama3.1, mistral, etc. | Local, free, requires [Ollama](https://ollama.ai) |
| **OpenAI** | GPT-4, GPT-5, o1, o3 | Best quality |
| **Gemini** | gemini-2.0-flash, 3.0-pro | Good free tier |
| **Anthropic** | Claude Sonnet/Haiku/Opus | Great at following instructions |
| **Groq** | Various open models | Ultra-fast inference |

### Programmatic Usage

You can use the core components directly without the UI:

```python
from src.llm import create_llm_client
from src.data_generator import DataGenerator
import pandas as pd

# Create client
client = create_llm_client(
    provider="openai",
    model="gpt-4o",
    api_key="your-api-key"
)

# Generate data
generator = DataGenerator(client)
df = generator.generate_features(
    instructions="Customer data: name, email, age 25-60",
    n_rows=50,
    existing_dataframe=pd.DataFrame()
)

print(df.head())
```

## Project Structure

```
├── app.py                  # Streamlit UI application
├── src/
│   ├── __init__.py         # Package exports
│   ├── llm.py              # LLM client implementations (Ollama, OpenAI, etc.)
│   ├── data_generator.py   # Core generation logic & prompt engineering
│   └── utils.py            # Helper functions (JSON parsing, normalization)
├── requirements.txt        # Python dependencies (pip/venv)
├── pyproject.toml          # Project metadata and dependencies (uv)
├── Dockerfile              # Container configuration
└── docker-compose.yml      # One-command deployment
```

