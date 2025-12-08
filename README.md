<div align="center">

# 🧬 Synthetic Data Generator AI

**Generate realistic synthetic datasets using natural language prompts powered by LLMs**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/familorujov/synthetic-data-generator-ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

<img src="https://raw.githubusercontent.com/FamilOrujov/synthetic-data-generator-ai/main/assets/demo.gif" alt="Demo" width="800">

*Describe your data in plain English → Get structured datasets instantly*

[🚀 Quick Start](#-quick-start) • [✨ Features](#-features) • [📖 Usage](#-usage) • [🔧 Configuration](#-configuration)

</div>

---

## 🎯 What is This?

A Streamlit-based tool that generates realistic tabular data using LLMs. Instead of writing scripts or using random generators, just describe what you need:

```
"Customer data with name, email, age 25-60, and US city"
```

And get a clean pandas DataFrame ready to use.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Natural Language** | Describe your data in plain English |
| 🔌 **Multi-Provider** | Ollama, OpenAI, Gemini, Anthropic, Groq |
| 📊 **Smart Generation** | Add columns incrementally to existing data |
| 🎛️ **Full Control** | Remove rows/columns, toggle index, export CSV |
| 🐳 **Docker Ready** | One command deployment |

## 🚀 Quick Start

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
# Clone
git clone https://github.com/FamilOrujov/synthetic-data-generator-ai.git
cd synthetic-data-generator-ai

# Install dependencies
pip install pandas requests streamlit

# Run
streamlit run app.py
```

## 📖 Usage

1. **Select Provider** — Choose Ollama (local), OpenAI, Gemini, Anthropic, or Groq
2. **Configure** — Enter API key (if using cloud provider) and click "Apply Settings"
3. **Describe Data** — Enter natural language instructions like:
   ```
   E-commerce products: name, price $10-500, category, in_stock boolean
   ```
4. **Generate** — Click Generate and get your dataset
5. **Export** — Download as CSV

### Example Prompts

```
👥 Customer database with first_name, last_name, email, age 25-65, city

🛒 Product catalog: name, price ($10-500), category, stock quantity, rating 1-5

👔 Employee records: name, department, job_title, salary, years_employed
```

## 🔧 Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| Temperature | Higher = more creative | 0.7 |
| Max Tokens | Response limit | 800 |
| Timeout | Request timeout (seconds) | 300 |

## 🔌 Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Ollama** | llama3.1, mistral, etc. | Local, free, requires [Ollama](https://ollama.ai) |
| **OpenAI** | GPT-4, GPT-5, o1, o3 | Best quality |
| **Gemini** | gemini-2.0-flash, 3.0-pro | Good free tier |
| **Anthropic** | Claude Sonnet/Haiku/Opus | Great at following instructions |
| **Groq** | Various open models | Ultra-fast inference |

## 📁 Project Structure

```
├── app.py              # Streamlit UI
├── src/
│   ├── llm.py          # LLM client implementations
│   ├── data_generator.py # Core generation logic
│   └── utils.py        # Helper functions
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 🐳 Docker Hub

```bash
docker pull familorujov/synthetic-data-generator-ai:v1.0
```

## 📄 License

MIT License - feel free to use in your projects.

---

<div align="center">

**Made with ❤️ by [Famil Orujov](https://github.com/FamilOrujov)**

⭐ Star this repo if you find it useful!

</div>