# Dora

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Mamo1031/Dora/graph/badge.svg?token=D2A2FU8CFY)](https://codecov.io/gh/Mamo1031/Dora)

Knowledge-augmented local LLM using RAG (Retrieval-Augmented Generation).

## Overview

Dora is a knowledge-augmented large language model (LLM) system that operates in a local environment. It uses RAG (Retrieval-Augmented Generation) technology to supplement accurate and up-to-date information from external knowledge sources, addressing issues such as hallucinations (incorrect knowledge generation) and outdated knowledge.

## Requirements

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (package manager)

## Setup

### 1. Install uv

If uv is not installed, you can install it with the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

Run the following command in the project root:

```bash
# Install main dependencies and development dependencies
uv sync --all-extras
```

This will install the following main dependencies:
- `langchain`: For building LLM chains
- `langchain-community`: Community integrations (e.g., Ollama)
- `sentence-transformers`: For document vectorization
- `chromadb`: Vector database

### 3. Activate Virtual Environment

uv automatically manages the virtual environment. When running scripts:

```bash
uv run python your_script.py
```

Alternatively, to activate the virtual environment directly in your shell:

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Development

### Code Quality Checks

```bash
# Lint and format with Ruff
uv run poe lint-ruff

# Type checking with MyPy
uv run poe lint-mypy

# Run all linting
uv run poe lint
```

### Project Structure

```
Dora/
├── src/
│   └── dora/          # Main package
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lock file
└── README.md
```
