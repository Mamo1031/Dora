# Dora

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Mamo1031/Dora/graph/badge.svg?token=IkuhZ1Tu3K)](https://codecov.io/gh/Mamo1031/Dora)

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

### 3. Activate Virtual Environment

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Usage

### Prerequisites

Before using the local LLM, ensure:

1. **Ollama is installed and running**
   
   Install Ollama on Linux/WSL:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   After installation, Ollama service should start automatically. If not, start it manually:
   ```bash
   ollama serve
   ```

2. **Llama 3.2 (3B) model is pulled**
   ```bash
   ollama pull llama3.2
   ```

### Testing the Local LLM

After installation, you can test the local LLM using the CLI command:

```bash
dora-test
```

This will:
- Initialize the LocalLLM with Llama 3.2 (3B)
- Run a test prompt
- Display the generated response

### Interactive Mode

Start an interactive chat session with the LLM:

```bash
dora
```

This will:
- Initialize the LocalLLM with Llama 3.2 (3B)
- Start an interactive session where you can chat with the LLM
- Type your prompts and receive responses
- Type `exit` or `quit` to end the session
- Press `Ctrl+C` to exit at any time

## Development

```bash
# Run tests
uv run poe test

# Run linting
uv run poe lint
```

### Project Structure

```
Dora/
├── src/
│   └── dora/          # Main package
│       ├── __init__.py
│       ├── llm.py     # LocalLLM implementation
│       └── cli.py     # CLI interface
├── tests/             # Test files
│   ├── __init__.py
│   └── test_llm.py    # LLM tests
├── pyproject.toml     # Project configuration
├── uv.lock            # Dependency lock file
├── .gitignore         # Git ignore rules
└── README.md
```
