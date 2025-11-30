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

This will install the following main dependencies:
- `langchain`: For building LLM chains
- `langchain-community`: Community integrations (e.g., Ollama)
- `sentence-transformers`: For document vectorization
- `chromadb`: Vector database

And development dependencies:
- `pytest`: Testing framework
- `pytest-cov`: Coverage plugin for pytest
- `ruff`: Fast Python linter and formatter
- `mypy`: Static type checker
- `pydoclint`: Docstring linter
- `poethepoet`: Task runner

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

## Usage

### Prerequisites

Before using the local LLM, ensure:

1. **Ollama is installed and running**
   - Install Ollama from [https://ollama.ai](https://ollama.ai)
   - Start the Ollama service

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

Alternatively, you can use it with `uv run`:

```bash
uv run dora-test
```

## Development

### Running Tests

Run tests with pytest:

```bash
# Run all tests
uv run pytest
```

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
│       ├── __init__.py
│       ├── llm.py     # LocalLLM implementation
│       └── cli.py      # CLI interface
├── tests/             # Test files
│   ├── __init__.py
│   └── test_llm.py    # LLM tests
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lock file
├── .gitignore         # Git ignore rules
└── README.md
```
