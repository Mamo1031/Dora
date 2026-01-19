# Dora

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Mamo1031/Dora/graph/badge.svg?token=IkuhZ1Tu3K)](https://codecov.io/gh/Mamo1031/Dora)

Knowledge-augmented local LLM using RAG (Retrieval-Augmented Generation).

## Overview

Dora is a knowledge-augmented large language model (LLM) system that operates in a local environment. It uses RAG (Retrieval-Augmented Generation) technology to supplement accurate and up-to-date information from external knowledge sources, addressing issues such as hallucinations (incorrect knowledge generation) and outdated knowledge.

### Features

- **Local LLM**: Run LLM inference locally using Ollama
- **RAG Support**: Enhance responses with knowledge from PDF documents
- **Vector Search**: Fast semantic search using ChromaDB and multilingual embeddings
- **Knowledge Base Management**: Easy document management via CLI commands

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
- Automatically detect if a knowledge base exists and enable RAG mode
- Start an interactive session where you can chat with the LLM
- Type your prompts and receive responses
- Type `exit` or `quit` to end the session
- Press `Ctrl+C` to exit at any time

If documents are added to the knowledge base, RAG mode will be automatically enabled to provide more accurate answers based on your documents.

### RAG (Retrieval-Augmented Generation)

Dora supports RAG to enhance LLM responses with information from your documents. The system uses:
- **ChromaDB**: Vector database for storing document embeddings
- **Multilingual Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` for semantic search
- **PDF Support**: Load and process PDF documents

#### Adding Documents to Knowledge Base

Add a PDF document to the knowledge base:

```bash
dora add-doc <path_to_pdf_file>
```

Example:
```bash
dora add-doc document.pdf
```

The document will be:
- Loaded and split into chunks (1000 characters with 200 character overlap)
- Embedded using the multilingual embedding model
- Stored in the vector database at `.dora/kb/`

#### Listing Documents

View all documents in your knowledge base:

```bash
dora list-docs
```

This shows:
- Total number of chunks in the knowledge base
- List of document sources

#### Clearing Knowledge Base

Remove all documents from the knowledge base:

```bash
dora clear-kb
```

This will prompt for confirmation before clearing.

#### Using RAG in Code

You can also use RAG programmatically:

```python
from dora import LocalLLM, KnowledgeBase

# Initialize knowledge base
kb = KnowledgeBase()

# Add a document
kb.add_document("document.pdf")

# Create LLM with RAG enabled
llm = LocalLLM(
    model_name="llama3.2",
    use_rag=True,
    knowledge_base=kb
)

# Get answer with sources
result = llm.invoke_with_sources("What is the document about?")
print(result["result"])
print(f"Sources: {len(result['source_documents'])} documents")
```

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
│   └── dora/              # Main package
│       ├── __init__.py
│       ├── llm.py         # LocalLLM implementation with RAG support
│       ├── cli.py         # CLI interface
│       ├── document.py    # Document loading and processing
│       ├── vectorstore.py # Vector store management (ChromaDB)
│       ├── knowledge_base.py  # Knowledge base management
│       └── rag.py         # RAG chain implementation
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_llm.py        # LLM tests
│   ├── test_document.py   # Document processing tests
│   ├── test_vectorstore.py # Vector store tests
│   ├── test_knowledge_base.py # Knowledge base tests
│   └── test_rag.py        # RAG tests
├── .dora/                 # Knowledge base storage (created automatically)
│   └── kb/                # ChromaDB data directory
├── pyproject.toml         # Project configuration
├── uv.lock                # Dependency lock file
├── .gitignore             # Git ignore rules
└── README.md
```

### Available CLI Commands

- `dora-test`: Test the local LLM with a simple prompt
- `dora`: Start interactive chat session (with automatic RAG detection)
- `dora add-doc <file_path>`: Add a PDF document to the knowledge base
- `dora list-docs`: List all documents in the knowledge base
- `dora clear-kb`: Clear all documents from the knowledge base
