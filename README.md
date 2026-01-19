# Dora

![python](https://img.shields.io/badge/python-3.10+-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mamo1031/Dora/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Mamo1031/Dora/graph/badge.svg?token=IkuhZ1Tu3K)](https://codecov.io/gh/Mamo1031/Dora)

**Dora** is a local LLM system that uses RAG (Retrieval-Augmented Generation) technology to enhance responses with knowledge from PDF documents. All processing runs locally on your machine, ensuring privacy and security.

## Features

- **Fully Local**: Runs completely offline after initial setup
- **RAG Support**: Enhances responses with knowledge from your PDF documents
- **Multilingual**: Supports documents in multiple languages including Japanese
- **Easy to Use**: Simple command-line interface
- **Privacy-First**: All data stays on your local machine

## Overview

Dora uses Ollama to run a large language model (LLM) locally and retrieves knowledge from PDF documents to provide more accurate and up-to-date answers. By using RAG technology, Dora addresses common LLM issues such as hallucinations (incorrect information) and outdated knowledge.

## Requirements

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (package manager)
- [Ollama](https://ollama.ai/) (local LLM runtime)

## Installation

### Step 1: Install uv

If you don't have uv installed, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install Dependencies

Navigate to the project root directory and run:

```bash
uv sync
```

### Step 3: Activate Virtual Environment

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Step 4: Install and Setup Ollama

Install Ollama:

**Linux/WSL:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download and install from the [Ollama website](https://ollama.ai/)

After installation, the Ollama service should start automatically. If not, start it manually:

```bash
ollama serve
```

### Step 5: Download Llama 3.2 Model

```bash
ollama pull llama3.2
```

You're all set!

## Quick Start

### Test the Installation

Verify that everything works:

```bash
dora-test
```

This will test if the LLM is working correctly.

### Basic Interactive Mode

Start a conversation with the LLM:

```bash
dora
```

In interactive mode:
- The LLM initializes automatically
- RAG mode is automatically enabled if you have documents in the knowledge base
- Type your questions and get answers
- Type `exit` or `quit` to end the session
- Press `Ctrl+C` to exit at any time

**Example:**
```
You: What is Python?
Assistant: Python is a programming language...

You: exit
Goodbye!
```

## Using RAG Features

RAG (Retrieval-Augmented Generation) allows Dora to retrieve information from your PDF documents to provide more accurate answers.

### Adding Documents

Add PDF documents to your knowledge base:

```bash
dora add-doc document.pdf
```

**Examples:**
```bash
# Add a PDF file from the current directory
dora add-doc my_document.pdf

# Add a PDF file with a full path
dora add-doc /path/to/document.pdf
```

Documents are automatically:
- Loaded and split into appropriate chunks (1000 characters with 200 character overlap)
- Converted to vectors using a multilingual embedding model
- Stored in the `.dora/kb/` directory

### Listing Documents

View all documents in your knowledge base:

```bash
dora list-docs
```

**Example Output:**
```
Knowledge base contains 45 chunks.

Documents:
  1. /path/to/document1.pdf
  2. /path/to/document2.pdf
```

### Clearing the Knowledge Base

Remove all documents from the knowledge base:

```bash
dora clear-kb
```

You'll be prompted for confirmation. Type `yes` to confirm deletion.

### Using RAG Mode

After adding documents, start interactive mode and RAG will be automatically enabled:

```bash
# 1. Add a document
dora add-doc manual.pdf

# 2. Start interactive mode (RAG mode automatically enabled)
dora
```

In RAG mode, answers are generated based on the content of your added documents.

**Example:**
```
You: According to the manual, how do I use this feature?
Assistant: [Answer based on the manual content]
```

## Available Commands

| Command | Description |
|---------|-------------|
| `dora-test` | Test LLM functionality (runs a test prompt) |
| `dora` | Start interactive mode (RAG auto-detection) |
| `dora add-doc <file_path>` | Add a PDF document to the knowledge base |
| `dora list-docs` | List all documents in the knowledge base |
| `dora clear-kb` | Remove all documents from the knowledge base |

## Troubleshooting

### Cannot Connect to Ollama

**Error:** `Failed to connect to Ollama`

**Solution:**
1. Check if Ollama is running:
   ```bash
   ollama serve
   ```
2. Verify the model is downloaded:
   ```bash
   ollama list
   ```
3. If the model is not listed, download it again:
   ```bash
   ollama pull llama3.2
   ```

### Cannot Add Document

**Error:** `File not found` or `Only PDF files are supported`

**Solution:**
- Verify the file path is correct
- Ensure the file is a PDF (has `.pdf` extension)
- Check that you have read permissions for the file

### RAG Mode Not Enabled

**Symptom:** RAG is not being used in interactive mode

**Solution:**
1. Check if documents are added:
   ```bash
   dora list-docs
   ```
2. If no documents are listed, add one:
   ```bash
   dora add-doc your_document.pdf
   ```

### Out of Memory Error

**Symptom:** Error occurs when processing large PDF files

**Solution:**
- Split large PDF files into smaller ones
- Increase system memory
- Close other applications to free up memory

## Frequently Asked Questions (FAQ)

**Q: What types of PDF files are supported?**\
A: Dora supports PDF files that contain text. PDFs with only images or scanned PDFs may require OCR processing beforehand.

**Q: Can I add multiple documents?**\
A: Yes, you can add multiple PDF files. All documents are integrated into the knowledge base and become searchable.

**Q: Where is the knowledge base data stored?**\
A: Data is stored in the `.dora/kb/` directory within the project directory. Deleting this directory will remove all data.

**Q: Can I use other LLM models?**\
A: Yes, you can use any model supported by Ollama. Change the `model_name` parameter in your code.

**Q: Do I need an internet connection?**\
A: Only for the initial model download. After that, Dora runs completely offline.

**Q: How does RAG improve answers?**\
A: RAG retrieves relevant information from your documents and includes it in the context when generating answers, making responses more accurate and grounded in your specific documents.

**Q: Can I use Dora without adding documents?**\
A: Yes, you can use Dora in regular LLM mode without RAG. Just start interactive mode without adding any documents.

## License

This project is open source. See the [LICENSE](LICENSE) file for details.

## Contributing

Bug reports and feature requests are welcome via GitHub Issues. Pull requests are also appreciated.
