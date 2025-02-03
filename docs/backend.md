# MindAI Backend

The backend is a Python FastAPI application that provides a chat completion API, as well as other REST APIs for managing conversations, memories, and documents.

## Installation

```bash
# Clone the repository
git clone https://github.com/54rt1n/mindai.git
cd mindai

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Set up your environment variables:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your configuration
# Required variables:
OPENAI_API_KEY=your_api_key
MEMORY_PATH=memory
PERSONA_PATH=configs/personas
```

2. Start the server:

```bash
python -m mindai.server
```

## Core Components

### Chat Manager

The ChatManager handles conversation state, history management, and interaction with the underlying models:

```python
from mindai.chat import ChatManager
from mindai.config import ChatConfig

config = ChatConfig.from_env()
chat = ChatManager.from_config(config)
```

### Conversation Model

The ConversationModel manages long-term memory and conversation storage:

```python
from mindai.conversation.model import ConversationModel

cvm = ConversationModel.from_config(config)
results = cvm.query(["your search query"], top_n=5)
```

## Memory

Memory is a tantivy indexed jsonl store of conversations.

### Tantivy

Tantivy is a full text search engine, based on the Lucene search engine but written in Rust. Lucene is one of the most sophisticated and mature search engine components, and drives many of the most popular search engines today such as Elasticsearch, Solr, and OpenSearch.

### JSONL

The JSONL format is a simple format for storing conversations in a text format. JSONL makes it easy to store and manage large amounts of data but provide a large amount of flexibility in how the data is managed. The format consists of a JSON object per line.

### Document Types

```python
DOC_ANALYSIS = "analysis" # A core document type that is used to store the analysis of the conversation. These are created by the analysis pipeline, and are highly weighted in the search index.
DOC_BRAINSTORM = "brainstorm" # An intermediate document type that is used to store the brainstorming of the conversation. These are created by many pipelines, and are weighted low in the search index.
DOC_CODEX = "codex" # A core document type that is used to store the semantic graph of the conversation. These are created by most pipelines.
DOC_CONVERSATION = "conversation" # A conversation entry made as either the user or the assistant.
DOC_DAYDREAM = "daydream" # A daydream entry made by the daydream pipeline. These are mostly used for the philosopher pipeline.
DOC_JOURNAL = "journal" # A journal entry made by the journal pipeline. Journal entries make up the most important aspects of an agent's memory.
DOC_MOTD = "motd" # A message to self, which is used to pass immediate short-term memory to the next conversation.
DOC_NER = "ner-task" # An intermediate document type that is used to store the NER task of the conversation. These are created by many pipelines, and are weighted low in the search index and are generally ignored.
DOC_PONDERING = "pondering" # A pondering entry made by the philosopher pipeline. Used for the journal pipeline.
DOC_REFLECTION = "reflection" # A reflection entry made by the journal pipeline.
DOC_REPORT = "report" # A report made on a document. Currently unused.
DOC_SELF_RAG = "self-rag" # A self-rag entry.
DOC_SOURCE_CODE = "source-code" # A source code entry, currently unused.
DOC_STEP = "step" # An intermediate document type that is used to store the step of the conversation. These are created by many pipelines, and are ignored in the search index.
DOC_SUMMARY = "summary" # A summary entry made by the summary pipeline.
```

## Pipeline Worker

The pipeline worker is a separate process that runs the pipelines. It is used to process the messages from a Redis queue.

### Installation

You will need to install redis, and update the .env file with the correct information.

### Running the Pipeline Worker

To start the pipeline worker, run the following command:

```bash
python -m mindai.app.worker
```

### Pipelines

MindAI includes several built-in pipelines for conversation processing:

- `analyst` - Analyze the conversation, and the summary if it exists, and create a new summary.
- `daydream` - Create a daydream from the conversation.
- `journaler` - Create a journal from the conversation.
- `philosopher` - Create a pondering from the conversation.
- `summarizer` - Create a summary from the conversation.

## Configuration

Key configuration options in `ChatConfig`:

- `memory_path`: Directory for conversation storage
- `embedding_model`: Model to use for text embeddings
- `persona_path`: Directory containing persona configurations
- `model_config_path`: Path to LLM model configurations
- `documents_dir`: Directory for document storage

## API Documentation

The REST API provides endpoints for:

- Chat completion
- Conversation management
- Memory search
- Pipeline execution
- Document management
- Agent management

See the API documentation in `docs/api.md` for detailed endpoint information.

## Development

### Running Tests

```bash
pytest tests/
```
