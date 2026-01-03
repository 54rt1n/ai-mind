# aim-core

Core AI-Mind library with persona management, memory, and LLM orchestration.

## Features

- **Persona Management**: Define and manage AI personas with psychological aspects
- **Memory System**: Hybrid full-text (Tantivy) + vector (FAISS) search with MMR reranking
- **LLM Orchestration**: Multi-provider LLM abstraction (OpenAI, Anthropic, Groq, Cohere, etc.)
- **Dreamer Pipeline**: DAG-based scenario execution with Redis state management
- **Refiner Module**: Autonomous exploration engine for idle-time processing
- **Tool System**: YAML-driven capability definitions with dynamic loading
- **Chat Coordination**: Strategy-based turn management with context gathering

## Installation

```bash
pip install aim-core
```

## License

CC BY-NC-SA 4.0
