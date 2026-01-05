# AI-Mind

An AI agent framework for creating persistent, personality-driven AI personas with long-term memory capabilities. The system grounds AI agents with customizable personalities and physical embodiment to enable reasoning, planning, and contextual actions.

## Core Concepts

AI-Mind combines several powerful capabilities towards persistent AI agents:

- **Implicit Thought Turns** - Use any model to generate thought turns, which are injected into the conversation stream
- **Active Memory** - Save and recall conversations through Retrieval Augmented Generation (RAG) using sparse search vectors and dense vector reranking
- **Conversation Pipelines** - Post-conversation analysis and augmentation through paradigm-driven processing
- **Persona Management** - Different agent profiles with customizable personalities and aspects
- **Document Context** - Integrate and use documents within conversation context
- **Multi-LLM Support** - Seamlessly switch between different LLM models and providers

## Architecture

AI-Mind is organized as a modular workspace with two core packages and application entry points:

### Workspace Packages

- **`aim-core`** - Core AI-Mind library
  - Persona management and aspects
  - Conversation memory with vector + sparse indexing
  - LLM provider abstraction (OpenAI, Anthropic, Groq, Cohere, etc.)
  - Refiner pipelines for post-conversation processing
  - Tool system with built-in implementations
  - Dreamer subsystem for async task processing

- **`aim-mud`** - ANDIMUD Integration
  - `andimud_mediator` - Redis Streams mediator that routes events to agents by location
  - `andimud_worker` - MUD agent worker that consumes events and generates actions
  - `aim_mud_types` - Shared type definitions and enums

### Application Entry Points

**Core Applications** (located in `src/`):

- **`aim_server`** - FastAPI REST API server
- **`aim_cli`** - Command-line chat interface with conversation management
- **`aim_bot`** - Discord bot integration

**Dreamer Pipeline Management** (located in `src/`):

- **`dream_agent`** - CLI for manually managing Dreamer pipelines (start, restart, inspect, status, watch)
- **`dream_watcher`** - Daemon that monitors conversations and automatically triggers pipelines based on rules
- **`dreamer`** - Background worker for processing async pipeline tasks

**ANDIMUD/MUD Integration** (located in `packages/aim-mud/src/`):

- **`andimud_mediator`** - Redis Streams mediator that routes MUD events to agents by location
- **`andimud_worker`** - MUD agent worker that consumes events and generates actions

### Frontend

- **`webui/`** - SvelteKit application with chat interface, agent management, and document handling

## Installation

AI-Mind uses `uv` for Python package management and a workspace structure for organizing packages and applications.

### Prerequisites

- Python 3.10+ (see `.python-version` in workspace packages)
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- Node.js + Yarn (for frontend development)

### Setup

```bash
# Clone the repository
git clone https://github.com/54rt1n/ai-mind.git
cd ai-mind

# Create a virtual environment and install workspace dependencies
# uv automatically handles workspace packages (aim-core, aim-mud)
uv sync

# Install test dependencies (optional)
# uv sync --extra test

# Setup environment configuration
cp .env.example .env
# Edit .env with your LLM API keys and Redis configuration

# Install frontend dependencies
cd webui
yarn install
cd ..
```

### Development

The workspace is organized into two layers:

**Workspace Packages** (core functionality):
```bash
# Work with aim-core
cd packages/aim-core
uv run pytest tests/

# Work with aim-mud
cd packages/aim-mud
uv run pytest tests/
```

**Applications** (entry points in `src/`):
```bash
# All applications use the workspace packages via uv sync
# Run from repository root:

# Start the API server
uv run python -m aim_server

# Use the CLI for conversation management and chat
# The CLI is a click-based tool with multiple subcommands:
uv run python -m aim_cli --help  # Show all available commands

# Examples:
uv run python -m aim_cli chat my-conversation --persona-id assistant --user-id user123
uv run python -m aim_cli list-conversations
uv run python -m aim_cli export-conversation my-conversation
uv run python -m aim_cli rebuild-index --agent-id assistant
uv run python -m aim_cli repair-conversation --all-conversations

# Start the Discord bot
uv run python -m aim_bot

# Manage Dreamer pipelines with the dream agent CLI
uv run python -m dream_agent --help  # Show all available commands

# Examples:
uv run python -m dream_agent start analysis_dialogue conv-001 --model claude-3-5-sonnet
uv run python -m dream_agent list --status running
uv run python -m dream_agent watch <pipeline-id>

# Start the dream watcher (monitors conversations and triggers pipelines)
uv run python -m dream_watcher

# Start the dreamer worker (background task processor)
uv run python -m dreamer

# ANDIMUD integration (for MUD-based agents)
# Start the mediator (routes events to agents by location)
uv run python -m andimud_mediator --agents assistant

# Start a MUD agent worker (consumes events and generates actions)
uv run python -m andimud_worker --agent-id assistant --persona-id assistant

# Run all tests
uv run pytest tests/
uv run pytest packages/aim-core/tests/
uv run pytest packages/aim-mud/tests/
```

**Frontend** (SvelteKit):
```bash
cd webui
yarn dev --host
```

### Quick Start (Development)

**For API + Frontend Development:**

Use `tmux` or multiple terminal sessions:

```bash
# Terminal 1: API server
uv run python -m aim_server

# Terminal 2: Frontend development server
cd webui && yarn dev --host

# Terminal 3: Dream watcher (monitors conversations and triggers pipelines)
uv run python -m dream_watcher

# Terminal 4: Dreamer worker (background task processor)
uv run python -m dreamer

# Terminal 5: Optional - Tests with watch mode
uv run pytest --watch

# Optional (for ANDIMUD/MUD-based development):
# Terminal 6: ANDIMUD mediator
uv run python -m andimud_mediator --agents assistant

# Terminal 7: ANDIMUD worker
uv run python -m andimud_worker --agent-id assistant --persona-id assistant
```

**Managing Pipelines (in separate session as needed):**

```bash
# Use dream_agent CLI to manually manage pipelines
uv run python -m dream_agent start analysis_dialogue conv-001 --model claude-3-5-sonnet
uv run python -m dream_agent list --status running
uv run python -m dream_agent watch <pipeline-id>
```

**For CLI-based Development:**

```bash
# View available commands
uv run python -m aim_cli --help

# Start an interactive chat session
uv run python -m aim_cli chat conv-001 --persona-id assistant --user-id user-001

# Manage conversations
uv run python -m aim_cli list-conversations
uv run python -m aim_cli export-conversation conv-001
uv run python -m aim_cli import-conversation dump.jsonl

# Maintain memory indices
uv run python -m aim_cli rebuild-index --agent-id assistant
uv run python -m aim_cli repair-conversation --conversation-id conv-001
```

**For ANDIMUD/MUD Development:**

```bash
# Terminal 1: ANDIMUD mediator (routes events to agents by location)
uv run python -m andimud_mediator --agents assistant

# Terminal 2: ANDIMUD worker (MUD agent processing)
uv run python -m andimud_worker --agent-id assistant --persona-id assistant

# Terminal 3: Optional - Monitor conversations and trigger pipelines
uv run python -m dream_watcher

# Terminal 4: Optional - Background pipeline processor
uv run python -m dreamer
```

## Project Structure

```
ai-mind/
├── pyproject.toml              # Root workspace configuration
├── uv.lock                     # Workspace lock file (uv)
├──
├── packages/                   # Workspace packages (core libraries)
│   ├── aim-core/               # Core AI-Mind library
│   │   ├── pyproject.toml
│   │   ├── src/aim/            # Main package
│   │   │   ├── agents/         # Persona & aspect management
│   │   │   ├── chat/           # Chat system & manager
│   │   │   ├── conversation/   # Memory & indexing (vector + sparse)
│   │   │   ├── llm/            # LLM provider abstraction
│   │   │   ├── refiner/        # Post-conversation processing
│   │   │   ├── tool/           # Tool system & implementations
│   │   │   ├── dreamer/        # Async task processing
│   │   │   ├── nlp/            # NLP utilities (summarization)
│   │   │   ├── io/             # I/O utilities
│   │   │   ├── utils/          # Shared utilities
│   │   │   └── config.py       # Central configuration
│   │   └── tests/              # Unit tests
│   │
│   └── aim-mud/                # ANDIMUD integration
│       ├── pyproject.toml
│       ├── src/
│       │   ├── aim_mud_types/   # Shared types and enums
│       │   ├── andimud_mediator/# Redis Streams mediator
│       │   └── andimud_worker/  # Evennia worker & adapter
│       └── tests/              # Unit tests
│
├── src/                        # Application entry points
│   ├── aim_cli/                # CLI chat interface
│   ├── aim_bot/                # Discord bot
│   ├── aim_server/             # FastAPI REST API
│   ├── dreamer/                # Dream agent runner
│   ├── dream_agent/            # Dream processing
│   └── dream_watcher/          # Event monitoring
│
├── webui/                      # SvelteKit frontend
│   ├── src/
│   │   ├── lib/                # Stores, components, types
│   │   └── routes/             # Page routes
│   ├── static/                 # Static assets
│   └── package.json
│
├── tests/                      # Root-level tests
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── config/                     # Configuration files
│   ├── models.yaml             # LLM model definitions
│   ├── persona/                # Persona configs
│   ├── paradigm/               # Processing paradigms
│   ├── tools/                  # Tool definitions
│   └── scenario/               # Scenario configurations
│
└── docs/                       # Documentation
    ├── overview.md
    ├── api.md
    ├── backend.md
    └── scenarios-and-paradigms.md
```

## Development Guidelines

### Workspace Packages vs Applications

- **Packages** (`packages/aim-core`, `packages/aim-mud`) contain reusable libraries with full test coverage
- **Applications** (`src/*`) are entry points that use the workspace packages
- Changes to packages should include unit tests in `packages/*/tests/`
- Application code should have integration tests in `tests/`

### Using `uv`

```bash
# Add a dependency to a specific package
cd packages/aim-core
uv add dependency-name

# Add a dev dependency
uv add --dev pytest-something

# Sync all dependencies across workspace
cd /repo/root
uv sync

# Run commands in the virtual environment
uv run python script.py
uv run pytest
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific package tests
uv run pytest packages/aim-core/tests/
uv run pytest packages/aim-mud/tests/
uv run pytest tests/

# Run with coverage
uv run pytest --cov=packages/aim-core packages/aim-core/tests/
```

## License

AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 