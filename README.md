# AI-MIND

As an AI researcher, I have been experimenting with the concept of mind and AI for several years now. I have a theory: if you anchor an AI agent with a personality, and ground them in a physical embodiment, they will be able to reason, plan, and act in ways that are beyond their original design.

AI-Mind is a powerful and flexible AI conversation that combines several required concepts towards persistent AI agents:
- Implicit thought turns (You can use any model to generate thought turns, which are then injected into the conversation stream)
- Active Memory - the ability to save and recall conversations through Retrieval Augmented Generation (RAG) using sparse search vectors, and dense vector reranking
- Pipelines - Post conversation analysis and augmentation
- Agents - The ability to have different agent profiles for different types of conversations
- Documents - The ability to save and use documents in the conversation context

## How?

AI-Mind has both a frontend and a backend piece.

The backend is a Python FastAPI application that provides a chat completion API that can easily switch between different LLM models and providers; as well as other REST APIs for managing conversations, memories, and documents.

The frontend is a SvelteKit application that provides a chat interface with the ability to build and inject thoughts in to the conversation stream; in addition to conversation, document, and agent management.

## Installation

```bash
# Clone the repository
git clone https://github.com/54rt1n/ai-mind.git
cd ai-mind

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

cp .env.example .env

# Edit .env with your configuration

cd webui
yarn install
cd ..

tmux

# Window 1: Start the server
python -m aim.server

# Window 2: Start the worker (optional)
python -m aim.app.worker

# Window 3: Start the web UI
cd webui
yarn dev --host

```

## License

AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 