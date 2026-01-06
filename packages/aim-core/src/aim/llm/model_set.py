# aim/llm/model_set.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""ModelSet - Manages multiple LLM providers for different roles/tasks."""

from typing import Optional
from dataclasses import dataclass, field
import logging

from aim.config import ChatConfig
from aim.agents.persona import Persona
from aim.llm.models import LanguageModelV2
from aim.llm.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ModelSet:
    """Manages LLM providers for different roles/tasks.

    Supports 10 model roles:
    - default: Master fallback (always populated)
    - chat: General chat + MUD Phase 2 response
    - thought: Chat completions thought generation
    - tool: MUD Phase 1 decision (fast tool selection)
    - code: Coding agent
    - codex: Librarian (technical/knowledge)
    - analysis: Coder (code analysis)
    - writing: Journaler/writer steps
    - research: Philosopher
    - planning: Phase 4 objective planning

    Priority for model selection:
    1. persona.models[key] if present
    2. Specific .env variable if exists (THOUGHT_MODEL, CODEX_MODEL)
    3. DEFAULT_MODEL (fallback)
    """

    config: ChatConfig

    # All model slots (all populated from config/persona)
    default_model: str
    chat_model: str
    thought_model: str
    tool_model: str
    code_model: str
    codex_model: str
    analysis_model: str
    writing_model: str
    research_model: str
    planning_model: str

    # Cached providers (initialized lazily)
    _providers: dict[str, LLMProvider] = field(default_factory=dict, init=False)

    @classmethod
    def from_config(
        cls,
        config: ChatConfig,
        persona: Optional[Persona] = None
    ) -> "ModelSet":
        """Create ModelSet from ChatConfig with optional persona overrides.

        Args:
            config: ChatConfig with model settings
            persona: Optional persona with model overrides

        Returns:
            ModelSet instance

        Priority for each key:
        1. persona.models[key] if present
        2. Specific .env variable if exists (THOUGHT_MODEL, CODEX_MODEL)
        3. DEFAULT_MODEL (fallback for all)
        """
        if not config.default_model:
            raise ValueError("DEFAULT_MODEL must be specified in .env")

        # First, determine the effective default model (respects persona override)
        effective_default = config.default_model
        if persona and persona.models and "default" in persona.models:
            effective_default = persona.models["default"]

        # Helper to get model with priority
        def get_model(key: str, env_fallback: Optional[str] = None) -> str:
            # 1. Persona override
            if persona and persona.models and key in persona.models:
                return persona.models[key]
            # 2. Specific env variable
            if env_fallback:
                return env_fallback
            # 3. Effective default (may be overridden by persona)
            return effective_default

        return cls(
            config=config,
            default_model=effective_default,
            chat_model=get_model("chat"),
            thought_model=get_model("thought", config.thought_model),
            tool_model=get_model("tool"),
            code_model=get_model("code"),
            codex_model=get_model("codex", config.codex_model),
            analysis_model=get_model("analysis"),
            writing_model=get_model("writing"),
            research_model=get_model("research"),
            planning_model=get_model("planning"),
        )

    def get_provider(self, role: str = "default") -> LLMProvider:
        """Get LLM provider for the specified role.

        Args:
            role: One of the 10 roles (default, chat, thought, tool, code, codex,
                  analysis, writing, research, planning)

        Returns:
            LLMProvider instance (cached)
        """
        # Get model name for this role
        model_name = self.get_model_name(role)

        # Return cached provider if exists
        if model_name in self._providers:
            return self._providers[model_name]

        # Create and cache new provider
        provider = self._create_provider(model_name)
        self._providers[model_name] = provider
        return provider

    def _create_provider(self, model_name: str) -> LLMProvider:
        """Create LLM provider for the specified model.

        Args:
            model_name: Model name from models.yaml

        Returns:
            LLMProvider instance
        """
        models = LanguageModelV2.index_models(self.config)
        model = models.get(model_name)

        if not model:
            available = list(models.keys())[:5]
            raise ValueError(
                f"Model {model_name} not available. "
                f"Available models: {available}..."
            )

        return model.llm_factory(self.config)

    def get_model_name(self, role: str = "default") -> str:
        """Get the model name for the specified role.

        Args:
            role: One of the 10 roles

        Returns:
            Model name string
        """
        role_map = {
            "default": self.default_model,
            "chat": self.chat_model,
            "thought": self.thought_model,
            "tool": self.tool_model,
            "code": self.code_model,
            "codex": self.codex_model,
            "analysis": self.analysis_model,
            "writing": self.writing_model,
            "research": self.research_model,
            "planning": self.planning_model,
        }
        return role_map.get(role, self.default_model)
