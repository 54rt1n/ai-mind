# aim/llm/model_set.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""ModelSet - Manages multiple LLM providers for different roles/tasks."""

from typing import Optional
from dataclasses import dataclass, field
import logging
import random

from aim.config import ChatConfig
from aim.agents.persona import Persona
from aim.llm.models import LanguageModelV2
from aim.llm.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ModelSet:
    """Manages LLM providers for different roles/tasks.

    Supports 13 model roles:
    - default: Master fallback (always populated)
    - chat: General chat + MUD Phase 2 response
    - thought: Chat completions thought generation
    - tool: General-purpose tool execution
    - decision: MUD Phase 1 decision (fast tool selection)
    - agent: MUD agent commands (structured actions)
    - code: Coding agent
    - codex: Librarian (technical/knowledge)
    - analysis: Coder (code analysis)
    - writing: Journaler/writer steps
    - research: Philosopher
    - planning: Phase 4 objective planning
    - fallback: Secondary fallback for chat when format fails

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
    decision_model: str
    agent_model: str
    code_model: str
    codex_model: str
    analysis_model: str
    writing_model: str
    research_model: str
    planning_model: str
    fallback_model: str

    # Cached providers (initialized lazily)
    _providers: dict[str, LLMProvider] = field(default_factory=dict, init=False)

    # Model pools for roles with multiple models (random selection)
    _model_pools: dict[str, list[str]] = field(default_factory=dict, init=False)

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

        # Track model pools for roles with multiple models
        model_pools: dict[str, list[str]] = {}

        # First, determine the effective default model (respects persona override)
        effective_default = config.default_model
        effective_default_pool: list[str] | None = None
        if persona and persona.models and "default" in persona.models:
            default_spec = persona.models["default"]
            if isinstance(default_spec, list):
                effective_default = default_spec[0]
                effective_default_pool = default_spec
                model_pools["default"] = default_spec
            else:
                effective_default = default_spec

        # Helper to resolve model spec to (first_model, pool)
        def resolve_model_spec(spec: str | list[str]) -> tuple[str, list[str] | None]:
            if isinstance(spec, list):
                return spec[0], spec
            return spec, None

        # Helper to get model with priority, returns (first_model, pool)
        def get_model(key: str, env_fallback: Optional[str] = None) -> tuple[str, list[str] | None]:
            # 1. Persona override
            if persona and persona.models and key in persona.models:
                return resolve_model_spec(persona.models[key])
            # 2. Specific env variable
            if env_fallback:
                return env_fallback, None
            # 3. Effective default (may be overridden by persona, may have pool)
            return effective_default, effective_default_pool

        # Get models for each role
        default_result = effective_default, effective_default_pool
        chat_result = get_model("chat")
        thought_result = get_model("thought", config.thought_model)
        tool_result = get_model("tool")
        decision_result = get_model("decision", config.decision_model)
        agent_result = get_model("agent", config.agent_model)
        code_result = get_model("code")
        codex_result = get_model("codex", config.codex_model)
        analysis_result = get_model("analysis")
        writing_result = get_model("writing")
        research_result = get_model("research")
        planning_result = get_model("planning")
        fallback_result = get_model("fallback", config.fallback)

        # Collect pools for roles with multiple models
        role_results = {
            "chat": chat_result,
            "thought": thought_result,
            "tool": tool_result,
            "decision": decision_result,
            "agent": agent_result,
            "code": code_result,
            "codex": codex_result,
            "analysis": analysis_result,
            "writing": writing_result,
            "research": research_result,
            "planning": planning_result,
            "fallback": fallback_result,
        }

        for role, (_, pool) in role_results.items():
            if pool is not None:
                model_pools[role] = pool

        instance = cls(
            config=config,
            default_model=effective_default,
            chat_model=chat_result[0],
            thought_model=thought_result[0],
            tool_model=tool_result[0],
            decision_model=decision_result[0],
            agent_model=agent_result[0],
            code_model=code_result[0],
            codex_model=codex_result[0],
            analysis_model=analysis_result[0],
            writing_model=writing_result[0],
            research_model=research_result[0],
            planning_model=planning_result[0],
            fallback_model=fallback_result[0],
        )

        # Set model pools after initialization
        instance._model_pools = model_pools

        return instance

    def get_provider(self, role: str = "default") -> LLMProvider:
        """Get LLM provider for the specified role.

        Args:
            role: One of the 13 roles (default, chat, thought, tool, decision, agent,
                  code, codex, analysis, writing, research, planning, fallback)

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

        If the role has a model pool configured, randomly selects from the pool.
        Otherwise returns the static model for the role.

        Args:
            role: One of the 13 roles

        Returns:
            Model name string
        """
        # Check if this role has a model pool (random selection)
        if role in self._model_pools:
            return random.choice(self._model_pools[role])

        role_map = {
            "default": self.default_model,
            "chat": self.chat_model,
            "thought": self.thought_model,
            "tool": self.tool_model,
            "decision": self.decision_model,
            "agent": self.agent_model,
            "code": self.code_model,
            "codex": self.codex_model,
            "analysis": self.analysis_model,
            "writing": self.writing_model,
            "research": self.research_model,
            "planning": self.planning_model,
            "fallback": self.fallback_model,
        }
        return role_map.get(role, self.default_model)

    def get_model_pool(self, role: str) -> list[str]:
        """Get the full model pool for a role.

        Args:
            role: One of the 13 roles

        Returns:
            List of model names in the pool, or single-item list with static model
        """
        if role in self._model_pools:
            return self._model_pools[role]
        # Return single-item list with the static model
        return [self.get_model_name(role)]

    def has_model_pool(self, role: str) -> bool:
        """Check if a role has multiple models configured.

        Args:
            role: One of the 13 roles

        Returns:
            True if the role has a model pool with multiple models
        """
        return role in self._model_pools and len(self._model_pools[role]) > 1
