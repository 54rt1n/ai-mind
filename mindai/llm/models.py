# mindai/llm/models.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any
import logging
from ..config import ChatConfig
from .llm import LLMProvider, OpenAIProvider, AIStudioProvider, GroqProvider
from .completion import CompletionProvider, OpenAICompletionProvider

logger = logging.getLogger(__name__)

def emptyornone(value: Any) -> bool:
    """Check if the value is empty or None."""
    return value is None or value == ""

class ModelProvider(str, Enum):
    """Supported model providers"""
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    COMPATIBLE = "compatible"
    FEATHERLESS = "featherless"
    GOOGLE = "google"
    GROQ = "groq"
    LOCAL = "local"
    OPENAI = "openai"
    OPENROUTER = "openrouter"

    def has_configuration(self, config: ChatConfig) -> bool:
        """Check if the model has a configuration that matches the provided `ChatConfig`."""
        if self == ModelProvider.ANTHROPIC:
            return not emptyornone(config.anthropic_api_key)
        elif self == ModelProvider.COHERE:
            return not emptyornone(config.cohere_api_key)
        elif self == ModelProvider.COMPATIBLE:
            return not emptyornone(config.compat_model_url) and not emptyornone(config.compat_api_key) and not emptyornone(config.compat_model_name)
        elif self == ModelProvider.LOCAL:
            return not emptyornone(config.local_model_url) and not emptyornone(config.local_api_key)
        elif self == ModelProvider.FEATHERLESS:
            return not emptyornone(config.featherless_api_key)
        elif self == ModelProvider.OPENAI:
            return not emptyornone(config.openai_api_key)
        elif self == ModelProvider.GOOGLE:
            return not emptyornone(config.ai_studio_api_key)
        elif self == ModelProvider.GROQ:
            return not emptyornone(config.groq_api_key)
        elif self == ModelProvider.OPENROUTER:
            return not emptyornone(config.openrouter_api_key)
        return False
    
class LLMProviderError(Exception):
    """Error for LLM provider errors"""
    def __init__(self, message: str, provider: ModelProvider):
        self.message = message
        self.provider = provider
        super().__init__(message)

class ModelCategory(str, Enum):
    """Model capability categories"""
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    COMPLETION = "completion"
    FUNCTIONS = "functions"
    THOUGHT = "thought"
    VISION = "vision"
    WORKSPACE = "workspace"

@dataclass
class SamplerConfig:
    """Configuration for model sampling parameters"""
    temperature: float = 0.7
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.05
    top_p: float = 0.9
    top_k: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'SamplerConfig':
        """Create a SamplerConfig from a dictionary, using only valid fields"""
        valid_fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class LanguageModelV2:
    """
    Enhanced language model configuration with architecture and sampling parameters.
    """
    name: str
    provider: ModelProvider
    architecture: str
    size: str
    category: set[ModelCategory]
    sampler: Optional[SamplerConfig] = None

    def can_provide(self, config: ChatConfig) -> bool:
        try:
            self.llm_factory(config)
            return True
        except LLMProviderError:
            return False
        except ValueError:
            return False

    def llm_factory(self, config: ChatConfig) -> LLMProvider:
        """
        Factory method to create an instance of `LLMProvider` based on the provided `ChatConfig`.
        
        Args:
            config (ChatConfig): The configuration settings for the chat generation.
        
        Returns:
            LLMProvider: An instance of the appropriate `LLMProvider` subclass.
        
        Raises:
            ValueError: If the provider is not recognized or required API key is missing.
        """
        if self.provider == ModelProvider.FEATHERLESS:
            return OpenAIProvider.from_url("https://api.featherless.ai/v1", config.featherless_api_key, model_name=self.name)
        elif self.provider == ModelProvider.ANTHROPIC:
            raise LLMProviderError(f"Anthropic is not supported yet", self.provider)
        elif self.provider == ModelProvider.COHERE:
            raise LLMProviderError(f"Cohere is not supported yet", self.provider)
        elif self.provider == ModelProvider.COMPATIBLE:
            return OpenAIProvider.from_url(config.compat_model_url, config.compat_api_key, model_name=self.name)
        elif self.provider == ModelProvider.OPENAI:
            return OpenAIProvider(api_key=config.openai_api_key, model_name=self.name)
        elif self.provider == ModelProvider.GOOGLE:
            raise LLMProviderError(f"Google is not supported yet", self.provider)
            return AIStudioProvider(api_key=config.ai_studio_api_key)
        elif self.provider == ModelProvider.GROQ:
            return GroqProvider(api_key=config.groq_api_key)
        elif self.provider == ModelProvider.OPENROUTER:
            return OpenAIProvider.from_url("https://openrouter.ai/api/v1", config.openrouter_api_key, model_name=self.name)
        elif self.provider == ModelProvider.LOCAL:
            return OpenAIProvider.from_url(config.compat_model_url, config.compat_api_key, model_name=self.name)
        else:
            raise LLMProviderError(f"Unknown LLM provider: {self.provider}", self.provider)

    def completion_factory(self, config: ChatConfig) -> CompletionProvider:
        """Factory method to create an instance of `CompletionProvider` based on the provided `ChatConfig`.
        
        Args:
            config (ChatConfig): The configuration settings for the chat generation.
        
        Returns:
            CompletionProvider: An instance of the appropriate `CompletionProvider` subclass.
        
        Raises:
            ValueError: If the provider is not recognized or required API key is missing.
        """
        if self.provider == ModelProvider.COMPATIBLE:
            return OpenAICompletionProvider.from_url(config.compat_model_url, config.compat_api_key, model_name=self.name)
        elif self.provider == ModelProvider.FEATHERLESS:
            return OpenAICompletionProvider.from_url("https://api.featherless.ai/v1", config.featherless_api_key, model_name=self.name)
        elif self.provider == ModelProvider.LOCAL:
            return OpenAICompletionProvider.from_url(config.local_model_url, config.local_api_key, model_name=self.name)
        elif self.provider == ModelProvider.OPENAI:
            return OpenAICompletionProvider(api_key=config.openai_api_key, model_name=self.name)
        else:
            raise LLMProviderError(f"Unknown LLM provider: {self.provider}", self.provider)

    @classmethod
    def load_yaml_config(cls, filename: str) -> 'list[LanguageModelV2]':
        """
        Load and process the YAML config file, merging sampler settings into model configs.

        Args:
            filename (str): Path to the YAML config file.

        Returns:
            List[LanguageModelV2]: List of configured language models.
        """
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)

        # Extract sampler settings
        default_sampler = SamplerConfig.from_dict(config['sampler_settings'].get('default', {}))
        architecture_samplers = {
            arch: SamplerConfig.from_dict({**default_sampler.__dict__, **settings})
            for arch, settings in config['sampler_settings'].items()
            if arch != 'default'
        }

        # Process models
        models : list[LanguageModelV2] = []
        for model_config in config['models']:
            # Get architecture sampler settings
            arch = model_config['architecture']
            sampler = architecture_samplers.get(arch, default_sampler)
            
            # Override with model-specific sampler settings if present
            if 'sampler' in model_config:
                sampler = SamplerConfig.from_dict({
                    **sampler.__dict__,
                    **model_config['sampler']
                })

            # Convert categories to enum set
            categories = {ModelCategory(cat) for cat in model_config['category']}

            provider = ModelProvider(model_config['provider'])

            model = cls(
                name=model_config['name'],
                provider=provider,
                architecture=arch,
                size=model_config['size'],
                category=categories,
                sampler=sampler
            )
            models.append(model)
        
        return models
    
    @classmethod
    def index_models(cls, config: ChatConfig) -> dict[str, 'LanguageModelV2']:
        """Index the models from the config file."""
        models = cls.load_yaml_config(config.model_config_path)
        return {model.name: model for model in models if model.provider.has_configuration(config)}

    @staticmethod
    def filter_category(models: 'list[LanguageModelV2]', category_filter: set[ModelCategory]) -> 'list[LanguageModelV2]':
        """Filter the models to only include those that match the category filter."""
        return [model for model in models if model.category.intersection(category_filter)]

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'LanguageModelV2':
        """Create a LanguageModelV2 instance from a ChatConfig."""
        return cls.index_models(config)[config.model]
