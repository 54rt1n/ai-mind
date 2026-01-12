# aim/config.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from dataclasses import dataclass, field, asdict
import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

def get_env(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    return {
        "conversation_id": os.getenv("CONVERSATION_ID", None),
        "device": os.getenv("DEVICE", "cpu"),
        "embedding_device": os.getenv("EMBEDDING_DEVICE", None),
        "documents_dir": os.getenv("DOCUMENTS_DIR", "local/documents"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "guidance": os.getenv("GUIDANCE", None),
        "memory_path": os.getenv("MEMORY_PATH", "memory"),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "max_tokens": int(os.getenv("MAX_TOKENS", 256)),
        "memory_window": int(os.getenv("MEMORY_WINDOW", 12)),

        "compat_api_key": os.getenv("COMPAT_API_KEY", None),
        "compat_model_url": os.getenv("COMPAT_MODEL_URL", None),
        "compat_model_name": os.getenv("COMPAT_MODEL_NAME", None),
        "local_model_url": os.getenv("LOCAL_MODEL_URL", None),
        "local_api_key": os.getenv("LOCAL_API_KEY", None),
        "ai_studio_api_key": os.getenv("AI_STUDIO_API_KEY", None),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", None),
        "cohere_api_key": os.getenv("COHERE_API_KEY", None),
        "featherless_api_key": os.getenv("FEATHERLESS_API_KEY", None),
        "groq_api_key": os.getenv("GROQ_API_KEY", None),
        "openai_api_key": os.getenv("OPENAI_API_KEY", None),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", None),
        "meta_api_key": os.getenv("META_API_KEY", None),

        "persona_id": os.getenv("PERSONA_ID", "assistant"),
        "persona_location": os.getenv("PERSONA_LOCATION", None),
        "persona_mood": os.getenv("PERSONA_MOOD", "Inquisitive"),
        "persona_path": os.getenv("PERSONA_PATH", "configs/personas"),

        "user_timezone": os.getenv("USER_TIMEZONE", None),
        
        # Redis configuration
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379)),
        "redis_db": int(os.getenv("REDIS_DB", 0)),
        "redis_password": os.getenv("REDIS_PASSWORD", None),
        "redis_namespace": os.getenv("REDIS_NAMESPACE", "aim:cache:"),
        "redis_cache_expiry": int(os.getenv("REDIS_CACHE_EXPIRY", 604800)),  # 1 week in seconds
        
        # Summarizer configuration
        "summarizer_model": os.getenv("SUMMARIZER_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        "summarizer_use_gpu": os.getenv("SUMMARIZER_USE_GPU", "false").lower() == "true",
        "summarizer_num_generations": int(os.getenv("SUMMARIZER_NUM_GENERATIONS", 2)),
        "summarizer_num_beams": int(os.getenv("SUMMARIZER_NUM_BEAMS", 3)),
        "summarizer_temperature": float(os.getenv("SUMMARIZER_TEMPERATURE", 0.7)),
        
        # History management
        "history_management_strategy": os.getenv("HISTORY_MANAGEMENT_STRATEGY", "sparsify"),
        
        "recall_size": int(os.getenv("RECALL_SIZE", 2)),
        "server_api_key": os.getenv("SERVER_API_KEY", None),
        "temperature": float(os.getenv("TEMPERATURE", 0.7)),
        "top_n": int(os.getenv("TOP_N", 3)),
        "user_id": os.getenv("USER_ID", "user"),
        "workdir_folder": os.getenv("OUTPUT_FOLDER", "export"),
        "tools_path": os.getenv("TOOLS_PATH", "config/tools"),
        "discord_app_id": os.getenv("DISCORD_APP_ID", None),
        "discord_bot_token": os.getenv("DISCORD_BOT_TOKEN", None),
        "discord_public_key": os.getenv("DISCORD_PUBLIC_KEY", None),
        "thought_model": os.getenv("THOUGHT_MODEL", None),
        "codex_model": os.getenv("CODEX_MODEL", None),
        "decision_model": os.getenv("DECISION_MODEL", None),
        "agent_model": os.getenv("AGENT_MODEL", None),
        "default_model": os.getenv("DEFAULT_MODEL", None),
    }


@dataclass
class ChatConfig:
    server_api_key: Optional[str] = None
    queue_name: str = "pipeline_tasks"
    device: str = "cpu"
    embedding_device: Optional[str] = None  # None = auto-detect CUDA, or specify "cpu", "cuda:0", etc.
    memory_path: str = "memory"
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    persona_path: str = "config/persona"
    tools_path: str = "config/tools"
    model_config_path: str = "config/models.yaml"
    workdir_folder: str = "export"
    documents_dir: Optional[str] = None
    user_id: str = "user"
    persona_id: str = "assistant"
    conversation_id: Optional[str] = None
    llm_provider: str = "openai"
    local_model_url: Optional[str] = None
    local_api_key: Optional[str] = None
    compat_model_url: Optional[str] = None
    compat_model_name: Optional[str] = None
    compat_api_key: Optional[str] = None
    ai_studio_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    featherless_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    meta_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    thought_model: Optional[str] = None
    codex_model: Optional[str] = None
    decision_model: Optional[str] = None
    agent_model: Optional[str] = None
    default_model: Optional[str] = None

    # Redis cache configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_namespace: str = "aim:cache:"
    redis_cache_expiry: int = 604800  # 1 week in seconds
    
    # Summarizer configuration
    summarizer_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    summarizer_use_gpu: bool = False
    summarizer_num_generations: int = 2
    summarizer_num_beams: int = 3
    summarizer_temperature: float = 0.7
    
    # History management
    history_management_strategy: str = "sparsify"  # Options: sparsify, random_removal, ai_summarize
    user_timezone: Optional[str] = None
    top_n: int = 3
    recall_size: int = 3
    memory_window: int = 8
    query_text: Optional[str] = None
    generations: int = 1
    presence: Optional[float] = None
    repetition: Optional[float] = None
    system_message: Optional[str] = None
    persona_location: Optional[str] = None
    persona_mood: str = "Inquisitive"
    debug: bool = False
    no_retry: bool = False
    guidance: Optional[str] = None
    stop_sequences: List[str] = field(
        default_factory=lambda: ['I cannot']
    )
    discord_app_id: Optional[str] = None
    discord_bot_token: Optional[str] = None
    discord_public_key: Optional[str] = None
    max_tokens: int = 512
    min_tokens: Optional[int] = None
    temperature: float = 0.7
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = 0.05
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: Optional[bool] = False
    response_format: Optional[str] = None

    show_llm_messages: bool = False


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_env(cls, dotenv_file: Optional[str] = None) -> "ChatConfig":
        env = get_env(dotenv_file)
        return cls(**env)
