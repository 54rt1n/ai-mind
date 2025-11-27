# aim/llm/llm.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, List, Optional, Generator

import tiktoken

from ..config import ChatConfig

logger = logging.getLogger(__name__)

# Retryable exceptions for API calls
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Includes network errors
)

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient network/API error)."""
    # Check direct instance
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True

    # Check error message for known retryable patterns
    error_msg = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "remote",
        "protocol",
        "peer closed",
        "incomplete",
        "reset by peer",
        "broken pipe",
        "rate limit",
        "429",
        "503",
        "502",
        "504",
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig, model_name: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """
        Streams the response for a series of chat messages.
        
        Args:
            messages (List[Dict[str, str]]): The list of chat messages to generate a response for.
            config (ChatConfig): The configuration settings for the chat generation.
            model_name (Optional[str]): The name of the LLM model to use. If not provided, the provider's default model will be used.
            **kwargs: Additional keyword arguments to pass to the underlying LLM provider.
        
        Returns:
            Generator[str, None, None]: A generator that yields the generated response text, one token at a time.
        """
        pass


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        import groq
        self.groq = groq.Groq(api_key=api_key)
    
    @property
    def model(self):
        return 'mixtral-8x7b-32768'

    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig, model_name: str=None, **kwargs) -> Generator[str, None, None]:
        from groq.types.chat import ChatCompletionChunk

        model = model_name or self.model

        # Groq wants their messages in a specific format: messages = [{'role':'user' or 'model', 'content': 'hello'}]
        messages = [
            { 'role': message['role'], 'content': message['content'] } for message in messages
        ]

        if config.system_message:
            system_message = {"role": "system", "content": config.system_message}
            messages = [system_message, *messages]

        for chunk in self.groq.chat.completions.create(
            messages=messages,
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            n=config.generations,
            stream=True,
            **kwargs
        ):
            c : ChatCompletionChunk = chunk
            yield c.choices[0].delta.content


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        show_llm_messages: bool = False,
        enable_retry: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        import openai
        self.openai = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.show_llm_messages = show_llm_messages
        self.enable_retry = enable_retry
        self.max_retries = max_retries if enable_retry else 0
        self.base_delay = base_delay
        self.max_delay = max_delay

    @property
    def model(self):
        return self.model_name

    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig, model_name: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        from openai.types.chat import ChatCompletionChunk
        from openai._types import NOT_GIVEN

        system_message = {"role": "system", "content": config.system_message} if config.system_message else None

        stop_sequences = [] if config.stop_sequences is None else config.stop_sequences

        if system_message:
            messages = [system_message, *messages]

            if self.show_llm_messages:
                logger.info(f"Using system message: {system_message}")
                for message in messages[:]:
                    logger.info(f"{message['role']}: {message['content']}")

        model = model_name or self.model
        logger.info(f"Using model: {model}")

        rargs = { "response_format": { "type": "json_object" } } if config.response_format == "json" else {}

        # Count tokens in request
        encoder = tiktoken.get_encoding("cl100k_base")
        request_tokens = sum(len(encoder.encode(m.get('content', ''))) for m in messages)
        logger.info(f"Request: {len(str(messages))} characters, {request_tokens} tokens. Generating {config.max_tokens} tokens.")

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                logger.warning(f"Retry {attempt}/{self.max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)

            progress = 0
            lastpct = 0
            tenpct = int(config.max_tokens * 0.1)

            try:
                for t in self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    stop=stop_sequences,
                    n=config.generations,
                    stream=True,
                    presence_penalty=config.presence_penalty if config.presence_penalty is not None else NOT_GIVEN,
                    frequency_penalty=config.frequency_penalty if config.frequency_penalty is not None else NOT_GIVEN,
                    top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
                    **rargs
                ):
                    c: Optional[ChatCompletionChunk] = t
                    progress += 1
                    pct = progress / config.max_tokens
                    if pct > (lastpct + tenpct):
                        lastpct = pct
                        logger.info(f"{pct*100}% done")
                    yield c.choices[0].delta.content

                # Success - log and return
                logger.info(f"Generation complete. {progress}/{config.max_tokens} tokens processed.")
                return

            except Exception as e:
                last_error = e
                logger.error(f"API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if hasattr(e, 'response'):
                    logger.error(f"Response body: {e.response.text if hasattr(e.response, 'text') else e.response}")

                # Check if error is retryable
                if is_retryable_error(e) and attempt < self.max_retries:
                    logger.warning(f"Retryable error detected, will retry...")
                    continue
                else:
                    # Non-retryable or max retries exceeded
                    if attempt >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded, giving up")
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    @classmethod
    def from_url(cls, url: str, api_key: str, model_name: Optional[str] = None):
        return cls(base_url=url, api_key=api_key, model_name=model_name)


class AIStudioProvider(LLMProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.gem = genai.GenerativeModel(self.model)

    @property
    def model(self):
        return 'gemini-1.5-flash'

    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig) -> Generator[str, None, None]:
        from google.generativeai import GenerationConfig

        config = GenerationConfig(candidate_count=1, stop_sequences=config.stop_sequences,
                                  max_output_tokens=config.max_tokens, temperature=config.temperature)

        # Google wants their messages in a specific format: messages = [{'role':'user' or 'model', 'parts': ['hello']}]

        rewrote = [
            { 'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']] } for m in messages
        ]

        try:
            for chunk in self.gem.generate_content(rewrote, generation_config=config, stream=True):
                yield chunk.text.strip()
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            yield ""
            
        return 
