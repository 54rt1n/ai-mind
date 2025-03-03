# aim/llm/completion.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from abc import ABC, abstractmethod
import logging
from typing import Dict, Generator, Optional

from ..config import ChatConfig

logger = logging.getLogger(__name__)

class CompletionProvider(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def stream_completion(self, prompt: str, config: ChatConfig, **kwargs) -> Generator[str, None, None]:
        pass

class OpenAICompletionProvider(CompletionProvider):
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        import openai
        self.openai = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.running = True

    @property
    def model(self):
        return self.model_name
    
    def stream_completion(self, prompt: str, config: ChatConfig, **kwargs) -> Generator[str, None, None]:
        from openai.types.completion import Completion
        from openai._streaming import Stream

        completion_params = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
            "presence_penalty": kwargs.get("presence_penalty", config.presence),
            "frequency_penalty": kwargs.get("frequency_penalty", config.repetition),
            "top_p": kwargs.get("top_p", None),
            "stop": kwargs.get("stop", config.stop_sequences),
        }

        completion_params["stop"].append("[DONE]")
        logger.info(completion_params)

        # Filter out None values
        completion_params = {k: v for k, v in completion_params.items() if v is not None}
        self.running = True

        try:
            response : Completion = self.openai.completions.create(
                stream=False,
                **completion_params)
            logger.info(response)
            yield response.choices[0].text
        except Exception as e:
            logger.error(f"Error during completion streaming: {e}")
            raise
        finally:
            self.running = False

    @classmethod
    def from_url(cls, url: str, api_key: str, model_name: Optional[str] = None):
        return cls(base_url=url, api_key=api_key, model_name=model_name)
