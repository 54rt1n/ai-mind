# aim/server/modules/chat/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import os
import re
import time
import uuid
import json
import asyncio
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse

from ....llm.models import LanguageModelV2, LLMProvider, ModelCategory
from ....chat import ChatManager, chat_strategy_for
from ....config import ChatConfig
from ....utils.turns import validate_turns
from ....utils.xml import XmlFormatter
from ....tool.formatting import ToolUser
from ....agents.roster import Roster

from .dto import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)

def word_count(text: str) -> int:
    """
    Counts the number of words in a string.

    Args:
        text (str): The string to count the words of.

    Returns:
        int: The number of words in the string.
    """
    # use a regular expression to convert newlines and any whitespace to spaces, and handle multiple spaces as one.
    text = re.sub(r"\s+", " ", text)
    return len(text.split())

class ModelClasses:
    def __init__(self,
                 analysis: Optional[list[str]] = None,
                 conversation: Optional[list[str]] = None,
                 thought: Optional[list[str]] = None,
                 vision: Optional[list[str]] = None,
                 functions: Optional[list[str]] = None,
                 completion: Optional[list[str]] = None,
                 workspace: Optional[list[str]] = None,
                 ):
        self.analysis = analysis
        self.conversation = conversation
        self.thought = thought
        self.vision = vision
        self.functions = functions
        self.completion = completion
        self.workspace = workspace
        
    @property
    def categories(self) -> dict[str, list[str]]:
        return {
            "analysis": [m for m in self.analysis],
            "conversation": [m for m in self.conversation],
            "thought": [m for m in self.thought],
            "vision": [m for m in self.vision],
            "functions": [m for m in self.functions],
            "completion": [m for m in self.completion],
            "workspace": [m for m in self.workspace],
        }

    @classmethod
    def from_models(cls, models: list[LanguageModelV2]) -> 'ModelClasses':
        return cls(
            analysis=[m.name for m in models if ModelCategory.ANALYSIS in m.category],
            conversation=[m.name for m in models if ModelCategory.CONVERSATION in m.category],
            thought=[m.name for m in models if ModelCategory.THOUGHT in m.category],
            vision=[m.name for m in models if ModelCategory.VISION in m.category],
            functions=[m.name for m in models if ModelCategory.FUNCTIONS in m.category],
            completion=[m.name for m in models if ModelCategory.COMPLETION in m.category],
            workspace=[m.name for m in models if ModelCategory.WORKSPACE in m.category],
        )


class ChatModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, shared_roster: Roster):
        self.router = APIRouter(prefix="/v1/chat", tags=["chat"])
        self.security = security
        self.config = config
        self.shared_roster = shared_roster
        self.chat = ChatManager.from_config_with_roster(config, shared_roster)
        self.chat_strategy = chat_strategy_for("xmlmemory", self.chat)
        self.models = LanguageModelV2.index_models(self.config)
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Handle the chat completion request."""
            try:
                return await self.handle_chat_completions(request, credentials)
            except HTTPException:
                # Let HTTPException bubble up to FastAPI - don't catch and re-raise as 500
                raise
            except ValueError as e:
                logger.error(e)
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(e)
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/models")
        async def chat_models():
            self.models = LanguageModelV2.index_models(self.config)
            model_classes = ModelClasses.from_models(list(self.models.values()))
            return {
                "categories": model_classes.categories,
                "models": list(self.models.values()),
            }

    async def handle_chat_completions(self, request: ChatCompletionRequest, credentials: Optional[HTTPAuthorizationCredentials]):
        """Handle the chat completion request."""
        if self.config.server_api_key is not None and (credentials is None or credentials.credentials != self.config.server_api_key):
            raise HTTPException(status_code=401, detail=f"Invalid API key")

        selected_model : LanguageModelV2 | None = self.models.get(request.model, None)
        if selected_model is None or type(selected_model) is not LanguageModelV2:
            raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")

        if len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        if request.metadata is None:
            # TODO either use default values or just pass the request directly to the llm
            raise HTTPException(status_code=400, detail="No metadata provided")

        metadata = request.metadata

        if metadata.persona_id is None or metadata.persona_id not in self.chat.roster.personas:
            logger.error(f"Persona {metadata.persona_id} not found. Available personas: {list(self.chat.roster.personas.keys())}")
            logger.error(f"Roster object ID: {id(self.chat.roster)}")
            raise HTTPException(status_code=400, detail=f"Invalid persona: {metadata.persona_id}")

        if metadata.user_id is None or metadata.user_id == "":
            raise HTTPException(status_code=400, detail="No user ID provided")

        if metadata.active_document is not None:
            logger.info(f"Found active document: {metadata.active_document}")
            self.chat.current_document = metadata.active_document
        else:
            logger.info("No active document found")
            self.chat.current_document = None

        if metadata.workspace_content is not None:
            logger.info(f"Found workspace content: {len(metadata.workspace_content)}")
            self.chat.current_workspace = metadata.workspace_content
        else:
            logger.info("No workspace content found")
            self.chat.current_workspace = None

        if metadata.pinned_messages:
            logger.info(f"Pinned messages: {metadata.pinned_messages}")
            self.chat_strategy.clear_pinned()
            for doc_id in metadata.pinned_messages:
                self.chat_strategy.pin_message(doc_id)
        else:
            logger.info("Clearing pinned messages")
            self.chat_strategy.clear_pinned()

        if metadata.thought_content:
            logger.info(f"Found thought content: {len(metadata.thought_content)}")
            self.chat_strategy.thought_content = metadata.thought_content
        else:
            logger.info("No thought content found")
            self.chat_strategy.thought_content = None

        if metadata.scratch_pad:
            logger.info(f"Found scratch pad: {len(metadata.scratch_pad)}")
            self.chat_strategy.scratch_pad = metadata.scratch_pad
        else:
            logger.info("No scratch pad found")
            self.chat_strategy.scratch_pad = None

        self.config.user_id = metadata.user_id
        self.config.persona_id = metadata.persona_id
        self.config.temperature = request.temperature
        self.config.max_tokens = request.max_tokens or self.config.max_tokens
        self.config.repetition = request.repetition_penalty

        persona = self.chat.roster.personas[metadata.persona_id]

        system_formatter = XmlFormatter()

        if request.system_message is not None:
            system_formatter.add_element("SystemMessage", content=request.system_message, priority=3)

        system_formatter = persona.xml_decorator(system_formatter,
            mood=self.config.persona_mood,
            user_id=metadata.user_id,
            location=metadata.location or persona.default_location,
            disable_guidance=metadata.disable_guidance or False,
            disable_pif=metadata.disable_pif or False,
            conversation_length=len(request.messages),
        )

        if request.tools is not None and len(request.tools) > 0:
            tool_user = ToolUser(request.tools)
            system_formatter = tool_user.xml_decorator(system_formatter)
            self.config.response_format = "json"
        else:
            self.config.response_format = None

        self.config.system_message = system_formatter.render().replace("{{user}}", metadata.user_id)

        content_len = len(self.config.system_message)
        user_turn = request.messages[-1].model_dump()['content']
        messages = [msg.model_dump() for msg in request.messages[:-1]]

        prepared_messages = self.chat_strategy.chat_turns_for(persona=persona, user_input=user_turn, history=messages, content_len=content_len)

        logger.info(f"Processing Length: {sum([word_count(v) for e in prepared_messages for k, v in e.items() if k == 'content'])}")

        validate_turns(prepared_messages)

        provider = selected_model.llm_factory(self.config)

        if request.stream:
            return StreamingResponse(self._generate_stream_response(provider, selected_model.name, prepared_messages), media_type="text/event-stream")
        
        response = ""
        for chunk in provider.stream_turns(prepared_messages, self.config, model_name=selected_model.name):
            if chunk:
                response += chunk

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

    async def _generate_stream_response(self, provider: LLMProvider, model_name: str, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Generate streaming response for chat completion."""
        response_id = str(uuid.uuid4())
        full_response = ""

        for chunk in provider.stream_turns(messages, self.config, model_name=model_name):
            if chunk:
                full_response += chunk
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.01)  # Simulate some processing time

        # Send the final chunk
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
