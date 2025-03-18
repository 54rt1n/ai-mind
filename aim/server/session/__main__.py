import asyncio
import json
import sys
import os
from typing import Dict, List, Any, AsyncGenerator, Optional

from ...config import ChatConfig
from ...llm.models import LanguageModelV2
from .workflow import SessionWorkflow, WorkflowRequest
from .state import SessionState, CurrentState


async def process_chat_completion(
    messages: List[Dict[str, Any]], 
    model: str,
    session_id: Optional[str] = None,
    user_id: str = "user",
    persona_id: str = "default",
    conversation_id: Optional[str] = None,
    system_message: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = True,
    save_conversation: bool = False,
    location: Optional[str] = None,
    enable_thought_turn: bool = True,
    enable_tool_turn: bool = True
) -> AsyncGenerator[str, None]:
    """Process a chat completion request and yield tokens."""
    
    # Initialize config from environment variables
    config = ChatConfig.from_env()
    
    # Override with parameters passed to the function
    config.user_id = user_id
    config.persona_id = persona_id
    config.conversation_id = conversation_id
    
    # Create session state
    state = SessionState(
        session_id=session_id or os.urandom(16).hex(),
        user_id=user_id,
        persona_id=persona_id,
        conversation_id=conversation_id,
        location=location,
        save_conversation=save_conversation,
        models={"conversation": model, "thought": model, "tool": model}
    )
    
    # Create workflow
    workflow = SessionWorkflow(state, config)
    
    # Create a workflow request
    request = WorkflowRequest(
        messages=messages,
        save_conversation=save_conversation,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        use_thought_turn=enable_thought_turn,
        use_tool_turn=enable_tool_turn
    )
    
    # Process through workflow
    try:
        async for token in workflow.run(request):
            yield token
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"\n\nError: {str(e)}"
            

async def main():
    """Example of using the session workflow."""
    # Example chat completion request
    request = {
        "messages": [
            {"role": "user", "content": "Hey Andi, what is the weather today?"}
        ],
        "model": "Qwen/QwQ-32B",
        "stream": True,
        "user_id": "Prax",
        "persona_id": "Andi",
        "conversation_id": "test-conversation",
        "system_message": "You are a helpful assistant.",
        "enable_thought_turn": True,
        "enable_tool_turn": False
    }
    
    print("Processing request...", file=sys.stderr)
    
    # Process request and print tokens
    async for token in process_chat_completion(**request):
        print(token, end="", flush=True)
    
    print("\nRequest completed.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())