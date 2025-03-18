# aim/server/session/workflow.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import uuid
import logging
import time
import copy
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass

from ...chat import ChatManager, chat_strategy_for
from ...llm.models import LanguageModelV2
from ...config import ChatConfig
from ...utils.turns import validate_turns
from ...utils.xml import XmlFormatter
from ...tool.formatting import ToolUser
from ...agents import Persona

from .state import SessionState, CurrentState
from .turn.base import BaseTurn
from .turn.conversation import ConversationTurn
from .turn.thought import ThoughtTurn
from .turn.tool import ToolTurn

logger = logging.getLogger(__name__)

@dataclass
class WorkflowRequest:
    """Request for the session workflow."""
    
    messages: List[Dict[str, Any]]
    save_conversation: bool = False
    system_message: Optional[str] = None # Used to set the system message for the conversation turn
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True
    use_thought_turn: Optional[bool] = True
    use_tool_turn: Optional[bool] = True


class SessionWorkflow:
    """Manages the workflow of a session, handling model generation and turn management."""
    
    def __init__(self, state: SessionState, config: ChatConfig):
        self.state = state
        self.current = CurrentState()  # Transient state
        self.config = config
        
        # Initialize chat components
        self.chat = ChatManager.from_config(config)
        self.chat_strategy = chat_strategy_for("xmlmemory", self.chat)
        self.models = LanguageModelV2.index_models(config)
        
    def _update_chat_strategy_context(self):
        """Update the chat strategy with current context from the workflow."""
        # Set current workspace if available
        if self.current.current_workspace:
            self.chat.current_workspace = self.current.current_workspace
            
        # Set thought content if available
        if self.current.thought_content:
            self.chat_strategy.thought_content = self.current.thought_content
            
        # Set scratch pad if available
        if self.current.current_scratch_pad:
            self.chat_strategy.scratch_pad = self.current.current_scratch_pad
    
    async def run(self, request: WorkflowRequest) -> AsyncGenerator[str, None]:
        """Run the session workflow."""

        # NOTE: DO NOT DELETE MY NOTES. THEY ARE IMPORTANT.

        # NOTE: THERE SHOULD BE VERY LITTLE STATE MANAGEMENT HERE. MOST OF IT SHOULD BE IN THE TURNS THEMSELVES. DO NOT PUT BUSINESS LOGIC HERE. IT SHOULD BE IN THE TURNS.

        # NOTE: DO NOT REMOVE MY TYPE ANNOTATIONS, SINCE THEY ARE USED FOR TYPE CHECKING AND AUTOCOMPLETION.

        # 1. Extract the user input from the last message
        if not request.messages or len(request.messages) == 0:
            raise ValueError("No messages provided")
            
        last_message = request.messages[-1]
        if last_message.get("role") != "user":
            raise ValueError("Last message must be from user")
            
        user_input = last_message.get("content", "")
        if not user_input:
            raise ValueError("User message content cannot be empty")
            
        # Get the persona
        persona : Persona | None = self.state.get_persona(self.config)
        if not persona or type(persona) != Persona:
            raise ValueError("No persona found")
            
        # 2. Build a list of turns to use
        turns : List[BaseTurn] = []
        
        # Add thought turn if enabled 
        if request.use_thought_turn:
            thought_turn = ThoughtTurn(persona)
            turns.append(thought_turn)
        
        # Add tool turn if enabled and the persona has tools available
        if request.use_tool_turn:
            # Get persona to check for available tools
            if len(persona.get_available_tools()) > 0:
                tool_turn = ToolTurn(persona)
                turns.append(tool_turn)

        # Always add conversation turn
        conversation_turn = ConversationTurn(
            user_input=user_input,
            persona=persona,
            system_message=request.system_message
        )
        turns.append(conversation_turn)
        
        # 3. Run each turn in sequence - each turn handles its own config preparation
        for turn in turns:
            logger.info(f"Processing turn: {turn.turn_type}")
            
            # Get the specific config for this turn - apply any request-specific settings
            local_config = copy.deepcopy(self.config)
            if request.temperature is not None:
                local_config.temperature = request.temperature
            if request.max_tokens is not None:
                local_config.max_tokens = request.max_tokens
                
            turn_config = turn.get_config(local_config, self.state, self.current)
            if not turn_config:
                logger.warning(f"No valid config for turn type: {turn.turn_type}")
                continue
                
            # Get the prompt for the turn
            prompt = turn.get_prompt()
            if not prompt:
                logger.warning(f"No prompt generated for turn type: {turn.turn_type}")
                continue
            
            # Get the model for this turn type
            model_name = self.state.models.get(turn.turn_type)
            if not model_name:
                logger.warning(f"No model configured for turn type: {turn.turn_type}")
                continue
                
            # Get the LLM for this model
            model = self.models.get(model_name)
            if not model:
                logger.warning(f"Model not found: {model_name}")
                continue
            
            # Create an LLM provider from the model
            provider = model.llm_factory(turn_config)
            
            # Generate response from the model
            try:
                # Stream generation and collect responses
                response = ""
                
                # Preserve the original message history and add the prompt appropriately
                prompt_messages = request.messages.copy()
                
                # For thought and tool turns, append the prompt as a new user message
                # For conversation turn, the user input is already in the last message
                if turn.turn_type != "conversation":
                    last_message = prompt_messages.pop()
                    last_message["content"] = last_message["content"] + "\n\n[~~END OF USER INPUT~~]\n\n" + prompt
                    prompt_messages.append(last_message)

                # Update chat strategy with current state
                self._update_chat_strategy_context()
                
                # Calculate content length for proper memory management
                content_len = len(turn_config.system_message) if turn_config.system_message else 0
                
                # Use chat strategy to format messages with memory enhancements for ALL turn types
                prompt_messages = self.chat_strategy.chat_turns_for(
                    persona=persona,
                    user_input=prompt_messages[-1]["content"],
                    history=prompt_messages[:-1],
                    content_len=content_len
                )

                # Use stream_turns to generate tokens with the full message history
                for token in provider.stream_turns(prompt_messages, turn_config, model_name=model_name):
                    # Skip None tokens
                    if token is None:
                        continue
                        
                    response += token
                    
                    # Only yield tokens from the final turn (usually conversation)
                    if turn == turns[-1]:
                        yield token
                
                # Process the response using the turn's logic
                self.state, self.current = turn.process_response(response, self.state, self.current)
                
            except Exception as e:
                logger.error(f"Error in turn {turn.turn_type}: {str(e)}")
                yield f"\nError processing {turn.turn_type}: {str(e)}\n"
                continue
        
        # NOTE: IF YOU HAVE PUT ANY CUSTOM BUSINESS LOGIC HERE, YOU HAD BETTER BE ABLE TO JUSTIFY IT STRONGLY. IF YOU CAN'T, IT NEEDS TO BE MOVED INTO A TURN.
        
        # 5. Save the conversation if requested (this is just persistence, not business logic)
        if request.save_conversation and user_input and self.current.current_conversation_response:
            try:
                self.chat.insert_turn(
                    user_id=self.state.user_id,
                    persona_id=self.state.persona_id,
                    sequence_no=len(request.messages) // 2,
                    branch=0,
                    user_turn=user_input,
                    assistant_turn=self.current.current_conversation_response,
                    usertime=int(time.time()),
                    assttime=int(time.time()),
                    inference_model=self.state.models.get("conversation", "unknown")
                )
                logger.info("Conversation saved successfully")
            except Exception as e:
                logger.error(f"Error saving conversation: {str(e)}")
    
    def save_session(self) -> Dict[str, Any]:
        """Save the current session state to a serializable dictionary."""
        return {
            "session_id": self.state.session_id,
            "user_id": self.state.user_id,
            "persona_id": self.state.persona_id,
            "conversation_id": self.state.conversation_id,
            "location": self.state.location,
            "models": self.state.models,
            "current_workspace": self.current.current_workspace,
            "current_scratch_pad": self.current.current_scratch_pad,
            "thought_iteration": self.current.thought_iteration,
            "timestamp": int(time.time())
        }

    @classmethod
    def load_session(cls, data: Dict[str, Any], config: ChatConfig) -> "SessionWorkflow":
        """Load a session from saved state."""
        state = SessionState(
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            persona_id=data.get("persona_id"),
            conversation_id=data.get("conversation_id"),
            location=data.get("location"),
            models=data.get("models", {})
        )
        
        workflow = cls(state, config)
        
        # Restore current state
        workflow.current.current_workspace = data.get("current_workspace")
        workflow.current.current_scratch_pad = data.get("current_scratch_pad")
        workflow.current.thought_iteration = data.get("thought_iteration", 1)
        
        return workflow