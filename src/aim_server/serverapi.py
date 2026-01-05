# aim/server/serverapi.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from aim.config import ChatConfig
from aim.agents.roster import Roster
from aim.chat import ChatManager

from .modules.admin.route import AdminModule
from .modules.chat.route import ChatModule
from .modules.completion.route import CompletionModule
from .modules.conversation.route import ConversationModule
from .modules.document.route import DocumentModule
from .modules.dreamer.route import DreamerModule
from .modules.memory.route import MemoryModule
from .modules.report.route import ReportModule
from .modules.roster.route import RosterModule
from .modules.tools.route import ToolsModule

logger = logging.getLogger(__name__)

class ServerApi:
    def __init__(self):
        # Initialize FastAPI app
        self.app = FastAPI(title="AI-Mind OpenAI-compatible API")

        # Setup security
        self.security = HTTPBearer(auto_error=False)

        # Load config
        self.config = ChatConfig.from_env()

        # Create shared roster instance
        self.shared_roster = Roster.from_config(self.config)

        # Lazy cache for persona-specific ChatManagers
        self.chat_managers: Dict[str, ChatManager] = {}

        # Initialize all modules with get_chat_manager callback
        admin_module = AdminModule(self.config, self.security)
        chat_module = ChatModule(self.config, self.security, self.get_chat_manager, self.shared_roster)
        completion_module = CompletionModule(self.config, self.security)
        conversation_module = ConversationModule(self.config, self.security, self.get_chat_manager)
        document_module = DocumentModule(self.config, self.security)
        dreamer_module = DreamerModule(self.config, self.security)
        memory_module = MemoryModule(self.config, self.security, self.get_chat_manager)
        report_module = ReportModule(self.config, self.security, self.get_chat_manager)
        roster_module = RosterModule(self.config, self.security, self.shared_roster)
        tools_module = ToolsModule(self.config, self.security)

        # Include all routers
        self.app.include_router(admin_module.router)
        self.app.include_router(chat_module.router)
        self.app.include_router(completion_module.router)
        self.app.include_router(conversation_module.router)
        self.app.include_router(document_module.router)
        self.app.include_router(dreamer_module.router)
        self.app.include_router(memory_module.router)
        self.app.include_router(report_module.router)
        self.app.include_router(roster_module.router)
        self.app.include_router(tools_module.router)

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount static files last
        self.app.mount("/", StaticFiles(directory="public", html=True), name="static")

    def get_chat_manager(self, persona_id: str) -> ChatManager:
        """Get or create ChatManager for a specific persona."""
        if persona_id not in self.shared_roster.personas:
            raise HTTPException(status_code=404, detail=f"Persona not found: {persona_id}")

        if persona_id not in self.chat_managers:
            config = ChatConfig.from_env()
            config.persona_id = persona_id
            self.chat_managers[persona_id] = ChatManager.from_config_with_roster(
                config,
                self.shared_roster
            )
            logger.info(f"Created ChatManager for persona: {persona_id}")

        return self.chat_managers[persona_id]

def create_app():
    """Create and configure a new FastAPI application instance."""
    server_api = ServerApi()
    return server_api.app
