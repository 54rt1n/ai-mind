# aim/server/modules/roster/route.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from aim.config import ChatConfig
from aim.agents.roster import Roster, PersonaNotFoundError, PersonaExistsError, RosterError

from .dto import CreatePersonaRequest, UpdatePersonaRequest, PersonaResponse, PersonaListResponse

logger = logging.getLogger(__name__)

class RosterModule:
    def __init__(self, config: ChatConfig, security: HTTPBearer, shared_roster: Roster):
        self.router = APIRouter(prefix="/api/roster", tags=["roster"])
        self.security = security
        self.config = config
        self.roster = shared_roster
        
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("")
        async def list_personas(
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """List all available personas"""
            try:
                personas = [
                    PersonaResponse(**self.roster.get_persona(pid).to_dict())
                    for pid in self.roster.persona_list
                ]
                return PersonaListResponse(personas=personas)
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/{persona_id}")
        async def get_persona(
            persona_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get a specific persona"""
            try:
                persona = self.roster.get_persona(persona_id)
                return PersonaResponse(**persona.to_dict())
            except PersonaNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("")
        async def create_persona(
            request: CreatePersonaRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Create a new persona"""
            try:
                persona = self.roster.create_persona(request.model_dump())
                return {
                    "status": "success", 
                    "message": f"Persona {persona.persona_id} created",
                    "data": PersonaResponse(**persona.to_dict())
                }
            except PersonaExistsError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except RosterError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.put("/{persona_id}")
        async def update_persona(
            persona_id: str,
            request: UpdatePersonaRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Update an existing persona"""
            try:
                persona = self.roster.update_persona(persona_id, request.model_dump())
                return {
                    "status": "success", 
                    "message": f"Persona {persona_id} updated",
                    "data": PersonaResponse(**persona.to_dict())
                }
            except PersonaNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except RosterError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/{persona_id}")
        async def delete_persona(
            persona_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Delete a persona"""
            try:
                self.roster.delete_persona(persona_id)
                return {
                    "status": "success", 
                    "message": f"Persona {persona_id} deleted"
                }
            except PersonaNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except RosterError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception(e)
                raise HTTPException(status_code=500, detail=str(e))