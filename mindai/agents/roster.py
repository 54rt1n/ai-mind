# mindai/agents/roster.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import json
import logging
import os
from typing import Dict, List, Optional

from ..config import ChatConfig
from .persona import Persona

logger = logging.getLogger(__name__)

class RosterError(Exception):
    """Base exception for roster operations"""
    pass

class PersonaNotFoundError(RosterError):
    """Raised when a persona is not found"""
    pass

class PersonaExistsError(RosterError):
    """Raised when attempting to create a persona that already exists"""
    pass

class Roster:
    def __init__(self, personas: dict[str, Persona], config: ChatConfig):
        self.config = config
        self.personas = personas
        
    @property
    def persona_list(self) -> List[str]:
        return list(self.personas.keys())
    
    def get_persona(self, persona_id: str) -> Persona:
        """Get a specific persona by ID"""
        if persona_id not in self.personas:
            raise PersonaNotFoundError(f"Persona {persona_id} not found")
        return self.personas[persona_id]

    def create_persona(self, persona_data: dict) -> Persona:
        """Create a new persona"""
        persona_id = persona_data["persona_id"]
        persona_path = os.path.join(self.config.persona_path, f"{persona_id}.json")
        
        if os.path.exists(persona_path):
            raise PersonaExistsError(f"Persona {persona_id} already exists")
        
        try:
            # Save the persona file
            with open(persona_path, 'w') as f:
                json.dump(persona_data, f, indent=2)
            
            # Create and store the persona object
            persona = Persona.from_dict(persona_data)
            self.personas[persona_id] = persona
            return persona
            
        except Exception as e:
            # Cleanup if file was created but something else failed
            if os.path.exists(persona_path):
                os.remove(persona_path)
            raise RosterError(f"Failed to create persona: {str(e)}") from e

    def update_persona(self, persona_id: str, update_data: dict) -> Persona:
        """Update an existing persona with partial data"""
        persona_path = os.path.join(self.config.persona_path, f"{persona_id}.json")
        
        if not os.path.exists(persona_path):
            raise PersonaNotFoundError(f"Persona {persona_id} not found")
        
        try:
            # Read existing persona with explicit UTF-8 encoding
            with open(persona_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Update only provided fields
            filtered_updates = {k: v for k, v in update_data.items() if v is not None}
            existing_data.update(filtered_updates)
            
            # Save updated data with explicit UTF-8 encoding
            with open(persona_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            # Update persona object
            persona = Persona.from_dict(existing_data)
            self.personas[persona_id] = persona
            return persona
            
        except Exception as e:
            raise RosterError(f"Failed to update persona: {str(e)}") from e

    def delete_persona(self, persona_id: str) -> None:
        """Delete a persona"""
        persona_path = os.path.join(self.config.persona_path, f"{persona_id}.json")
        
        if not os.path.exists(persona_path):
            raise PersonaNotFoundError(f"Persona {persona_id} not found")
        
        try:
            # Remove the file
            os.remove(persona_path)
            
            # Remove from roster
            if persona_id in self.personas:
                del self.personas[persona_id]
                
        except Exception as e:
            raise RosterError(f"Failed to delete persona: {str(e)}") from e

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'Roster':
        persona_path = config.persona_path
        if not os.path.exists(persona_path):
            raise ValueError(f"Persona path {persona_path} does not exist")

        personas = {}
        for persona_file in os.listdir(persona_path):
            if persona_file.endswith(".json"):
                persona = Persona.from_json_file(os.path.join(persona_path, persona_file))
                personas[persona.persona_id] = persona

        return cls(personas=personas, config=config)