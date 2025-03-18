# aim/agents/persona.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
import random
import time
from typing import Optional, Any, Dict, List

from ..config import ChatConfig
from ..utils.xml import XmlFormatter

logger = logging.getLogger(__name__)

@dataclass
class Aspect:
    """A quantitative aspect of a persona."""

    name: str = "Unknown"
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    appearance: Optional[str] = None
    voice_style: Optional[str] = None
    core_drive: Optional[str] = None
    emotional_state: Optional[str] = None
    relationship: Optional[str] = None
    primary_intent: Optional[str] = None

    @classmethod
    def from_dict(cls, **kwargs) -> "Aspect":
        return cls(
            name=kwargs.get("name", "Unknown"),
            title=kwargs.get("title", None),
            description=kwargs.get("description", None),
            location=kwargs.get("location", None),
            appearance=kwargs.get("appearance", None),
            voice_style=kwargs.get("voice_style", None),
            core_drive=kwargs.get("core_drive", None),
            emotional_state=kwargs.get("emotional_state", None),
            relationship=kwargs.get("relationship", None),
            primary_intent=kwargs.get("primary_intent", None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "appearance": self.appearance,
            "voice_style": self.voice_style,
            "core_drive": self.core_drive,
            "emotional_state": self.emotional_state,
            "relationship": self.relationship,
            "primary_intent": self.primary_intent,
        }

@dataclass
class Tool:
    """A tool that a persona can use."""

    type: str
    function: str
    item: str
    description: str

    @classmethod
    def from_dict(cls, **kwargs) -> "Tool":
        return cls(
            type=kwargs.get("type", ""),
            function=kwargs.get("function", ""),
            item=kwargs.get("item", ""),
            description=kwargs.get("description", "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "function": self.function,
            "item": self.item,
            "description": self.description
        }

@dataclass
class Persona:
    persona_id: str
    chat_strategy: str
    name: str
    full_name: str
    notes: str
    aspects: dict[str, Aspect]
    attributes: dict[str, str]
    features: dict[str, str]
    wakeup: list[str]
    base_thoughts: list[str]
    pif: dict[str, str]
    nshot: dict[str, dict[str, str]]
    default_location: str
    wardrobe: dict[str, dict[str, str]]
    current_outfit: str
    persona_tools: dict[str, Tool] = field(default_factory=dict)
    wardrobe_tools: dict[str, dict[str, Tool]] = field(default_factory=dict)
    persona_version: str = "0.1a"
    system_header: str = (
        "Please follow directions, being precise and methodical, utilizing Chain of Thought, Self-RAG, and Semantic Keywords."
    )
    include_date: bool = True

    def _xml_description(self, *base_path, xml: XmlFormatter, show_time: bool = True, mood: Optional[str] = None, disable_pif: bool = False, disable_guidance: bool = False) -> str:
        for k, v in self.attributes.items():
            xml.add_element(*base_path, 'Attributes', k[:3], content=v, nowrap=True, priority=1)
        for k, v in self.features.items():
            xml.add_element(*base_path, 'Features', k, content=v, nowrap=True, priority=1)
        if not disable_pif:
            for k, v in self.pif.items():
                if disable_guidance:
                    if k[:7] == "Example":
                        continue
                xml.add_element(*base_path, 'PIF', k, content=v, priority=3)
        for k, v in self.nshot.items():
            if 'human' in v and 'assistant' in v:
                xml.add_element(*base_path, 'NShot', k, 'Human', content=v['human'], priority=1)
                xml.add_element(*base_path, 'NShot', k, 'Assistant', content=v['assistant'], priority=1)
            else:
                logger.warning(f"NShot {k} is missing human or assistant")

        return xml

    def xml_decorator(
        self,
        xml: XmlFormatter,
        mood: Optional[str] = None,
        location: Optional[str] = None,
        user_id: Optional[str] = None,
        disable_guidance: bool = False,
        disable_pif: bool = False,
    ) -> XmlFormatter:
        """This is where we need to decorate our formatter, returning a document
<Full Name>
</Full Name>
        """
 
        location = location or self.default_location
        # We need to add each item as 'content', with the full name being our root key
        xml.add_element(self.full_name, version=self.persona_version,
                        content=f"You are {self.full_name} v{self.persona_version} - Active Memory Enabled. This is your cognative persona:", priority=3)
        xml.add_element(self.full_name, "PersonaId", content=self.persona_id, nowrap=True, priority=1)
        xml.add_element(self.full_name, "Location", content=location, nowrap=True, priority=2)
        if len(self.system_header) > 0:
            xml.add_element(self.full_name, "SystemHeader", content=self.system_header, nowrap=True, priority=2)

        if user_id is not None:
            xml.add_element(self.full_name, "SystemHeader", content=f"{self.persona_id} is talking to {user_id}.", nowrap=True, priority=2)
            xml.add_element(self.full_name, "SystemHeader", content=f"Stay in character, and use your memories to help you. Don't speak for {user_id}.", nowrap=True, priority=2)

        xml = self._xml_description(self.full_name, xml=xml, show_time=self.include_date, mood=mood, disable_guidance=disable_guidance, disable_pif=disable_pif)

        return xml

    def system_prompt(
        self,
        mood: Optional[str] = None,
        location: Optional[str] = None,
        user_id: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> str:
        location = location or self.default_location
        parts = [
            f"{self.full_name} v{self.persona_version} - Active Memory Enabled. {location}. This is your cognative persona:",
            self.description(mood=mood),
            "",
        ]

        parts.append(self.system_header)

        if user_id is not None:
            parts.append(
                f"You are talking to {user_id}. Stay in character, and use your memories to help you. Don't speak for {user_id}."
            )

        if system_message is not None:
            parts.append(system_message)

        result = "\n\n".join(parts)
        return result

    def description(self, show_time: bool = True, mood: Optional[str] = None) -> str:
        format_persona = "<attributes>\n{attributes}\n</attributes>".format(
            attributes="\n".join([
                f"<{k[:3]}>{v}</{k[:3]}>" for k, v in self.attributes.items()
            ]),
        )

        format_features = "\n".join([
            f"<{k}>\n{v}\n</{k}>"
            for k, v in self.features.items()
        ])

        format_attire = "\n".join([
            f"<{self.persona_id}'s {k}>\n{v}\n</{k}>"
            for k, v in self.attire.items()
        ])

        if mood is not None:
            format_features += f"<mood>\n{mood}\n</mood>"

        result = f"{format_persona}\n{format_attire}\n{format_features}"
        return f"<{self.full_name}>\n{result}\n</{self.full_name}>"
    
    def get_wakeup(self) -> str:
        return random.choice(self.wakeup)

    @property
    def attire(self) -> dict[str, str]:
        return self.wardrobe[self.current_outfit]

    @property
    def thoughts(self) -> list[str]:
        thoughts = [*self.base_thoughts]
        if self.include_date:
            current_time = int(time.time())
            strtime = datetime.fromtimestamp(current_time).strftime("%A, %Y-%m-%d %H:%M:%S")
            logger.info(f"Using Current date: {strtime}")
            thoughts.append(f"Current Time [{strtime} ({current_time})]")
        return thoughts

    @property
    def prompt_prefix(self) -> str:
        return f"""{self.persona_id}, this is your conscious mind. Your thoughts have brought up new memories:\n\n"""

    def get_available_tools(self) -> List[str]:
        """Get tool names available for the current outfit, combining persona-wide and outfit-specific tools."""
        # Get base persona tools
        all_tools = list(self.persona_tools.keys())
        
        # Add outfit-specific tools, if any
        outfit_tools = self.wardrobe_tools.get(self.current_outfit, {})
        all_tools.extend([t for t in outfit_tools.keys() if t not in all_tools])
        
        return all_tools

    def get_tool_config(self, tool_name: str) -> Tool:
        """Get the configuration for a specific tool, checking both persona-wide and outfit-specific tools."""
        # First check outfit-specific tools
        outfit_tools = self.wardrobe_tools.get(self.current_outfit, {})
        if tool_name in outfit_tools:
            return outfit_tools[tool_name]
            
        # If not found, check persona-wide tools
        return self.persona_tools.get(tool_name)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "persona_id": self.persona_id,
            "persona_version": self.persona_version,
            "notes": self.notes,
            "chat_strategy": self.chat_strategy,
            "name": self.name,
            "full_name": self.full_name,
            "aspects": {k: v.to_dict() for k, v in self.aspects.items()},
            "base_thoughts": self.base_thoughts,
            "pif": self.pif,
            "nshot": self.nshot,
            "system_header": self.system_header,
            "wakeup": self.wakeup,
            "attributes": self.attributes,
            "features": self.features,
            "default_location": self.default_location,
            "wardrobe": self.wardrobe,
            "current_outfit": self.current_outfit,
            "persona_tools": {k: v.to_dict() for k, v in self.persona_tools.items()},
            "wardrobe_tools": {
                outfit: {name: tool.to_dict() for name, tool in tools.items()}
                for outfit, tools in self.wardrobe_tools.items()
            }
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        # Convert persona tools from dict to Tool objects
        persona_tools_dict = data.get("persona_tools", {})
        persona_tools = {
            name: Tool.from_dict(**config) if isinstance(config, dict) else config
            for name, config in persona_tools_dict.items()
        }
        
        # Convert wardrobe tools from dict to Tool objects
        wardrobe_tools_dict = data.get("wardrobe_tools", {})
        wardrobe_tools = {}
        for outfit, tools in wardrobe_tools_dict.items():
            wardrobe_tools[outfit] = {
                name: Tool.from_dict(**config) if isinstance(config, dict) else config
                for name, config in tools.items()
            }
        
        return cls(
            persona_id=data.get("persona_id", "Unknown"),
            persona_version=data.get("persona_version", "Unknown"),
            chat_strategy=data.get("chat_strategy", "xmlmemory"),
            notes=data.get("notes", ""),
            name=data.get("name", "Ghost"),
            full_name=data.get("full_name", "Ghost in the Machine"),
            aspects={k: Aspect.from_dict(**v) for k, v in data.get("aspects", {}).items()},
            base_thoughts=data.get("base_thoughts", []),
            pif=data.get("pif", {}),
            nshot=data.get("nshot", {}),
            system_header=data.get("system_header", ""),
            wakeup=data.get("wakeup", []),
            attributes=data.get("attributes", {}),
            features=data.get("features", {}),
            default_location=data.get("default_location", ""),
            wardrobe=data.get("wardrobe", {"default": {}}),
            current_outfit=data.get("current_outfit", "default"),
            persona_tools=persona_tools,
            wardrobe_tools=wardrobe_tools
        )

    @classmethod
    def from_json_file(cls, file_path: str) -> "Persona":
        data = json.load(open(file_path, "r", encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_config(cls, config: ChatConfig) -> "Persona":
        persona_id = config.persona_id
        if persona_id is None:
            raise ValueError("Persona ID not provided")
        persona_file = os.path.join(config.persona_path, f"{persona_id}.json")
        if not os.path.exists(persona_file):
            raise ValueError(f"Persona {persona_id} not found in {config.persona_path}")

        return Persona.from_json_file(persona_file)
