# aim-mud-types
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""World state snapshot for MUD agents."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from .state import RoomState, EntityState


class InventoryItem(BaseModel):
    """An item in the agent's inventory."""

    item_id: str = ""
    name: str
    description: str = ""
    quantity: int = 1
    tags: list[str] = Field(default_factory=list)


class WhoEntry(BaseModel):
    """An entry in the who list."""

    name: str
    status: str = ""
    location: str = ""
    is_self: bool = False


class WorldState(BaseModel):
    """Aggregated world state snapshot for an agent turn."""

    room_state: Optional[RoomState] = None
    entities_present: list[EntityState] = Field(default_factory=list)
    inventory: list[InventoryItem] = Field(default_factory=list)
    who: list[WhoEntry] = Field(default_factory=list)
    time: Optional[str] = None
    home: Optional[str] = None

    def to_xml(self, include_self: bool = False) -> str:
        """Render the world state as an XML string."""
        lines: list[str] = ["<world_state>"]

        if self.room_state:
            room = self.room_state
            attrs = f' name="{room.name}"'
            if room.room_id:
                attrs += f' id="{room.room_id}"'
            if room.tags:
                tags = ", ".join(room.tags)
                attrs += f' tags="{tags}"'
            lines.append(f"  <location{attrs}>")
            if room.description:
                lines.append(f"    {room.description}")
            if room.exits:
                exits = ", ".join(room.exits.keys())
                lines.append(f"    Exits: {exits}")
            lines.append("  </location>")

        entities = [
            e for e in self.entities_present if include_self or not e.is_self
        ]
        if entities:
            characters = [
                e for e in entities if e.entity_type in ("player", "ai", "npc")
            ]
            objects = [
                e for e in entities if e.entity_type not in ("player", "ai", "npc")
            ]

            if characters:
                lines.append("  <present>")
                for entity in characters:
                    attrs = f' name="{entity.name}" type="{entity.entity_type}"'
                    if entity.entity_id:
                        attrs += f' id="{entity.entity_id}"'
                    if entity.is_self:
                        attrs += ' self="true"'
                    if entity.tags:
                        tag_str = ", ".join(entity.tags)
                        attrs += f' tags="{tag_str}"'
                    if entity.agent_id:
                        attrs += f' agent_id="{entity.agent_id}"'
                    if entity.description:
                        lines.append(f"    <entity{attrs}>{entity.description}</entity>")
                    else:
                        lines.append(f"    <entity{attrs}/>")
                lines.append("  </present>")

            if objects:
                lines.append("  <objects>")
                for entity in objects:
                    attrs = f' name="{entity.name}" type="{entity.entity_type}"'
                    if entity.entity_id:
                        attrs += f' id="{entity.entity_id}"'
                    if entity.tags:
                        tag_str = ", ".join(entity.tags)
                        attrs += f' tags="{tag_str}"'
                    if entity.agent_id:
                        attrs += f' agent_id="{entity.agent_id}"'
                    if entity.description:
                        lines.append(f"    <object{attrs}>{entity.description}</object>")
                    else:
                        lines.append(f"    <object{attrs}/>")
                lines.append("  </objects>")

        if self.inventory:
            lines.append("  <inventory>")
            for item in self.inventory:
                attrs = f' name="{item.name}"'
                if item.item_id:
                    attrs += f' id="{item.item_id}"'
                if item.quantity != 1:
                    attrs += f' qty="{item.quantity}"'
                if item.tags:
                    tag_str = ", ".join(item.tags)
                    attrs += f' tags="{tag_str}"'
                if item.description:
                    lines.append(f"    <item{attrs}>{item.description}</item>")
                else:
                    lines.append(f"    <item{attrs}/>")
            lines.append("  </inventory>")

        if self.who:
            lines.append("  <who>")
            for entry in self.who:
                attrs = f' name="{entry.name}"'
                if entry.status:
                    attrs += f' status="{entry.status}"'
                if entry.location:
                    attrs += f' location="{entry.location}"'
                if entry.is_self:
                    attrs += ' self="true"'
                lines.append(f"    <player{attrs}/>")
            lines.append("  </who>")

        if self.time:
            lines.append(f"  <time>{self.time}</time>")

        if self.home:
            lines.append(f"  <home>{self.home}</home>")

        lines.append("</world_state>")
        return "\n".join(lines)
