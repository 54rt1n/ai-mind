# aim-mud-types

Shared types for AI-Mind and Evennia MUD integration.

## Overview

This package provides the common data types and utilities used by both the AI-Mind agent workers and the Evennia MUD server for event-driven communication via Redis streams.

## Architecture

```
Evennia -> mud:events -> Mediator -> agent:{id}:events -> AIM Worker
AIM Worker -> mud:actions -> Mediator -> Evennia
```

## Types

### Enums
- `EventType` - Types of world events (speech, emote, movement, object, ambient, system)
- `ActorType` - Types of actors (player, ai, npc, system)

### State Models
- `RoomState` - Current state of a room (id, name, description, exits)
- `EntityState` - State of an entity in a room (id, name, type, description)

### Event/Action Models
- `MUDEvent` - A world event from the MUD (published by Evennia, consumed by AIM)
- `MUDAction` - An action to execute (emitted by AIM, consumed by Evennia)

### Utilities
- `RedisKeys` - Consistent Redis key name generation

## Usage

```python
from aim_mud_types import MUDEvent, MUDAction, EventType, ActorType, RedisKeys

# Create an event
event = MUDEvent(
    event_type=EventType.SPEECH,
    actor="Prax",
    actor_type=ActorType.PLAYER,
    room_id="#123",
    content="Hello, Andi!",
)

# Serialize for Redis
redis_data = event.to_redis_dict()

# Create an action
action = MUDAction(tool="say", args={"message": "Hello, Papa!"})

# Convert to command
command = action.to_command()  # "say Hello, Papa!"

# Get stream keys
agent_stream = RedisKeys.agent_events("andi")  # "agent:andi:events"
```

## License

CC-BY-NC-SA-4.0
