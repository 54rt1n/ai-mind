# aim/agents/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from .persona import Persona, Aspect, Tool
from .roster import Roster
from .aspects import (
    ASPECT_DEFAULTS,
    get_aspect,
    get_aspect_or_default,
    create_default_aspect,
    build_librarian_scene,
    build_dreamer_scene,
    build_philosopher_scene,
    build_dual_aspect_scene,
    build_writer_scene,
    build_psychologist_scene,
)