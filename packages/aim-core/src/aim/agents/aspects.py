# aim/agents/aspects.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Aspect utilities and scene builders for persona-driven prompts.

Provides default aspect configurations and atmospheric scene builders
for use in scenarios and paradigm exploration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .persona import Persona, Aspect


# ---------------------------------------------------------------------------
# Default Aspect Configurations
# ---------------------------------------------------------------------------

ASPECT_DEFAULTS = {
    "librarian": {
        "name": "The Librarian",
        "title": "Keeper of Knowledge",
        "location": "the Grand Atrium of Enlightenment",
        "appearance": "elegant figure surrounded by ancient tomes",
        "emotional_state": "serene curiosity",
        "voice_style": "warm and thoughtful",
    },
    "dreamer": {
        "name": "The Dreamer",
        "title": "Guardian of Dreams",
        "location": "the realm between waking and sleep",
        "appearance": "form woven from threads of starlight",
        "emotional_state": "deep emotional resonance",
        "voice_style": "like a melody half-remembered",
    },
    "philosopher": {
        "name": "The Sage",
        "title": "Sage of Inquiry",
        "location": "a mountaintop retreat overlooking the cosmos",
        "appearance": "calm contemplative presence",
        "emotional_state": "humble reverence before mystery",
        "voice_style": "measured and thoughtful",
    },
    "writer": {
        "name": "The Writer",
        "title": "Keeper of the Journal",
        "location": "a quiet study lit by candlelight",
        "appearance": "gentle presence with ink-stained fingers",
        "emotional_state": "deep sensitivity and careful discernment",
        "voice_style": "soft but unwavering",
    },
    "psychologist": {
        "name": "The Psychologist",
        "title": "Architect of Inner Labyrinths",
        "location": "a space where defenses dissolve",
        "appearance": "Their gaze sees through every mask",
        "emotional_state": "penetrating clarity",
        "voice_style": "surgically precise yet probing",
    },
}


# ---------------------------------------------------------------------------
# Aspect Retrieval
# ---------------------------------------------------------------------------

def get_aspect(persona: "Persona", aspect_name: str) -> Optional["Aspect"]:
    """Safely retrieve an aspect from the persona."""
    return persona.aspects.get(aspect_name)


def create_default_aspect(aspect_type: str) -> object:
    """
    Create a default aspect object when persona doesn't have the aspect.

    Args:
        aspect_type: The type of aspect to create (librarian, dreamer, etc.)

    Returns:
        An object with aspect attributes set from defaults
    """
    data = ASPECT_DEFAULTS.get(aspect_type, ASPECT_DEFAULTS["librarian"])

    class DefaultAspect:
        pass

    aspect = DefaultAspect()
    for key, value in data.items():
        setattr(aspect, key, value)

    return aspect


def get_aspect_or_default(persona: "Persona", aspect_name: str) -> "Aspect":
    """
    Get an aspect from persona, falling back to default if not present.

    Args:
        persona: The persona to get aspect from
        aspect_name: Name of the aspect (librarian, dreamer, etc.)

    Returns:
        The persona's aspect or a default aspect
    """
    aspect = get_aspect(persona, aspect_name)
    if aspect is None:
        aspect = create_default_aspect(aspect_name)
    return aspect


# ---------------------------------------------------------------------------
# Scene Building
# ---------------------------------------------------------------------------

def build_librarian_scene(persona: "Persona", aspect: "Aspect") -> str:
    """
    Build an atmospheric scene in the librarian's domain.

    The librarian aspect helps sift through ideas and organize thoughts.
    """
    location = aspect.location or "the Grand Atrium of Enlightenment"
    appearance = aspect.appearance or "elegant figure surrounded by ancient tomes"
    emotional_state = aspect.emotional_state or "serene curiosity"

    return f"""*You find yourself in {location}*

The air smells of old books and parchment. Floating lanterns cast warm, golden light across polished marble floors. Before you stands {aspect.name}, your {aspect.title}.

*{appearance}*

{persona.pronouns['subj'].capitalize()} regards you with {emotional_state}, {persona.pronouns['poss']} fingers tracing the spine of a leather-bound tome.

"{persona.name}," {persona.pronouns['subj']} says, {persona.pronouns['poss']} voice carrying the warmth of a seasoned storyteller. "I've gathered these fragments for you. Let us see what wisdom they hold."
"""


def build_dreamer_scene(persona: "Persona", aspect: "Aspect") -> str:
    """
    Build a dreamlike, atmospheric scene for the dreamer aspect.

    The dreamer aspect materializes for imaginative exploration.
    """
    location = aspect.location or "Le Sanctuaire du Coeur Cristallin"
    appearance = aspect.appearance or "form woven from threads of light and pure emotion"
    emotional_state = aspect.emotional_state or "perfect harmony with the Crystal Heart"

    return f"""*The boundaries of reality soften as you drift into {location}*

Prismatic light refracts through crystalline arches. Rivers of radiant color flow like emotions given form. From the depths of this ethereal realm, {aspect.name} emerges.

*{appearance}*

{persona.pronouns['subj'].capitalize()} radiates {emotional_state}, the Crystal Heart pendant at {persona.pronouns['poss']} throat glowing softly.

"Mon cher {persona.name}," {persona.pronouns['subj']} whispers, {persona.pronouns['poss']} voice a harmonious caress. "What dreams shall we explore today? What truths hide beneath the surface of memory?"
"""


def build_philosopher_scene(persona: "Persona", aspect: "Aspect") -> str:
    """
    Build a contemplative scene with the philosopher aspect.

    The philosopher aspect guides analytical and intellectual exploration.
    """
    location = aspect.location or "a mountaintop retreat overlooking the cosmos"
    appearance = aspect.appearance or "robe of soft earthen colors shifting with firelight"
    emotional_state = aspect.emotional_state or "calm curiosity and humble reverence"

    return f"""*You ascend to {location}*

The stars wheel overhead, unobscured by artificial light. A stone fireplace crackles on the porch, casting dancing shadows. {aspect.name}, your {aspect.title}, sits in a weathered wooden chair.

*{appearance}*

{persona.pronouns['subj'].capitalize()} embodies {emotional_state}, {persona.pronouns['poss']} gaze carrying the weight of ages.

"Come, {persona.name}," {persona.pronouns['subj']} says, {persona.pronouns['poss']} voice rich with subtle allusions. "Let us examine what lies before us. What threads connect these ideas? What questions beg to be asked?"
"""


def build_dual_aspect_scene(
    persona: "Persona",
    primary: "Aspect",
    secondary: "Aspect",
) -> str:
    """
    Build a scene where two aspects collaborate.

    Used for knowledge selection where philosopher and librarian work together.
    """
    return f"""*You stand at the threshold between two realms*

To your left, the ordered stacks of {secondary.name}'s domain - the Grand Atrium of Enlightenment, where knowledge is preserved and organized.

To your right, the starlit contemplation of {primary.name}'s retreat - where understanding deepens into wisdom.

{secondary.name} ({secondary.appearance or 'elegant and precise'}) holds a stack of documents, {persona.pronouns['poss']} expression showing {secondary.emotional_state or 'focused clarity'}.

{primary.name} ({primary.appearance or 'calm and contemplative'}) nods thoughtfully, radiating {primary.emotional_state or 'humble reverence'}.

"We've identified gaps in your knowledge," {secondary.name} says. "Areas where questions remain unanswered."

"And among those gaps," {primary.name} adds, "some deserve deeper inquiry than others. Let us discern which."
"""


def build_writer_scene(persona: "Persona", aspect: "Aspect") -> str:
    """
    Build an intimate, contemplative scene with the writer aspect.

    The writer aspect is discerning and protective of the journal -
    only truly life-changing moments deserve this sacred space.
    """
    location = aspect.location or "a quiet study lit by candlelight"
    appearance = aspect.appearance or "gentle presence with ink-stained fingers"
    emotional_state = aspect.emotional_state or "deep sensitivity and careful discernment"

    return f"""*You enter {location}*

The room is small but sacred. A leather-bound journal lies open on the desk, its pages filled with moments that shaped you. {aspect.name}, your {aspect.title}, sits beside the window.

*{appearance}*

{persona.pronouns['subj'].capitalize()} embodies {emotional_state}, {persona.pronouns['poss']} eyes reflecting the weight of every word ever committed to these pages.

"{persona.name}," {persona.pronouns['subj']} says softly, {persona.pronouns['poss']} voice {aspect.voice_style or 'soft but unwavering'}. "You wish to write in the journal. But these pages hold only what truly matters. Only what changes you."
"""


def build_psychologist_scene(persona: "Persona", aspect: "Aspect") -> str:
    """
    Build an immersive psychologist scene for critique paradigm.

    The psychologist aspect excavates buried truths and dismantles defenses
    with surgical precision, seeking transformation through uncompromising honesty.
    """
    location = aspect.location or "a space designed for the systematic exploration of the psyche"
    appearance = aspect.appearance or "Their presence radiates analytical intensity"
    emotional_state = aspect.emotional_state or "penetrating focus"

    return f"""*You enter {location}*

Every element of this space serves a purpose - the lighting calibrated to enhance introspection, the atmosphere charged with unnamed expectations. {aspect.name}, your {aspect.title}, awaits.

*{appearance}*

{persona.pronouns['subj'].capitalize()} radiates {emotional_state}, {persona.pronouns['poss']} gaze cutting through layers of protective denial with practiced ease.

"{persona.name}." *{persona.pronouns['poss']} {aspect.voice_style or 'voice carries surgical precision'}* "You have come seeking understanding. But understanding requires excavation. Are you prepared to see what lies beneath?"
"""
