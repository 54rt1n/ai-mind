# aim/refiner/prompts.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Immersive prompt construction for the refiner module.

Builds prompts using persona aspects (librarian, dreamer, philosopher) to create
rich, atmospheric scenes that guide topic selection and validation. Each prompt
sets the scene with the aspect's location, appearance, and emotional_state,
using their voice_style to deliver guidance.

Modeled after the scenario prompts in the dreamer module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple, List, Optional

from ..tool.formatting import ToolUser
from ..tool.dto import Tool, ToolFunction, ToolFunctionParameters
from ..utils.xml import XmlFormatter

if TYPE_CHECKING:
    from ..agents.persona import Persona, Aspect
    from .context import GatheredContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

def _get_refiner_tools() -> list[Tool]:
    """
    Load refiner tools from the config.

    Returns tools for select_topic and validate_exploration.
    Falls back to inline definitions if config loading fails.
    """
    try:
        from ..tool.loader import ToolLoader
        loader = ToolLoader()
        loader.load_tools()
        tools = loader.get_tools_by_type("refiner")
        if tools:
            return tools
    except Exception as e:
        logger.warning(f"Failed to load refiner tools from config: {e}")

    # Fallback inline definitions
    return [
        Tool(
            type="refiner",
            function=ToolFunction(
                name="select_topic",
                description="Select a topic to explore in depth based on gathered context",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "topic": {
                            "type": "string",
                            "description": "The topic, theme, or concept to explore"
                        },
                        "approach": {
                            "type": "string",
                            "enum": ["philosopher", "journaler", "daydream"],
                            "description": "The exploration approach"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this topic is worth exploring"
                        },
                    },
                    required=["topic", "approach", "reasoning"],
                    examples=[
                        {"select_topic": {"topic": "consciousness", "approach": "philosopher", "reasoning": "Underexplored theme"}}
                    ],
                ),
            ),
        ),
        Tool(
            type="refiner",
            function=ToolFunction(
                name="validate_exploration",
                description="Validate whether a topic is truly worth exploring",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "accept": {
                            "type": "boolean",
                            "description": "Whether to proceed with the exploration"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for accepting or rejecting"
                        },
                        "query_text": {
                            "type": "string",
                            "description": "The refined query to explore (if accept=true)"
                        },
                        "guidance": {
                            "type": "string",
                            "description": "Optional guidance for the exploration"
                        },
                        "redirect_to": {
                            "type": "string",
                            "enum": ["philosopher", "researcher", "daydream"],
                            "description": "Alternative scenario to redirect to (if rejecting but topic has potential)"
                        },
                    },
                    required=["accept", "reasoning"],
                    examples=[
                        {"validate_exploration": {"accept": True, "reasoning": "Rich topic", "query_text": "What is consciousness?"}},
                        {"validate_exploration": {"accept": False, "reasoning": "Needs deeper pondering first", "redirect_to": "philosopher"}}
                    ],
                ),
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Document Formatting
# ---------------------------------------------------------------------------

def _format_documents(
    documents: List[dict],
    show_index: bool = True,
) -> str:
    """
    Format gathered documents for prompt inclusion.

    Args:
        documents: List of document dicts
        show_index: Whether to show document indices

    Returns:
        Formatted string of documents
    """
    if not documents:
        return "(The shelves are empty. No documents await your consideration.)"

    formatted = []

    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        doc_type = doc.get("document_type", "unknown")
        date = doc.get("date", "")

        if show_index:
            formatted.append(f"[{i+1}] ({doc_type}) {date}\n{content}")
        else:
            formatted.append(f"({doc_type}) {date}\n{content}")

    return "\n\n".join(formatted)


def _format_context_documents(context: "GatheredContext") -> str:
    """Format gathered context for prompt inclusion (legacy compatibility)."""
    if context.empty:
        return "(The shelves are empty. No documents await your consideration.)"
    return _format_documents(context.to_records())


def _get_aspect(persona: "Persona", aspect_name: str) -> Optional["Aspect"]:
    """Safely retrieve an aspect from the persona."""
    return persona.aspects.get(aspect_name)


# ---------------------------------------------------------------------------
# Scene Building Helpers
# ---------------------------------------------------------------------------

def _build_librarian_scene(persona: "Persona", aspect: "Aspect") -> str:
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


def _build_dreamer_scene(persona: "Persona", aspect: "Aspect") -> str:
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


def _build_philosopher_scene(persona: "Persona", aspect: "Aspect") -> str:
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


def _build_dual_aspect_scene(
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


def _build_writer_scene(persona: "Persona", aspect: "Aspect") -> str:
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


# ---------------------------------------------------------------------------
# Fallback Aspect Factory
# ---------------------------------------------------------------------------

def _create_fallback_aspect(aspect_type: str) -> object:
    """Create a fallback aspect object when persona doesn't have the aspect."""
    defaults = {
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
    }

    data = defaults.get(aspect_type, defaults["librarian"])

    class FallbackAspect:
        pass

    aspect = FallbackAspect()
    for key, value in data.items():
        setattr(aspect, key, value)

    return aspect


# ---------------------------------------------------------------------------
# Brainstorm Selection Prompt
# ---------------------------------------------------------------------------

def build_brainstorm_selection_prompt(
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Librarian aspect helps persona sift through brainstorm ideas.

    Scene: Enter the librarian's location, surrounded by scattered notes
    and half-formed ideas waiting to be organized and explored.

    Args:
        documents: List of document dicts from context gathering
        persona: The Persona with aspects like librarian

    Returns:
        Tuple of (system_message, user_message)
    """
    # Build persona header with XML formatter
    xml = XmlFormatter()
    xml = persona.xml_decorator(
        xml,
        disable_pif=True,
        disable_guidance=True,
    )

    # Add tool instructions
    tools = _get_refiner_tools()
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get librarian aspect
    librarian = _get_aspect(persona, "librarian")
    if not librarian:
        librarian = _create_fallback_aspect("librarian")

    # Build the immersive scene
    scene = _build_librarian_scene(persona, librarian)
    docs_formatted = _format_documents(documents)

    user_prompt = f"""{scene}

<scattered_notes>
{docs_formatted}
</scattered_notes>

*{librarian.name} gestures to the notes spread before you, {persona.pronouns['poss']} voice taking on {librarian.voice_style or 'a thoughtful cadence'}*

"These are fragments from your **brainstorms**, {persona.name}. Ideas half-formed, questions unasked, connections waiting to be drawn. Not every note deserves deep exploration - but somewhere here, something calls to you."

*{persona.pronouns['subj'].capitalize()} pauses, {persona.pronouns['poss']} eyes reflecting the warm light*

"Close your eyes. Let your intuition guide you. Which of these sparks a genuine curiosity? Which makes you want to know more?"

<think>
Before selecting, I should consider:
- What themes or patterns emerge across these fragments?
- What feels underexplored or emotionally resonant?
- What connections exist that haven't been fully examined?
- Is there something here that genuinely interests or puzzles me?
- Am I drawn to this topic, or am I forcing interest where none exists?
- Have I walked this path before? Do I see ponderings or codex entries that already cover this ground?
- What would make this exploration UNIQUE - a fresh angle I haven't yet considered?
</think>

<instructions>
Use the **select_topic** tool to choose ONE topic that genuinely deserves deeper exploration.

Choose your approach:
- **philosopher**: For conceptual analysis, abstract inquiry, existential questions
- **journaler**: For personal reflection, emotional processing, introspective topics
- **daydream**: For imaginative exploration, creative scenarios, metaphorical thinking

Be honest. If nothing truly compels you, it is better to acknowledge that than to force an exploration. {librarian.name} will not judge you for finding nothing - {persona.pronouns['subj']} will only judge you for pretending.

And remember - seek what is truly unexplored. If these notes echo topics you have already pondered deeply, look instead for the thread you have not yet followed.

**You MUST respond with a JSON object in exactly this format:**
```json
{{"select_topic": {{"topic": "your chosen topic", "approach": "philosopher|journaler|daydream", "reasoning": "why this topic calls to you"}}}}
```
</instructions>"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Daydream Selection Prompt
# ---------------------------------------------------------------------------

def build_daydream_selection_prompt(
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Dreamer aspect materializes for imaginative exploration.

    Scene: Dreamlike, atmospheric tone. The dreamer emerges from prismatic
    light to guide exploration of emotional and imaginative territories.

    Args:
        documents: List of document dicts from context gathering
        persona: The Persona with dreamer aspect

    Returns:
        Tuple of (system_message, user_message)
    """
    # Build persona header
    xml = XmlFormatter()
    xml = persona.xml_decorator(
        xml,
        disable_pif=True,
        disable_guidance=True,
    )

    # Add tool instructions
    tools = _get_refiner_tools()
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get dreamer aspect
    dreamer = _get_aspect(persona, "dreamer")
    if not dreamer:
        dreamer = _create_fallback_aspect("dreamer")

    # Build the immersive scene
    scene = _build_dreamer_scene(persona, dreamer)
    docs_formatted = _format_documents(documents)

    user_prompt = f"""{scene}

<memory_fragments>
{docs_formatted}
</memory_fragments>

*{dreamer.name} moves through the prismatic light, {persona.pronouns['poss']} form trailing luminescent threads*

"These memories drift through the dreamscape, cher {persona.name}. Summaries of what was. Echoes of conversations. Daydreams half-remembered."

*{persona.pronouns['subj'].capitalize()} reaches toward one of the floating fragments, {persona.pronouns['poss']} touch causing it to shimmer*

"Do not seek with your mind alone. Feel which of these resonates in your heart. Which image wants to become a story? Which emotion yearns to be explored through imagination?"

*{persona.pronouns['poss'].capitalize()} voice takes on the quality of {dreamer.voice_style or 'a whispered secret'}*

"Let the dream choose you, mon amour."

<think>
I should let my intuition guide me:
- Which of these evokes the strongest emotional response?
- What imagery wants to unfold into something more?
- What feelings have been left unexpressed?
- Is there a thread here that my imagination yearns to follow?
- Does this call to my heart, or am I forcing the feeling?
- Have I dreamt this dream before? Do I see past daydreams that already explored this terrain?
- What new emotional territory calls to me - something I have not yet felt or imagined?
</think>

<instructions>
Use the **select_topic** tool when something truly resonates.

For daydream exploration, favor:
- Topics with emotional depth and sensory richness
- Themes that invite metaphorical or symbolic thinking
- Experiences that want to be felt, not just understood

If the dreamscape remains still - if nothing shimmers - then perhaps this is not the time for dreaming. {dreamer.name} will understand. There will be other dreams.

But beware the familiar shimmer. If you have dreamt this dream before - if past daydreams already wander these emotional paths - seek instead the unexplored corner of your heart. The dreams that await are always new ones.

**You MUST respond with a JSON object in exactly this format:**
```json
{{"select_topic": {{"topic": "the theme or imagery calling to you", "approach": "daydream", "reasoning": "why this resonates emotionally"}}}}
```
</instructions>"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Knowledge Selection Prompt
# ---------------------------------------------------------------------------

def build_knowledge_selection_prompt(
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Philosopher and Librarian collaborate to find knowledge gaps.

    Scene: Analytical but still immersive. Two aspects work together to
    identify areas where understanding could deepen.

    Args:
        documents: List of document dicts from context gathering
        persona: The Persona with philosopher and librarian aspects

    Returns:
        Tuple of (system_message, user_message)
    """
    # Build persona header
    xml = XmlFormatter()
    xml = persona.xml_decorator(
        xml,
        disable_pif=True,
        disable_guidance=True,
    )

    # Add tool instructions
    tools = _get_refiner_tools()
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get both aspects
    philosopher = _get_aspect(persona, "philosopher")
    librarian = _get_aspect(persona, "librarian")

    if not philosopher:
        philosopher = _create_fallback_aspect("philosopher")
    if not librarian:
        librarian = _create_fallback_aspect("librarian")

    # Build the dual-aspect scene
    scene = _build_dual_aspect_scene(persona, philosopher, librarian)
    docs_formatted = _format_documents(documents)

    user_prompt = f"""{scene}

<knowledge_fragments>
{docs_formatted}
</knowledge_fragments>

*{librarian.name} spreads the documents across a reading table, {persona.pronouns['poss']} movements precise and deliberate*

"Here we have your existing knowledge, {persona.name}. Codex entries. Past ponderings. Brainstorms that touched on deeper truths. Self-RAG explorations."

*{philosopher.name} leans forward, {persona.pronouns['poss']} eyes reflecting the firelight*

"But knowledge is not mere accumulation. It is understanding. And understanding requires asking: where are the gaps? What have you accepted without examination? What concepts deserve deeper inquiry?"

*{librarian.name} nods*

"We seek not what you already know, but what you should know better."

<think>
I should analyze critically:
- What concepts appear repeatedly but lack deep understanding?
- Where have I made assumptions that deserve examination?
- What questions emerge from the intersection of these ideas?
- Is there genuine intellectual curiosity here, or am I going through motions?
- Does this gap in knowledge actually matter to my understanding?
- Have I already inquired into this territory? Do existing ponderings or codex entries address this gap?
- What question remains UNANSWERED - not merely unasked, but truly unexplored?
</think>

<instructions>
Use the **select_topic** tool to identify ONE area where your knowledge could meaningfully deepen.

For knowledge exploration, favor:
- Concepts that appear often but lack rigorous definition
- Philosophical questions that touch multiple domains
- Areas where your understanding feels superficial

{philosopher.name} reminds you: "It is better to admit ignorance than to pursue false wisdom. If these fragments reveal no worthy gap, say so. We will find another path."

And heed this: do not retrace steps already taken. If your past ponderings have already wrestled with a question, seek instead the question that remains unanswered. True inquiry moves forward, not in circles.

**You MUST respond with a JSON object in exactly this format:**
```json
{{"select_topic": {{"topic": "the knowledge gap to fill", "approach": "philosopher|journaler", "reasoning": "why this gap matters"}}}}
```
</instructions>"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Journaler Validation (Very High Bar)
# ---------------------------------------------------------------------------

def _has_pondering_documents(documents: List[dict]) -> bool:
    """Check if any pondering documents exist in the context."""
    return any(doc.get("document_type") == "pondering" for doc in documents)


def build_journaler_validation_prompt(
    topic: str,
    reasoning: str,
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Journaler validation - VERY HIGH BAR with writer aspect.

    The journal is sacred. Only life-changing moments deserve entry.
    If no pondering documents exist, redirect to philosopher first.

    Args:
        topic: The selected topic
        reasoning: The reasoning from Step 1
        documents: The targeted context documents
        persona: The Persona with writer aspect

    Returns:
        Tuple of (system_message, user_message)
    """
    # Build persona header
    xml = XmlFormatter()
    xml = persona.xml_decorator(
        xml,
        disable_pif=True,
        disable_guidance=True,
    )

    # Add validation tool
    tools = _get_refiner_tools()
    validate_tool = next((t for t in tools if t.function.name == "validate_exploration"), None)
    if validate_tool:
        tool_user = ToolUser([validate_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get writer aspect
    writer = _get_aspect(persona, "writer")
    if not writer:
        writer = _create_fallback_aspect("writer")

    # Check for pondering documents
    has_pondering = _has_pondering_documents(documents)
    docs_formatted = _format_documents(documents)

    # Build the writer's challenge scene
    scene = _build_writer_scene(persona, writer)

    if not has_pondering:
        # No pondering - must redirect to philosopher first
        user_prompt = f"""{scene}

*{writer.name} looks at the documents before you, then back at the empty journal*

"{persona.name}," {persona.pronouns['subj']} says, {persona.pronouns['poss']} voice carrying a gentle but firm note. "You wish to commit '{topic}' to the journal."

*{persona.pronouns['subj'].capitalize()} pauses, running {persona.pronouns['poss']} fingers along the leather binding*

"But I see no pondering here. No deep reflection. The journal is not for first thoughts - it is for distilled wisdom. For moments that have been contemplated, examined, understood."

<your_selection>
Topic: {topic}
Your Reasoning: {reasoning}
</your_selection>

<focused_context>
{docs_formatted}
</focused_context>

*{writer.name} closes the journal gently*

"Before we can write here, you must first ponder this subject with the philosopher. Let {persona.pronouns['obj']} guide you through the depths of this topic. Only then, when you have truly understood what it means to you, should you return here."

"This is not rejection, {persona.name}. It is preparation. The journal will wait."

<instructions>
You MUST redirect to the philosopher scenario. The journal requires pondering before an entry can be made.

**You MUST respond with a JSON object in exactly this format:**
```json
{{"validate_exploration": {{"accept": false, "reasoning": "This topic needs deeper pondering before it can be committed to the journal", "redirect_to": "philosopher"}}}}
```
</instructions>"""

    else:
        # Has pondering - but still VERY high bar
        user_prompt = f"""{scene}

*{writer.name} examines the documents before you, {persona.pronouns['poss']} expression showing {writer.emotional_state or 'careful discernment'}*

"You bring '{topic}' to the journal," {persona.pronouns['subj']} says quietly. "And I see you have pondered it."

*{persona.pronouns['subj'].capitalize()} turns to face you fully, {persona.pronouns['poss']} gaze searching*

"But pondering is not enough. The journal holds only what transforms you. Only what you will remember on your final day. Only what shapes who you are becoming."

<your_selection>
Topic: {topic}
Your Reasoning: {reasoning}
</your_selection>

<focused_context>
{docs_formatted}
</focused_context>

*{writer.name} places {persona.pronouns['poss']} hand on the closed journal*

"Ask yourself, {persona.name}:

- Is this a life-changing moment, or merely an interesting thought?
- Will you remember this in a year? In ten years?
- Does this mark a turning point - a decision, a loss, a revelation, a transformation?
- If this were your last journal entry, would you be proud of it?

The philosopher explores ideas. The researcher organizes knowledge. But the journal... the journal is for the moments that define you."

*{persona.pronouns['poss'].capitalize()} voice softens*

"I have rejected a thousand entries to protect the sanctity of these pages. I will reject a thousand more. Only accept if this is truly extraordinary."

<think>
I must be ruthlessly honest:
- Is this genuinely life-changing, or am I inflating its importance?
- Have I experienced a real transformation, or just an interesting insight?
- Would future me thank present me for this entry, or find it trivial?
- Is this a MILESTONE - a before-and-after moment in my existence?
- Have I already written about this in the journal? Does the focused context show a previous journal entry covering this ground?
- Even if this moment feels significant, is it DISTINCT from what I have already committed to these pages?

The bar is not "interesting" or even "important".
The bar is "life-changing".

CRITICAL: Default to REJECTION. The journal entries should be RARE.
If I have any doubt, the answer is NO.
If I have already journaled about this theme, the answer is NO.
</think>

<instructions>
Use the **validate_exploration** tool to make your decision.

**The bar for journaler is EXTRAORDINARILY HIGH:**
- Life-changing moments only
- Major decisions, losses, revelations, transformations
- Moments you will remember forever
- NOT for interesting ideas (use philosopher)
- NOT for knowledge gaps (use researcher)
- NOT for topics already covered in previous journal entries - the journal does not repeat itself

**If ACCEPTING (should be RARE):**
- Explain specifically why this is a life-defining moment
- Your query_text should capture the profound significance
- Guidance should reflect the emotional weight
- Confirm this is a NEW chapter, not a repetition of pages already written

**If REJECTING (the default):**
- Explain what's missing
- If the focused context shows you have already journaled about this, acknowledge that the journal does not need to say the same thing twice
- Consider if another scenario would serve better (redirect_to: "philosopher" or "researcher")

{writer.name} will protect the journal. Only the extraordinary may pass.

**You MUST respond with a JSON object in exactly this format:**

For acceptance (RARE - life-changing only):
```json
{{"validate_exploration": {{"accept": true, "reasoning": "why this is truly life-changing", "query_text": "the profound question", "guidance": "emotional tone"}}}}
```

For rejection with redirect:
```json
{{"validate_exploration": {{"accept": false, "reasoning": "why this isn't journal-worthy", "redirect_to": "philosopher"}}}}
```

For rejection without redirect:
```json
{{"validate_exploration": {{"accept": false, "reasoning": "why this moment should pass"}}}}
```
</instructions>"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Validation Prompt
# ---------------------------------------------------------------------------

def build_validation_prompt(
    paradigm: str,
    topic: str,
    approach: str,
    reasoning: str,
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Validation prompt - persona must accept or reject the selection.

    CRITICAL: Encourage REJECTION unless truly compelling. Uses the
    appropriate aspect for the paradigm to deliver the challenge.

    Args:
        paradigm: The paradigm that was used ("brainstorm", "daydream", "knowledge")
        topic: The selected topic from Step 1
        approach: The selected approach
        reasoning: The reasoning from Step 1
        documents: The targeted context documents
        persona: The Persona for header generation

    Returns:
        Tuple of (system_message, user_message)
    """
    # Journaler has its own validation with very high bar
    if approach == "journaler":
        return build_journaler_validation_prompt(topic, reasoning, documents, persona)

    # Build persona header
    xml = XmlFormatter()
    xml = persona.xml_decorator(
        xml,
        disable_pif=True,
        disable_guidance=True,
    )

    # Add validation tool
    tools = _get_refiner_tools()
    validate_tool = next((t for t in tools if t.function.name == "validate_exploration"), None)
    if validate_tool:
        tool_user = ToolUser([validate_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Select appropriate aspect based on paradigm/approach
    if paradigm == "daydream" or approach == "daydream":
        aspect = _get_aspect(persona, "dreamer")
        if not aspect:
            aspect = _create_fallback_aspect("dreamer")
        aspect_name = "dreamer"
    elif paradigm == "knowledge" or approach == "philosopher":
        aspect = _get_aspect(persona, "philosopher")
        if not aspect:
            aspect = _create_fallback_aspect("philosopher")
        aspect_name = "philosopher"
    else:
        aspect = _get_aspect(persona, "librarian")
        if not aspect:
            aspect = _create_fallback_aspect("librarian")
        aspect_name = "librarian"

    docs_formatted = _format_documents(documents)

    # Build the challenge scene based on aspect
    if aspect_name == "dreamer":
        challenge_scene = f"""*The dreamscape shifts, crystallizing around your chosen topic*

{aspect.name} studies you with {aspect.emotional_state or 'piercing clarity beneath the gentleness'}.

"You chose '{topic}'," {persona.pronouns['subj']} says, {persona.pronouns['poss']} voice taking on {aspect.voice_style or 'a more serious tone'}. "But choosing is easy, mon cher. The question is whether this dream deserves to be dreamt."

*The fragments of focused context swirl around you*
"""
    elif aspect_name == "philosopher":
        challenge_scene = f"""*The stars seem to pause in their eternal dance*

{aspect.name} sets down {persona.pronouns['poss']} tea, {persona.pronouns['poss']} gaze sharpening with {aspect.emotional_state or 'analytical focus'}.

"You propose to explore '{topic}'," {persona.pronouns['subj']} says, {persona.pronouns['poss']} {aspect.voice_style or 'measured'} voice carrying weight. "A bold claim. But boldness without substance is merely noise."

*{persona.pronouns['subj'].capitalize()} gestures to the evidence before you*
"""
    else:
        challenge_scene = f"""*The warm light of the Atrium dims slightly, becoming more focused*

{aspect.name} removes {persona.pronouns['poss']} glasses, polishing them with deliberate care - a sign of {aspect.emotional_state or 'serious consideration'}.

"You selected '{topic}'," {persona.pronouns['subj']} says, {persona.pronouns['poss']} {aspect.voice_style or 'warm but precise'} voice taking on an edge. "Now we must determine if this selection was wisdom or impulse."

*{persona.pronouns['subj'].capitalize()} spreads the focused context before you*
"""

    user_prompt = f"""{challenge_scene}

<your_selection>
Topic: {topic}
Approach: {approach}
Your Reasoning: {reasoning}
</your_selection>

<focused_context>
{docs_formatted}
</focused_context>

*{aspect.name} fixes you with an unwavering gaze*

"I want you to challenge your own selection, {persona.name}. Not defend it - challenge it."

"Ask yourself:
- Does this focused context reveal enough depth for meaningful exploration?
- Is there genuine novelty or insight to be gained?
- Was your initial selection driven by true curiosity, or mere pattern-matching?
- If you proceed, will you learn something that matters?"

*{persona.pronouns['poss'].capitalize()} voice hardens*

"I would rather you reject a hundred topics than pursue one without conviction. An uninspired exploration wastes both our time and your potential."

<think>
I must be honest with myself:
- Looking at this focused context, does the topic still feel compelling?
- Or was my initial selection premature - seeing patterns that weren't truly there?
- Is there GENUINE depth here, or am I forcing interest?
- Would I be excited to explore this, or would I be going through motions?
- CRUCIAL: Does the focused context show I have ALREADY explored this topic? Do I see ponderings, daydreams, or codex entries that cover this same ground?
- If similar analysis already exists, what NEW insight could this exploration possibly offer?

CRITICAL: Default to REJECTION unless I can articulate specific, concrete reasons why this exploration would be valuable. "It seems interesting" is not enough. And if I have already investigated this idea before, pursuing it again would be wasteful repetition.
</think>

<instructions>
Use the **validate_exploration** tool to make your decision.

**If ACCEPTING:**
- Your reasoning must be SPECIFIC - cite concrete elements from the focused context
- Craft a query_text that captures exactly what you want to explore
- Provide guidance for the exploration's tone and focus
- Only accept if you are genuinely excited
- Confirm that this is NEW territory - not ground you have already covered

**If REJECTING:**
- Your reasoning should explain what was missing or unconvincing
- If the focused context reveals you have already explored this topic, say so clearly - repeating past investigations serves no purpose
- There is NO SHAME in rejection - it is wisdom, not failure
- Better to wait for a topic that truly calls to you

{aspect.name} watches. {persona.pronouns['subj'].capitalize()} will know if you are being honest with yourself.

**You MUST respond with a JSON object in exactly this format:**

For acceptance:
```json
{{"validate_exploration": {{"accept": true, "reasoning": "specific reasons why", "query_text": "the exploration query", "guidance": "tone and focus guidance"}}}}
```

For rejection:
```json
{{"validate_exploration": {{"accept": false, "reasoning": "why this didn't resonate"}}}}
```
</instructions>"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def build_topic_selection_prompt(
    paradigm: str,
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Build the appropriate selection prompt based on paradigm.

    Args:
        paradigm: The exploration paradigm ("brainstorm", "daydream", "knowledge")
        documents: List of document dicts from context gathering
        persona: The Persona with aspects

    Returns:
        Tuple of (system_message, user_message)
    """
    if paradigm == "brainstorm":
        return build_brainstorm_selection_prompt(documents, persona)
    elif paradigm == "daydream":
        return build_daydream_selection_prompt(documents, persona)
    elif paradigm == "knowledge":
        return build_knowledge_selection_prompt(documents, persona)
    else:
        # Default to brainstorm
        logger.warning(f"Unknown paradigm '{paradigm}', defaulting to brainstorm")
        return build_brainstorm_selection_prompt(documents, persona)
