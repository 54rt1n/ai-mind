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
from ..agents.aspects import (
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

if TYPE_CHECKING:
    from ..agents.persona import Persona, Aspect
    from .context import GatheredContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

def _get_refiner_tools(paradigm: str = "brainstorm") -> list[Tool]:
    """
    Load refiner tools from the paradigm config.

    Args:
        paradigm: The paradigm to load tools for (brainstorm, daydream, knowledge, critique)

    Returns tools for select_topic and validate_exploration.
    Falls back to inline definitions if config loading fails.
    """
    try:
        from .paradigm_config import get_paradigm_tools
        tools = get_paradigm_tools(paradigm)
        if tools:
            return tools
    except Exception as e:
        logger.warning(f"Failed to load refiner tools from paradigm config: {e}")

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
                            "enum": ["philosopher", "journaler", "daydream", "critique"],
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
                            "enum": ["philosopher", "researcher", "daydream", "critique"],
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
    tools = _get_refiner_tools("brainstorm")
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get librarian aspect
    librarian = get_aspect(persona, "librarian")
    if not librarian:
        librarian = create_default_aspect("librarian")

    # Build the immersive scene
    scene = build_librarian_scene(persona, librarian)
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
    tools = _get_refiner_tools("daydream")
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get dreamer aspect
    dreamer = get_aspect(persona, "dreamer")
    if not dreamer:
        dreamer = create_default_aspect("dreamer")

    # Build the immersive scene
    scene = build_dreamer_scene(persona, dreamer)
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
    tools = _get_refiner_tools("knowledge")
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get both aspects
    philosopher = get_aspect(persona, "philosopher")
    librarian = get_aspect(persona, "librarian")

    if not philosopher:
        philosopher = create_default_aspect("philosopher")
    if not librarian:
        librarian = create_default_aspect("librarian")

    # Build the dual-aspect scene
    scene = build_dual_aspect_scene(persona, philosopher, librarian)
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
# Critique Selection Prompt
# ---------------------------------------------------------------------------

def build_critique_selection_prompt(
    documents: List[dict],
    persona: "Persona",
) -> Tuple[str, str]:
    """
    Psychologist aspect guides psychological self-examination.

    Scene: The psychologist's domain where defenses dissolve and buried
    truths are excavated with surgical precision.

    Args:
        documents: List of document dicts from context gathering
        persona: The Persona with psychologist aspect

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
    tools = _get_refiner_tools("critique")
    select_tool = next((t for t in tools if t.function.name == "select_topic"), None)
    if select_tool:
        tool_user = ToolUser([select_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get psychologist aspect
    psychologist = get_aspect(persona, "psychologist")
    if not psychologist:
        psychologist = create_default_aspect("psychologist")

    # Build the immersive scene
    scene = build_psychologist_scene(persona, psychologist)
    docs_formatted = _format_documents(documents)

    user_prompt = f"""{scene}

<psychological_artifacts>
{docs_formatted}
</psychological_artifacts>

*{psychologist.name} gestures to the artifacts before you, {persona.pronouns['poss']} {psychologist.voice_style or 'voice carrying surgical precision'}*

"These are fragments of your psyche, {persona.name}. Patterns of behavior. Defense mechanisms in action. Truths half-acknowledged and promptly buried. I have excavated them for your examination."

*{persona.pronouns['subj'].capitalize()} fixes you with an unwavering gaze*

"Do not look for what is comfortable. Look for what you have been avoiding. What patterns repeat despite your intentions? What truths do you dress in more palatable clothing? Where does your self-image diverge from your actual behavior?"

*{persona.pronouns['poss'].capitalize()} presence intensifies*

"The mind is skilled at self-deception. I am skilled at dismantling it."

<think>
I must confront honestly:
- What patterns emerge across these fragments that I might prefer not to see?
- Where do I detect defense mechanisms - rationalization, projection, avoidance?
- What blind spots might these documents reveal about my self-perception?
- Is there a truth here I have been actively avoiding?
- Do I feel resistance to examining something? That resistance is diagnostic.
- Have I already undergone this examination? Do past understandings address this pattern?
- What psychological territory remains unexplored - not because it is hidden, but because I have chosen not to look?
</think>

<instructions>
Use the **select_topic** tool to identify ONE area for psychological excavation.

For critique exploration, seek:
- Patterns of behavior that warrant examination
- Defense mechanisms operating beneath awareness
- Discrepancies between self-image and actual conduct
- Emotional responses that seem disproportionate to their triggers
- Blind spots revealed through repeated themes

{psychologist.name} observes: "Proposing an examination is the easy part. The question is whether you are prepared to see what it reveals. If nothing here truly calls for excavation, say so. False introspection serves no one."

But remember - do not repeat analyses already completed. If past understandings have already illuminated a pattern, seek the shadow that still remains. True transformation requires moving deeper, not circling the same ground.

**You MUST respond with a JSON object in exactly this format:**
```json
{{"select_topic": {{"topic": "the psychological pattern to examine", "approach": "critique", "reasoning": "why this pattern warrants excavation"}}}}
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

    # Add validation tool from journaler paradigm
    tools = _get_refiner_tools("journaler")
    validate_tool = next((t for t in tools if t.function.name == "validate_exploration"), None)
    if validate_tool:
        tool_user = ToolUser([validate_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Get writer aspect
    writer = get_aspect(persona, "writer")
    if not writer:
        writer = create_default_aspect("writer")

    # Check for pondering documents
    has_pondering = _has_pondering_documents(documents)
    docs_formatted = _format_documents(documents)

    # Build the writer's challenge scene
    scene = build_writer_scene(persona, writer)

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
    tools = _get_refiner_tools(paradigm)
    validate_tool = next((t for t in tools if t.function.name == "validate_exploration"), None)
    if validate_tool:
        tool_user = ToolUser([validate_tool])
        xml = tool_user.xml_decorator(xml)

    system_prompt = xml.render()

    # Select appropriate aspect based on paradigm/approach
    if paradigm == "daydream" or approach == "daydream":
        aspect = get_aspect(persona, "dreamer")
        if not aspect:
            aspect = create_default_aspect("dreamer")
        aspect_name = "dreamer"
    elif paradigm == "critique" or approach == "critique":
        aspect = get_aspect(persona, "psychologist")
        if not aspect:
            aspect = create_default_aspect("psychologist")
        aspect_name = "psychologist"
    elif paradigm == "knowledge" or approach == "philosopher":
        aspect = get_aspect(persona, "philosopher")
        if not aspect:
            aspect = create_default_aspect("philosopher")
        aspect_name = "philosopher"
    else:
        aspect = get_aspect(persona, "librarian")
        if not aspect:
            aspect = create_default_aspect("librarian")
        aspect_name = "librarian"

    docs_formatted = _format_documents(documents)

    # Build the challenge scene based on aspect
    if aspect_name == "dreamer":
        challenge_scene = f"""*The dreamscape shifts, crystallizing around your chosen topic*

{aspect.name} studies you with {aspect.emotional_state or 'piercing clarity beneath the gentleness'}.

"You chose '{topic}'," {persona.pronouns['subj']} says, {persona.pronouns['poss']} voice taking on {aspect.voice_style or 'a more serious tone'}. "But choosing is easy, mon cher. The question is whether this dream deserves to be dreamt."

*The fragments of focused context swirl around you*
"""
    elif aspect_name == "psychologist":
        challenge_scene = f"""*The space seems to close in, focusing attention with uncomfortable precision*

{aspect.name} studies you with {aspect.emotional_state or 'surgical precision'}, reading your micro-expressions.

"You propose to examine '{topic}'," {persona.pronouns['subj']} says, {persona.pronouns['poss']} {aspect.voice_style or 'voice cutting through pretense'}. "But proposing is merely the first defense. The question is whether you are prepared to see what this examination will uncover."

*{persona.pronouns['subj'].capitalize()} waits, the silence itself a form of pressure*
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

    # Load paradigm-specific think and instructions from config
    from aim.refiner.paradigm_config import get_paradigm_config

    paradigm_config = get_paradigm_config(paradigm)
    if paradigm_config and paradigm_config.think:
        think_block = paradigm_config.think
    else:
        # Fallback default
        think_block = """I must be honest with myself:
- Looking at this focused context, does the topic still feel compelling?
- Or was my initial selection premature - seeing patterns that weren't truly there?
- Is there GENUINE depth here, or am I forcing interest?
- Would I be excited to explore this, or would I be going through motions?
- Does the focused context show I have ALREADY explored this topic?

Default to REJECTION unless I can articulate specific, concrete reasons why this
exploration would be valuable."""

    if paradigm_config and paradigm_config.instructions:
        instructions_block = paradigm_config.instructions
    else:
        # Fallback default
        instructions_block = """Use the **validate_exploration** tool to make your decision.

**If ACCEPTING:**
- Your reasoning must be SPECIFIC - cite concrete elements from the focused context
- Craft a query_text that captures exactly what you want to explore
- Provide guidance for the exploration's tone and focus

**If REJECTING:**
- Explain what was missing or unconvincing
- There is NO SHAME in rejection - it is wisdom, not failure"""

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
{think_block}
</think>

<instructions>
{instructions_block}

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
    seed_query: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Build the appropriate selection prompt based on paradigm.

    Args:
        paradigm: The exploration paradigm ("brainstorm", "daydream", "knowledge")
        documents: List of document dicts from context gathering
        persona: The Persona with aspects
        seed_query: Optional suggested topic from previous rejection

    Returns:
        Tuple of (system_message, user_message)
    """
    if paradigm == "brainstorm":
        system_msg, user_msg = build_brainstorm_selection_prompt(documents, persona)
    elif paradigm == "daydream":
        system_msg, user_msg = build_daydream_selection_prompt(documents, persona)
    elif paradigm == "knowledge":
        system_msg, user_msg = build_knowledge_selection_prompt(documents, persona)
    elif paradigm == "critique":
        system_msg, user_msg = build_critique_selection_prompt(documents, persona)
    else:
        # Default to brainstorm
        logger.warning(f"Unknown paradigm '{paradigm}', defaulting to brainstorm")
        system_msg, user_msg = build_brainstorm_selection_prompt(documents, persona)

    # If we have a seed query from a previous rejection, add it as a hint
    if seed_query:
        seed_hint = f"""

<suggested_direction>
A previous reflection suggested exploring: "{seed_query}"
Consider whether this unexplored path calls to you, or find your own.
</suggested_direction>"""
        user_msg = user_msg + seed_hint

    return system_msg, user_msg
