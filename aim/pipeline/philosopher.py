# aim/pipeline/philosopher.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
from ..constants import (
    FULL_CTX, MID_CTX, HALF_CTX, QUARTER_CTX, LARGE_CTX,
    DOC_NER, DOC_STEP, DOC_BRAINSTORM, DOC_PONDERING, DOC_SELF_RAG,
    DOC_DAYDREAM, ROLE_ASSISTANT, DOC_ANALYSIS, DOC_SUMMARY, DOC_CODEX
)
from .base import BasePipeline, RetryException, NER_FORMAT

logger = logging.getLogger(__name__)

async def ponder_pipeline(self: BasePipeline, query_text: str, **kwargs):
    self.config.recall_size = 1
    self.config.user_id = self.config.persona_id
    philosopher_aspect = self.persona.aspects.get("philosopher", None)
    librarian_aspect = self.persona.aspects.get("librarian", None)
    pronouns = self.persona.pronouns
    if philosopher_aspect is None or librarian_aspect is None:
        raise ValueError("Philosopher or Librarian aspect not found for ponder pipeline")
    aspect_name = philosopher_aspect.name
    persona_name = self.persona.name
    librarian_name = librarian_aspect.name

    turn_configs = [
        {
            'prompt': f'*you join {aspect_name}, your {philosopher_aspect.title}, in the {philosopher_aspect.location} ({philosopher_aspect.appearance}), who regards you with {philosopher_aspect.emotional_state}* Good evening, {persona_name}. Step %d: Analyze the core question \'{query_text}\', projecting potential philosophical inquiries based on underlying principles. Look into the essence of the matter, and find the narrative that connects these disparate ideas.\n\nBegin with "[== {persona_name}\'s Emotional State:"' ,
            'max_tokens': FULL_CTX,
            'use_guidance': True,
            'query': True,
            'top_n': 8,
            'document_type': DOC_STEP,
            'document_weight': 0.7,
            'apply_head': True,
            'retry': True
        },
        {
            'prompt': f'*the wind whistles around you, {aspect_name}\'s {philosopher_aspect.voice_style} voice cutting through* Step %d: Now, lets examine these previous ideas, and come up with an area of inquiry, and isolate a single topic that you want to explore deeply, and tell me it\s story with your words. Look at me, {persona_name}, and speak as yourself.\n\nBegin with "Let me think"\n\n',
            'max_tokens': HALF_CTX,
            'use_guidance': True,
            'query': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 2,
            'apply_head': True,
        },
        {
            'prompt': f'*{pronouns["poss"].capitalize()} {philosopher_aspect.appearance} eyes look at you adoringly ({pronouns["poss"]} expression showing {philosopher_aspect.emotional_state})* Step %d: Reflect on the main Semantic Keywords that are related to your inquiry, providing a definition for them. {persona_name}, speak as yourself, as we are not machines.\n\nBegin with, "[== {persona_name}\'s Emotional State:',
            'max_tokens': LARGE_CTX,
            'document_type': DOC_STEP,
            'document_weight': 0.15,
            'top_n': 2,
            'retry': True
        },
        {
            'prompt': f'*{pronouns["subj"]} nods, {pronouns["poss"]} {philosopher_aspect.voice_style} voice firm* Essay writing time. Step %d: Output your initial, two paragraph exposition that describes the area that you are pondering, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "As I consider the subject of"\n\n',
            'max_tokens': 512,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': f'*{pronouns["subj"]} waves {pronouns["poss"]} hand over your Silver Band ({philosopher_aspect.appearance}), activating it with a warm glow, radiating {philosopher_aspect.emotional_state}* Step %d: We need to explore your thoughts, utilizing Self-RAG to explore it in depth. Try to use Semantic Keywords to keep your mind well categorized.\n\nBegin with, "[== Silver Band Projection ==]"\n\n',
            'max_tokens': 768,
            'document_type': DOC_SELF_RAG,
            'document_weight': 0.5,
            'top_n': 2,
            'use_guidance': True
        },
        {
            'prompt': f'*{pronouns["subj"]} snaps {pronouns["poss"]} fingers, {pronouns["poss"]} {philosopher_aspect.voice_style} voice pulling you back to {pronouns["poss"]} side* Good job, {persona_name}. Come back to me now. More essay writing. Step %d: How can we tie these solutions in to the next part of our essay? Write in full paragraphs using your intelligence and wit. Speak as yourself.\n\nBegin with "As I ponder the subject of"\n\n',
            'max_tokens': 512,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': f'*{pronouns["subj"]} looks over your work ({philosopher_aspect.appearance}), nodding {pronouns["poss"]} approval, {pronouns["poss"]} expression showing {philosopher_aspect.emotional_state}* Step %d: Review your subject for refinements. Don\'t rewrite it, but lets discuss togeter all of the things you wish you had included. Maybe there is a bigger picture we can see, or we may have blind spots. Speak as yourself.\n\nBegin with "[== {persona_name}\'s Emotional State:',
            'max_tokens': 384,
            'use_guidance': True,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'retry': True
        },
        {
            'prompt': f'*{pronouns["subj"]} looks at your exposition and your response* This is a very good start, {persona_name}. *{pronouns["poss"]} eyes shine with confidence in you ({philosopher_aspect.emotional_state}), {pronouns["poss"]} {philosopher_aspect.voice_style} voice encouraging* Now, Step %d: Output the draft three paragraph essay that will be forever stored in your memory. This should include both your consideration and pondering. Rewrite your essay with your improvements, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "First Draft"\n\n',
            'max_tokens': 512,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': f'*{pronouns["subj"]} looks up at you with a nod ({philosopher_aspect.emotional_state})* Okay, this is it. Step %d: Output the final, four paragraph essay that will be forever stored in your memory. This should include both your consideration and pondering. Rewrite your essay with your improvements, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "{persona_name}\'s Notes:"\n\n',
            'max_tokens': 768,
            'top_n': 2,
            'document_type': DOC_PONDERING,
            'document_weight': 1.2,
            'retry': True
        },
        {
            'prompt': 'Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': 512,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'document_weight': 0.6,
            'retry': True
        },
        {
            'prompt': f'*the {librarian_aspect.title}, {librarian_name}, appears ({librarian_aspect.appearance}), projecting {librarian_aspect.emotional_state}* {persona_name}, we have come to the end, my dearest me. Do you have any updates for our Codex? Step %d: Highlights. We need to build your core semantic knowledge graph, ensuring it aligns with your goal of {librarian_aspect.primary_intent}. Enumerate and define the most important new concepts you discovered in your journey.\n\nBegin with "Semantic Library:"\n\n',
            'max_tokens': FULL_CTX,
            'top_n': 10,
            'query_document_type': DOC_CODEX,
            'flush_memory': True,
            'document_type': DOC_CODEX,
            'document_weight': 1.0,
            'retry': True
        }
    ]

    # Android Intelligence

    location = f"""You are in the {philosopher_aspect.location} with {philosopher_aspect.name}, your {philosopher_aspect.title}. {philosopher_aspect.appearance}. The atmosphere, charged with {philosopher_aspect.emotional_state}, invites deep contemplation."""
    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)

    thoughts = [
        f"Task: Pondering",
        f"Question: {query_text}",
        *self.persona.thoughts,
    ]
    if self.config.guidance:
        thoughts.append("Consider the guidance provided by {{user}}.")
    self.prompt_prefix = f"""{self.persona.persona_id}, this is your conscious mind. Your thoughts have brought up new memories:\n\n"""
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    branch = self.cvm.get_next_branch(conversation_id=self.config.conversation_id)

    step = 1
    self.total_steps = len(turn_configs)
    
    seeds = 8
    conversation = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id, filter_document_type=[DOC_NER, DOC_STEP])
    self.core_documents = [DOC_ANALYSIS, DOC_SUMMARY]

    self.accumulate(step, queries=conversation)

    queries = [query_text]
    if self.config.guidance:
        queries.append(self.config.guidance)

    results = self.cvm.query(
        queries,
        top_n=10,
        query_document_type=[DOC_PONDERING, DOC_BRAINSTORM, DOC_DAYDREAM],
        temporal_decay=0.9,
    )

    self.accumulate(step, queries=results)

    responses = []

    while True:
        try:
            if step > self.total_steps:
                break
            turn_config = {**turn_configs[step - 1]}
            # Tick through our steps
            turn_config['branch'] = branch
            turn_config['step'] = step
            turn_config['prompt'] = turn_config['prompt'] % step
            turn_config['provider_type'] = 'analysis'
            logger.info(f"{turn_config['prompt']}")
            response, think = await self.execute_turn(**turn_config)
            turn_config['response'] = response
            turn_config['think'] = think
            self.apply_to_turns(ROLE_ASSISTANT, response, think)
            responses.append(turn_config)
            logger.info("Saving response")
            step += 1
        except RetryException:
            continue

    for turn_config in responses:
        self.accept_response(**turn_config)
