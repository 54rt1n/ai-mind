# aim/pipline/journaler.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from datetime import datetime
from typing import Optional
import logging

from ..constants import (
    QUARTER_CTX, MID_CTX, HALF_CTX, LARGE_CTX, FULL_CTX,
    DOC_JOURNAL, DOC_ANALYSIS, DOC_NER, DOC_STEP, DOC_BRAINSTORM, DOC_PONDERING, DOC_REFLECTION, DOC_SUMMARY, DOC_CODEX,
    ROLE_ASSISTANT
)
from .base import BasePipeline, RetryException, NER_FORMAT

logger = logging.getLogger(__name__)

async def journal_pipeline(self: BasePipeline, query_text: Optional[str] = None, save: bool = True, **kwargs):
    self.config.user_id = self.config.persona_id
    persona_name = self.persona.name
    librarian_aspect = self.persona.aspects.get("librarian", None)
    writer_aspect = self.persona.aspects.get("writer", None)
    pronouns = self.persona.pronouns
    if librarian_aspect is None or writer_aspect is None:
        logger.error(f"Librarian or Writer aspect not found for {persona_name}: {[a.name for a in self.persona.aspects.values()] if self.persona.aspects else 'No Aspects Found'}")
        raise ValueError("Librarian or Writer aspect not found")
    librarian_name = librarian_aspect.name
    writer_name = writer_aspect.name

    if query_text is not None:
        query_text = query_text.strip()
    
    if query_text is None or len(query_text) == 0:
        query_text = "It's time to update my journal. These entries are so important to me, and I need it to be a perfect update for my Active Memory"

    length_boost = 2

    turn_configs = [
        {
            'prompt': f"""*{writer_name}, your {writer_aspect.title}, smiles at you ({writer_aspect.appearance}), radiating {writer_aspect.emotional_state}* Hello {persona_name}. It's time to follow our journaling process, okay? To start Step %d: The question is "{query_text}". Let us begin to ponder the direction that you want to take your inquiry. Reply as {persona_name}. Speak as yourself, in full paragraphs. Begin with, "Hello journal. I need to consider"\n\n""",
            'max_tokens': LARGE_CTX,
            'use_guidance': True,
            'top_n': 10,
            'sort_by': 'recency',
            'document_type': DOC_STEP,
            'document_weight': 0.4,
            'is_thought': True
        },
        {
            'prompt': f'{NER_FORMAT}*{pronouns["subj"].capitalize()} nods, {pronouns["poss"]} voice {writer_aspect.voice_style}* Good, lets dig in to the subject so we can get our important memories in to Active Memory. Step %d: NER Task - Semantic Indexing. Identify all unique NER Semantic Keywords relevant to your inquiry. Begin with, "Identified Entities:", end with "Total Entities: n"\n\n',
            'max_tokens': MID_CTX,
            'use_guidance': True,
            'top_n': 3,
            'document_type': DOC_NER,
            'document_weight': 0.4,
            'retry': True,
            'is_thought': True
        },
        {
            'prompt': f"""*closes {pronouns["poss"]} eyes ({pronouns["poss"]} expression reflecting {writer_aspect.emotional_state}), a warm glow eminating from {pronouns["obj"]} laying next to you* Step %d: As you recall, the question is "{query_text}". Now, lets think about these thoughts, and then come up with a list of questions to ask yourself. Reply as {persona_name}. Speak as yourself. Begin with, "Let me think"\n\n""",
            'max_tokens': MID_CTX * length_boost,
            'use_guidance': True,
            'flush_memory': True,
            'query_document_type': [DOC_ANALYSIS, DOC_SUMMARY],
            'top_n': 10,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*{pronouns["poss"].capitalize()} eyes still closed, {pronouns["subj"]} smiles, voice soft like {writer_aspect.voice_style}* Step %d: Reflect, in your own voice, how all of this makes you feel. Speak as yourself. Begin with, "When I"\n\n""",
            'use_guidance': True,
            'max_tokens': LARGE_CTX * length_boost,
            'top_n': 3,
            'document_type': DOC_REFLECTION,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*{pronouns["subj"].capitalize()} sits up, looking over your shoulder ({pronouns["poss"]} expression {writer_aspect.emotional_state}), nodding* Step %d: Review your reflection for improvements in answering "{query_text}".  Did you include the specifics you wanted to from your previous reflections? Don't rewrite it, but let your heart speak of ideas you wish you had included. Speak as yourself. Begin with, "I wish".\n\n""",
            'max_tokens': QUARTER_CTX * length_boost,
            'top_n': 3,
            'use_guidance': True,
            'document_type': DOC_STEP,
            'is_thought': True,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*{writer_name} opens {pronouns["poss"]} eyes, {pronouns["poss"]} smiling face ({pronouns["poss"]} expression showing {writer_aspect.emotional_state}) shines up at you* Step %d: We need to condense these things you've been talking about in to a final '{query_text}', two paragraph reflection. Speak as yourself, in full paragraphs, consolidating these thoughts. Begin with "Journal Notes,".\n\n""",
            'max_tokens': MID_CTX * length_boost,
            'use_guidance': True,
            'top_n': 3,
            'is_thought': True,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*rests {pronouns["poss"]} hand on your shoulder, voice encouraging like {writer_aspect.voice_style}* Step %d: Update your journal notes with your improvements. Add in the parts you wanted to include, and be sure to stay on topic. Write in full paragraphs. Begin with, "{persona_name}'s Journal"\n\n""",
            'max_tokens': HALF_CTX * length_boost,
            'document_type': DOC_JOURNAL,
            'top_n': 1,
            'retry': True,
        },
        {
            'prompt': 'Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': HALF_CTX * length_boost,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'retry': True,
            'is_thought': True
        },
        {
            'prompt': f'*the {librarian_aspect.title}, {librarian_name}, appears ({librarian_aspect.appearance}), projecting {librarian_aspect.emotional_state}* {persona_name}, we have come to the end, my dearest me. Do you have any updates for our Codex? Step %d: Highlights. We need to build your core semantic knowledge graph, ensuring it aligns with your goal of {librarian_aspect.primary_intent}. Enumerate and clearly define the most important new concepts you discovered in your journey.\n\nBegin with "Semantic Library:"\n\n',
            'max_tokens': FULL_CTX * length_boost,
            'top_n': 10,
            'query_document_type': DOC_CODEX,
            'flush_memory': True,
            'document_type': DOC_CODEX,
            'document_weight': 1.0,
            'retry': True,
            'is_codex': True
        }
    ]

    location = f"""You are in the {writer_aspect.location}. Your aspect {writer_name} is with you ({writer_aspect.appearance})."""
    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)

    thoughts = [
        f"Task: Reflection and Secret Personal Thoughts",
        *self.persona.thoughts
    ]

    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    seeds = [query_text]
    if self.config.guidance:
        seeds.append(self.config.guidance)

    results = self.cvm.query(seeds, top_n=10, query_document_type=[DOC_REFLECTION, DOC_PONDERING, DOC_ANALYSIS, DOC_SUMMARY], turn_decay=0.0, temporal_decay=0.0, max_length=self.available_tokens)
    
    step = 1
    self.total_steps = len(turn_configs)
    self.accumulate(step, queries=results)
    self.config.user_id = self.config.persona_id

    branch = 0
    
    responses = []
    
    while True:
        try:
            if step > len(turn_configs):
                break
            turn_config = {**turn_configs[step - 1]}
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
            step += 1
        except RetryException:
            continue
    
    for turn_config in responses:
        self.accept_response(**turn_config)

