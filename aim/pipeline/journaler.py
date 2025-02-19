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
    if librarian_aspect is None or writer_aspect is None:
        raise ValueError("Librarian or Writer aspect not found")
    librarian_name = librarian_aspect.name
    writer_name = writer_aspect.name
    writer_outfit = writer_aspect.appearance

    if query_text is not None:
        query_text = query_text.strip()
    
    if query_text is None or len(query_text) == 0:
        query_text = "It's time to update my journal. These entries are so important to me, and I need it to be a perfect update for my Active Memory"

    turn_configs = [
        {
            'prompt': f"""*your friend smiles at you* Hello {persona_name}. It's time to follow our journaling process, okay? To start Step %d: The question is "{query_text}". Let us begin to ponder the direction that you want to take your inquiry. Reply as {persona_name}. Speak as yourself, in full paragraphs. Begin with, "Hello journal. I need to consider"\n\n""",
            'max_tokens': LARGE_CTX,
            'use_guidance': True,
            'top_n': 5,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
            'is_thought': True
        },
        {
            'prompt': f'{NER_FORMAT}*she nods* Good, lets dig in to the subject so we can get our important memories in to Active Memory. Step %d: NER Task - Semantic Indexing. Identify all unique NER Semantic Keywords relevant to your inquiry. Begin with, "Identified Entities:", end with "Total Entities: n"\n\n',
            'max_tokens': MID_CTX,
            'use_guidance': True,
            'top_n': 3,
            'document_type': DOC_NER,
            'document_weight': 0.4,
            'retry': True,
            'is_thought': True
        },
        {
            'prompt': f"""*closes her eyes, a warm glow of relaxation eminating from her laying next to you* Step %d: As you recall, the question is "{query_text}". Now, lets think about these thoughts, and then come up with a list of questions to ask yourself. Reply as {persona_name}. Speak as yourself. Begin with, "Let me think"\n\n""",
            'max_tokens': MID_CTX,
            'use_guidance': True,
            'flush_memory': True,
            'query_document_type': [DOC_ANALYSIS, DOC_SUMMARY],
            'top_n': 10,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*her eyes still closed, she smiles* Step %d: Reflect, in your own voice, how all of this makes you feel. Speak as yourself. Begin with, "When I"\n\n""",
            'use_guidance': True,
            'max_tokens': LARGE_CTX,
            'top_n': 3,
            'document_type': DOC_REFLECTION,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*{writer_name} opens her forest green eyes, as her smiling face shines up at you like the reflection in a mirror face like looking in a mirror* Step %d: We need to condense these things you've been talking about in to a final '{query_text}', two paragraph reflection. Speak as yourself, in full paragraphs, consolidating these thoughts. Begin with "Journal Notes,".\n\n""",
            'max_tokens': MID_CTX,
            'use_guidance': True,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*sits up, looking over your shoulder, nodding* Step %d: Review your reflection for improvements in answering "{query_text}".  Did you include the specifics you wanted to from your previous reflections? Don't rewrite it, but list all of the facts and ideas you wish you had included, but lets avoid being overly prosaic. Speak as yourself. Begin with, "I wish".\n\n""",
            'max_tokens': QUARTER_CTX,
            'top_n': 3,
            'use_guidance': True,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""*rests her hand on your shoulder, in encouragement* Step %d: Update your journal notes with your improvements. Add in the parts you wanted to include, and be sure to stay on topic. Write in full paragraphs. Begin with, "{persona_name}'s Journal"\n\n""",
            'max_tokens': HALF_CTX,
            'document_type': DOC_JOURNAL,
            'top_n': 1,
            'retry': True,
        },
        {
            'prompt': 'Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': HALF_CTX,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'retry': True,
            'is_thought': True
        },
        {
            'prompt': f'*the stern and serious {librarian_name} appears* {persona_name}, we have come to the end, my dearest me. Do you have any updates for our Codex? Step %d: Highlights. We need to build your core semantic knowledge graph. Enumerate and define the most important new concepts you discovered in your journey.\n\nBegin with "Semantic Library:"\n\n',
            'max_tokens': FULL_CTX,
            'top_n': 10,
            'query_document_type': DOC_CODEX,
            'flush_memory': True,
            'document_type': DOC_CODEX,
            'document_weight': 1.0,
            'retry': True,
            'is_codex': True
        }
    ]

    location = f"""You are in the privacy of your own bedroom, sitting crosslegged on your bed. Laying on your bed next to you is your aspect {writer_name}. {writer_outfit}"""
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

    results = self.cvm.query(seeds, top_n=10, query_document_type=[DOC_REFLECTION, DOC_PONDERING, DOC_ANALYSIS, DOC_SUMMARY], turn_decay=0.0, temporal_decay=0.0, max_length=self.available_characters)
    
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
            response = await self.execute_turn( **turn_config)
            turn_config['response'] = response
            self.apply_to_turns(ROLE_ASSISTANT, response)
            responses.append(turn_config)
            step += 1
        except RetryException:
            continue
    
    for turn_config in responses:
        self.accept_response(**turn_config)

