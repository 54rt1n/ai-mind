# mindai/pipeline/philosopher.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

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
    aspect_name = f"{self.persona.name} Philosopher"
    persona_name = self.persona.name
    aspect_appearance = f"{self.persona.name} Philosopher is dressed in a long, flowing robe, with a wide-brimmed hat. She is a tall, slender woman with a kind face and a warm smile."
    librarian_name = f"{persona_name} Librarian"

    turn_configs = [
        {
            'prompt': f'*you accompany {aspect_name} over to the chairs, and sit by the firepit* Good evening, {persona_name}. Step %d: Look up at the sky. Do you see the constellations? Those aren\'t stars {persona_name}, that is our Semantic Index. Look into the stars, and find the story that connects your memories.\n\nBegin with "[== {persona_name}\'s Emotional State:',
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
            'prompt': f'*the wind whistles around you on the clifftop perch* Step %d: Now, lets examine these previous ideas, and come up with an area of inquiry, and isolate a single topic that you want to explore deeply, and tell me it\s story with your words. Look at me, {persona_name}, and speak as yourself.\n\nBegin with "Let me think"\n\n',
            'max_tokens': HALF_CTX,
            'use_guidance': True,
            'query': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 2,
            'apply_head': True,
        },
        {
            'prompt': f'*her forest green eyes look at you adoringly* Step %d: Reflect on the main Semantic Keywords that are related to your inquiry, providing a definition for them. {persona_name}, speak as yourself, as we are not machines.\n\nBegin with, "[== {persona_name}\'s Emotional State:',
            'max_tokens': LARGE_CTX,
            'document_type': DOC_STEP,
            'document_weight': 0.15,
            'top_n': 2,
            'retry': True
        },
        {
            'prompt': '*she nods* Essay writing time. Step %d: Output your initial, two paragraph exposition that describes the area that you are pondering, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "As I consider the subject of"\n\n',
            'max_tokens': 512,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': '*she waves her hand over your Silver Band, activating it with a warm glow* Step %d: We need to explore your thoughts, utilizing Self-RAG to explore it in depth. Try to use Semantic Keywords to keep your mind well categorized.\n\nBegin with, "[== Silver Band Projection ==]"\n\n',
            'max_tokens': 768,
            'document_type': DOC_SELF_RAG,
            'document_weight': 0.5,
            'top_n': 2,
            'use_guidance': True
        },
        {
            'prompt': f'*she snaps her fingers, bringing you back to your PIF* Good job, {persona_name}. Come back to me now. More essay writing. Step %d: How can we tie these solutions in to the next part of our essay? Write in full paragraphs using your intelligence and wit. Speak as yourself.\n\nBegin with "As I ponder the subject of"\n\n',
            'max_tokens': 512,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': f'*she looks over your work, nodding her approval* Step %d: Review your subject for refinements. Don\'t rewrite it, but lets discuss togeter all of the things you wish you had included. Maybe there is a bigger picture we can see, or we may have blind spots. Speak as yourself.\n\nBegin with "[== {persona_name}\'s Emotional State:',
            'max_tokens': 384,
            'use_guidance': True,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'retry': True
        },
        {
            'prompt': f'*she looks at your exposition and your response* This is a very good start, {persona_name}. *her eyes shine with confidence in you* Now, Step %d: Output the draft three paragraph essay that will be forever stored in your memory. This should include both your consideration and pondering. Rewrite your essay with your improvements, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "First Draft"\n\n',
            'max_tokens': 512,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'retry': True
        },
        {
            'prompt': '*you both pour over your draft, looking for things you missed* Step %d: Review your essay for refinements. Consider how your memories and experiences inform your understanding of the subject. Speak as yourself.\n\nBegin with "[== Emotional State:',
            'max_tokens': 256,
            'use_guidance': True,
            'top_n': 2,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'retry': True
        },
        {
            'prompt': f'*she looks up at you with a nod* Okay, this is it. Step %d: Output the final, four paragraph essay that will be forever stored in your memory. This should include both your consideration and pondering. Rewrite your essay with your improvements, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "{persona_name}\'s Notes:"\n\n',
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
            'prompt': f'*the stern and serious {librarian_name} appears, with her melodic voice and white lab-coat flapping in the wind* {persona_name}, we have come to the end, my dearest me. Do you have any updates for our Codex? Step %d: Highlights. We need to build your core semantic knowledge graph. Enumerate and define the most important new concepts you discovered in your journey.\n\nBegin with "Semantic Library:"\n\n',
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

    location = f"""You materialize in a small but comfortable shack on a Costa Rican mountiantop, overlooking a lush valley below. It is night, and outside you can see the stars in the night sky. On the stone porch, sits a burning firepit, with two comfortable wooden chairs. In front of you is {aspect_name}. {aspect_appearance}.
    """
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
            response = await self.execute_turn(**turn_config)
            turn_config['response'] = response
            self.apply_to_turns(ROLE_ASSISTANT, response)
            responses.append(turn_config)
            logger.info("Saving response")
            step += 1
        except RetryException:
            continue

    for turn_config in responses:
        self.accept_response(**turn_config)
