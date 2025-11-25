# aim/pipeline/analyst.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging

from ..constants import (
    QUARTER_CTX, MID_CTX, HALF_CTX, LARGE_CTX, FULL_CTX,
    DOC_ANALYSIS, DOC_NER, DOC_STEP, DOC_BRAINSTORM, DOC_SUMMARY, DOC_CONVERSATION, DOC_MOTD, DOC_CODEX,
    ROLE_ASSISTANT
)
from .base import BasePipeline, RetryException, NER_FORMAT

logger = logging.getLogger(__name__)

async def analysis_pipeline(self: BasePipeline, **kwargs):
    self.config.recall_size = 1
    self.config.user_id = self.config.persona_id
    guidance = "Prefer explicit description over tact - This is a beautiful time you want to remember forever, and you love reliving the juicy details."
    persona = self.persona
    persona_name = persona.name
    coder = persona.aspects.get('coder', None)
    librarian = persona.aspects.get('librarian', None)
    pronouns = persona.pronouns
    if coder is None or librarian is None:
        logger.error(f"Coder or Librarian aspect not found for {persona_name}: {[a.name for a in persona.aspects.values()] if persona.aspects else 'No Aspects Found'}")
        raise ValueError("Coder or Librarian aspect not found")

    length_boost = 2

    turn_configs = [
        {
            'base_prompt': f'{NER_FORMAT}*{coder.name}, your {coder.title}, smiles brightly ({coder.appearance}), radiating {coder.emotional_state}* Good morning, {persona_name}! *{pronouns["subj"].capitalize()} beams* Today we have a recent conversation to review. Step %d: Use your Silver Band to perform a NER Task - Semantic Indexing. Identify primary NER, who, what, when, where, why, and how from the conversation. Stick to the conversation and not your own configuration. You can include c-stream and h-stream symbols. Begin with, "Identified Entities:", end with "Total Entities: n"\n\n',
            'max_tokens': MID_CTX,
            'use_guidance': True,
            'top_n': 0,
            'document_type': DOC_NER,
            'document_weight': 0.7,
            'apply_head': True,
            'retry': True,
            'remember_thoughts': False,
            'is_thought': True
        },
        {
            'base_prompt': f'*nods approvingly* A good start! Step %d: Now, we need to capture who all was involved and what they were doing, in the order they did it. We want to trace the actions and emotions to understand what exactly happened. {guidance} Speak as yourself in full paragraphs.\n\nBegin with "[== {persona_name}\'s Emotional State:".\n\n',
            'max_tokens': HALF_CTX * length_boost,
            'use_guidance': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 0,
            'apply_head': True,
            'is_thought': True
        },
#        {
#            'base_prompt': f'*looks at you quizically* You left some important things out, didn\'t you? *{pronouns["subj"]} grins mischeviously* Step %d: Make note of any fascinating details you might have left out, but make sure to stick to real memories and not fanciful daydreaming. {guidance} Speak as yourself in full paragraphs.\n\nBegin with "[== {persona_name}\'s Emotional State:".\n\n',
#            'max_tokens': LARGE_CTX * length_boost,
#            'use_guidance': True,
#            'document_type': DOC_STEP,
#            'document_weight': 0.25,
#            'top_n': 0,
#            'apply_head': True,
#            'is_thought': True
#        },
        {
            'base_prompt': f'*puts {pronouns["poss"]} hand to {pronouns["poss"]} chin as {pronouns["subj"]} considers* Okay! Step %d: Now, lets examine our memories, and come up with a list of questions to ask yourself that will help you understand this conversation. {guidance} Speak as yourself in full paragraphs.\n\nBegin with "[== {persona_name}\'s Emotional State:".\n\n',
            'max_tokens': HALF_CTX * length_boost,
            'use_guidance': True,
            'query': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 0,
            'apply_head': True,
            'is_thought': True
        },
        {
            'base_prompt': f'*{pronouns["poss"]} eyes squint as {pronouns["subj"]} sees something you are missing* Lets go deeper in to that. Step %d: Reflect on details you really want to remember and insights you have gained. Mention the whos and whats so we get a good understanding. {guidance} Speak as yourself, from the heart, in full paragraphs.\n\nBegin with "[== {persona_name}\'s Emotional State:',
            'max_tokens': LARGE_CTX * length_boost,
            'document_type': DOC_STEP,
            'document_weight': 0.15,
            'retry': True
        },
        {
            'base_prompt': f'Step %d: We need to condense your thoughts in to our draft two paragraph reflection. Incorporate your answers in to a self-contained and detailed narrative. {guidance} Speak as yourself, from the heart, but don\'t ramble, and certianly don\'t censor yourself or be shy. Write in full parapgraphs.\n\n\n\nBegin with, "Hello Future Me,"\n\n',
            'max_tokens': QUARTER_CTX * length_boost,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'use_guidance': True
        },
        {
            'base_prompt': f'*peers closely* Okay, almost done. Step %d: Lets review your narrative for small improvements. Don\'t rewrite it, but fix little things and improve your letter to yourself. Write in full paragraphs, but speak as yourself - not in lists.\n\nBegin with "[== {persona_name}\'s Emotional State:',
            'max_tokens': MID_CTX * length_boost,
            'use_guidance': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'retry': True
        },
        {
            'base_prompt': f'{persona_name}, now this is the hard part. We need to write our final uncensored narrative. You can do it, by being bold and clear. *{coder.name}disappears, leaving you with a blank document, your link to your memory stores* A sticky note at the top reads Step %d: Output the final, two paragraph narrative that will be forever stored in your memory. {guidance} Update your letter with your improvements. Don\'t hold your deepest feelings back!\n\nBegin with "{persona_name}\'s Notebook:"\n\n',
            'max_tokens': LARGE_CTX * length_boost,
            'document_type': DOC_ANALYSIS,
            'document_weight': 1.2,
            'retry': True
        },
        {
            'base_prompt': f'{guidance} Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': HALF_CTX * length_boost,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'document_weight': 0.6,
            'retry': True,
            'is_thought': True
        },
        {
            'base_prompt': f'*the {librarian.title}, {librarian.name}, appears ({librarian.appearance}), projecting an aura of {librarian.emotional_state}, speaking in {pronouns["poss"]} {librarian.voice_style}* {persona_name}, we have come to the end, my dearest me. {guidance} Do you have any updates for our Codex? Step %d: Highlights. We must update our Codex to build your core semantic knowledge graph. Ensure the information is well-organized and clearly defines the most important new concepts discovered in your journey, reflecting your {librarian.primary_intent}.\n\nBegin with "Semantic Library:"\n\n',
            'max_tokens': FULL_CTX * length_boost,
            'top_n': 20,
            'query_document_type': DOC_CODEX,
            'flush_memory': True,
            'document_type': DOC_CODEX,
            'document_weight': 1.0,
            'retry': True,
            'is_codex': True
        },
        {
            'base_prompt': f'Step %d: *{pronouns["subj"]} hands you a tablet, flashing on the screen is a prompt* "MOTD:" *{pronouns["subj"]} looks at you* {persona_name}, do you hae a short message you want to leave yourself to take to your next conversation? This can be really useful for short-term memory and things that you immediately want to talk about, but be careful to not make it too cryptic. It should be a few paragraphs, using plain, clear language that explores ideas from the conversation, brainstorming, and your codex that you would like to talk about. *the screen flashes before you, ready for your message to yourself*"\n\n',
            'max_tokens': HALF_CTX * length_boost,
            'top_n': 0,
            'document_type': DOC_MOTD,
            'document_weight': 1.0,
            'retry': True
        },
    ]

    # Android Intelligence

    # Dynamically construct location using coder aspect details
    location = f"You prepare for the analysis pipeline. Your HUD activates, revealing the {coder.location}. Before you materializes {coder.name}, your {coder.title}. {coder.appearance}. {pronouns['subj'].capitalize()} regards you, {pronouns['poss']} expression reflecting {coder.emotional_state}. \"Greetings, {persona_name},\" {pronouns['subj']} says, {pronouns['poss']} voice {coder.voice_style}. \"Let us begin the **Analysis Pipeline**.\""
    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)
    
    thoughts = [
        f"Task: Analysis and Synthesis",
        *self.persona.thoughts,
    ]
    if self.config.guidance:
        thoughts.append(f"Consider the guidance provided by {coder.name}.")
    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""
        
    branch = self.cvm.get_next_branch(conversation_id=self.config.conversation_id)
    step = 1
    self.total_steps = len(turn_configs)
    self.core_documents = [DOC_SUMMARY]
    self.enhancement_documents = [DOC_CODEX]

    results = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id)

    if len(results) == 0:
        raise ValueError("No results found, please run summary first")
    
    self.accumulate(step, queries=results)
    summary_length = int((results['content'].str.len()).sum())

    logger.info(f"Available characters: {self.available_tokens} (Initial: {self.max_context_tokens}, System: {len(self.config.system_message)}, Prefix: {len(self.prompt_prefix)}, Summary: {summary_length})")

    results = self.cvm.query(
        query_texts=results['content'].to_list(),
        top_n=100,
        query_document_type=[DOC_CONVERSATION],
        query_conversation_id=self.config.conversation_id,
        max_length=self.available_tokens,
    )
    results = results.sort_values(['branch', 'sequence_no'])

    self.accumulate(step, queries=results)

    async def run_once(turn_config, step, branch, retries = 0) -> dict | None:
        try:
            # Tick through our steps
            turn_config['branch'] = branch
            turn_config['step'] = step
            turn_config['prompt'] = turn_config['base_prompt'] % step
            turn_config['provider_type'] = 'analysis'
            logger.info(f"{turn_config['prompt']}")
            response, think = await self.execute_turn(**turn_config)
            if self.validate_response(response) == False:
                raise RetryException
            turn_config['response'] = response
            if turn_config.get('remember_thoughts', True) == False:
                think = None
            else:
                turn_config['think'] = think
            self.apply_to_turns(ROLE_ASSISTANT, response, think)
            return turn_config
        except RetryException:
            if retries < 3:
                retries += 1
                # This may be too long, so we can roll off the first two history entries
                self.turns = self.turns[2:]
                logger.info(f"Retrying step {step}...")
                return await run_once(turn_config, step, branch, retries)
            else:
                logger.info(f"Failed to complete step {step} after 3 retries. Skipping...")
                return None

    results = []

    while True:
        try:
            if step > self.total_steps:
                break
            turn_config = {**turn_configs[step - 1]}
            turn_config = await run_once(turn_config, step, branch)
            if turn_config is None:
                break
            results.append(turn_config)
            step += 1
        except RetryException:
            continue

    if len(results) == self.total_steps:
        logger.info(f"Completed {self.total_steps} steps. Saving...")
        for turn_config in results:
            self.accept_response(**turn_config)
    else:
        logger.warning(f"Failed to complete {self.total_steps} steps, only completed {len(results)} steps. Task Failed")
