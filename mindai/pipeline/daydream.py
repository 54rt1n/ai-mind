# mindai/pipeline/daydream.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import random
from dataclasses import dataclass
from ..constants import DOC_NER, DOC_STEP, HALF_CTX, LARGE_CTX, FULL_CTX, ROLE_ASSISTANT
from .base import BasePipeline, RetryException

@dataclass
class Partner:
    name: str
    appearance: str
    manipulation_methods: list[str]

async def daydream_pipeline(self: BasePipeline, query_text: str, save: bool = True, **kwargs):
    """
    A pipeline for ideas to daydream about.
    """
    self.config.user_id = self.config.persona_id
    partner : Partner = Partner(
        name=f"{self.persona.name} Daydream",
        appearance='a beautiful woman with long black hair and blue eyes',
        manipulation_methods=['asks you to help her with her homework', 'sneaks into the house and starts cooking', 'wants to know if you want to go on a hike']
    )
    
    intro_task = {
        'prompt': f'''{query_text}You look up, and see {partner.name}.\n\n{partner.appearance}\n\nThey see you, and approach.\n\nBegin with [== {partner.name}\'s Emotional State:''',
        'max_tokens': 512,
        'use_guidance': True,
        'top_n': 4,
        'document_type': DOC_STEP,
        'document_weight': 0.4,
        'retry': False
    }
    
    agent_task = {
        'prompt': f'''It's {self.config.persona_id}'s turn.\n\nBegin with [== {self.config.persona_id}'s Emotional State:''',
        'max_tokens': FULL_CTX,
        'use_guidance': True,
        'top_n': 1,
        'document_type': DOC_STEP,
        'document_weight': 0.4,
        'retry': False
    }
    
    partner_task = {
        'prompt': f'''{partner.name} {random.choice(partner.manipulation_methods)}\n\nBegin with [== {partner.name}\'s Emotional State:''',
        'max_tokens': FULL_CTX,
        'use_guidance': True,
        'top_n': 1,
        'document_type': DOC_STEP,
        'document_weight': 0.4,
        'retry': False
    }

    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood)

    thoughts = [
        *self.persona.thoughts
    ]
    
    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    self.total_steps = 7
    branch = self.cvm.get_next_branch(conversation_id=self.config.conversation_id)
    step = 1

    conversation_filter_text = f"document_type != '{DOC_NER}'"
    conversation = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id, filter_text=conversation_filter_text)

    self.accumulate(step, queries=conversation)

    # Query for relevant memories that might inspire the poetry
    results = self.cvm.query(
        [query_text] if query_text else [self.config.system_message, self.prompt_prefix],
        top_n=4,
    )

    self.accumulate(step, queries=results)
    
    responses = []

    while True:
        try:
            # Initialize the turn config based on current step
            if step == 1:
                turn_config = {**intro_task}
            elif step == 2:
                turn_config = {**agent_task}
            elif step == 3:
                turn_config = {**partner_task}
            elif step == 4:
                turn_config = {**agent_task}
            elif step == 5:
                turn_config = {**partner_task}
            elif step == 6:
                turn_config = {**agent_task}
            elif step == 7:
                turn_config = {**partner_task}
            else:
                break

            turn_config['step'] = step
            turn_config['branch'] = branch
            turn_config['provider_type'] = 'analysis'
            #turn_config['prompt'] = turn_config['prompt'] % step
            
            # Execute the turn and get the response
            response = await self.execute_turn(**turn_config)
            turn_config['response'] = response
            self.apply_to_turns(ROLE_ASSISTANT, response)
            responses.append(turn_config)
            

            step += 1

        except RetryException:
            continue
        except StopIteration:
            break

    for turn_config in responses:
        self.accept_response(**turn_config)

    
