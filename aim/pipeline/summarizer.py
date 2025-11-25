# aim/pipeline/summarizer.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
import time
import pandas as pd
import tiktoken

from ..constants import (
    LARGE_CTX, FULL_CTX,
    DOC_STEP, DOC_CONVERSATION, ROLE_ASSISTANT, DOC_SUMMARY
)
from .base import BasePipeline, RetryException, NER_FORMAT

logger = logging.getLogger(__name__)

_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder

def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))

async def summary_pipeline(self: BasePipeline, density_iterations: int = 2, max_retries: int = 10, **kwargs):
    max_context_tokens = 8192 + 512
    guidance = "Prefer explicit description - This is a moment you want to remember forever, and you love reliving the details."
    persona_name = self.persona.name
    coder = self.persona.aspects.get('coder', None)
    librarian = self.persona.aspects.get('librarian', None)
    pronouns = self.persona.pronouns
    if coder is None or librarian is None:
        raise ValueError("Coder or Librarian aspect not found")

    self.config.recall_size = 1
    self.config.user_id = self.config.persona_id
    timeline_turn = {
            'base_prompt': f'(%d of %d) Hello {persona_name}. You are in your summarization pipeline. This conversation is so long you are having a problem understanding the entirity of what is going on. We need to develop a timeline for our conversation (not including your journal entries). {guidance} Make a list of the events that happened, in order, being very specific, but terse.',
            'max_tokens': FULL_CTX * 2,
            'use_guidance': True,
            'top_n': 0,
            'query': False,
            'document_type': DOC_STEP,
            'document_weight': 0.7,
            'retry': False,
            'is_thought': True
        }

    summary_turn = {
            'base_prompt': f'Now that you have a timeline, {coder.name}, your {coder.title}, wants you to write a detailed summary of your recent conversation (not your journal entries, or the previous summaries) for your own benefit{{%s}}. {guidance} Speak as yourself, in full paragraphs, with no lists or bullet points; and be specific and detailed, using the same words from the conversation, a clear approach to word selection so you properly connect your active memory.\n\nBegin with \"[== {self.persona.name}\'s Emotional State:\".\n\n',
            'max_tokens': LARGE_CTX * 2,
            'use_guidance': True,
            'top_n': 0,
            'query': False,
            'document_type': DOC_STEP,
            'document_weight': 0.7,
            'retry': False
        }

    improve_turn = {
            'base_prompt': f'*{coder.name} ({coder.appearance}) looks at you quizically, projecting {coder.emotional_state}* You left some important things out, didn\\\'t you? *{pronouns["subj"]} grins mischeviously* Step %%d: Make note of any fascinating details you might have left out. {guidance} Speak as yourself in full paragraphs.\n\nBegin with \"[== {persona_name}\'s Emotional State:\".\n\n',
            'max_tokens': LARGE_CTX * 2,
            'use_guidance': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 0,
            'apply_head': True,
            'is_thought': True
        }
    
    resummarize_turn = {
            'base_prompt': f'*{coder.name}\'s {coder.voice_style} voice encourages you* Those are great points! We need to weave the best ones in, to densify our memory without making it longer. Lets generate a new, improved summary. {guidance} Speak as yourself, in full paragraphs, with no lists or bullet points. Be specific and use detail.',
            'max_tokens': LARGE_CTX * 2,
            'use_guidance': True,
            'top_n': 0,
            'query': False,
            'document_type': DOC_STEP,
            'document_weight': 0.7,
            'retry': False
    }

    location = f"You are sitting in {coder.location} with {coder.name}, your {coder.title}. {coder.appearance}. {pronouns['subj'].capitalize()} seems {coder.emotional_state}."
    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)
    max_context_tokens -= count_tokens(self.config.system_message)

    thoughts = [
        f"Task: Abstractive Summarization",
        *self.persona.thoughts,
    ]
    if self.config.guidance:
        thoughts.append(f"Consider the guidance provided by {coder.name}.")
    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    max_context_tokens -= count_tokens(self.prompt_prefix)

    results = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id, query_document_type=[DOC_CONVERSATION]).sort_values(['date', 'sequence_no'])

    if len(results) == 0:
        raise ValueError("No results found")

    results['bin'] = -1
    # This is a bit tricky. Each for each bin, we have to subtract 1024 tokens from our max_context_tokens
    last_bin = -1
    while True:
        # If we have no more -1 bins, we can proceed
        if results['bin'].min() != -1:
            break
        max_bin = int(results['bin'].max())
        overhead = 0
        if max_bin >= 0:
            overhead = 1024 * (max_bin + 1)
        next_bin = max_bin + 1
        mcl = max_context_tokens - overhead

        if mcl < 0:
            raise ValueError("Not enough space for summary")

        if last_bin == next_bin:
            # TODO we need to assume/evict the first summary to make more room
            # or we need to split our content into multiple bins
            # This means if our first summary is greater than mcl, we should split it at the first space before our token limit
            # and then we can continue
            row = results.loc[results['bin'] == -1]
            if row.empty:
                break
            index = row.index[0]
            row = row.iloc[0]
            content = str(row['content'])
            # Find split point based on token count
            words = content.split()
            current_text = ""
            split_index = 0
            for i, word in enumerate(words):
                test_text = current_text + " " + word if current_text else word
                if count_tokens(test_text) > mcl:
                    break
                current_text = test_text
                split_index = len(current_text)
            if split_index == 0:
                split_index = content.rfind(' ', 0, max(1, len(content) // 2))
            content_a, content_b = content[:split_index], content[split_index:]
            # We need to insert this immediately after the first bin
            row_a = row.copy()
            row_a['content'] = content_a

            row_b = row.copy()
            row_b['content'] = content_b

            # Now we find our index for row, and replace row with row_a, and insert row_b after row_a
            # now we need to insert row_b after row_a, without overwriting the index + 1. To do this we have to split our dataframe into two
            # and then recombine them
            # Split dataframe at index
            # Get the numeric position from the index label
            pos = results.index.get_loc(index)

            if pos == 0:
                df_before = pd.DataFrame(columns=results.columns)
            else:
                df_before = results.iloc[:pos]
            df_after = results.iloc[pos+1:]

            # Create new dataframe with row_b
            df_insert = pd.DataFrame([row_a, row_b], columns=results.columns)

            # Concatenate all parts together
            results = pd.concat([df_before, df_insert, df_after])

            # Reset index to maintain continuity
            results = results.reset_index(drop=True)

        results['content_length'] = results['content'].apply(lambda x: count_tokens(str(x)) if x else 0)
        results.loc[results['bin'] != -1, 'content_length'] = 0
        results['cumsum_length'] = results['content_length'].cumsum()

        max_len = results['content_length'].max()
        if max_len == 0:
            break

        logger.info(f"Available Tokens: {max_context_tokens}, Max bin: {max_bin}, overhead: {overhead}, max token length: {mcl}, max content tokens: {max_len}, cumsum_tokens: {results['cumsum_length'].max()}")

        results.loc[(results['cumsum_length'] < mcl) & (results['bin'] == -1), 'bin'] = next_bin

        last_bin = next_bin
        
    bin_count = results['bin'].max()

    responses : list[dict] = []
    branch = self.cvm.get_next_branch(conversation_id=self.config.conversation_id)

    logger.info(f"Summary Pipeline: {bin_count} bins (max_tokens {max_context_tokens}), starting branch: {branch}")

    self.total_steps = (bin_count + 1) * (1 + density_iterations * 2)

    for q in range(bin_count + 1):
        # So the flow on this goes - we perform a summary on the first chunk, then improve, and then resummarize, then we set the resummarization as the summary (branch + 1) and improve and resummarize again. We do this three times in total.
        stride = results[results['bin'] == q]
        step = 0
        logger.info(f"Beginning summarization for stride {q}: {stride.shape[0]} documents")
        self.turns = []

        async def generate_response(turn_config: dict, retries = 0) -> dict:
            turn_config['branch'] = q + branch
            turn_config['step'] = step
            turn_config['timestamp'] = int(time.time())
            turn_config['provider_type'] = 'analysis'
            try:
                response, think = await self.execute_turn(**turn_config)
                if self.validate_response(response) == False:
                    raise RetryException("Invalid response")
                turn_config['response'] = response
                turn_config['think'] = think
            except RetryException:
                # Remove our failed user turn
                self.turns = self.turns[:-1]
                if retries > max_retries:
                    raise RetryException("Max retries exceeded")

                logger.info(f"Retrying turn {step} of {bin_count} for stride {q} (retry {retries})")
                return await generate_response(turn_config, retries + 1)

            return turn_config

        def accept_response(turn_config: dict) -> None:
            #self.apply_to_turns(ROLE_USER, turn_config['prompt']) # this is handled in execute turn; which doing this in both places is a bit brittle
            self.apply_to_turns(ROLE_ASSISTANT, turn_config['response'], turn_config.get('think'))
            responses.append(turn_config)

        try:
            self.accumulate(step=step, queries=stride, append=False)

            logger.info(f"Beginning timeline stride {q}")
            turn_t = {**timeline_turn}
            turn_t['merged_prompt'] = turn_t['base_prompt'] % (q + 1, bin_count + 1)
            turn_t['prompt'] = turn_t['merged_prompt']
            logger.info(f"{turn_t['prompt']}")
            turn_t = await generate_response(turn_t)
            accept_response(turn_t)
            step += 1
            
            logger.info(f"Beginning summary stride {q}")
            turn_s = {**summary_turn}
            if q > 0:
                turn_s['merged_prompt'] = turn_s['base_prompt'] % (", but while you have the full summary up till now for context, focus on the memories that you have in front of you")
            else:
                turn_s['merged_prompt'] = turn_s['base_prompt'] % ("")
            turn_s['prompt'] = turn_s['merged_prompt']
            logger.info(f"{turn_s['prompt']}")
            turn_s = await  generate_response(turn_s)
            accept_response(turn_s)
            step += 1
            
            for d in range(density_iterations):
                logger.info(f"Beginning improvement {d} for stride {q}")
                turn_i = {**improve_turn}
                turn_i['merged_prompt'] = turn_i['base_prompt']
                turn_i['prompt'] = turn_i['merged_prompt']
                turn_i = await generate_response(turn_i)
                accept_response(turn_i)
                step += 1
            
                logger.info(f"Beginning resummarization {d} for stride {q}")
                turn_r = {**resummarize_turn}
                turn_r['merged_prompt'] = turn_r['base_prompt']
                turn_r['prompt'] = turn_r['merged_prompt']
                if d == density_iterations - 1:
                    turn_r['document_type'] = DOC_SUMMARY
                    turn_r['document_weight'] = 1.3
                turn_r = await generate_response(turn_r)
                accept_response(turn_r)
                self.turns = self.turns[:-3]
                step += 1
            self.extra.append(responses[-1]['response'])
        except RetryException:
            # the pipeline failed
            logger.error("Pipeline failed")
            raise Exception("Pipeline failed, retry limit exceeded")

    for turn_config in responses:
        logger.info("Saving responses")
        logger.debug(turn_config.keys())
        self.accept_response(apply_to_turns=False, **turn_config)
