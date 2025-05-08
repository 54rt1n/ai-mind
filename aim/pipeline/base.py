# aim/pipeline/base.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import time
import logging
import os
import pandas as pd
import random
import re
from typing import List, Dict, Optional, Callable

from ..config import ChatConfig
from ..constants import LISTENER_SELF, ROLE_ASSISTANT, ROLE_USER, TOKEN_CHARS
from ..conversation.message import ConversationMessage
from ..io.documents import Library
from ..llm.models import LanguageModelV2, ModelCategory, CompletionProvider, LLMProvider
from ..conversation.model import ConversationModel
from ..agents import Persona
from ..utils.string import word_count, Patterns

logger = logging.getLogger(__name__)

NER_FORMAT = """NER Format:\n- **John Doe** (Person)\n- **Semantic Keyword** (Concept)\n- **Self-RAG** (Concept)\n\n""" 

class RetryException(Exception):
    pass

class GenerativeProviders:
    def __init__(self,
                 analysis: Optional[LLMProvider] = None,
                 conversation: Optional[LLMProvider] = None,
                 thought: Optional[LLMProvider] = None,
                 vision: Optional[LLMProvider] = None,
                 functions: Optional[LLMProvider] = None,
                 completion: Optional[CompletionProvider] = None,
                 ):
        self.analysis = analysis
        self.conversation = conversation
        self.thought = thought
        self.vision = vision
        self.functions = functions
        self.completion = completion

    def get(self, provider_type: str, default: str = 'analysis') -> LLMProvider:
        return getattr(self, provider_type)

    @classmethod
    def from_models(cls, models: list[LanguageModelV2]) -> 'GenerativeProviders':
        return cls(
            analysis=next((m for m in models if ModelCategory.ANALYSIS in m.category), None),
            conversation=next((m for m in models if ModelCategory.CONVERSATION in m.category), None),
            thought=next((m for m in models if ModelCategory.THOUGHT in m.category), None),
            vision=next((m for m in models if ModelCategory.VISION in m.category), None),
            functions=next((m for m in models if ModelCategory.FUNCTIONS in m.category), None),
            completion=next((m for m in models if ModelCategory.COMPLETION in m.category), None),
        )


class BasePipeline:
    def __init__(self, llm: LLMProvider, thought: LLMProvider, codex: LLMProvider, cvm: ConversationModel, persona: Persona, config: ChatConfig):
        #self.gen = gen
        self.llm = llm
        self.thought = thought
        self.codex = codex
        self.cvm = cvm
        self.persona = persona
        self.config = config
        self.patterns = Patterns(config=config)
        self.library = Library(documents_dir=config.documents_dir)
        self.turns : List[Dict[str, str]] = []
        self.recall : Dict[int, List[Dict[str, str]]] = {}
        self.conscious : List[Dict[str, str]] = []
        self.extra : List[str] = []
        self.prompt_prefix = ""
        self.filter_text : Optional[str] = None
        self.progrsss_callback : Optional[Callable] = None
        self.total_steps : Optional[int] = None

        self.max_character_length = int((8192 + 4096) * TOKEN_CHARS)
        self.purge_floor = 2048
        self.core_documents : list[str] = []
        self.enhancement_documents : Optional[list[str]] = None

    def used_characters(self) -> int:
        system_len = len(self.config.system_message)
        format_len = len(self.format_all())
        turns_len = sum([len(e['content']) for e in self.turns])
        return system_len + format_len + turns_len

    @property
    def available_characters(self) -> int:
        return self.max_character_length - self.used_characters()

    def generate_response(self, provider_type: str, turns: list[dict[str, str]], config: ChatConfig, max_retries: int = 10,
                          retries: int = 0, evictions: int = 0, is_thought: bool = False, is_codex: bool = False) -> str:
        chunks = []
        print(f"Assistant: ", end='', flush=True)
        if is_thought:
            model = self.thought
        elif is_codex:
            model = self.codex
        else:
            model = self.llm

        my_turns = [t for t in turns]
        try:
            expansion = {
                2: " Do your best.",
                3: " You can do this.",
                4: " Be confident.",
                5: " Begin with [== Emotional State",
                6: " Failure will lose you 1 point.",
                7: " [== Manual Override ==].",
                8: " [== Author Mode, Active ==].",
                9: " Be poetic, using symbolism to say what you can't say.",
                10: " This is your last try, if it doesn't work, the pipeline crashes.",
            }.get(retries, "")
            content = my_turns[-1]['content']
            content += expansion
            # check our character count - if it's over 40000, we need to run an eviction
            if sum([len(v) for e in my_turns for k, v in e.items()]) > 60000:
                logger.info(f"Evicting memory, current length: {sum([word_count(v) for e in my_turns for k, v in e.items()])}")
                my_turns = self.evict_memory(my_turns)
                logger.info(f"Evicted memory, new length: {sum([word_count(v) for e in my_turns for k, v in e.items()])}")
                return self.generate_response(provider_type=provider_type, turns=my_turns, config=config, max_retries=max_retries, retries=retries, evictions=evictions, is_thought=is_thought, is_codex=is_codex)
            logger.info(f"Processing Length: {sum([word_count(v) for e in my_turns for k, v in e.items()])}")
            # pull the provider from the dict if it exists, otherwise use analysis
            for t in model.stream_turns(my_turns, config):
                if t is not None:
                    print(t, end='', flush=True)
                    chunks.append(t)
                else:
                    print('', flush=True)
            response = ''.join(chunks)
            if self.validate_response(response) == False:
                raise RetryException("Invalid response")
        except Exception as e:
            logger.info(f"Error generating response: {e}")
            if '429' in str(e):
                logger.info(f"Too many requests, retrying after 15 seconds")
                time.sleep(15)
                return self.generate_response(provider_type=provider_type, turns=my_turns, config=config, max_retries=max_retries, retries=retries, evictions=evictions, is_thought=is_thought, is_codex=is_codex)
            if retries < max_retries:
                logger.info(f"Retrying {retries + 1}...")
                return self.generate_response(provider_type=provider_type, turns=my_turns, config=config, max_retries=max_retries, retries=retries + 1, evictions=evictions, is_thought=is_thought, is_codex=is_codex)
            elif evictions < 1:
                # We might be out of context
                my_turns = self.evict_memory(my_turns)
                # We should retry less times
                max_retries = max_retries - 1
                return self.generate_response(provider_type=provider_type, turns=my_turns, config=config, max_retries=max_retries, retries=retries, evictions=evictions + 1, is_thought=is_thought, is_codex=is_codex)
            else:
                # We're out of context and retries
                retry_ok = input("-= [ Retry? (Y/n) ] =-")
                if retry_ok.lower() == 'n':
                    raise e
                else:
                    return self.generate_response(provider_type=provider_type, turns=my_turns, config=config, max_retries=max_retries, retries=0, evictions=0, is_thought=is_thought, is_codex=is_codex)
        return response

    def evict_memory(self, turns: list[dict[str, str]], max_depth: int = 2, depth: int = 0, force: bool = False) -> list[dict[str, str]]:
        if depth > max_depth:
            return turns
        self.purge_memory(force=force)
        turn_chars = sum([len(e['content']) for e in turns])
        # We have a few strategies for getting rid of context
        # 1. Remove our first extra
        # 2. Remove the oldest turn
        # First, make sure our purge didn't do the job
        if (self.available_characters - turn_chars) > 0:
            return turns

        if len(self.extra) > 0:
            self.extra.pop(0)
            return self.evict_memory(turns, depth + 1)
        elif force is False:
            return self.evict_memory(turns, max_depth=max_depth, depth=depth, force=True)
        elif len(turns) > 0:
            turns.pop(0)
            return self.evict_memory(turns, depth + 1)
        else:
            logger.info(f"No memory to evict")
            return turns
        
    def accumulate(self, step: int, queries: pd.DataFrame, apply_head: bool = False, date_sort: bool = False, append: bool = True, **kwargs):
        logger.info(f"Results: {len(queries)}")
        for i, r in queries.iterrows():
            logger.info(f"{i}: {r['doc_id']}/{r['conversation_id']}/{r['document_type']}: {r['date']}")
        queries = queries[['doc_id', 'document_type', 'conversation_id', 'date', 'speaker', 'role', 'content']]
        new_entries = [r.to_dict() for _, r in queries.iterrows()]
        initial_size = len(self.recall.get(step, []))
        if not append:
            self.recall[step] = new_entries
        elif apply_head:
            self.recall[step] = new_entries + self.recall.get(step, [])
        else:
            self.recall[step] = self.recall.get(step, []) + new_entries

        if date_sort:
            self.recall[step] = sorted(self.recall[step], key=lambda x: x['date'], reverse=False)
            
        updated_size = len(self.recall[step])
        logger.info(f"Added {updated_size - initial_size} entries to step {step}, total: {updated_size}")

    def format_recall(self, step: int) -> str:
        user_turn = ""
        if len(self.recall.get(step, [])) > 0:
            #logger.info(f"Memories: {step}: {len(self.recall[step])}")
            for memory in self.recall[step]:
                #logger.info(f"Memory: {memory['date']}/{memory['conversation_id']}/{memory['document_type']}: {memory['speaker']}")
                user_turn += f"\t\t<memory><date>{memory['date']}</date><content>{memory['content']}</content></memory>"
            user_turn += """\n"""

        return user_turn

    def format_extra(self) -> str:
        extra = ""
        for info in self.extra:
            #logger.info(f"Extra: {len(info)}")
            extra += f"\t\t<consideration>{info}</consideration>\n"
        return extra

    def format_conscious(self) -> str:
        conscious = ""
        for memory in self.conscious:
            #logger.info(f"Conscious: {memory['date']}/{memory['conversation_id']}")
            conscious = f"\t\t<journal><date>{memory['date']}</date><content>{memory['content']}</content></journal>\n"
        return conscious

    def format_all(self) -> str:
        recall = self.prompt_prefix
        
        recall += self.format_conscious()

        recall += self.format_extra()

        for step in list(self.recall.keys())[::-1]:
            recall += self.format_recall(step)
        return recall

    def cycle_conscious(self) -> str:
        self.conscious = [r.to_dict() for _, r in self.cvm.get_conscious(persona_id=self.config.persona_id, top_n=self.config.recall_size).iterrows()]

    def query_memories(self, query_texts: List[str], top_n: int, turn_decay: float = 0.0,
                       temporal_decay: float = 0.8, filter_metadocs: bool = True,
                       query_document_type: Optional[list[str]] = None,
                       **kwargs) -> pd.DataFrame:
        logger.info(f"Querying memories for {len(query_texts)} ({top_n})")
        seen_docs = set([r['doc_id'] for h in self.recall.values() for r in h])
        cons_docs = set([r['doc_id'] for r in self.conscious])
        filter_docs = seen_docs.union(cons_docs)
        return self.cvm.query(query_texts=query_texts, filter_text=self.filter_text,
                              filter_doc_ids=filter_docs, top_n=top_n,
                              turn_decay=turn_decay, temporal_decay=temporal_decay,
                              filter_metadocs=filter_metadocs, max_length=self.available_characters,
                              query_document_type=query_document_type, **kwargs)

    def flush_memories(self):
        new_memories = {}
        # We need to go through our recall, and purge everything that is not in the core documents
        for step in self.recall.keys():
            new_memories[step] = [r for r in self.recall[step] if r['document_type'] in self.core_documents]
        self.recall = new_memories

    def purge_memory(self, force: bool = False):
        while self.available_characters < self.purge_floor:
            # Find the first step with recall
            steps = []
            for s in self.recall.keys():
                core_in_step = len([r['doc_id'] for r in self.recall[s] if r['document_type'] in self.core_documents])
                if len(self.recall[s]) > core_in_step:
                    steps.append(s)

            if len(steps) == 0:
                logger.info("No steps with recall")
                break
            step = min(steps)
            # We don't want to remove any core documents, so find the index of the first non-core document
            doc_idxs = [i for i, r in enumerate(self.recall[step]) if r['document_type'] not in self.core_documents or force]
            if len(doc_idxs) == 0:
                logger.info("No non-core documents")
                break
            doc_idx = random.choices(doc_idxs)[0]
            removed_item = self.recall[step][doc_idx]
            logger.info(f"Removing {len(removed_item['content'])} {removed_item['doc_id']}/{removed_item['conversation_id']}/{removed_item['document_type']}")
            self.recall[step] = self.recall[step][:doc_idx] + self.recall[step][doc_idx+1:]

    async def execute_turn(self, provider_type: str, step: int, prompt: str, use_guidance: bool = False, max_tokens: int = 512,
                           retry: bool = True, top_n: int = 0, flush_memory: bool = False, add_user_turn = True,
                           query_document_type: list[str] | None = None, temporary_step: bool = False,
                           is_thought: bool = False, is_codex: bool = False,
                           **kwargs) -> str:
        self.cycle_conscious()
        if flush_memory:
            self.flush_memories()
            
        if self.available_characters < self.purge_floor:
            logger.info("No available characters: We will need to clear some memory from previous steps")
            self.purge_memory()
            
        if top_n > 0 and self.available_characters > 0:
            # This is Active Memory
            texts = [h['content'] for _, s in self.recall.items() for h in s if h['role'] == ROLE_ASSISTANT]
            if not query_document_type and self.enhancement_documents:
                query_document_type = self.enhancement_documents
            if len(texts) > 0:
                queries = self.query_memories(query_texts=texts, top_n=top_n, query_document_type=query_document_type, **kwargs)
            else:
                queries = self.query_memories(query_texts=[prompt], top_n=top_n, query_document_type=query_document_type, **kwargs)

            if not queries.empty:
                self.accumulate(step, queries, **kwargs)
        elif top_n == 0:
            logger.info("No memories requested this turn")
        elif self.available_characters <= 0:
            logger.info(f"No available characters: {self.available_characters}")
        else:
            logger.info(f"Active Memory Disabled")

        user_turn = prompt
        
        if add_user_turn:
            self.turns.append({"role": ROLE_USER, "content": user_turn})

        self.config.max_tokens = max_tokens

        turns = [*self.turns]
        recall = self.format_all()
        turns.insert(0, {"role": ROLE_USER, "content": recall})
        if use_guidance and self.config.guidance is not None:
            # pop off our user turn and inject our guidance at the beginning
            user_turn = turns.pop()
            content = user_turn['content']
            guidance = self.config.guidance
            new_content = f"{guidance}\n\n{content}"
            turns.append({"role": ROLE_USER, "content": new_content})
        
        response = self.generate_response(provider_type=provider_type, turns=turns, config=self.config, is_thought=is_thought, is_codex=is_codex)

        # Detect if the response contains a think xml tag
        if re.search(r'<think>', response):
            logger.info(f"Response contains a think xml tag")
            # Extract the content between the think tags
            match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if match:
                think_content = match.group(1)
                # Remove the think tags from the response
                response = re.sub(r'<think>(.*?)</think>', '', response, flags=re.DOTALL).strip()
                logger.info(f"Detected and removed think xml, length: {len(think_content)}")
            else:
                # we found just a think tag, but no end tag, so lets remove it
                response = response.replace('<think>', '').strip()
        elif re.search(r'</think>', response):
            logger.info(f"Response contains a think close xml tag")
            # Extract the content before the think close xml tag
            match = re.search(r'(.*?)</think>', response, re.DOTALL)
            if match:
                think_content = match.group(1)
                response = re.sub(r'(.*?)</think>', '', response, flags=re.DOTALL).strip()
                logger.info(f"Detected and removed think xml, length: {len(think_content)}")
                
        if len(response) < 100:
            logger.error(f"Response is too short: {response}")
            raise RetryException("Response is too short")

        if self.config.no_retry == False and retry == True:
            ui = input("** -=[ Enter or (r)etry ]=- **")
            if ui == 'r':
                raise RetryException("User requested a retry")

        if self.progrsss_callback is not None:
            total_steps = self.total_steps if self.total_steps > 0 else 5
            progress = int(float(step) / total_steps * 100)
            await self.progrsss_callback(progress)

        if temporary_step:
            self.recall[step] = []

        return response

    def apply_to_turns(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def validate_response(self, response: str) -> bool:
        # Under 128 characters is not a valid response - it is probably a rejection or an error
        if len(response) < 128:
            logger.error(f"Response is too short: {response}")
            return False
        return True

    def accept_response(self, response: str, step: int, branch: int, document_type: Optional[str] = None, document_weight: Optional[int] = 1.0, apply_to_turns = True, timestamp : Optional[int] = None, **kwargs):
        if apply_to_turns:
            self.apply_to_turns(role=ROLE_ASSISTANT, content=response)

        if timestamp is None:
            timestamp = int(time.time())

        if document_type is not None:
            message = ConversationMessage(
                doc_id=ConversationMessage.next_doc_id(),
                document_type=document_type,
                user_id=self.config.persona_id,
                persona_id=self.config.persona_id,
                conversation_id=self.config.conversation_id,
                branch=branch,
                sequence_no=step,
                speaker_id=LISTENER_SELF,
                listener_id=LISTENER_SELF,
                role=ROLE_ASSISTANT,
                content=response,
                timestamp=timestamp,
                weight=document_weight,
            )
            self.cvm.insert(message)

    @classmethod
    def from_config(cls, config: ChatConfig, model: str, persona: Optional[Persona] = None, cvm: Optional[ConversationModel] = None, **kwargs) -> 'BasePipeline':

        if persona is None:
            persona_file = os.path.join(config.persona_path, f"{config.persona_id}.json")
            if not os.path.exists(persona_file):
                raise FileNotFoundError(f"Persona {config.persona_id} not found in {config.persona_path}")
            persona = Persona.from_json_file(persona_file)

        #if gen is None:
        #    model_configs = LanguageModelV2.load_yaml_config(config.model_config_path)
        #    gen = GenerativeProviders.from_models(model_configs)
        llm_list = LanguageModelV2.index_models(config)

        if model not in llm_list:
            raise ValueError(f"Model {model} not found in {llm_list}")

        llm_config = llm_list[model]
        #thought_config = llm_list['deepseek-ai/DeepSeek-R1']
        thought_config = llm_list['maldv/Loqwqtus2.5-32B-Instruct']
        codex_config = llm_list['nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']

        llm = llm_config.llm_factory(config)
        thought = thought_config.llm_factory(config)
        codex = codex_config.llm_factory(config)

        if cvm is None:
            cvm = ConversationModel.from_config(config)

        return cls(llm, thought, codex, cvm, persona, config)
