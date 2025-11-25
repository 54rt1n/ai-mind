# aim/chat/app.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import time
from typing import List, Dict, Callable, Tuple, Optional, Generator

from .strategy import chat_strategy_for
from .manager import ChatManager, ChatConfig, ConversationModel
from ..agents import Persona
from ..llm.llm import LLMProvider

HELP = """Commands:
- (b)ack: Go back to the previous message
- (h)elp: Show this help message
- new: Start a new chat
- (p)rompt <message>: Update the system message
- redraw: Redraw the screen
- (d)ocument <name>: Set the current document
- (l)ist: List the documents in the library
- (r)etry: Retry the previous user input
- (s)earch <query>: Search your documents
- top <n>: Set the top n results to use
- temp <n>: Set the temperature
- (q)uit/exit: End the chat
"""


class ChatApp:
    def __init__(self, chat: ChatManager, clear_output=Callable[[], None]):
        self.chat = chat
        self.persona = chat.roster.personas[chat.config.persona_id]
        self.clear_output = clear_output
        self.chat_strategy = chat_strategy_for(self.persona.chat_strategy, chat)

        self.running = False
        self.history : List[Dict[str, str]] = []

    def render_conversation(self, messages: List[Dict[str, str]]) -> None:
        if self.clear_output is not None:
            self.clear_output()
        for message in messages:
            rolename = message['role'].capitalize()
            print(f"{rolename}: {message['content']}\n")
        print(flush=True)

    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.history.clear()

    def render_conversation(self, messages: List[Dict[str, str]]) -> None:
        self.clear_output()
        for message in messages:
            print(f"{message['role']}: {message['content']}\n")
        print(flush=True)

    def handle_user_input(self) -> Generator[Tuple[str, str, Optional[str]], None, None]:
        # Next, we get the user input, and handle special commands
        user_input = input("You (h for help): ").strip()

        # If the user input is empty, we skip this iteration
        if not user_input or user_input.strip() == '':
            print("Type 'h' for help, 'q' to quit.")
            yield 'pass', 'No input provided', None
            return

        # Handle single word commands
        lowered = user_input.lower()
        if lowered in ['q', 'quit', 'exit']:
            yield 'quit', None, None
            return

        if lowered in ['b', 'back']:
            if len(self.history) >= 2:
                self.history = self.history[:-2]
            yield 'back', 'Back one turn', None
            return

        if lowered == 'new':
            self.clear_history()
            yield 'new', 'New chat started', None
            return

        if lowered == 'redraw':
            yield 'redraw', 'Redrew the screen', None
            return

        if lowered in ['h', 'help']:
            yield 'help', 'Help', None
            return

        if lowered in ['l', 'list']:
            yield 'list', 'List of documents', None
            return

        # Handle multi-word commands
        multi = lowered.split()
        # 'prompt' allows the user to update the system message
        if len(multi) > 1 and multi[0] in ['p', 'prompt']:
            # TODO re-add persona.
            self.chat.config.persona_location = ' '.join(multi[1:])
            yield 'prompt', 'System Prompt Updated', None
            return

        # Set our top-n
        if len(multi) > 1 and multi[0] in ['top']:
            try:
                intval = int(multi[1])
                if intval > 0:
                    self.chat.config.top_n = intval
                    yield 'top_n_set', f'Top N set to {self.chat.config.top_n}', None
                else:
                    yield 'error', 'Invalid top N value', None
            except ValueError:
                yield 'error', 'Invalid top N value', None

            return

        # Set our temperature
        if len(multi) > 1 and multi[0] in ['temp']:
            try:
                self.chat.config.temperature = float(multi[1])
                yield 'temperature_set', f'Temperature set to {self.chat.config.temperature}', None
            except ValueError:
                yield 'invalid_temperature', 'Invalid temperature value', None

            return

        # Search for memories
        if len(multi) > 1 and multi[0] in ['s', 'search']:
            search_results = self.chat.search_memory(' '.join(multi[1:]))
            if len(search_results) == 0:
                yield 'no_results', 'No memories found', None
            else:
                for descr, mem in search_results:
                    print(descr)
                    print(mem)
                yield 'found_memory', None, None
            return
        
        if len(multi) > 0 and multi[0] in ['d', 'document']:
            try:
                if len(multi) > 1:
                    document = multi[1]
                    if self.chat.library.exists(document):
                        self.current_doucment = document
                        yield 'document_set', f'Document set to {self.current_doucment}', None
                    else:
                        yield 'document_set', f'Document does not exist, current document {self.current_doucment}.', None
                else:
                    self.current_doucment = None
                    yield 'document_set', 'Document cleared', None
            except Exception as e:
                yield 'error', f'Error setting document: {e}', None
            return

        if lowered in ['r', 'retry']:
            if len(self.history) >= 2:
                user_input = self.history[-2]["content"]
                self.history = self.history[:-2]
                yield 'retry', 'Retried the last turn', None
            else:
                yield 'pass', 'No previous turn to retry', None

        yield 'user', user_input, int(time.time())

    def run_once(self) -> Generator[Tuple[str, str, Optional[int]], None, None]:
        # Render our current turn - we could optionally just include only the history, but for debugging purposes,
        # we'll render the entire turn
        self.render_conversation(self.history)

        user_input = None
        for event in self.handle_user_input():
            action, message, _ = event
            
            yield event

            if action == 'user':
                user_input = message

        if user_input is None:
            return

        user_turn = self.chat_strategy.user_turn_for(self.persona, user_input)
        chat_turns = self.chat_strategy.chat_turns_for(self.persona, user_input, self.history)

        if self.chat.config.debug:
            self.render_conversation(chat_turns)

        print(f"Assistant: ", end='', flush=True)

        chunks = []

        # TODO add a provider
        for t in self.chat.llm.stream_turns(chat_turns, self.chat.config):
            if t is not None:
                print(t, end='', flush=True)
                chunks.append(t)
            else:
                print('', flush=True)

        response = ''.join(chunks)

        yield 'assistant', response, int(time.time())

        ui = input("** -=[ <enter> or (r)etry ]=- **")
        if ui == 'r':
            yield 'pass', 'Retrying', None
            return

        self.add_history(**user_turn)
        self.add_history("assistant", response)

        yield 'continue', None, None
        return

    def chat_loop(self, save: bool=True) -> None:
        history = self.chat.cvm.get_conversation_history(conversation_id=self.chat.config.conversation_id).sort_values(['date', 'sequence_no', 'branch']).reset_index(drop=True)
        if history.empty:
            self.sequence_no = 0
            self.branch = 0
            self.history = []
        else:
            last = history.iloc[-1]
            self.sequence_no = last['sequence_no'] + 1
            self.branch = last['branch']
            self.history = history[['role', 'content', 'think']].to_dict(orient='records')

        self.chat.config.system_message = self.chat.get_system_prompt()

        self.running = True
        while self.running:
            try:
                enter = True
                user_input : Optional[str] = None
                usertime : Optional[int] = None

                assistant_response : Optional[str] = None
                assttime : Optional[int] = None

                for event in self.run_once():
                    result, message, ts = event
                    
                    if result == 'quit':
                        self.running = False
                        enter = False
                    elif result == 'redraw':
                        enter = False
                    elif result == 'new':
                        self.branch = 0
                        self.sequence_no = 0
                        self.history = []
                        self.chat.new_conversation()
                        enter = False
                    elif result == 'help':
                        print()
                        print(HELP)
                    elif result == 'back' or result == 'retry':
                        self.branch += 1
                    elif result == 'user':
                        user_input = message
                        usertime = ts
                        print('user', ts, len(message))
                    elif result == 'assistant':
                        assistant_response = message
                        assttime = ts
                        print('assistant', ts, len(message))
                    elif result == 'list':
                        print(f"Library {self.chat.library.documents_dir}:")
                        for f, t, s in self.chat.library.list_documents:
                            print('  ', f, t, s)
                    elif result == 'continue':
                        enter = False

                        if save:
                            self.chat.insert_turn(
                                sequence_no=self.sequence_no, branch=self.branch,
                                user_turn=user_input, assistant_turn=assistant_response,
                                usertime=usertime, assttime=assttime
                            )
                            self.sequence_no += 2
                    else:
                        if message is not None:
                            print(f"{result}: {message}")
                        else:
                            print(f"{result}", end='')

                if enter:
                    print()
                    input("-=[ Hit <enter> to continue... ]=-")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"An error occurred: {e}")

                print()
                input("-=[ Hit <enter> to continue... ]=-")
            
        self.running = False
        print("Chat session ended.")

    @classmethod
    def factory(cls, llm: LLMProvider, cvm: ConversationModel, config: ChatConfig, persona: Persona, clear_output=Callable[[], None]) -> 'ChatApp':
        return cls(ChatManager(llm, cvm, config, persona), clear_output)

