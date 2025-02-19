# aim/app/cli/__main__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

# This is the CLI for AI-Mind. It is used to manage the application and the data it contains.
# It is out of date and needs to be rewritten for the new architecture.

import asyncio
import click
from collections import defaultdict
import logging
import os
import pandas as pd
import sys
from typing import Any, Dict, Optional

from ...agents import Persona
from ...chat.app import ChatApp
from ...conversation.model import ConversationModel
from ...io.jsonl import write_jsonl, read_jsonl
from ...llm.llm import LLMProvider, OpenAIProvider, ChatConfig

logger = logging.getLogger(__name__)


class ContextObject:
    config : ChatConfig
    cvm : ConversationModel
    persona : Persona
    llm : LLMProvider

    def __init__(self):
        self.config = None
        self.cvm = None
        self.persona = None
        self.llm = None

    def accept(self, **kwargs) -> 'ContextObject':
        config_dict = self.config_dict
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in config_dict:
                setattr(self.config, k, v)

        return self

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()

    def init_cvm(self) -> None:
        self.cvm = ConversationModel.from_config(self.config)

    def init_persona(self) -> None:
        persona_id = self.config.persona_id
        if persona_id is None:
            click.echo("No persona ID provided")
            sys.exit(1)
        persona_file = os.path.join(self.config.persona_path, f"{persona_id}.json")
        if not os.path.exists(persona_file):
            click.echo(f"Persona {persona_id} not found in {self.config.persona_path}")
            sys.exit(1)

        self.persona = Persona.from_json_file(persona_file)

    def build_chat(self) -> ChatApp:
        if self.llm is None:
            raise ValueError("LLM not initialized")
        if self.cvm is None:
            raise ValueError("ConversationModel not initialized")
        if self.persona is None:
            raise ValueError("Persona not initialized")
        return ChatApp.factory(llm=self.llm, cvm=self.cvm, config=self.config, persona=self.persona, clear_output=lambda: click.clear())

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'ContextObject':
        co = ContextObject()
        if env_file is not None:
            co.config = ChatConfig.from_env(env_file)
        else:
            co.config = ChatConfig.from_env()
        return co

@click.group()
@click.option('--env-file', default=None, help='Path to environment file')
@click.pass_context
def cli(ctx, env_file):
    co = ContextObject.from_env(env_file=env_file)
    
    co.cvm = ConversationModel.from_config(co.config)
    ctx.obj = co

@cli.command()
@click.pass_obj
def list_conversations(co: ContextObject):
    """List all conversations"""
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 100)
    df: pd.DataFrame = co.cvm.to_pandas()
    conversations = df.groupby(['document_type', 'user_id', 'persona_id', 'conversation_id']).size().reset_index(name='messages')
    click.echo(conversations)

@cli.command()
@click.pass_obj
def matrix(co: ContextObject):
    """List all conversations"""
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 100)
    df: pd.DataFrame = co.cvm.get_conversation_report()
    df.columns = [s[:2] for s in df.columns]
    click.echo(df)

@cli.command()
@click.argument('conversation_id')
@click.pass_obj
def display_conversation(co: ContextObject, conversation_id):
    """Display a specific conversation"""
    history = co.cvm.get_conversation_history(conversation_id=conversation_id)
    for _, row in history.iterrows():
        click.echo(f"{row['role']}: {row['content']}\n")

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.pass_obj
def delete_conversation(co: ContextObject, user_id, persona_id, conversation_id):
    """Delete a specific conversation"""
    co.cvm.delete_conversation(conversation_id)
    co.cvm.collection.delete(f"user_id = '{user_id}' and persona_id = '{persona_id}' and conversation_id = '{conversation_id}'")
    click.echo(f"Conversation {conversation_id} for user {user_id} with {persona_id} has been deleted.")

@cli.command()
@click.option('--workdir_folder', default=None, help='working directory')
@click.option('--filename', default=None, help='output file')
@click.argument('conversation_id')
@click.pass_obj
def export_conversation(co: ContextObject, conversation_id, workdir_folder, filename):
    """Export a conversation as a jsonl file"""
    if filename is None:
        filename = f"{conversation_id}.jsonl"

    workdir_folder = co.accept(workdir_folder=workdir_folder).config.workdir_folder

    output_file = os.path.join(workdir_folder if workdir_folder is not None else '.', filename)

    history = co.cvm.get_conversation_history(conversation_id=conversation_id)
    history = [r.to_dict() for _, r in history.iterrows()]
    write_jsonl(history, output_file)

    click.echo(f"Conversation {conversation_id} has been exported to {output_file}. ({len(history)} messages)")

@cli.command()
@click.option('--workdir_folder', default=None, help='working directory')
@click.option('--filename', default=None, help='output file')
@click.pass_obj
def export_all(co: ContextObject, workdir_folder, filename):
    """Export a conversation as a jsonl file"""
    if filename is None:
        filename = f"dump.jsonl"

    workdir_folder = co.accept(workdir_folder=workdir_folder).config.workdir_folder
    output_file = os.path.join(workdir_folder if workdir_folder is not None else '.', filename)

    history = co.cvm.dataframe
    history.drop(columns=['index'], inplace=True)
    history = [r.to_dict() for _, r in history.iterrows()]
    write_jsonl(history, output_file)

    click.echo(f"All data has been exported to {output_file}. ({len(history)} messages)")

@cli.command()
@click.option('--user-id', default=None, help='User ID for whom to apply the conversation')
@click.option('--persona-id', default=None, help='Persona ID for whom to apply the conversation')
@click.argument('conversation_filename')
@click.pass_obj
def import_conversation(co: ContextObject, conversation_filename, user_id, persona_id):
    """Export a conversation as a jsonl file"""

    conversation_ids = defaultdict(int)
    data = read_jsonl(conversation_filename)
    for row in data:
        conversation_ids[row['conversation_id']] += 1
        if user_id is not None:
            row['user_id'] = user_id
        if persona_id is not None:
            row['persona_id'] = persona_id
        co.cvm.insert(**row)

    click.echo(f"Conversation {conversation_filename} has been imported.")
    
    for conversation_id, count in conversation_ids.items():
        click.echo(f"Conversation {conversation_id} has been imported. ({count} messages)")

@cli.command()
@click.argument('dump_filename')
@click.pass_obj
def import_all(co: ContextObject, dump_filename):
    """Import the contents of a conversation dump from a jsonl file"""

    conversation_ids = defaultdict(int)
    data = read_jsonl(dump_filename)
    for row in data:
        conversation_ids[row['conversation_id']] += 1
        co.cvm.insert(**row)

    click.echo(f"Conversation {dump_filename} has been imported.")
    
    for conversation_id, count in conversation_ids.items():
        click.echo(f"Conversation {conversation_id} has been imported. ({count} messages)")


@cli.command()
@click.argument('conversation-id')
@click.option('--model-url', default=None, help='URL for the OpenAI-compatible API')
@click.option('--api-key', default=None, help='API key for the LLM service')
@click.option('--user-id', default=None, help='User ID for the conversation')
@click.option('--persona-id', default=None, help='Persona ID for the conversation')
@click.option('--max-tokens', default=None, help='Maximum number of tokens for LLM response')
@click.option('--mood', default=None, help='Mood for the chat')
@click.option('--temperature', default=None, help='Temperature for LLM response')
@click.option('--test-mode', is_flag=True, help='Test mode')
@click.option('--top-n', default=None, help='Top N for LLM response')
@click.pass_obj
def chat(co: ContextObject, model_url, api_key, user_id, persona_id, conversation_id, max_tokens, temperature, mood, test_mode, top_n):
    """Start a new chat session"""
    co.accept(
        model_url=model_url,
        api_key=api_key,
        user_id=user_id,
        persona_id=persona_id,
        conversation_id=conversation_id,
        max_tokens=max_tokens,
        mood=mood,
        temperature=temperature,
        top_n=top_n
    )
    co.llm = OpenAIProvider.from_url(co.config.model_url, co.config.api_key)

    user_id = co.config.user_id
    persona_id = co.config.persona_id

    if co.config.conversation_id is None:
        co.config.conversation_id = co.cvm.next_conversation_id(user_id=user_id, persona_id=persona_id)

    # So the AI doesn't try and speak in the user's voice
    co.config.stop_sequences.append(f"{co.config.user_id}:")
    co.init_persona()

    # Build and run the chat
    cm = co.build_chat()
    save = not test_mode
    cm.chat_loop(save=save)

@cli.command()
@click.argument('pipeline_type')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--mood', default=None, help='The mood of the persona')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.option('--guidance', is_flag=True, help='Prompt for guidance for the conversation')
@click.argument('query', nargs=-1)
@click.pass_obj
def pipeline(co: ContextObject, pipeline_type, persona_id, conversation_id, mood, query, no_retry, guidance):
    """Run the journal pipeline"""
    from ...pipeline.factory import pipeline_factory, BasePipeline
    co.accept(
        persona_id=persona_id,
        conversation_id=conversation_id,
        no_retry=no_retry,
        mood=mood,
        query_text=' '.join(query),
    )

    if guidance:
        value = click.prompt('Enter your guidance', type=str)
        co.config.guidance = value
        print(f"Guidance: {co.config.guidance}")

    base = BasePipeline.from_config(co.config)
    pipeline = pipeline_factory(pipeline_type=pipeline_type)
    asyncio.run(pipeline(self=base, **(co.config_dict)))

@cli.command()
@click.option('--conversations-dir', default="memory/conversations", help='Directory containing conversation JSONL files')
@click.option('--index-dir', default="memory/indices", help='Directory for storing indices')
@click.option('--debug', is_flag=True, help='Enable debug output')
@click.option('--device', default="cpu", help='Device to use for indexing')
@click.pass_obj
def rebuild_index(co: ContextObject, conversations_dir: str, index_dir: str, device:str, debug: bool):
    """Rebuild search indices from conversation JSONL files"""
    from ...conversation.loader import ConversationLoader
    from ...conversation.index import SearchIndex
    from pathlib import Path

    try:
        # Initialize loader and index
        loader = ConversationLoader(conversations_dir)
        index = SearchIndex(index_path=Path(index_dir), embedding_model=co.config.embedding_model, device=device)
        
        # Load all conversations
        click.echo("Loading conversations...")
        messages = loader.load_all()
        if debug:
            click.echo(f"Message sample: {messages[0].content[:100]}")
        click.echo(f"Loaded {len(messages)} messages")
        
        if len(messages) == 0:
            click.echo("No messages found to index!", err=True)
            return
        
        # Convert to index documents
        click.echo("Converting to index documents...")
        documents = [msg.to_dict() for msg in messages]
        
        # Build the index
        click.echo("Building index...")
        if debug:
            click.echo("Document sample:")
            click.echo(f"ID: {documents[0]['doc_id']}")
            click.echo(f"Content: {documents[0]['content'][:100]}")
            
        index.add_documents(documents)
        
        click.echo("Index rebuild complete!")
        
    except Exception as e:
        click.echo(f"Error rebuilding index: {e}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc())
        raise click.Abort()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cli()
