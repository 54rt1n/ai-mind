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
from ...pipeline.factory import pipeline_factory, BasePipeline
from ...conversation.message import ConversationMessage
from ...utils.turns import process_think_tag_in_message, extract_and_update_emotions_from_header
from ...conversation.loader import ConversationLoader
from ...constants import DOC_STEP, DOC_NER

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


def _show_chunk_stats(index, click):
    """Display chunk level statistics for the index."""
    from ...constants import CHUNK_LEVEL_256, CHUNK_LEVEL_768, CHUNK_LEVEL_FULL

    searcher = index.index.searcher()

    # Count entries per chunk level
    stats = {}
    for level in [CHUNK_LEVEL_FULL, CHUNK_LEVEL_768, CHUNK_LEVEL_256]:
        query = index.index.parse_query(query=level, default_field_names=["chunk_level"])
        results = searcher.search(query, limit=1)
        stats[level] = results.count

    total_entries = sum(stats.values())
    parent_docs = stats.get(CHUNK_LEVEL_FULL, 0)

    click.echo("\nChunk Level Statistics:")
    click.echo(f"  Parent documents: {parent_docs}")
    click.echo(f"  768-token chunks: {stats.get(CHUNK_LEVEL_768, 0)}")
    click.echo(f"  256-token chunks: {stats.get(CHUNK_LEVEL_256, 0)}")
    click.echo(f"  Total entries: {total_entries}")


@cli.command()
@click.option('--conversations-dir', default="memory/conversations", help='Directory containing conversation JSONL files')
@click.option('--index-dir', default="memory/indices", help='Directory for storing indices')
@click.option('--debug', is_flag=True, help='Enable debug output')
@click.option('--device', default="cpu", help='Device to use for indexing')
@click.option('--batch-size', default=64, help='Batch size for indexing')
@click.option('--full', is_flag=True, help='Force full rebuild instead of incremental update')
@click.pass_obj
def rebuild_index(co: ContextObject, conversations_dir: str, index_dir: str, device: str, debug: bool, batch_size: int, full: bool):
    """Rebuild or incrementally update search indices from conversation JSONL files"""
    from ...conversation.loader import ConversationLoader
    from ...conversation.index import SearchIndex
    from pathlib import Path

    try:
        # Initialize loader and index
        loader = ConversationLoader(conversations_dir)
        index_path = Path(index_dir)
        
        # Check if index exists
        index_exists = index_path.exists() and any(index_path.iterdir())
        
        if full or not index_exists:
            click.echo("Performing full index rebuild...")
            index = SearchIndex(index_path=index_path, embedding_model=co.config.embedding_model, device=device)
            
            # Load all conversations
            click.echo("Loading conversations...")
            messages = loader.load_all()
            if debug and messages:
                click.echo(f"Message sample: {messages[0].content[:100]}")
            click.echo(f"Loaded {len(messages)} messages")
            
            if len(messages) == 0:
                click.echo("No messages found to index!", err=True)
                return
            
            # Convert to index documents
            click.echo("Converting to index documents...")
            documents = [msg.to_dict() for msg in messages]

            if debug:
                click.echo(f"Found {len(documents)} documents")
                for doc in documents[:5]:  # Show first 5 only
                    click.echo(f"Document: {doc['doc_id']} - {doc['document_type']}")

            # Build the index
            click.echo("Building index...")
            if debug and documents:
                click.echo("Document sample:")
                click.echo(f"ID: {documents[0]['doc_id']}")
                click.echo(f"Content: {documents[0]['content'][:100]}")
                
            index.rebuild(documents)

            # Show chunk level stats
            _show_chunk_stats(index, click)

            click.echo("Full index rebuild complete!")
            
        else:
            click.echo("Performing incremental index update...")
            index = SearchIndex(index_path=index_path, embedding_model=co.config.embedding_model, device=device)
            
            # Load all conversations
            click.echo("Loading conversations...")
            messages = loader.load_all()
            click.echo(f"Loaded {len(messages)} messages")
            
            if len(messages) == 0:
                click.echo("No messages found to index!", err=True)
                return
            
            # Convert to index documents
            click.echo("Converting to index documents...")
            documents = [msg.to_dict() for msg in messages]

            if debug:
                click.echo(f"Found {len(documents)} documents")

            # Perform incremental update
            click.echo("Comparing with existing index...")
            added_count, updated_count, deleted_count = index.incremental_update(
                documents, use_tqdm=True, batch_size=batch_size
            )
            
            click.echo(f"Incremental update complete!")
            click.echo(f"  Added: {added_count} documents")
            click.echo(f"  Updated: {updated_count} documents")
            click.echo(f"  Deleted: {deleted_count} documents")

            # Show chunk level stats
            _show_chunk_stats(index, click)

            if added_count == 0 and updated_count == 0 and deleted_count == 0:
                click.echo("Index was already up to date!")
        
    except Exception as e:
        click.echo(f"Error rebuilding index: {e}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc())
        raise click.Abort()

@cli.command()
@click.option('--conversation-id', 'target_conversation_id', default=None, help='ID of a specific conversation to repair.')
@click.option('--all-conversations', is_flag=True, help='Repair all conversations.')
@click.option('--dry-run', is_flag=True, help='Show what would change without writing to files.')
@click.option('--skip-steps', is_flag=True, help='Skip processing messages identified as step documents.')
@click.pass_obj
def repair_conversation(co: ContextObject, target_conversation_id: Optional[str], all_conversations: bool, dry_run: bool):
    """
    Scans conversation messages to perform repairs:
    - Extracts </think> tag content into the 'think' field.
    - Extracts emotions from 'Emotional State:' headers into emotion fields.
    Writes changes back to the conversation files unless --dry-run is specified.
    """
    if not target_conversation_id and not all_conversations:
        click.echo("Please specify either --conversation-id or --all-conversations.", err=True)
        return
    if target_conversation_id and all_conversations:
        click.echo("Cannot use both --conversation-id and --all-conversations.", err=True)
        return

    # Ensure CVM and loader are initialized
    if co.cvm is None:
        co.init_cvm() # This should initialize the loader as well
    if co.cvm.loader is None:
        # This case shouldn't happen if CVM init works, but as a safeguard:
        if not hasattr(co.config, 'memory_path') or not co.config.memory_path:
            click.echo("ERROR: Cannot initialize loader - memory_path not configured.", err=True)
            return
        conversations_dir = os.path.join(co.config.memory_path, 'conversations')
        co.cvm.loader = ConversationLoader(conversations_dir=conversations_dir)
        click.echo(f"Initialized loader with path: {conversations_dir}", err=True) # Info/Debug log

    loader = co.cvm.loader
    conversations_dir_path = loader.conversations_dir

    conversation_ids_to_process = []
    if target_conversation_id:
        target_file_path = conversations_dir_path / f"{target_conversation_id}.jsonl"
        if not target_file_path.exists():
            click.echo(f"Conversation file '{target_file_path}' not found.", err=True)
            return
        conversation_ids_to_process.append(target_conversation_id)
    elif all_conversations:
        found_files = list(conversations_dir_path.glob("*.jsonl"))
        if not found_files:
            click.echo(f"No conversation files found in {conversations_dir_path}.")
            return
        conversation_ids_to_process = [f.stem for f in found_files]

    click.echo(f"Found {len(conversation_ids_to_process)} conversations to process in {conversations_dir_path}.")
    if dry_run:
        click.echo("DRY RUN: No changes will be written.")

    overall_messages_changed_count = 0

    for conv_id in conversation_ids_to_process:
        click.echo(f"Processing conversation: {conv_id}")
        conversation_file_path = conversations_dir_path / f"{conv_id}.jsonl"
        try:
            # Load directly using the loader
            original_messages: list[ConversationMessage] = loader.load_conversation(conv_id)
            
            if not original_messages:
                click.echo(f"  No messages loaded for conversation {conv_id} (file might be empty or only contain invalid lines). Skipping.")
                continue

            updated_messages_for_conv: list[ConversationMessage] = []
            conversation_modified_locally = False
            messages_changed_in_conv_count = 0

            # Process the list of ConversationMessage objects directly
            for original_msg in original_messages:
                current_msg = original_msg
                think_updated = False
                emotion_updated = False
                
                # 1. Process think tag
                msg_after_think_proc = process_think_tag_in_message(current_msg)
                if msg_after_think_proc:
                    current_msg = msg_after_think_proc
                    think_updated = True
                    
                # 2. Process emotions (on the potentially updated message)
                msg_after_emotion_proc = extract_and_update_emotions_from_header(current_msg)
                if msg_after_emotion_proc:
                    current_msg = msg_after_emotion_proc
                    emotion_updated = True
                    
                # Check if any update occurred for this message
                if think_updated or emotion_updated:
                    updated_messages_for_conv.append(current_msg) # Append the final state of the message
                    conversation_modified_locally = True
                    messages_changed_in_conv_count += 1
                    if dry_run:
                        click.echo(f"  [DRY RUN] Message seq={original_msg.sequence_no} (doc={original_msg.doc_id}) needs update.")
                        if think_updated:
                            click.echo(f"      Think tag processed.")
                        if emotion_updated:
                            click.echo(f"      Emotions extracted/updated: a={current_msg.emotion_a}, b={current_msg.emotion_b}, c={current_msg.emotion_c}, d={current_msg.emotion_d}")
                else:
                    updated_messages_for_conv.append(original_msg) # No changes, append original

            if conversation_modified_locally:
                overall_messages_changed_count += messages_changed_in_conv_count
                click.echo(f"  Conversation {conv_id}: {messages_changed_in_conv_count} message(s) marked for update.")
                if not dry_run:
                    messages_to_write_dicts = [msg.to_dict() for msg in updated_messages_for_conv]
                    try:
                        # Ensure directory exists (though loader init should handle base dir)
                        if not conversations_dir_path.exists():
                            conversations_dir_path.mkdir(parents=True, exist_ok=True)
                            
                        write_jsonl(messages_to_write_dicts, str(conversation_file_path))
                        click.echo(f"  SUCCESS: Conversation {conv_id} updated and saved to {conversation_file_path}.")
                        # Note: This does not update the ConversationModel's internal index or cache if it has one.
                        # Consider adding co.cvm.index.update_documents(...) or similar if needed.
                    except Exception as e:
                        click.echo(f"  ERROR: Failed to write updated conversation {conv_id} to {conversation_file_path}: {e}", err=True)
            else:
                click.echo(f"  Conversation {conv_id}: No messages needed updates.")

        except FileNotFoundError:
            click.echo(f"  ERROR: File not found for conversation {conv_id} at {conversation_file_path}. Skipping.", err=True)
        except Exception as e:
            click.echo(f"  ERROR processing conversation {conv_id}: {e}", err=True)
            # Safely check for debug_mode
            debug_is_on = False
            if hasattr(co.config, 'debug_mode') and co.config.debug_mode:
                debug_is_on = True
            elif hasattr(co.config, 'env_config') and hasattr(co.config.env_config, 'debug') and co.config.env_config.debug:
                debug_is_on = True # Example of checking a nested debug flag
            
            if debug_is_on:
                 import traceback
                 click.echo(traceback.format_exc())

    if overall_messages_changed_count > 0:
        if dry_run:
            click.echo(f"\nDRY RUN SUMMARY: {overall_messages_changed_count} message(s) across all processed conversations would be updated.")
        else:
            click.echo(f"\nSUMMARY: {overall_messages_changed_count} message(s) across all processed conversations were updated and saved.")
    else:
        click.echo("\nNo messages required updates across all processed conversations.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cli()
