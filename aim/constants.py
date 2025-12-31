# aim/constants.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

DOC_ANALYSIS = "analysis"
DOC_BRAINSTORM = "brainstorm"
DOC_CODEX = "codex"
DOC_CONVERSATION = "conversation"
DOC_DAYDREAM = "daydream"
DOC_DIALOGUE_ARTIST = "dialogue-artist"
DOC_DIALOGUE_CODER = "dialogue-coder"
DOC_DIALOGUE_DREAMER = "dialogue-dreamer"
DOC_DIALOGUE_LIBRARIAN = "dialogue-librarian"
DOC_DIALOGUE_PHILOSOPHER = "dialogue-philosopher"
DOC_DIALOGUE_PSYCHOLOGIST = "dialogue-psychologist"
DOC_DIALOGUE_REVELATOR = "dialogue-revelator"
DOC_DIALOGUE_WRITER = "dialogue-writer"
DOC_INSPIRATION = "inspiration"
DOC_JOURNAL = "journal"
DOC_MOTD = "motd"
DOC_NER = "ner-task"
DOC_PONDERING = "pondering"
DOC_REFLECTION = "reflection"
DOC_REPORT = "report"
DOC_SELF_RAG = "self-rag"
DOC_SOURCE_CODE = "source-code"
DOC_STEP = "step"
DOC_SUMMARY = "summary"
DOC_UNDERSTANDING = "understanding"

# Chunk levels for multi-resolution indexing
CHUNK_LEVEL_256 = "chunk_256"
CHUNK_LEVEL_768 = "chunk_768"
CHUNK_LEVEL_FULL = "full"
CHUNK_LEVELS = [CHUNK_LEVEL_256, CHUNK_LEVEL_768, CHUNK_LEVEL_FULL]

# Chunk sizes in tokens (tiktoken cl100k_base)
CHUNK_SIZE_256 = 256
CHUNK_SIZE_768 = 768
CHUNK_SLIDE_256 = 256  # Non-overlapping
CHUNK_SLIDE_768 = 384  # 50% overlap

LISTENER_ALL = "all"
LISTENER_SELF = "self"

ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"

PIPELINE_ANALYSIS = "analysis_dialogue"
PIPELINE_CODER = "coder"
PIPELINE_DAYDREAM = "dreamer"
PIPELINE_JOURNAL = "journaler"
PIPELINE_PHILOSOPHER = "philosopher"
PIPELINE_REPORTER = "reporter"
PIPELINE_SUMMARIZER = "summarizer"

QUARTER_CTX = 512
MID_CTX = 768
HALF_CTX = 1024
LARGE_CTX = 1536
FULL_CTX = 2048
MASSIVE_CTX = 3072
MAXIMUM_CTX = 4096

TOKEN_CHARS = 4