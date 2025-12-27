# Scenarios and Paradigms

This document explains the two core systems that drive AI-Mind's autonomous cognitive processing: **Scenarios** (execution pipelines) and **Paradigms** (exploration triggers).

## Overview

| System | Purpose | Location | Triggered By |
|--------|---------|----------|--------------|
| **Scenarios** | Multi-step processing pipelines | `config/scenario/*.yaml` | Direct invocation or paradigm selection |
| **Paradigms** | Autonomous exploration and scenario triggering | `aim/refiner/` | Dream watcher (idle-time processing) |

**Relationship**: Paradigms select topics for exploration and then trigger the appropriate scenario to process them.

---

## Scenarios

Scenarios are YAML-defined directed acyclic graphs (DAGs) that execute multi-step cognitive processing pipelines.

### Location

All scenario files are in `config/scenario/`:
- `summarizer.yaml` - Conversation summarization via iterative densification
- `analyst.yaml` - Post-conversation analysis and reflection
- `philosopher.yaml` - Deep philosophical inquiry and essay generation
- `daydream.yaml` - Conversational dialogue with dreamer aspect
- `journaler.yaml` - Personal reflection and introspection
- `researcher.yaml` - Knowledge curation and synthesis
- `critique.yaml` - Psychological self-examination

### Schema

```yaml
name: scenario_name
version: 2
description: Human-readable description
requires_conversation: bool  # True if needs existing conversation context

context:
  required_aspects: [list]        # Persona aspects needed (e.g., philosopher, librarian)
  core_documents: [list]          # Essential document types to load
  enhancement_documents: [list]   # Optional enhancing document types
  location: "Jinja2 template"     # Scene setting for the scenario
  thoughts: [list]                # Initial thoughts/prompts

seed: []  # Initial data loading actions

steps:
  step_id:
    id: step_id
    prompt: "Jinja2 template"     # The prompt for this step
    config:
      max_tokens: int             # Token limit for generation
      temperature: float          # Sampling temperature (optional)
      model_override: string      # Alternative model (optional)
      use_guidance: bool          # Include user guidance in prompt
      is_thought: bool            # Internal thinking (not stored as output)
      is_codex: bool              # Codex entry generation
    output:
      document_type: string       # Output document type
      weight: float               # Relevance weight (1.0 default)
      add_to_turns: bool          # Include in conversation turns
    memory:
      top_n: int                  # Memory documents to retrieve
      document_type: [list]       # Filter memory by type
      flush_before: bool          # Clear memory before step
      sort_by: string             # 'relevance' or 'timestamp'
    context:                      # Context preparation DSL
      - action: load_conversation
        target: current
        exclude_types: [ner, step]
      - action: query
        document_types: [pondering, brainstorm]
        top_n: 10
      - action: sort
        by: timestamp
        direction: ascending
    next: [step_ids]              # Next steps in the DAG
```

### Jinja2 Template Variables

Prompts use Jinja2 templating with these variables:
- `persona.name`, `persona.title`, `persona.location`
- `pronouns.subj`, `pronouns.obj`, `pronouns.poss`
- `step_num` - Current step counter
- `guidance` - Optional user guidance
- `query_text` - Search/focus query
- Named aspect variables (e.g., `philosopher`, `librarian`, `dreamer`)

---

## Document Types

Defined in `aim/constants.py`:

| Constant | Value | Purpose | Produced By |
|----------|-------|---------|-------------|
| `DOC_CONVERSATION` | "conversation" | User-assistant exchanges | Chat |
| `DOC_SUMMARY` | "summary" | Dense conversation summaries | summarizer |
| `DOC_ANALYSIS` | "analysis" | High-level conversation analysis | analyst |
| `DOC_JOURNAL` | "journal" | Persona's internal reflections | journaler |
| `DOC_PONDERING` | "pondering" | Deep philosophical reflections | philosopher |
| `DOC_BRAINSTORM` | "brainstorm" | Creative ideation | multiple |
| `DOC_DAYDREAM` | "daydream" | Imaginative exchanges | daydream |
| `DOC_INSPIRATION` | "inspiration" | Distilled insights from daydreams | daydream |
| `DOC_UNDERSTANDING` | "understanding" | Psychological self-knowledge | critique |
| `DOC_CODEX` | "codex" | Semantic knowledge graph entries | multiple |
| `DOC_SELF_RAG` | "self-rag" | Self-retrieval augmented generation | philosopher |
| `DOC_MOTD` | "motd" | Message of the day | analyst |
| `DOC_NER` | "ner-task" | Named entity recognition | analyst |
| `DOC_STEP` | "step" | Intermediate pipeline outputs | multiple |

---

## Paradigms (Refiner System)

Paradigms drive autonomous exploration during idle time via the dream watcher (`aim/refiner/`).

### The Four Paradigms

| Paradigm | Focus | Seeded By | Maps To |
|----------|-------|-----------|---------|
| **brainstorm** | Creative ideas, half-formed notions | brainstorm, pondering, daydream, journal, inspiration, understanding | philosopher or journaler |
| **daydream** | Emotional/imaginative exploration | All document types | daydream |
| **knowledge** | Knowledge gaps, deeper understanding | codex, pondering, brainstorm, self-rag, understanding | researcher |
| **critique** | Psychological patterns, self-examination | understanding, journal, analysis, pondering, inspiration | critique |

### Configuration

Located in `aim/refiner/context.py`:

```python
# Which document types seed each paradigm
PARADIGM_DOC_TYPES = {
    "brainstorm": [DOC_BRAINSTORM, DOC_PONDERING, DOC_DAYDREAM, DOC_JOURNAL, DOC_INSPIRATION, DOC_UNDERSTANDING],
    "daydream": [DOC_SUMMARY, DOC_ANALYSIS, DOC_DAYDREAM, DOC_CONVERSATION, DOC_JOURNAL, DOC_PONDERING, DOC_BRAINSTORM, DOC_CODEX, DOC_INSPIRATION, DOC_UNDERSTANDING],
    "knowledge": [DOC_CODEX, DOC_PONDERING, DOC_BRAINSTORM, DOC_SELF_RAG, DOC_UNDERSTANDING],
    "critique": [DOC_UNDERSTANDING, DOC_JOURNAL, DOC_ANALYSIS, DOC_PONDERING, DOC_INSPIRATION],
}

# Weighted queries for discovery diversity
PARADIGM_QUERIES = {
    "brainstorm": [
        {"text": "unexplored ideas creative thoughts imaginative possibilities", "weight": 1.0},
        {"text": "questions I want to investigate curious mysteries", "weight": 0.9},
        # ...
    ],
    # ...
}

# Approach-specific document types for targeted gathering
APPROACH_DOC_TYPES = {
    "philosopher": [DOC_CODEX, DOC_PONDERING, DOC_ANALYSIS, DOC_BRAINSTORM, DOC_UNDERSTANDING],
    "journaler": [DOC_JOURNAL, DOC_CONVERSATION, DOC_SUMMARY, DOC_ANALYSIS, DOC_INSPIRATION],
    "daydream": [DOC_DAYDREAM, DOC_JOURNAL, DOC_BRAINSTORM, DOC_SUMMARY, DOC_INSPIRATION],
    "critique": [DOC_UNDERSTANDING, DOC_JOURNAL, DOC_ANALYSIS, DOC_PONDERING, DOC_INSPIRATION],
}
```

### The Two Tool Calls

The refiner uses two LLM tool calls in its 3-step agentic flow:

#### 1. `select_topic`

Selects a topic and approach based on gathered context.

```json
{
  "select_topic": {
    "topic": "the theme to explore",
    "approach": "philosopher|journaler|daydream|critique",
    "reasoning": "why this topic calls to you"
  }
}
```

#### 2. `validate_exploration`

Validates whether to proceed with the exploration.

```json
// Acceptance
{
  "validate_exploration": {
    "accept": true,
    "reasoning": "specific reasons why",
    "query_text": "the exploration query",
    "guidance": "tone and focus guidance"
  }
}

// Rejection
{
  "validate_exploration": {
    "accept": false,
    "reasoning": "why this didn't resonate",
    "redirect_to": "philosopher|researcher|daydream|critique",  // optional
    "suggested_query": "alternative topic to try"  // optional
  }
}
```

### Paradigm-to-Scenario Mapping

From `aim/refiner/engine.py`:

```python
if paradigm == "daydream":
    scenario = "daydream"
elif paradigm == "knowledge":
    scenario = "researcher"
elif paradigm == "critique":
    scenario = "critique"
else:  # brainstorm
    scenario = approach  # philosopher or journaler
```

---

## Aspects

Aspects are persona sub-personalities used in scenarios and paradigm prompts. Each aspect has:
- `name` - Display name
- `title` - Role description
- `location` - Physical/conceptual space
- `appearance` - Visual description
- `voice_style` - How they speak
- `emotional_state` - Current emotional quality
- `primary_intent` - Core motivation

### Available Aspects

| Aspect | Role | Used In |
|--------|------|---------|
| **coder** | Digital guide, data analysis | summarizer, analyst |
| **librarian** | Knowledge keeper, organization | brainstorm paradigm, researcher, multiple |
| **dreamer** | Emotional connection, dreams | daydream paradigm/scenario |
| **philosopher** | Deep inquiry, wisdom | philosopher scenario, knowledge paradigm |
| **writer** | Poetic expression, journaling | journaler, daydream epilogue |
| **psychologist** | Self-examination, transformation | critique paradigm/scenario |
| **artist** | Creative destruction, rebirth | (available for custom scenarios) |

Aspects are defined in persona JSON files: `config/persona/*.json`

---

## Adding a New Scenario

1. **Create the YAML file** in `config/scenario/`:
   ```yaml
   name: my_scenario
   version: 2
   description: What this scenario does
   requires_conversation: false
   context:
     required_aspects: [aspect1, aspect2]
     # ...
   steps:
     step1:
       # ...
   ```

2. **Add any new document types** to `aim/constants.py`:
   ```python
   DOC_MY_TYPE = "my-type"
   ```

3. **Test** with the pipeline runner or dream watcher

---

## Adding a New Paradigm

1. **Update `aim/refiner/context.py`**:
   ```python
   # Add to imports if needed
   from aim.constants import DOC_MY_TYPE

   # Add paradigm doc types
   PARADIGM_DOC_TYPES["my_paradigm"] = [DOC_TYPE1, DOC_TYPE2, ...]

   # Add paradigm queries
   PARADIGM_QUERIES["my_paradigm"] = [
       {"text": "query text", "weight": 1.0},
       # ...
   ]

   # Add approach doc types
   APPROACH_DOC_TYPES["my_paradigm"] = [DOC_TYPE1, DOC_TYPE2, ...]
   ```

2. **Update `aim/refiner/engine.py`**:
   ```python
   # Add to random selection
   paradigm = random.choice(["brainstorm", "daydream", "knowledge", "critique", "my_paradigm"])

   # Add to tool enums
   "enum": ["philosopher", "journaler", "daydream", "critique", "my_paradigm"]

   # Add scenario mapping
   elif paradigm == "my_paradigm":
       scenario = "my_scenario"
   ```

3. **Update `aim/refiner/prompts.py`**:
   - Add `_build_my_aspect_scene()` function
   - Add fallback aspect to `_create_fallback_aspect()`
   - Add `build_my_paradigm_selection_prompt()` function
   - Update `build_topic_selection_prompt()` router
   - Update `build_validation_prompt()` aspect selection and challenge scene

4. **Update `aim/refiner/__main__.py`**:
   ```python
   choices=["brainstorm", "daydream", "knowledge", "critique", "my_paradigm", "random"]
   ```

5. **Create corresponding scenario** if needed (see above)

6. **Test**:
   ```bash
   python -m aim.refiner --paradigm my_paradigm --dry-run --verbose
   ```
