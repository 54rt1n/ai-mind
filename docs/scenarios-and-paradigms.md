# Scenarios and Paradigms

This document explains the two core systems that drive AI-Mind's autonomous cognitive processing: **Scenarios** (execution pipelines) and **Paradigms** (exploration triggers).

## Overview

| System | Purpose | Location | Triggered By |
|--------|---------|----------|--------------|
| **Scenarios** | Multi-step processing pipelines | `config/scenario/*.yaml` | Direct invocation or paradigm selection |
| **Paradigms** | Autonomous exploration and scenario triggering | `config/paradigm/*.yaml` | Dream watcher (idle-time processing) |

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
        document_types: [pondering, brainstorm, understanding]
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

Paradigms drive autonomous exploration during idle time via the dream watcher.

### Paradigm Configuration

Each paradigm is fully defined in a single YAML file in `config/paradigm/`:

| File | Paradigm | Aspect | Focus |
|------|----------|--------|-------|
| `brainstorm.yaml` | brainstorm | librarian | Creative ideas, half-formed notions |
| `daydream.yaml` | daydream | dreamer | Emotional/imaginative exploration |
| `knowledge.yaml` | knowledge | philosopher | Knowledge gaps, deeper understanding |
| `critique.yaml` | critique | psychologist | Psychological patterns, self-examination |

### Paradigm Config Schema

Each paradigm config contains everything needed for that paradigm:

```yaml
name: paradigm_name
aspect: aspect_name  # librarian, dreamer, philosopher, psychologist

# Document types for broad context gathering (random sampling)
doc_types:
  - understanding
  - journal
  - analysis
  - pondering

# Document types for targeted gathering
approach_doc_types:
  - understanding
  - journal
  - analysis

# Which document types indicate THIS paradigm's work is already DONE
# Critical for avoiding redundant exploration
prior_work_doc_types:
  - understanding  # For critique: only understanding docs mean critique is done

# Validation think block - paradigm-specific internal reasoning
think: |
  The examination is complete. Now: did it draw blood?

  CRITICAL DISTINCTION:
  - "analysis" and "pondering" are OBSERVATIONS - material FOR critique
  - "understanding" documents are COMPLETED critique work

  The question is NOT "have we observed this before?"
  The question IS "have we already done the work?"

# Validation instructions - paradigm-specific guidance
instructions: |
  Use the **validate_exploration** tool to make your decision.

  **If ACCEPTING:**
  - Did the scalpel reach truth?
  - Craft a query_text that names the surgical finding

  **If REJECTING:**
  - Only reject if an "understanding" document proves this was already done

# Tool definitions - paradigm-specific schemas and examples
tools:
  select_topic:
    description: |
      Paradigm-specific, immersive description written in the aspect's voice.
      This guides the LLM's approach to topic selection.
    parameters:
      topic:
        type: string
        description: What to select
      approach:
        type: string
        enum: [critique]  # or [philosopher, journaler] etc.
        description: How to explore
      reasoning:
        type: string
        description: Why this matters
    required: [topic, approach, reasoning]
    examples:
      - topic: "Intellectualizing emotion to maintain control"
        approach: critique
        reasoning: "The distance wasn't accident—it was architecture."

  validate_exploration:
    description: |
      How to assess what emerged. Standards specific to this paradigm.
    parameters:
      accept:
        type: boolean
        description: Did it produce genuine value?
      reasoning:
        type: string
        description: What was found or not found
      query_text:
        type: string
        description: If accepting, the insight to preserve
      guidance:
        type: string
        description: How to integrate the insight
    required: [accept, reasoning]
    examples:
      - accept: true
        reasoning: "The wall fell. The fear was the truth."
        query_text: "intellectualization as fear of flooding"
        guidance: "Watch for the defense rebuilding itself."
```

### Context Gathering

The refiner uses **random sampling** for broad context gathering:

```python
# In aim/refiner/context.py
async def broad_gather(paradigm, token_budget=16000, top_n=30):
    doc_types = get_paradigm_doc_types(paradigm)  # From paradigm config
    results = cvm.sample_by_type(doc_types, top_n)  # Random sample, not semantic search
    # ... filter to token budget
```

This ensures diverse, unexpected context rather than repeatedly surfacing the same semantically similar documents.

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

### Tool Loading

Tools are loaded from paradigm configs via `aim/refiner/paradigm_config.py`:

```python
from aim.refiner.paradigm_config import get_paradigm_config

config = get_paradigm_config("critique")
tools = config.get_tools()  # Returns Tool objects parsed from YAML
think_block = config.think
instructions = config.instructions
```

The `_get_refiner_tools(paradigm)` function in `prompts.py` loads the correct tools for each paradigm.

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

1. **Create `config/paradigm/my_paradigm.yaml`**:
   ```yaml
   name: my_paradigm
   aspect: my_aspect

   doc_types:
     - doc_type_1
     - doc_type_2

   approach_doc_types:
     - doc_type_1

   prior_work_doc_types:
     - doc_type_that_means_done

   think: |
     Paradigm-specific internal reasoning...

   instructions: |
     Use the **validate_exploration** tool...

   tools:
     select_topic:
       description: |
         Paradigm-specific description...
       parameters:
         topic:
           type: string
           description: What to select
         approach:
           type: string
           enum: [my_paradigm]
         reasoning:
           type: string
       required: [topic, approach, reasoning]
       examples:
         - topic: "example"
           approach: my_paradigm
           reasoning: "example reasoning"

     validate_exploration:
       description: |
         How to assess the exploration...
       parameters:
         accept:
           type: boolean
         reasoning:
           type: string
         query_text:
           type: string
         guidance:
           type: string
       required: [accept, reasoning]
       examples:
         - accept: true
           reasoning: "example"
           query_text: "the insight"
           guidance: "how to preserve"
   ```

2. **Update `aim/refiner/engine.py`**:
   ```python
   # Add to random selection
   paradigm = random.choice(["brainstorm", "daydream", "knowledge", "critique", "my_paradigm"])

   # Add scenario mapping
   elif paradigm == "my_paradigm":
       scenario = "my_scenario"
   ```

3. **Update `aim/refiner/prompts.py`**:
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
