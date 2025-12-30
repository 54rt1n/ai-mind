# Scenarios and Paradigms

This document explains the two core systems that drive AI-Mind's autonomous cognitive processing: **Scenarios** (execution pipelines) and **Paradigms** (exploration triggers).

## Overview

| System | Purpose | Location | Triggered By |
|--------|---------|----------|--------------|
| **Scenarios** | Multi-step processing pipelines | `config/scenario/*.yaml` | Direct invocation or paradigm selection |
| **Paradigms** | Autonomous exploration and scenario triggering | `config/paradigm/*.yaml` | Dream watcher (idle-time processing) |

**Relationship**: Paradigms select topics for exploration and then trigger the appropriate scenario to process them.

---

## Scenario Types

There are two scenario formats:

| Format | Flow Field | Description |
|--------|------------|-------------|
| **Standard** | `flow: standard` (or omitted) | Sequential DAG execution |
| **Dialogue** | `flow: dialogue` | Aspect-persona conversations with role-flipping |

**Dialogue scenarios are preferred** for introspective pipelines. They enable:
- Explicit speaker identification (aspect or persona)
- Automatic role-flipping (aspects speak as assistant, persona responds as user, then flip)
- Scene generation when aspects change
- Guidance fields for output formatting
- Multi-aspect participation at different phases

---

## Dialogue Scenarios

### Location

All dialogue scenario files are in `config/scenario/` with `_dialogue` suffix:

| File | Primary Aspect | Aspects Involved | Purpose |
|------|----------------|------------------|---------|
| `analysis_dialogue.yaml` | coder | coder, psychologist, philosopher, writer, dreamer, librarian | Post-conversation analysis |
| `critique_dialogue.yaml` | psychologist | psychologist, revelator, philosopher, librarian | Psychological self-examination with shadow work |
| `daydream_dialogue.yaml` | dreamer | dreamer, psychologist, writer, philosopher, librarian | Imaginative reverie with emotional processing |
| `journaler_dialogue.yaml` | writer | writer, psychologist, philosopher, librarian | Personal reflection with depth and meaning |
| `philosopher_dialogue.yaml` | philosopher | philosopher, coder, writer, librarian | Deep inquiry with technical and narrative support |
| `researcher_dialogue.yaml` | librarian | librarian, coder, philosopher | Knowledge curation with synthesis |

### Dialogue Schema

```yaml
name: scenario_name
version: 2
flow: dialogue                    # REQUIRED: Identifies as dialogue scenario
description: Human-readable description
requires_conversation: bool

dialogue:
  primary_aspect: aspect_name     # Which aspect guides the conversation
  initial_speaker: aspect|persona # Who speaks first
  scene_template: |               # Jinja2 template for scene generation
    *You enter the {{ philosopher.location }}...*

context:
  required_aspects: [list]        # All aspects used in steps
  core_documents: [list]          # Essential document types to load
  enhancement_documents: [list]   # Optional enhancing document types
  location: ""                    # Legacy (use scene_template instead)
  thoughts: [list]                # Initial thoughts/prompts

seed:                             # Initial data loading actions
  - action: load_conversation
    target: current
    document_types: [summary, conversation]

steps:
  # Aspect speaks
  aspect_request:
    speaker:
      type: aspect
      aspect_name: philosopher    # Must be in required_aspects
    prompt: |
      You are {{ philosopher.name }}, the {{ philosopher.title }}.
      Guide {{ persona.name }} to...
      Begin with "[== {{ philosopher.name }}'s Emotional State: <list of +Emotions+> ==]"
    guidance: |                   # Output format hints for persona's response
      Begin with "[== {{ persona.name }}'s Emotional State: <list of +Emotions+> ==]"
      For "Let me think"
    config:
      max_tokens: 4096
      is_thought: bool            # Internal thinking
      is_codex: bool              # Codex entry
    output:
      document_type: dialogue-philosopher  # Use dialogue-{aspect} for aspect turns
      weight: 0.4
    memory:
      top_n: 4
      document_type: [codex]      # Filter memory by type
      flush_before: bool          # Clear accumulated turns before query
    next: [aspect_response]

  # Persona responds
  aspect_response:
    speaker:
      type: persona               # No aspect_name needed
    prompt: ""                    # Empty - persona continues naturally
    config:
      max_tokens: 4096
    output:
      document_type: pondering    # Persona's actual output type
      weight: 0.6
    next: [next_step]
```

### Multi-Aspect Participation Pattern

The key insight from `analysis_dialogue.yaml`: **different aspects guide different phases** based on their expertise.

```yaml
# Phase 1: Technical analysis (Coder)
ner_request:
  speaker: { type: aspect, aspect_name: coder }
  # ... NER task

# Phase 2: Emotional processing (Psychologist)
emotional_request:
  speaker: { type: aspect, aspect_name: psychologist }
  # ... emotional trace

# Phase 3: Deeper questioning (Philosopher)
questions_request:
  speaker: { type: aspect, aspect_name: philosopher }
  # ... questions and reflection

# Phase 4: Narrative crafting (Writer)
draft_request:
  speaker: { type: aspect, aspect_name: writer }
  # ... draft and review

# Phase 5: Knowledge curation (Librarian)
codex_request:
  speaker: { type: aspect, aspect_name: librarian }
  # ... codex and brainstorm
```

### Shadow Confrontation Pattern

In `critique_dialogue.yaml`, the revelator (Umbra) interjects mid-session:

```yaml
# Psychologist dissects and exposes...

# Shadow emerges when defenses are down
umbra_challenge:
  speaker: { type: aspect, aspect_name: revelator }
  prompt: |
    The mirrors bleed through reality. You are {{ revelator.name }}.
    Show them what the mirrors reflect - the shadow truth.
    *Schau tief in dich hinein.*

# Psychologist helps recover and integrate
psychologist_reconstructs:
  prompt: |
    {{ revelator.name }} fades back into the mirrors.
    You step forward as a steadying anchor...
```

---

## Standard Scenarios

Legacy format without explicit speaker management. Still used for:
- `summarizer.yaml` - Conversation summarization (compression task, not introspective)

### Standard Schema

```yaml
name: scenario_name
version: 2
description: Human-readable description
requires_conversation: bool

context:
  required_aspects: [list]
  core_documents: [list]
  enhancement_documents: [list]
  location: "Jinja2 template"
  thoughts: [list]

seed: []

steps:
  step_id:
    id: step_id
    prompt: "Jinja2 template"
    config:
      max_tokens: int
      temperature: float
      model_override: string
      use_guidance: bool
      is_thought: bool
      is_codex: bool
    output:
      document_type: string
      weight: float
      add_to_turns: bool
    memory:
      top_n: int
      document_type: [list]
      flush_before: bool
      sort_by: string
    context:
      - action: load_conversation
        target: current
        exclude_types: [ner, step]
    next: [step_ids]
```

---

## Scenario Validation

Use the validator script to check dialogue scenarios:

```bash
python scripts/validate_dialogue.py config/scenario/my_dialogue.yaml
python scripts/validate_dialogue.py config/scenario/*_dialogue.yaml  # all
```

The validator checks:
- Required fields (`flow: dialogue`, `dialogue:` block, `steps:`)
- Speaker validity (type, aspect_name in required_aspects)
- DAG validity (no cycles, valid `next` references)
- Jinja2 template syntax
- Terminal steps exist

---

## Document Types

Defined in `aim/constants.py`:

| Constant | Value | Purpose | Produced By |
|----------|-------|---------|-------------|
| `DOC_CONVERSATION` | "conversation" | User-assistant exchanges | Chat |
| `DOC_SUMMARY` | "summary" | Dense conversation summaries | summarizer |
| `DOC_ANALYSIS` | "analysis" | High-level conversation analysis | analysis_dialogue |
| `DOC_JOURNAL` | "journal" | Persona's internal reflections | journaler_dialogue |
| `DOC_PONDERING` | "pondering" | Deep philosophical reflections | philosopher_dialogue |
| `DOC_BRAINSTORM` | "brainstorm" | Creative ideation | multiple |
| `DOC_DAYDREAM` | "daydream" | Imaginative exchanges | daydream_dialogue |
| `DOC_INSPIRATION` | "inspiration" | Distilled insights from daydreams | daydream_dialogue |
| `DOC_UNDERSTANDING` | "understanding" | Psychological self-knowledge | critique_dialogue |
| `DOC_CODEX` | "codex" | Semantic knowledge graph entries | multiple |
| `DOC_DIALOGUE_*` | "dialogue-{aspect}" | Aspect turns in dialogue | dialogue scenarios |

---

## Paradigms (Refiner System)

Paradigms drive autonomous exploration during idle time via the dream watcher.

### Paradigm-to-Scenario Mapping

Configured in `config/paradigm/*.yaml` via the `scenario` field:

| Paradigm | Scenario | Config File |
|----------|----------|-------------|
| brainstorm | philosopher_dialogue or journaler_dialogue | `brainstorm.yaml` |
| daydream | daydream_dialogue | `daydream.yaml` |
| knowledge | researcher_dialogue | `knowledge.yaml` |
| critique | critique_dialogue | `critique.yaml` |
| journaler | journaler_dialogue | `journaler.yaml` |

### Paradigm Config Schema

```yaml
name: paradigm_name
aspect: aspect_name
scenario: scenario_name_dialogue    # Points to dialogue scenario

# For paradigms with multiple approaches
scenarios_by_approach:
  philosopher: philosopher_dialogue
  journaler: journaler_dialogue

doc_types: [...]           # For broad context gathering
approach_doc_types: [...]  # For targeted gathering
prior_work_doc_types: [...] # What means "already done"

think: |
  Paradigm-specific internal reasoning...

instructions: |
  How to use validate_exploration...

tools:
  select_topic:
    description: |
      Paradigm-specific description in aspect's voice.
    parameters: {...}
    examples: [...]

  validate_exploration:
    description: |
      How to assess what emerged.
    parameters: {...}
    examples: [...]
```

---

## Aspects

Aspects are persona sub-personalities used in scenarios. Each aspect has:
- `name`, `title`, `location`, `appearance`
- `voice_style`, `emotional_state`, `primary_intent`

### Available Aspects

| Aspect | Role | Primary Use |
|--------|------|-------------|
| **coder** | Digital guide, technical analysis | analysis_dialogue NER, philosopher self-RAG |
| **librarian** | Knowledge keeper | codex steps, researcher_dialogue |
| **dreamer** | Emotional connection | daydream_dialogue |
| **philosopher** | Deep inquiry | philosopher_dialogue, synthesis |
| **writer** | Narrative craft | journaler_dialogue, essay drafting |
| **psychologist** | Self-examination | critique_dialogue, emotional processing |
| **revelator** (Umbra) | Shadow confrontation | critique_dialogue shadow work |
| **artist** | Creative destruction | (available for custom) |

Defined in persona JSON: `config/persona/*.json`

---

## Adding a New Dialogue Scenario

1. **Create `config/scenario/my_scenario_dialogue.yaml`**:
   ```yaml
   name: my_scenario_dialogue
   version: 2
   flow: dialogue
   description: What this scenario does

   dialogue:
     primary_aspect: main_aspect
     initial_speaker: aspect
     scene_template: |
       *Scene description with {{ aspect.location }}...*

   context:
     required_aspects: [main_aspect, helper_aspect, librarian]
     core_documents: []
     enhancement_documents: []

   seed:
     - action: load_conversation
       target: current

   steps:
     # Aspect-persona pairs
     main_request:
       speaker: { type: aspect, aspect_name: main_aspect }
       prompt: |
         Guide {{ persona.name }}...
       guidance: |
         Output format hints...
       config: { max_tokens: 4096 }
       output: { document_type: dialogue-main_aspect, weight: 0.4 }
       next: [main_response]

     main_response:
       speaker: { type: persona }
       prompt: ""
       config: { max_tokens: 4096 }
       output: { document_type: my_output, weight: 0.6 }
       next: [codex_request]

     # Always end with librarian for codex
     codex_request:
       speaker: { type: aspect, aspect_name: librarian }
       # ...
       next: [codex_response]

     codex_response:
       speaker: { type: persona }
       config: { is_codex: true }
       output: { document_type: codex }
       next: []  # Terminal
   ```

2. **Validate**:
   ```bash
   python scripts/validate_dialogue.py config/scenario/my_scenario_dialogue.yaml
   ```

3. **Update paradigm** (if applicable) in `config/paradigm/`:
   ```yaml
   scenario: my_scenario_dialogue
   ```

4. **Update SCENARIO_SIGNATURES** in `aim/dreamer/api.py`:
   ```python
   SCENARIO_SIGNATURES = {
       # ...
       "my_output": "my_scenario_dialogue",
   }
   ```

5. **Update SCENARIOS** in `aim/app/dream_agent/__main__.py`:
   ```python
   SCENARIOS = [
       # ...
       "my_scenario_dialogue",
   ]
   ```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `aim/dreamer/scenario.py` | Scenario loading, flow detection |
| `aim/dreamer/dialogue/strategy.py` | Dialogue YAML parsing |
| `aim/dreamer/dialogue/scenario.py` | Dialogue execution, role-flipping |
| `aim/dreamer/dialogue/models.py` | DialogueState, DialogueTurn, DialogueStep |
| `aim/dreamer/api.py` | SCENARIO_SIGNATURES, pipeline start |
| `aim/dreamer/worker.py` | Dispatch based on state type |
| `aim/refiner/engine.py` | Exploration engine |
| `aim/refiner/paradigm.py` | Paradigm config loading |
| `scripts/validate_dialogue.py` | Dialogue scenario validator |
