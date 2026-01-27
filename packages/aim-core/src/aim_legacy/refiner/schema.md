# Paradigm Configuration Schema

## Architecture

```
YAML Config → Paradigm (domain object) → Engine (orchestration)
```

- **YAML**: All paradigm-specific data
- **Paradigm**: Domain object that knows how to use that data
- **Engine**: Orchestrates the flow using Paradigm objects

## YAML Schema

```yaml
aspect: string  # Guiding persona aspect

# Document gathering
doc_types: list[string]
approach_doc_types:
  <approach>: list[string]

# Scenario routing
scenario: string | null
scenarios_by_approach:
  <approach>: string

# Phase 1: Selection
prompts:
  selection:
    scene: string      # Immersive scene setting
    think: string      # What to consider before selecting
    instructions: string  # How to use select_topic tool
  validation:
    scene: string      # Scene for validation phase
    think: string      # What to consider before validating
    instructions: string  # How to use validate_exploration tool

# Tool definitions
tools:
  select_topic:
    description: string
    parameters:
      <param>:
        type: string
        description: string
        enum: list[string]  # optional
    required: list[string]
    examples: list[object]
  validate_exploration:
    description: string
    parameters: ...
    required: ...
    examples: ...
```

## Paradigm Class

```python
class Paradigm:
    """Domain object representing an exploration paradigm."""

    aspect: str
    doc_types: list[str]
    approach_doc_types: dict[str, list[str]]
    scenario: str | None
    scenarios_by_approach: dict[str, str]
    prompts: dict
    tools: dict

    @classmethod
    def load(cls, name: str) -> "Paradigm":
        """Load from YAML. Raises if config invalid."""

    @classmethod
    def available(cls) -> list[str]:
        """List available paradigm names from config directory."""

    def build_selection_prompt(self, documents: list, persona: Persona) -> tuple[str, str]:
        """Build Phase 1 system/user messages."""

    def build_validation_prompt(self, documents: list, persona: Persona,
                                 topic: str, approach: str) -> tuple[str, str]:
        """Build Phase 2 system/user messages."""

    def get_scenario(self, approach: str | None) -> str:
        """Get scenario for given approach."""

    def get_approach_doc_types(self, approach: str) -> list[str]:
        """Get doc types for targeted gathering."""

    def get_select_tool(self) -> Tool:
        """Get select_topic tool definition."""

    def get_validate_tool(self) -> Tool:
        """Get validate_exploration tool definition."""
```

## Engine Usage

```python
class ExplorationEngine:
    async def run_exploration(self):
        # Pick paradigm
        name = random.choice(Paradigm.available())
        paradigm = Paradigm.load(name)

        # Phase 1: Selection
        docs = await self.gather_broad(paradigm.doc_types)
        prompt = paradigm.build_selection_prompt(docs, self.persona)
        selection = await self.llm_call(prompt, paradigm.get_select_tool())

        topic = selection["topic"]
        approach = selection.get("approach", name)

        # Phase 2: Validation
        docs = await self.gather_targeted(paradigm.get_approach_doc_types(approach), topic)
        prompt = paradigm.build_validation_prompt(docs, self.persona, topic, approach)
        validation = await self.llm_call(prompt, paradigm.get_validate_tool())

        if not validation["accept"]:
            return None, validation.get("suggested_query")

        # Trigger pipeline
        scenario = paradigm.get_scenario(approach)
        return await self.trigger_pipeline(scenario, validation["query_text"])
```

## File Structure After Refactor

```
aim/refiner/
  __init__.py
  paradigm.py      # Paradigm domain class
  engine.py        # ExplorationEngine (simplified)
  context.py       # ContextGatherer (unchanged mostly)
  __main__.py      # CLI

config/paradigm/
  brainstorm.yaml  # Full config with prompts
  daydream.yaml
  knowledge.yaml
  critique.yaml
  journaler.yaml
```

## What Gets Deleted

- `prompts.py` - 1000+ lines of hardcoded prompts → moves to YAML
- `paradigm_config.py` - replaced by Paradigm class
- `tools.py` - tool defs come from YAML
- `plan.py` / old `paradigm.py` - ExplorationPlan unused
