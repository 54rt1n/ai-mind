# aim/refiner/tools.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Refiner tool definitions.

Single source of truth for the select_topic and validate_exploration tools
used by the ExplorationEngine.
"""

from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters

# Valid exploration approaches - shared constant
EXPLORATION_APPROACHES = ["philosopher", "journaler", "daydream", "critique"]


def get_select_topic_tool() -> Tool:
    """
    Get the select_topic tool definition.

    This tool is used in the first step of exploration to select
    a topic and approach based on gathered context.
    """
    return Tool(
        type="refiner",
        function=ToolFunction(
            name="select_topic",
            description="Select a topic to explore in depth based on gathered context",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "topic": {
                        "type": "string",
                        "description": "The topic, theme, or concept to explore"
                    },
                    "approach": {
                        "type": "string",
                        "enum": EXPLORATION_APPROACHES,
                        "description": "The exploration approach"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this topic is worth exploring"
                    },
                },
                required=["topic", "approach", "reasoning"],
                examples=[
                    {"select_topic": {"topic": "consciousness", "approach": "philosopher", "reasoning": "Underexplored"}}
                ],
            ),
        ),
    )


def get_validate_tool() -> Tool:
    """
    Get the validate_exploration tool definition.

    This tool is used in the second step to validate whether
    the selected topic is truly worth exploring.
    """
    return Tool(
        type="refiner",
        function=ToolFunction(
            name="validate_exploration",
            description="Validate whether a topic is truly worth exploring",
            parameters=ToolFunctionParameters(
                type="object",
                properties={
                    "accept": {
                        "type": "boolean",
                        "description": "Whether to proceed with the exploration"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for accepting or rejecting"
                    },
                    "query_text": {
                        "type": "string",
                        "description": "The refined query to explore (if accept=true)"
                    },
                    "guidance": {
                        "type": "string",
                        "description": "Optional guidance for the exploration"
                    },
                    "redirect_to": {
                        "type": "string",
                        "enum": ["philosopher", "researcher", "daydream", "critique"],
                        "description": "Alternative scenario to redirect to (if rejecting but topic has potential)"
                    },
                    "suggested_query": {
                        "type": "string",
                        "description": "If rejecting, suggest an alternative unexplored topic to try next"
                    },
                },
                required=["accept", "reasoning"],
                examples=[
                    {"validate_exploration": {"accept": True, "reasoning": "Rich topic", "query_text": "What is consciousness?"}},
                    {"validate_exploration": {"accept": False, "reasoning": "Needs deeper pondering first", "redirect_to": "philosopher"}},
                    {"validate_exploration": {"accept": False, "reasoning": "Already explored this", "suggested_query": "the nature of forgetting"}}
                ],
            ),
        ),
    )
