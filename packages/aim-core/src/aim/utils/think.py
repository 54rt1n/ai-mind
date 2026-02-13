# aim/utils/think.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

def extract_think_tags(response: str) -> tuple[str, Optional[str]]:
    """
    Extract <think> tags from response, returning (content, think).

    Removes all <think>...</think> blocks from the response and
    concatenates their content as the think output.

    Handles edge cases:
    - Truncated <think> tag (no closing </think>): treat as content
    - Orphan </think> tag (started mid-stream): content before it is think
    - Malformed close tag "/think>": normalized to </think>

    Args:
        response: Raw response from LLM

    Returns:
        Tuple of (cleaned_content, think_content or None)
    """
    think_content = None

    # Normalize malformed close tags "/think>" (missing "<") to "</think>"
    # so standard extraction logic can handle them.
    normalized_response, normalized_count = re.subn(r'(?<!<)/think>', '</think>', response)
    if normalized_count:
        logger.info(
            "Malformed think close tag '/think>' detected; normalizing %d occurrence(s)",
            normalized_count,
        )
        response = normalized_response

    if re.search(r'<think>', response):
        logger.info("Response contains a think xml tag")
        # Find complete think blocks
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response, re.DOTALL)

        if think_matches:
            # Concatenate all think content
            think_content = "\n\n".join(match.strip() for match in think_matches)
            # Remove think tags from response
            cleaned_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
            # Remove any stray closing tags left behind by malformed or nested tags.
            cleaned_response = re.sub(r'</think>', '', cleaned_response).strip()
            logger.info(f"Extracted think content, length: {len(think_content)}")
            return cleaned_response, think_content
        else:
            # Only <think> tag with no </think> means truncated output
            # Treat everything after <think> as content, not think
            logger.info("Truncated think tag detected, treating as content")
            response = response.replace('<think>', '').strip()
            return response, None

    elif re.search(r'</think>', response):
        # Only </think> means we started mid-stream, content before it is think
        logger.info("Response contains orphan think close tag")
        match = re.search(r'(.*?)</think>', response, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            cleaned_response = re.sub(r'(.*?)</think>', '', response, flags=re.DOTALL).strip()
            # Remove any additional stray closing tags.
            cleaned_response = re.sub(r'</think>', '', cleaned_response).strip()
            logger.info(f"Extracted think content from orphan tag, length: {len(think_content)}")
            return cleaned_response, think_content

    return response, None


def extract_reasoning_block(response: str) -> tuple[str, Optional[str]]:
    """Extract <reasoning>...</reasoning> block from response.

    Args:
        response: Raw response from LLM (after think tag extraction)

    Returns:
        Tuple of (cleaned_response, reasoning_content or None)
    """
    pattern = r'<reasoning>(.*?)</reasoning>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        cleaned = re.sub(pattern, '', response, flags=re.DOTALL).strip()
        return cleaned, reasoning
    return response, None
