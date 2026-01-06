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

    Args:
        response: Raw response from LLM

    Returns:
        Tuple of (cleaned_content, think_content or None)
    """
    think_content = None

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
            logger.info(f"Extracted think content from orphan tag, length: {len(think_content)}")
            return cleaned_response, think_content

    return response, None

