# mindai/utils/string.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import logging
import re
from typing import Optional

from ..config import ChatConfig


logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    """
    Counts the number of words in a string.

    Args:
        text (str): The string to count the words of.

    Returns:
        int: The number of words in the string.
    """
    # use a regular expression to convert newlines and any whitespace to spaces, and handle multiple spaces as one.
    text = re.sub(r"\s+", " ", text)
    return len(text.split())


class Patterns:
    def __init__(self, config: ChatConfig):
        self.config = config
        
        self.patterns = {
            "total_steps": re.compile(r"(?:Total\s+Steps?|Steps?\s+Total):\s*(\d+)", re.IGNORECASE)
        }

    def extract_total_steps(self, response: str) -> Optional[str]:
        matches = self.patterns['total_steps'].findall(response)
        if matches:
            for match in matches:
                try:
                    return int(match)
                except ValueError:
                    pass
        return None