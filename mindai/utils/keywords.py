# mindai/utils/keywords.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
import logging
import re
from typing import Optional

from ..conversation.model import ConversationModel

logger = logging.getLogger(__name__)

def extract_semantic_keywords(text : str) -> dict[str, int]:
    """
    Extract semantic keywords from a given text, and joins then into a single string.
    
    Args:
        text (str): The text to extract keywords from.

    """
    # semantic keywords are in the format **One-Two Threee**
    # It should exclude parens
    matcher = re.compile(r'(\*\*.*?\*\*)') # This is incorrect, as it includes (), commas
    matches = matcher.findall(text)
    
    # Second phase: Filter out entries with invalid characters
    invalid_chars = r'[{}<>,;]'
    filtered_matches = defaultdict(int)
    for match in matches:
        if len(match) < 5:
            pass
            #logger.info(f"Invalid match: {match}")
        elif match[-3] in (":", ".", "!", "?", ' '):
            pass
            #print(f"Invalid match: {match}")
        elif re.search(invalid_chars, match):
            pass
            #logger.info(f"Invalid match: {match}")
        else:
            # Filter out quotes and parens from the text
            filtered = re.sub(r'[\[\]"]', '', match)
            filtered_matches[filtered] += 1
    
    return filtered_matches

def get_all_keywords(cvm: ConversationModel, document_type: Optional[str] = None) -> list[tuple[str, int]]:
        
    queries = cvm.to_pandas()
    if document_type is not None:
        queries = queries[queries['document_type'] == document_type]
    keyword_dict = queries['content'].apply(extract_semantic_keywords)
    # Union of all the sets
    if queries.empty:
        return {}
    flattened = defaultdict(int)
    for query in keyword_dict:
        for  key, value in query.items():
            flattened[key] += value

    return sorted([(k, v) for k, v in flattened.items()], key=lambda x: x[1], reverse=True)