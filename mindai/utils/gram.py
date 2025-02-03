# mindai/utils/gram.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import re

def find_capitalized_ngrams(text: str, max_n: int = 3) -> list[str]:
    """
    Find capitalized 1-grams, 2-grams, and 3-grams in the text.

    Args:
        text (str): The input text.
        max_n (int): The maximum number of consecutive capitalized words to consider as an n-gram.

    Returns:
        List[str]: A list of matched capitalized n-grams.
    """
    # Patterns to match
    mixed_case_pattern = re.compile(r'\b[A-Z][a-z]*[A-Z][a-z]*\b')
    capitalized_word_pattern = re.compile(r'\b[A-Z][a-z]+\b')

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    matches = []

    for sentence in sentences:
        # Tokenize words, remove punctuation
        words = re.findall(r'\b\w+\b', sentence)
        if not words:
            continue

        idx = 0
        while idx < len(words):
            word = words[idx]

            # Check if the word is mixed-case
            if mixed_case_pattern.match(word):
                matches.append(word)

            # Handle sentence-initial word
            if idx == 0:
                idx += 1
                continue

            # Check for capitalized n-grams (excluding sentence-initial word)
            ngram_words = []
            n = 0
            while n < max_n and idx + n < len(words):
                next_word = words[idx + n]
                if capitalized_word_pattern.match(next_word):
                    ngram_words.append(next_word)
                    n += 1
                else:
                    break

            if ngram_words:
                matches.append(' '.join(ngram_words))
                idx += n
            else:
                idx += 1

    # Remove duplicates
    matches = set(matches)
    
    # Add singular and/or plural
    for match in list(matches):
        singular_match = match.rstrip('s')
        # TODO add es's
        plural_match = match + 's'
        if singular_match in matches:
            matches.add(plural_match)
        if plural_match in matches:
            matches.add(singular_match)

    return list(matches)