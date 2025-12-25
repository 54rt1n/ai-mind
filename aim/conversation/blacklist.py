# aim/conversation/blacklist.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

# Application-specific blacklist words (metadata, headers, etc.)
BLACKLIST_WORDS = frozenset({
    "Emotional", "State", "Header",
})

# Common contractions - removed entirely from queries
CONTRACTIONS = frozenset({
    "i'm", "i'll", "i've", "i'd",
    "you're", "you'll", "you've", "you'd",
    "he's", "he'll", "he'd", "she's", "she'll", "she'd",
    "it's", "it'll", "it'd",
    "we're", "we'll", "we've", "we'd",
    "they're", "they'll", "they've", "they'd",
    "that's", "that'll", "that'd",
    "who's", "who'll", "who'd", "what's", "what'll", "what'd",
    "where's", "where'll", "where'd", "when's", "when'll", "when'd",
    "why's", "why'll", "why'd", "how's", "how'll", "how'd",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "doesn't", "don't", "didn't",
    "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mightn't", "mustn't",
    "let's", "here's", "there's",
})

# Standard English stopwords (based on NLTK's English stopword list)
STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
})

# Combined set for efficient lookup
ALL_FILTERED_WORDS = STOPWORDS | BLACKLIST_WORDS
