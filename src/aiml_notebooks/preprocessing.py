"""
Text preprocessing utilities.

This module provides reusable text preprocessing functions for NLP tasks:
- HTML and special character removal
- Tokenization
- Normalization
- Text cleaning pipelines
"""

import re
from typing import List, Optional, Callable
from collections import Counter


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Example:
        >>> preprocessor = TextPreprocessor(
        ...     lowercase=True,
        ...     remove_html=True,
        ...     remove_punctuation=False
        ... )
        >>> clean_text = preprocessor("This is <b>HTML</b> text!")
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        remove_extra_whitespace: bool = True,
        custom_replacements: Optional[dict] = None
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Convert to lowercase
            remove_html: Remove HTML tags
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_numbers: Remove numbers
            remove_punctuation: Remove punctuation
            remove_extra_whitespace: Remove extra whitespace
            custom_replacements: Dict of custom replacements {pattern: replacement}
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.custom_replacements = custom_replacements or {}

    def __call__(self, text: str) -> str:
        """Preprocess text."""
        return self.preprocess(text)

    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', ' ', text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', ' ', text)

        # Custom replacements
        for pattern, replacement in self.custom_replacements.items():
            text = re.sub(pattern, replacement, text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())

        return text


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization.

    Args:
        text: Input text

    Returns:
        List of tokens

    Example:
        >>> tokens = simple_tokenize("Hello world!")
        >>> print(tokens)  # ['Hello', 'world!']
    """
    return text.split()


def tokenize_with_punctuation(text: str) -> List[str]:
    """
    Tokenize while separating punctuation.

    Args:
        text: Input text

    Returns:
        List of tokens

    Example:
        >>> tokens = tokenize_with_punctuation("Hello, world!")
        >>> print(tokens)  # ['Hello', ',', 'world', '!']
    """
    # Add spaces around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    # Split on whitespace
    tokens = text.split()
    return tokens


def build_vocabulary_from_texts(
    texts: List[str],
    max_size: Optional[int] = None,
    min_freq: int = 1,
    special_tokens: Optional[List[str]] = None,
    tokenizer: Optional[Callable] = None
) -> dict:
    """
    Build vocabulary from a list of texts.

    Args:
        texts: List of text strings
        max_size: Maximum vocabulary size (None = unlimited)
        min_freq: Minimum frequency to include a word
        special_tokens: List of special tokens to add (e.g., ['<PAD>', '<UNK>'])
        tokenizer: Custom tokenizer function (default: simple_tokenize)

    Returns:
        Dictionary mapping words to indices

    Example:
        >>> texts = ["hello world", "hello there"]
        >>> vocab = build_vocabulary_from_texts(texts, special_tokens=['<PAD>', '<UNK>'])
        >>> print(vocab)  # {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3, 'there': 4}
    """
    if tokenizer is None:
        tokenizer = simple_tokenize

    # Count words
    word_counts = Counter()
    for text in texts:
        tokens = tokenizer(text)
        word_counts.update(tokens)

    # Filter by frequency
    word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit vocabulary size
    if max_size is not None:
        n_special = len(special_tokens) if special_tokens else 0
        sorted_words = sorted_words[:max_size - n_special]

    # Build vocabulary
    vocab = {}

    # Add special tokens first
    if special_tokens:
        for idx, token in enumerate(special_tokens):
            vocab[token] = idx

    # Add regular words
    start_idx = len(vocab)
    for idx, (word, _) in enumerate(sorted_words, start=start_idx):
        vocab[word] = idx

    return vocab


def encode_text(
    text: str,
    vocab: dict,
    tokenizer: Optional[Callable] = None,
    unk_token: str = '<UNK>'
) -> List[int]:
    """
    Encode text to list of indices using vocabulary.

    Args:
        text: Input text
        vocab: Vocabulary dictionary (word -> index)
        tokenizer: Custom tokenizer function
        unk_token: Token to use for unknown words

    Returns:
        List of token indices

    Example:
        >>> vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3}
        >>> indices = encode_text("hello world", vocab)
        >>> print(indices)  # [2, 3]
    """
    if tokenizer is None:
        tokenizer = simple_tokenize

    tokens = tokenizer(text)
    unk_idx = vocab.get(unk_token, 1)  # Default UNK index is 1

    indices = [vocab.get(token, unk_idx) for token in tokens]
    return indices


def decode_text(
    indices: List[int],
    vocab: dict,
    join_str: str = ' '
) -> str:
    """
    Decode list of indices back to text using vocabulary.

    Args:
        indices: List of token indices
        vocab: Vocabulary dictionary (word -> index)
        join_str: String to join tokens with

    Returns:
        Decoded text

    Example:
        >>> vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3}
        >>> indices = [2, 3]
        >>> text = decode_text(indices, vocab)
        >>> print(text)  # "hello world"
    """
    # Create reverse vocabulary (index -> word)
    idx2word = {idx: word for word, idx in vocab.items()}

    tokens = [idx2word.get(idx, '<UNK>') for idx in indices]
    return join_str.join(tokens)


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_value: int = 0,
    truncate: str = 'right',
    pad_side: str = 'right'
) -> List[int]:
    """
    Pad or truncate sequence to fixed length.

    Args:
        sequence: Input sequence
        max_length: Target length
        pad_value: Value to use for padding
        truncate: Which side to truncate ('left' or 'right')
        pad_side: Which side to pad ('left' or 'right')

    Returns:
        Padded/truncated sequence

    Example:
        >>> seq = [1, 2, 3]
        >>> padded = pad_sequence(seq, max_length=5, pad_value=0)
        >>> print(padded)  # [1, 2, 3, 0, 0]
    """
    seq_len = len(sequence)

    if seq_len > max_length:
        # Truncate
        if truncate == 'right':
            return sequence[:max_length]
        else:  # truncate left
            return sequence[-max_length:]
    elif seq_len < max_length:
        # Pad
        padding = [pad_value] * (max_length - seq_len)
        if pad_side == 'right':
            return sequence + padding
        else:  # pad left
            return padding + sequence
    else:
        return sequence


def batch_encode_texts(
    texts: List[str],
    vocab: dict,
    max_length: Optional[int] = None,
    pad_value: int = 0,
    tokenizer: Optional[Callable] = None,
    unk_token: str = '<UNK>'
) -> List[List[int]]:
    """
    Encode and pad a batch of texts.

    Args:
        texts: List of input texts
        vocab: Vocabulary dictionary
        max_length: Maximum sequence length (None = use longest in batch)
        pad_value: Value for padding
        tokenizer: Custom tokenizer
        unk_token: Unknown token

    Returns:
        List of encoded and padded sequences

    Example:
        >>> texts = ["hello world", "hi"]
        >>> vocab = {'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3, 'hi': 4}
        >>> encoded = batch_encode_texts(texts, vocab, max_length=3)
        >>> print(encoded)  # [[2, 3, 0], [4, 0, 0]]
    """
    # Encode all texts
    encoded = [encode_text(text, vocab, tokenizer, unk_token) for text in texts]

    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in encoded)

    # Pad all sequences
    padded = [pad_sequence(seq, max_length, pad_value) for seq in encoded]

    return padded


def remove_stopwords(
    text: str,
    stopwords: Optional[List[str]] = None,
    tokenizer: Optional[Callable] = None
) -> str:
    """
    Remove stopwords from text.

    Args:
        text: Input text
        stopwords: List of stopwords (None = use default English stopwords)
        tokenizer: Custom tokenizer

    Returns:
        Text with stopwords removed

    Example:
        >>> text = "this is a test"
        >>> cleaned = remove_stopwords(text, stopwords=['is', 'a'])
        >>> print(cleaned)  # "this test"
    """
    if stopwords is None:
        # Default English stopwords
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
    else:
        stopwords = set(stopwords)

    if tokenizer is None:
        tokenizer = simple_tokenize

    tokens = tokenizer(text)
    filtered = [token for token in tokens if token.lower() not in stopwords]

    return ' '.join(filtered)


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_accents: bool = False,
    replace_contractions: bool = False
) -> str:
    """
    Normalize text with various options.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_accents: Remove accented characters
        replace_contractions: Expand contractions (e.g., "don't" -> "do not")

    Returns:
        Normalized text

    Example:
        >>> text = "Don't café"
        >>> normalized = normalize_text(text, remove_accents=True, replace_contractions=True)
        >>> print(normalized)  # "do not cafe"
    """
    if lowercase:
        text = text.lower()

    if replace_contractions:
        # Common English contractions
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

    if remove_accents:
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ASCII', 'ignore').decode('ASCII')

    return text
