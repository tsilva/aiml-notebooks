"""
Tokenizer classes for text data.

This module provides tokenizer classes for converting text to sequences of indices
and vice versa. These are reused across multiple notebooks for consistency.

Available tokenizers:
- CharacterTokenizer: Character-level tokenization
- WordTokenizer: Word-level tokenization with vocabulary building
"""

from collections import Counter
from typing import List, Optional


class CharacterTokenizer:
    """
    Character-level tokenizer that builds a vocabulary from input texts.

    This tokenizer:
    - Builds a sorted vocabulary from all unique characters in the input
    - Adds a special token (default '.') for start/end of sequence
    - Provides encode/decode methods for converting between text and indices

    Args:
        texts: List of strings to build vocabulary from
        special_token: Character to use as start/end token (default '.')

    Attributes:
        chars: List of all characters in vocabulary (special token first)
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        special_token: The special start/end token character
        vocab_size: Total number of characters in vocabulary

    Example:
        >>> tokenizer = CharacterTokenizer(['hello', 'world'])
        >>> tokenizer.encode('hello')
        [4, 2, 5, 5, 6]
        >>> tokenizer.decode([4, 2, 5, 5, 6])
        'hello'
    """

    def __init__(self, texts, special_token='.'):
        """Build vocabulary from a list of texts."""
        # Build character vocabulary
        chars = sorted(list(set(''.join(texts))))
        self.chars = [special_token] + chars

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        self.special_token = special_token
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """
        Convert text to list of indices.

        Args:
            text: String to encode

        Returns:
            List of integer indices corresponding to each character
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """
        Convert list of indices to text.

        Args:
            indices: List of integer indices

        Returns:
            String formed by concatenating characters at each index
        """
        return ''.join([self.idx_to_char[i] for i in indices])

    def encode_char(self, char):
        """
        Convert a single character to its index.

        Args:
            char: Single character to encode

        Returns:
            Integer index of the character
        """
        return self.char_to_idx[char]

    def decode_char(self, idx):
        """
        Convert a single index to its character.

        Args:
            idx: Integer index to decode

        Returns:
            Character at the given index
        """
        return self.idx_to_char[idx]

    def get_special_token_idx(self):
        """
        Get the index of the special token.

        Returns:
            Integer index of the special token
        """
        return self.char_to_idx[self.special_token]

    def is_special_token(self, char):
        """
        Check if a character is the special token.

        Args:
            char: Character to check

        Returns:
            Boolean indicating if the character is the special token
        """
        return char == self.special_token

    def __repr__(self):
        return f"CharacterTokenizer(vocab_size={self.vocab_size}, chars={''.join(self.chars)})"


class WordTokenizer:
    """
    Word-level tokenizer that builds vocabulary from input texts.

    This tokenizer:
    - Builds vocabulary from words (whitespace-separated tokens)
    - Supports special tokens (<PAD>, <UNK>, <SOS>, <EOS>)
    - Can limit vocabulary size and set minimum word frequency
    - Provides encode/decode methods

    Args:
        special_tokens: List of special tokens (default: ['<PAD>', '<UNK>'])
        max_vocab_size: Maximum vocabulary size (None = unlimited)
        min_freq: Minimum frequency for a word to be included

    Attributes:
        word_to_idx: Dictionary mapping words to indices
        idx_to_word: Dictionary mapping indices to words
        vocab_size: Total vocabulary size (including special tokens)
        word_counts: Counter of word frequencies

    Example:
        >>> texts = ['hello world', 'hello there']
        >>> tokenizer = WordTokenizer()
        >>> tokenizer.build_vocab(texts)
        >>> tokenizer.encode('hello world')
        [2, 3]
        >>> tokenizer.decode([2, 3])
        'hello world'
    """

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        max_vocab_size: Optional[int] = None,
        min_freq: int = 1
    ):
        """Initialize word tokenizer."""
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>']

        self.special_tokens = special_tokens
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Initialize mappings with special tokens
        self.word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx_to_word = {idx: token for idx, token in enumerate(special_tokens)}
        self.word_counts = Counter()
        self.vocab_size = len(special_tokens)

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts.

        Args:
            texts: List of text strings to build vocabulary from
        """
        # Count words
        for text in texts:
            words = text.split()
            self.word_counts.update(words)

        # Filter by minimum frequency
        filtered_words = {
            word: count for word, count in self.word_counts.items()
            if count >= self.min_freq
        }

        # Sort by frequency (most common first)
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        # Limit vocabulary size
        if self.max_vocab_size is not None:
            n_special = len(self.special_tokens)
            sorted_words = sorted_words[:self.max_vocab_size - n_special]

        # Add words to vocabulary
        start_idx = len(self.special_tokens)
        for idx, (word, _) in enumerate(sorted_words, start=start_idx):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)

    def encode(self, text: str, unk_token: str = '<UNK>') -> List[int]:
        """
        Convert text to list of indices.

        Args:
            text: String to encode
            unk_token: Token to use for unknown words

        Returns:
            List of integer indices
        """
        words = text.split()
        unk_idx = self.word_to_idx.get(unk_token, 1)
        return [self.word_to_idx.get(word, unk_idx) for word in words]

    def decode(self, indices: List[int], skip_special: bool = False) -> str:
        """
        Convert list of indices to text.

        Args:
            indices: List of integer indices
            skip_special: Whether to skip special tokens in output

        Returns:
            String formed by joining words
        """
        words = []
        for idx in indices:
            word = self.idx_to_word.get(idx, '<UNK>')
            if skip_special and word in self.special_tokens:
                continue
            words.append(word)
        return ' '.join(words)

    def encode_word(self, word: str, unk_token: str = '<UNK>') -> int:
        """
        Convert single word to its index.

        Args:
            word: Word to encode
            unk_token: Token for unknown words

        Returns:
            Integer index
        """
        unk_idx = self.word_to_idx.get(unk_token, 1)
        return self.word_to_idx.get(word, unk_idx)

    def decode_word(self, idx: int) -> str:
        """
        Convert single index to its word.

        Args:
            idx: Index to decode

        Returns:
            Word string
        """
        return self.idx_to_word.get(idx, '<UNK>')

    def get_pad_idx(self) -> int:
        """Get padding token index."""
        return self.word_to_idx.get('<PAD>', 0)

    def get_unk_idx(self) -> int:
        """Get unknown token index."""
        return self.word_to_idx.get('<UNK>', 1)

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self):
        return (
            f"WordTokenizer(vocab_size={self.vocab_size}, "
            f"special_tokens={self.special_tokens})"
        )
