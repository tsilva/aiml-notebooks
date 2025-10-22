"""
Character-level tokenizer for text data.

This module provides tokenizer classes for converting text to sequences of indices
and vice versa. These are reused across multiple notebooks for consistency.
"""


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
