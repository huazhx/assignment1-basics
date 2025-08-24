"""
bpe_tokenizer.py
Author: huazhx
Date: 2025-08-22
Description: Byte Pair Encoding (BPE) tokenizer implementation with full training pipeline.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import json


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer implementation that learns a subword vocabulary
    from raw text data through iterative merging of frequent byte pairs.

    The tokenizer starts with individual bytes as tokens and merges the most frequent
    adjacent pair in each training step until the target vocabulary size is reached.

    Attributes:
        vocab (Dict[int, bytes]): Mapping from token ID to byte string.
        inv_vocab (Dict[bytes, int]): Inverse mapping from byte string to token ID.
        special_tokens (Dict[str, int]): Reserved special tokens (e.g., <unk>, <pad>).
    """

    def __init__(self, special_tokens: Optional[List[str]] = None):
        """
        Initialize the BPETokenizer with base byte-level vocabulary.

        Args:
            special_tokens (Optional[List[str]]): Optional list of special tokens.
            If not provided, no special tokens are added initially.
        """
        # Start with 256 single-byte tokens
        self.vocab: Dict[int, bytes] = {}
        self.inv_vocab: Dict[bytes, int] = {}
        self.next_token_id = 0

        # Handle special tokens
        self.special_tokens = special_tokens or []
        for token in self.special_tokens:
            self.vocab[self.next_token_id] = token.encode("utf-8")
            self.inv_vocab[token.encode("utf-8")] = self.next_token_id
            self.next_token_id += 1

        for i in range(256):
            byte_val = bytes([i])
            self.vocab[i] = byte_val
            self.inv_vocab[byte_val] = i
            self.next_token_id += 1

    @staticmethod
    def tokenize_word(word: str) -> List[int]:
        """
        Convert a str into a list of UTF-8 encoded bytes.

        Args:
            word (str): Input word.

        Returns:
            List[int]: List of single-byte sequences.

        Raises:
            AssertionError: If input word is empty.
        """
        assert word, "Input word cannot be empty"
        return [b for b in word.encode()]

    @staticmethod
    def count_pairs(token_list: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count frequency of adjacent byte pairs in a token list.

        Args:
            token_list (List[bytes]): List of byte tokens.

        Returns:
            Dict[Tuple[bytes, bytes], int]: Dictionary mapping byte pairs to their frequencies.
        """
        if len(token_list) < 2:
            return {}

        pair_freq = defaultdict(int)
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_freq[pair] += 1
        return dict(pair_freq)

    @staticmethod
    def merge_pair_in_word(token: Tuple[int, ...], pair: Tuple[int, int]) -> Tuple[int, ...]:
        """
        Merge all occurrences of a given pair in a tokenized word.

        Args:
            word (Tuple[bytes, ...]): Tokenized word as list of byte sequences.
            pair (Tuple[bytes, bytes]): The pair of tokens to merge.

        Returns:
            Tuple[bytes, ...]: New token list with merged pairs.
        """
        if not token:
            return ()

        new_word = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                # Concatenate the byte sequences
                new_word.append(token[i] + token[i + 1])
                i += 2
            else:
                new_word.append(token[i])
                i += 1
        return tuple(new_word)

    def find_most_frequent_pair(
        self, tokenized_word_freq: Dict[Tuple[int, ...], int]
    ) -> Optional[Tuple[int, int]]:
        """
        Find the most frequent adjacent byte pair across all words.

        Ties are broken lexicographically by first then second token.

        Args:
            tokenized_word_freq (Dict[Tuple[bytes, ...], int]): Frequency dictionary of tokenized words.

        Returns:
            Optional[Tuple[bytes, bytes]]: Most frequent pair, or None if no pairs exist.
        """
        if not tokenized_word_freq:
            return None

        pair_freq = defaultdict(int)
        for token_tuple, count in tokenized_word_freq.items():
            token_list = list(token_tuple)
            part_pairs = self.count_pairs(token_list)
            for pair, freq in part_pairs.items():
                pair_freq[pair] += freq * count

        if not pair_freq:
            return None

        # Sort by frequency (desc), then lexicographically by pair
        sorted_pairs = sorted(
            pair_freq.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1])  # Negative frequency for descending order
        )
        return sorted_pairs[0][0]

    def train_epoch(
        self, tokenized_word_freq: Dict[Tuple[bytes, ...], int]
    ) -> Tuple[Optional[Dict[Tuple[bytes, ...], int]], Optional[Tuple[bytes, bytes]]]:
        """
        Perform one BPE merge step.

        Args:
            tokenized_word_freq (Dict[Tuple[bytes, ...], int]): Current tokenized word frequencies.

        Returns:
            Tuple of:
                - Updated tokenized word frequencies after merging.
                - The merged token pair.
            Returns (None, None) if no merge is possible.
        """
        target_pair = self.find_most_frequent_pair(tokenized_word_freq)
        if not target_pair:
            return None, None

        new_tokenized_word_freq = defaultdict(int)
        for token_tuple, count in tokenized_word_freq.items():
            merged = self.merge_pair_in_word(token_tuple, target_pair)
            new_tokenized_word_freq[merged] = count

        return dict(new_tokenized_word_freq), target_pair

    def train(self, word_freq: dict[str, int], vocab_size: int = 30000):
        """
        Train the BPE tokenizer on a text corpus.

        Args:
            vocab_size (int): Target vocabulary size (including base bytes).

        Raises:
            ValueError: If vocab_size <= 256.
        """
        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256")
        
        # Remember merged elements
        merges = []

        # Initialize tokenized word frequencies
        tokenized_word_freq = defaultdict(int)
        for word, count in word_freq.items():
            tokenized = tuple(self.tokenize_word(word))
            tokenized_word_freq[tokenized] = count

        tokenized_word_freq = dict(tokenized_word_freq)
        next_id = len(self.vocab)

        # Iteratively merge until target vocab size
        while next_id < vocab_size:
            tokenized_word_freq, merged_pair = self.train_epoch(tokenized_word_freq)
            if not merged_pair:
                print("No more merges possible.")
                break

            # Add new token to vocabulary
            merges.append(merged_pair)
            new_token_bytes = merged_pair[0] + merged_pair[1]
            self.inv_vocab[new_token_bytes] = next_id
            self.vocab[next_id] = new_token_bytes
            next_id += 1

        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")

        return (self.vocab, merges)
    


    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: List of token IDs.
        """
        # Convert text to bytes and then to list of single-byte tokens
        byte_tokens = [bytes([b]) for b in text.encode('utf-8')]
        tokens = tuple(byte_tokens)
        
        # Apply all learned merges
        for token_id in range(256, self.next_token_id):
            token_bytes = self.vocab[token_id]
            if len(token_bytes) == 1:  # Skip single bytes
                continue
                
            # Find the component bytes that make up this token
            first_byte = token_bytes[:-1]
            second_byte = token_bytes[-1:]
            # In a real implementation, we'd need to track the merge history
            # For simplicity, we're assuming we can decompose tokens
            
            # This is a simplified approach - a complete implementation
            # would require storing the merge operations
            new_tokens = []
            i = 0
            while i < len(tokens):
                # This is a simplified merge check
                if (i < len(tokens) - 1 and 
                    tokens[i] + tokens[i + 1] == token_bytes):
                    new_tokens.append(token_bytes)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = tuple(new_tokens)
            
        # Convert final tokens to IDs
        return [self.inv_vocab[token] for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text.
        """
        byte_string = b"".join(self.vocab[token_id] for token_id in token_ids)
        return byte_string.decode('utf-8', errors='replace')

    def save(self, file_path: str):
        """
        Save tokenizer vocabulary to a JSON file.

        Args:
            file_path (str): Path to save the vocabulary.
        """
        data = {
            "vocab": {k: v.hex() for k, v in self.vocab.items()},
            "inv_vocab": {k.hex(): v for k, v in self.inv_vocab.items()},
            "special_tokens": self.special_tokens,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str):
        """
        Load tokenizer vocabulary from a JSON file.

        Args:
            file_path (str): Path to the saved vocabulary.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.inv_vocab = {bytes.fromhex(k): int(v) for k, v in data["inv_vocab"].items()}
        self.special_tokens = data["special_tokens"]



if __name__ == "__main__":
    ...