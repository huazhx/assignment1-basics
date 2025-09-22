"""
bpe_tokenizer.py
Author: huazhx
Date: 2025-08-22
Description: Byte Pair Encoding (BPE) tokenizer implementation with full training pipeline.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Set, Iterable
import json
import ast
import regex as re
import heapq
from collections import defaultdict

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
        self.merges: List[Tuple[bytes, bytes]] = []

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.merge_ranks = []
        self.next_token_id = 0

        # Handle special tokens
        self.special_tokens = special_tokens or []
        for token in self.special_tokens:
            self.vocab[self.next_token_id] = token.encode("utf-8")
            self.inv_vocab[token.encode("utf-8")] = self.next_token_id
            self.next_token_id += 1

        for i in range(256):
            byte_val = bytes([i])
            self.vocab[self.next_token_id] = byte_val
            self.inv_vocab[byte_val] = self.next_token_id
            self.next_token_id += 1

    def bytes2id(self, word: str):
        assert word
        return tuple([b + len(self.special_tokens) for b in word.encode(errors='ignore')])

    @staticmethod
    def count_pairs(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count frequency of adjacent byte pairs in a token list.

        Args:
            token_list (List[bytes]): List of byte tokens.

        Returns:
            Dict[Tuple[bytes, bytes], int]: Dictionary mapping byte pairs to their frequencies.
        """
        if len(ids) < 2:
            return {}

        pair_freq = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            pair_freq[pair] += 1

        return dict(pair_freq)

    @staticmethod
    def merge_pair_in_word(ids: Tuple[int, ...], pair: Tuple[int, int], id: int) -> Tuple[int, ...]:
        """
        Merge all occurrences of a given pair in a tokenized word.

        Args:
            word (Tuple[int, ...]): Tokenized word as list of byte sequences.
            pair (Tuple[int, int]): The pair of tokens to merge.

        Returns:
            Tuple[int, ...]: New token list with merged pairs.
        """
        if len(ids) < 2:
            return ids
        # main
        new_word = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                # Concatenate the byte sequences
                new_word.append(id)
                i += 2
            else:
                new_word.append(ids[i])
                i += 1
        return tuple(new_word)

    def find_most_frequent_pair(
        self, pair_freq: Dict[Tuple[int, int], int]
    ) -> Optional[Tuple[int, int]]:
        """
        Find the most frequent adjacent byte pair across all words.

        Ties are broken lexicographically by first then second token.

        Args:
            tokenized_word_freq (Dict[Tuple[bytes, ...], int]): Frequency dictionary of tokenized words.

        Returns:
            Optional[Tuple[bytes, bytes]]: Most frequent pair, or None if no pairs exist.
        """
        assert pair_freq, 'pair_freq should not be empty'

        sorted_pairs = sorted(
            pair_freq.items(),
            key=lambda x: (x[1], tuple(self.vocab[e] for e in x[0])),
            reverse=True
        )

        return sorted_pairs[0][0]

    def train_epoch(
        self, vocab: Dict[Tuple[int, ...], int], pair_freq: Dict[Tuple[int, int], int], pair_to_words
    ) -> Tuple[Optional[Dict[Tuple[int, ...], int]], Optional[Tuple[int, int]]]:
        # If no pairs left to merge, return None
        if not pair_freq:
            return vocab, pair_freq, pair_to_words, None
        
        # Find and merge the most frequent pair
        new_pair = self.find_most_frequent_pair(pair_freq)
        vocab_out, pair_freq_out, pair_to_words_out = self.merge_vocab_with_cache(
            vocab, new_pair, pair_freq, pair_to_words)

        bytes_value = self.vocab[new_pair[0]] + self.vocab[new_pair[1]]
        self.vocab[self.next_token_id] = bytes_value
        self.inv_vocab[bytes_value] = self.next_token_id
        self.next_token_id += 1
        merge = tuple([self.vocab[id] for id in new_pair])
        self.merges.append(merge)

        return vocab_out, pair_freq_out, pair_to_words_out, merge

    def train(self, word_freq: dict[str, int], vocab_size: int = 30000):
        """
        Train the BPE tokenizer on a text corpus.

        Args:
            vocab_size (int): Target vocabulary size (including base bytes).

        Raises:
            ValueError: If vocab_size <= 256.
        """
        vocab = {self.bytes2id(word): freq for word, freq in word_freq.items()}
        pair_freq, pair_to_words = self.get_status_with_idx(vocab)
        # Iteratively merge until target vocab size
        while self.next_token_id < vocab_size:
            vocab, pair_freq, pair_to_words, merge = self.train_epoch(
                vocab, pair_freq, pair_to_words)
            if not merge:
                print("No more merges possible.")
                break

        # print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
        return (self.vocab, self.merges)

    def get_status_with_idx(self, vocab: Dict[Tuple[int, ...], int]) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[Tuple[int, ...]]]]:
        """
        Get pair frequencies and build an index mapping pairs to words that contain them.

        Args:
            vocab: Dictionary mapping token sequences to their frequencies

        Returns:
            Tuple containing:
            - pair_freq: Dictionary mapping pairs to their total frequencies
            - pair_to_words: Dictionary mapping pairs to sets of words containing them
        """
        pair_freq = defaultdict(int)
        pair_to_words = defaultdict(set)

        for ids, freq in vocab.items():
            word_pairs = self.count_pairs(ids)
            for pair, count in word_pairs.items():
                pair_freq[pair] += freq * count  # multiply by word frequency
                pair_to_words[pair].add(ids)

        return dict(pair_freq), dict(pair_to_words)

    def merge_vocab_with_cache(self, vocab_in: Dict[Tuple[int, ...], int],
                               new_pair: Tuple[int, int],
                               pair_freq: Dict[Tuple[int, int], int],
                               pair_to_words: Dict[Tuple[int, int], Set[Tuple[int, ...]]]) -> Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[Tuple[int, ...]]]]:
        """
        Merge vocabulary with cached pair frequencies and indexes.

        Args:
            vocab_in: Input vocabulary
            new_pair: Pair to merge
            pair_freq: Current pair frequencies (will be modified)
            pair_to_words: Index mapping pairs to words containing them (will be modified)

        Returns:
            Tuple containing:
            - vocab_out: Updated vocabulary
            - updated pair_freq: Updated pair frequencies
            - updated pair_to_words: Updated pair-to-words index
        """
        words_to_update = pair_to_words.get(new_pair, set()).copy()

        # Process words that contain the merged pair
        for ids in words_to_update:
            if ids not in vocab_in:
                continue

            freq = vocab_in[ids]
            new_ids = self.merge_pair_in_word(
                ids, new_pair, self.next_token_id)
            # vocab_out[new_ids] = freq
            

            # Update pair frequencies and indexes
            if ids != new_ids:  # Only update if the word actually changed
                if len(new_ids) > 1:
                    vocab_in[new_ids] = freq
                    
                # Remove old pair counts for this word
                old_word_pairs = self.count_pairs(ids)
                for old_pair, count in old_word_pairs.items():
                    if old_pair in pair_freq:
                        pair_freq[old_pair] -= freq * count
                        if pair_freq[old_pair] <= 0:
                            del pair_freq[old_pair]
                            if old_pair in pair_to_words:
                                del pair_to_words[old_pair]
                        else:
                            if old_pair in pair_to_words:
                                pair_to_words[old_pair].discard(ids)

                # Add new pair counts for this word
                new_word_pairs = self.count_pairs(new_ids)
                for new_pair_local, count in new_word_pairs.items():
                    if new_pair_local not in pair_freq:
                        pair_freq[new_pair_local] = 0
                        pair_to_words[new_pair_local] = set()
                    pair_freq[new_pair_local] += freq * count
                    pair_to_words[new_pair_local].add(new_ids)

        # Copy unchanged words
        # for ids, freq in vocab_in.items():
        #     if ids not in words_to_update:
        #         vocab_out[ids] = freq

        # Delete changed words
        for ids in words_to_update:
            if ids not in vocab_in:
                continue
            del vocab_in[ids]

        # Clean up the merged pair from indexes
        if new_pair in pair_freq:
            del pair_freq[new_pair]
        if new_pair in pair_to_words:
            del pair_to_words[new_pair]

        # Clean up empty entries
        pairs_to_remove = []
        for pair, word_set in pair_to_words.items():
            if not word_set or pair not in pair_freq:
                pairs_to_remove.append(pair)

        for pair in pairs_to_remove:
            pair_to_words.pop(pair, None)
            pair_freq.pop(pair, None)

        return vocab_in, pair_freq, pair_to_words

    def _build_merge_ranks(self) -> Dict[Tuple[int, int], int]:
        """Precompute pair -> rank dict for O(1) lookup."""
        merge_ranks = {}
        for rank, (a, b) in enumerate(self.merges):
            merge_ranks[(self.inv_vocab[a], self.inv_vocab[b])] = rank
        return merge_ranks
    
    def wencode(self, word: str) -> List[int]:
        """Encode a single word using BPE with heap optimization."""

        def rebuild_heap(word_ids, merge_ranks):
            heap = []
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i+1])
                if pair in merge_ranks:
                    heapq.heappush(heap, (merge_ranks[pair], i, pair))
            return heap
        # 0. ensure self.merge_ranks is up to date
        if len(self.merge_ranks) != len(self.merges):
            self.merge_ranks = self._build_merge_ranks()

        # 1. Convert word -> byte tokens -> token IDs
        word_bytes = word.encode("utf-8")
        word_ids = [self.inv_vocab[bytes([b])] for b in word_bytes]

        # 2. Build initial pairs (with ranks)
        def get_pairs(ids):
            return [((ids[i], ids[i+1]), i) for i in range(len(ids)-1)]

        heap = []
        for pair, i in get_pairs(word_ids):
            # i is the left index of the pair in the origin word_ids
            if pair in self.merge_ranks:
                heapq.heappush(heap, (self.merge_ranks[pair], i, pair))

        # 3. Iteratively merge
        while heap:
            rank, i, pair = heapq.heappop(heap)

            # Check still valid (i might be outdated)
            if i >= len(word_ids)-1 or (word_ids[i], word_ids[i+1]) != pair:
                continue

            # Merge into new symbol
            merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            assert merged_bytes in self.inv_vocab, "Merged token must exist in vocab!"
            new_token = self.inv_vocab[merged_bytes]

            # Replace [i, i+1] with new_token
            # Python List slice assignment
            word_ids[i:i+2] = [new_token]

            # Push new local pairs into heap
            if i > 0:
                new_pair = (word_ids[i-1], word_ids[i])
                if new_pair in self.merge_ranks:
                    heapq.heappush(heap, (self.merge_ranks[new_pair], i-1, new_pair))
            if i < len(word_ids)-1:
                new_pair = (word_ids[i], word_ids[i+1])
                if new_pair in self.merge_ranks:
                    heapq.heappush(heap, (self.merge_ranks[new_pair], i, new_pair))
                    
            heap = rebuild_heap(word_ids, self.merge_ranks)

        return word_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode input text using BPE, preserving special tokens.

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: List of token IDs representing the encoded text, with special tokens preserved.
        """
        token_ids = []
        if self.special_tokens:
            # Build regex to split and capture special tokens
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            spat = "|".join(f"({re.escape(token)})" for token in sorted_special_tokens)
            # Split and keep special tokens
            chunks = re.split(spat, text)
            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in self.special_tokens:
                    # Preserve special token as is
                    token_id = self.inv_vocab.get(chunk.encode("utf-8"))
                    if token_id is not None:
                        token_ids.append(token_id)
                else:
                    # Encode normal text
                    for match in re.finditer(self.PAT, chunk):
                        token_ids.extend(self.wencode(match.group()))
        else:
            # No special tokens, just encode
            for match in re.finditer(self.PAT, text):
                token_ids.extend(self.wencode(match.group()))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Lazily encode an iterable of strings and yield token ids one-by-one.

        Previously this returned lists from wencode which made consumers
        receive lists instead of a flat stream of ints. This yields ints.
        """
        for e in iterable:
            for tid in self.encode(e):
                yield tid

    def decode(self, ids: List[int]) -> str:
        decoded_bytes = b"".join([self.vocab[i] for i in ids])
        return decoded_bytes.decode("utf-8", errors="ignore")
    
    def save_merges(self, filename):
        with open(filename, "w") as f:
            for a, b in self.merges:
                f.write(f"{repr(a)} {repr(b)}\n")

    def load_merges(filename):
        merges = []
        with open(filename, "r") as f:
            while line := f.readline():
                a, b = line.strip().split()
                merges.append((ast.literal_eval(a), ast.literal_eval(b)))
        return merges
    
    def save_vocab(self, filename):
        eval_vocab = {str(k): repr(v) for k, v in self.vocab.items()}
        with open(filename, "w") as f:
            json.dump(eval_vocab, f)

    def load_vocab(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        restored_vocab = {int(k): ast.literal_eval(v) for k, v in data.items()}
        return restored_vocab
    
    def to_files(self, vocab_filename: str = 'vocab.json', merges_filename: str = 'merges.txt'):
        self.save_vocab(self.vocab, vocab_filename)
        self.save_merges(self.merges, merges_filename)

    def from_files(cls, vocab_filename: str = 'vacab.json', merges_filename: str = 'merges.txt', special_tokens = None):
        vocab = cls.load_vocab(vocab_filename)
        merges = cls.load_merges(merges_filename)
        tokenizer = cls(special_tokens=special_tokens)
        tokenizer.vocab = vocab
        tokenizer.inv_vocab = {v: k for k, v in vocab.items()}
        tokenizer.merges = merges
        tokenizer.next_token_id = max(vocab.keys()) + 1
        return tokenizer