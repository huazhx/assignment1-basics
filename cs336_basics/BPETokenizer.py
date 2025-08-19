import regex as re
import multiprocessing
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


class BPETokenizer:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        num_processes: int,
        special_tokens: list[str] = []
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes
        
        self.merges = []
        self.vocab: dict[int, bytes] = {}
        self.inv_vocab: dict[bytes, int] = {}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.end_of_text = "<|endoftext|>"
        self.encoding = 'utf-8'

        if not self.end_of_text in self.special_tokens:
            self.special_tokens.append(self.end_of_text)
        self.next_id: int = 0

        self._init_vocabulary()


    def _init_vocabulary(self):
        """
        initialize vocabulary with 256 byte values and special tokens
        The vocabulary maps token IDs to their byte representations and 
        inv_vocab provides the reverse mapping for efficient lookup
        """
        for i in range(0, 256):
            byte_val = bytes([i])
            self.vocab[i] = byte_val
            self.inv_vocab[byte_val] = i

        self.next_id = 256
        for token in self.special_tokens:
            token_bytes = token.encode(self.encoding)
            self.vocab[self.next_id] = token_bytes
            self.inv_vocab[token_bytes] = self.next_id
            self.next_id += 1

    def _pre_tokenize_worker(self, chunk: str):
        """
        Args:
            chunk: Text chunk to pre-tokenize

        Returns:
            Dictionary mapping int tuples to their frequencies. 
            For example:
            {(b'h', b'e', b'l', b'l', b'o'): 103}
        """
        frequency_map = {}

        for st in self.special_tokens:
            chunk = chunk.replace(st, "")

        matches = re.findall(self.PAT, chunk.strip())
        for match in matches:
            bytes_tuple = tuple(
                [bytes(e, encoding=self.encoding) for e in match])
            frequency_map[bytes_tuple] = frequency_map.get(bytes_tuple, 0)
            frequency_map[bytes_tuple] += 1

        return frequency_map

    def _pre_tokenize(self, debug = False):
        """
        Returns:
            Dictionary mapping byte tuples to their frequencies across all text 
            For example:

        """

        if not debug:
            with open(self.input_path, "rb") as f:
                num_processes = self.num_processes
                boundaries = find_chunk_boundaries(
                    f, num_processes, b"<|endoftext|>")
                chunks = []

                # The following is a serial implementation, but you can parallelize this
                # by sending each start/end pair to a set of processes.
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    chunks.append(chunk)

                with multiprocessing.Pool(self.num_processes) as pool:
                    frequency_results = pool.map(self._pre_tokenize_worker, chunks)

                combined_frequencies = defaultdict(int)
                for fre_map in frequency_results:
                    for bytes_tuple, freq in fre_map.items():
                        combined_frequencies[bytes_tuple] += freq

                return dict(combined_frequencies)
        else:
            combined_frequencies = defaultdict(int)
            with open(self.input_path, "rb") as f:
                content = f.read()
                fm = self._pre_tokenize_worker(chunk=content.decode("utf-8", errors="ignore"))
                for bytes_tuple, freq in fm.items():
                    combined_frequencies[bytes_tuple] += freq

                return dict(combined_frequencies)

    def _count_pairs(self, word_freqs: dict[tuple[bytes, bytes], int]):
        """
        Args:
            word_freqs: Dictionary mapping word byte-tuples to frequencies.

        Returns:
            pair_counts: Dictionary mapping byte pairs to their total frequencies
        """
        pair_counts = defaultdict(int)

        for bytes_tuple, freq in word_freqs.items():
            pairs = []
            splitted_word = self._split_word(bytes_tuple)
            if len(splitted_word) < 2:
                continue
            for i in range(len(splitted_word) - 1):
                pair = (splitted_word[i], splitted_word[i + 1])
                pair_counts[pair] += freq

        return dict(pair_counts)

    def _split_word(self, bytes_t):
        """
        Returns:
            results: A list containing all possible tokens in self.vocab from a given word
        """
        assert bytes_t
        results = []
        for i in range(len(bytes_t)):
            current = bytes()
            for j in range(i, len(bytes_t)):
                current += bytes_t[j]
                if not current in self.inv_vocab:
                    results.append(current[:-1])
                    break
        return results

    def _merge_vocab(self, pair_freq: dict[tuple[bytes, bytes], int]):
        """
        self.vocab updates with new vocab and 
        self.merges updates with new merged pair

        Args:
            pair_freq: The byte pairs with frequency to merge

        """
        if len(pair_freq) < 2:
            return
        sorted_pairs = sorted(pair_freq.items(), key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
        target = sorted_pairs[0]

        new_bytes_pair = target[0][0] + target[0][1]
        self.vocab[self.next_id] = new_bytes_pair
        self.inv_vocab[new_bytes_pair] = self.next_id
        self.next_id += 1
        self.merges.append(target[0])
        
    def train(self):
        bytes_freq = self._pre_tokenize(debug=True)
        real_id = self.next_id
        while real_id < self.vocab_size:
            pair_freq = self._count_pairs(bytes_freq)
            self._merge_vocab(pair_freq)
            real_id += 1
        return 

