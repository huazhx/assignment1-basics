import pretokenization_example
import regex as re

class BPETokenizer:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int,
    ):
        self.chunk_boundaries = None
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

        self.vocab : dict[int, bytes] = {}
        self.inv_vocab : dict[bytes, int] = {}
        self.merge_vocab: dict[tuple[bytes], int] = {}
        self.end_of_text = b"<|endoftext|>"

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.encoding = 'utf-8'


    def _init_vocabulary(self):
        for i in range(0, 256):
            self.vocab[i] = bytes(i)
            self.inv_vocab[bytes] = i

    def _pre_tokenize_worker(self, chunk: str):
        """
        Use the rule from self.merge_vocab to count new pairs.
        In the main progress remember to sync the data using .join()
        """
        partial_merge_vocab : dict[tuple[bytes], int] = {}
            
        for pattern in re.finditer(self.PAT, chunk):
            pattern_bytes = bytes(pattern, encoding=self.encoding)
            for i in range(0, len(pattern)):
                if tuple(pattern_bytes[0:len(pattern_bytes) - i]) in self.merge_vocab:
                    pass
                
            

    def _chunk_boundries(self):
        with open(self.input_path, "rb") as f:
            self.chunk_boundaries = pretokenization_example.find_chunk_boundaries(
                f, self.num_processes, self.end_of_text
            )

    