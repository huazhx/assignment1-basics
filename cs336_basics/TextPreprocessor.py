import os
import multiprocessing
from collections import defaultdict
from typing import BinaryIO, List, Dict, Optional
import regex as re


class TextPreprocessor:
    """
    A class for efficiently counting word frequencies in large text files using multiprocessing.
    
    Features:
    - Handles large files by chunking them for parallel processing
    - Supports custom special tokens for splitting
    - Uses regex pattern matching for tokenization
    - Multiprocessing support for improved performance
    """
    
    def __init__(self, num_processes: Optional[int] = None, 
                 special_tokens: List[str] = ["<|endoftext|>"], 
                 mini_chunk_size: int = 4096):
        """
        Initialize the WordFrequencyCounter.
        
        Args:
            num_processes: Number of processes to use. Defaults to CPU count.
            split_token: Special token used for chunk boundaries
            mini_chunk_size: Size of mini chunks when finding boundaries (bytes)
        """
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.special_tokens = special_tokens
        self.special_tokens_bytes = [token.encode(errors='ignore') for token in self.special_tokens]
        self.mini_chunk_size = mini_chunk_size
        
        # Regex pattern for tokenization (handles contractions, letters, numbers, etc.)
        self.tokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def _word_counter_worker(self, chunk: str) -> Dict[str, int]:
        """
        Worker function to count words in a single chunk.
        
        Args:
            chunk: Text chunk to process
            special_tokens: List of special tokens to split on
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        word_count = defaultdict(int)
        
        if not chunk:
            return word_count
        
        # Split on special tokens if provided
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            mini_chunks = re.split(pattern, chunk)
        else:
            mini_chunks = [chunk]
        
        # Count words in each mini chunk
        for mini_chunk in mini_chunks:
            if mini_chunk:
                for match in re.finditer(self.tokenize_pattern, mini_chunk):
                    word_count[match.group()] += 1
        
        return dict(word_count)
    
    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes = b"<|endoftext|>",
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    
    def count_words(self, file_path: str) -> Dict[str, int]:
        """
        Count word frequencies in a file using multiprocessing.
        
        Args:
            file_path: Path to the text file
            special_tokens: Optional list of special tokens to handle separately
            
        Returns:
            Dictionary mapping words to their frequencies
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file can't be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        word_freq = defaultdict(int)
        chunks = []
        
        # Read and chunk the file
        with open(file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(file=f, desired_num_chunks=self.num_processes)
            
            # Extract chunks
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                chunks.append(chunk_text)
        
        # Process chunks in parallel
        if len(chunks) == 1:
            # Single chunk, no need for multiprocessing
            chunk_word_freqs = [self._word_counter_worker(chunks[0])]
        else:
            # Use multiprocessing for multiple chunks
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                chunk_word_freqs = pool.map(self._word_counter_worker, chunks)
        
        # Combine results
        for chunk_word_freq in chunk_word_freqs:
            for word, freq in chunk_word_freq.items():
                word_freq[word] += freq
        
        return dict(word_freq)
    
    def count_words_from_text(self, text: str, special_tokens: List[str] = None) -> Dict[str, int]:
        """
        Count word frequencies in a text string (no file I/O).
        
        Args:
            text: Input text string
            special_tokens: Optional list of special tokens to handle separately
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        return dict(self._word_counter_worker(text))
    
    def get_top_words(self, word_freq: Dict[str, int], n: int = 10) -> List[tuple]:
        """
        Get the top N most frequent words.
        
        Args:
            word_freq: Word frequency dictionary
            n: Number of top words to return
            
        Returns:
            List of (word, frequency) tuples sorted by frequency (descending)
        """
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]


# Example usage
if __name__ == '__main__':
    # Create counter instance
    counter = TextPreprocessor(num_processes=4)
    
    # Count words in a file
    file_path = '../tests/fixtures/corpus.en'
    word_frequencies = counter.count_words(file_path)
    
    print(f"Total unique words: {len(word_frequencies)}")
    print(f"Total word count: {sum(word_frequencies.values())}")
    
    # Get top 10 most frequent words
    top_words = counter.get_top_words(word_frequencies, 10)
    print("\nTop 10 most frequent words:")
    for word, freq in top_words:
        print(f"'{word.encode()}': {freq}")
        print(f"---This is the unicode: {[b for b in word.encode()]}")
            