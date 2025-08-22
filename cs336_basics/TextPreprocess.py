import regex as re
from pathlib import Path
from typing import List, Optional, Union


class TextPreprocessor:
    """
    A class for preprocessing text files with tokenization capabilities.
    
    This preprocessor handles file reading, text cleaning, and tokenization
    using regex patterns suitable for natural language processing tasks.
    """
    
    # Default regex pattern for tokenization
    DEFAULT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, 
                 pattern: Optional[str] = None,
                 encoding: str = 'utf-8',
                 remove_tokens: Optional[List[str]] = None):
        """
        Initialize the TextPreprocessor.
        
        Args:
            pattern: Regex pattern for tokenization. If None, uses DEFAULT_PATTERN.
            encoding: Text encoding to use when decoding file contents.
            remove_tokens: List of tokens to remove from text (e.g., ["<|endoftext|>"]).
        """
        self.pattern = pattern or self.DEFAULT_PATTERN
        self.encoding = encoding
        self.remove_tokens = remove_tokens or ["<|endoftext|>"]
        
    def read_file(self, file_path: Union[str, Path]) -> str:
        """
        Read and decode a file.
        
        Args:
            file_path: Path to the file to read.
            
        Returns:
            Decoded file contents as string.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            UnicodeDecodeError: If file cannot be decoded with specified encoding.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            return data.decode(self.encoding)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                f"Cannot decode file {file_path} with encoding {self.encoding}: {e}"
            )
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing specified tokens and extra whitespace.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text string.
        """
        # Remove specified tokens
        for token in self.remove_tokens:
            text = text.replace(token, "")
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the configured regex pattern.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        return re.findall(self.pattern, text)
    
    def preprocess_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Complete preprocessing pipeline: read file, clean text, and tokenize.
        
        Args:
            file_path: Path to the file to preprocess.
            
        Returns:
            List of tokens from the preprocessed file.
        """
        # Read file
        raw_data = self.read_file(file_path)
        
        # Clean text
        cleaned_data = self.clean_text(raw_data)
        
        # Tokenize
        corpus = self.tokenize(cleaned_data)
        
        return corpus
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess raw text string (clean and tokenize).
        
        Args:
            text: Raw text string to preprocess.
            
        Returns:
            List of tokens from the preprocessed text.
        """
        cleaned_data = self.clean_text(text)
        return self.tokenize(cleaned_data)
    
    def update_pattern(self, new_pattern: str) -> None:
        """
        Update the tokenization pattern.
        
        Args:
            new_pattern: New regex pattern for tokenization.
        """
        self.pattern = new_pattern
    
    def add_remove_token(self, token: str) -> None:
        """
        Add a token to the list of tokens to remove during cleaning.
        
        Args:
            token: Token to add to removal list.
        """
        if token not in self.remove_tokens:
            self.remove_tokens.append(token)
    
    def remove_remove_token(self, token: str) -> None:
        """
        Remove a token from the list of tokens to remove during cleaning.
        
        Args:
            token: Token to remove from removal list.
        """
        if token in self.remove_tokens:
            self.remove_tokens.remove(token)


# Example usage
if __name__ == "__main__":
    # Basic usage - equivalent to your original code
    
    # # Advanced usage with custom configuration
    # custom_preprocessor = TextPreprocessor(
    #     remove_tokens=["<|endoftext|>", "<|startoftext|>"],
    #     encoding='utf-8'
    # )
    
    # # Process a file
    # tokens = custom_preprocessor.preprocess_file('data/test.txt')
    ...