import logging
import pickle

from tqdm import tqdm
from multiprocessing import cpu_count

from cs336_basics.TextPreprocessor import TextPreprocessor
from cs336_basics.BPETokenizer import BPETokenizer

# Initialize logging
logging.basicConfig(filename='tranning.log', filemode='a',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 
# Configuration
# 
dataset = 'tinystories'
special_tokens = ["<|endoftext|>"]
input_path = f"../data/{dataset}.txt"
pickle_path = f'{dataset}_bpe.pkl'

vocab_size = 300

# Initialize preprocessing and tokenizer
tp = TextPreprocessor(num_processes=cpu_count(), special_tokens=special_tokens)
bpe = BPETokenizer(special_tokens=special_tokens)

# Count words and create initial vocabulary
logger.info("Starting word frequency count...")
word_freq = tp.count_words(file_path=input_path)
logger.info(
    f"Word frequency count completed. Total unique words: {len(word_freq)}")

# Create vocab and initialize pair frequency
logger.info("Initializing BPE tokenizer with word frequencies...")
vocab = {bpe.bytes2id(word): freq for word, freq in word_freq.items()}
pair_freq, pair_to_words = bpe.get_status_with_idx(vocab)

# Start merging process with progress bar
logger.info(
    f"Starting BPE training to reach vocab size {vocab_size}, Current size: {len(bpe.vocab)}")
with tqdm(total=vocab_size - len(bpe.vocab), desc="Training BPE", unit="merge") as pbar:
    while bpe.next_token_id < vocab_size:
        vocab, pair_freq, pair_to_words, merge = bpe.train_epoch(
            vocab, pair_freq, pair_to_words)
        logger.info(f"Merged: {merge}, New vocab size: {len(bpe.vocab)}")
        if merge:
            pbar.update(1)  # Update progress bar after each successful merge
        else:
            logger.warning("No more merges possible. Stopping early.")
            break

# Final log once training completes
logger.info(f"Training completed. Final vocabulary size: {len(bpe.vocab)}")

with open('bpe.pkl', 'wb') as f:
    pickle.dump((bpe.vocab, bpe.merges), f)

logger.info("BPE vocabulary and merges saved to bpe.pkl")