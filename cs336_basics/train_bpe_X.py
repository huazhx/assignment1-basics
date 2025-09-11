import logging
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import cpu_count

from cs336_basics.TextPreprocessor import TextPreprocessor
from cs336_basics.BPETokenizer import BPETokenizer

# Defination for recovering from log file
import regex as re
from typing import Tuple
import ast
from typing import Dict, Optional


#
# Configuration
#
PREFIX = Path('../data/')
dataset = 'TinyStoriesV2-GPT4-valid.txt'
dataset = 'tinystories.txt'
special_tokens = ["<|endoftext|>"]

input_path = PREFIX / dataset
pickle_path = f'{dataset}_bpe.pkl'
log_file_path = f'{dataset.split(".")[0]}_training.log'
new_log_file_path = f'{dataset.split(".")[0]}_training_recovered.log'
recover_pickle_path = f'{dataset}_bpe_recovered.pkl'

vocab_size = 1000


# Functions for recovering from log file
def get_tuple_from_text(text: str) -> Tuple[bytes, bytes]:
    pattern = r"Merged: (\(b'.*?', b'.*?'\))"
    match = re.search(pattern, text)
    if match:
        return ast.literal_eval(match.group(1))
    return None


def get_tuples_from_log(file_path: Path):
    with open(file=file_path, mode='r') as f:
        logs = f.readlines()
    for line in logs:
        if 'Merged:' in line:
            yield get_tuple_from_text(line)


def recover_from_pair(
    bpe, new_pair_ids: Tuple[int, int], vocab: Dict[Tuple[int, ...], int], pair_freq: Dict[Tuple[int, int], int], pair_to_words
) -> Tuple[Optional[Dict[Tuple[int, ...], int]], Optional[Tuple[int, int]]]:
    # If no pairs left to merge, return None
    if not pair_freq:
        return vocab, pair_freq, pair_to_words, None
    new_pair = tuple([bpe.inv_vocab[x] for x in new_pair_ids])
    # Find and merge the most frequent pair
    vocab_out, pair_freq_out, pair_to_words_out = bpe.merge_vocab_with_cache(
        vocab, new_pair, pair_freq, pair_to_words)

    bytes_value = bpe.vocab[new_pair[0]] + bpe.vocab[new_pair[1]]
    bpe.vocab[bpe.next_token_id] = bytes_value
    bpe.inv_vocab[bytes_value] = bpe.next_token_id
    bpe.next_token_id += 1
    merge = tuple([bpe.vocab[id] for id in new_pair])
    bpe.merges.append(merge)

    return vocab_out, pair_freq_out, pair_to_words_out, merge


# Initialize preprocessing and tokenizer
tp = TextPreprocessor(num_processes=cpu_count(), special_tokens=special_tokens)
bpe = BPETokenizer(special_tokens=special_tokens)

word_freq = tp.count_words(file_path=input_path)
vocab = {bpe.bytes2id(word): freq for word, freq in word_freq.items()}
pair_freq, pair_to_words = bpe.get_status_with_idx(vocab)


# Whether log file exists, if not skip recovering
if Path(log_file_path).exists():
    pickle_path = recover_pickle_path
    # Initialize logging
    logging.basicConfig(filename=new_log_file_path, filemode='a',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Recovering from existing log file...")
    # Recover from existing log file
    new_pair_generator = get_tuples_from_log(Path(log_file_path))
    for new_pair in new_pair_generator:
        logger.info(f"Recovering merge: {new_pair}")
        if new_pair:
            vocab, pair_freq, pair_to_words, merge = recover_from_pair(
                bpe, new_pair, vocab, pair_freq, pair_to_words)

else:
    # Initialize logging
    logging.basicConfig(filename=log_file_path, filemode='a',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()


# Create vocab and initialize pair frequency
logger.info("Initializing BPE tokenizer with word frequencies...")

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

with open(pickle_path, 'wb') as f:
    pickle.dump((bpe.vocab, bpe.merges), f)

logger.info("BPE vocabulary and merges saved to bpe.pkl")
