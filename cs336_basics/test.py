import pytest
from collections import defaultdict
from typing import Dict, Tuple, Set

# 假设 BPETokenizer 类已经定义在 bpe_tokenizer.py 中
from .BPETokenizer import BPETokenizer

def test_train_epoch_normal_case():
    tokenizer = BPETokenizer()
    tokenizer.next_token_id = 256  # 初始 token ID 为 256，无特殊 token

    # 构造输入参数
    vocab_in = {(1, 2): 1, (1, 3): 1}
    pair_freq_in = {(1, 2): 1, (1, 3): 1}
    pair_to_words_in = {(1, 2): {(1, 2)}, (1, 3): {(1, 3)}}

    # 执行 train_epoch
    vocab_out, pair_freq_out, pair_to_words_out, merge = tokenizer.train_epoch(
        vocab_in, pair_freq_in, pair_to_words_in
    )

    # 预期结果
    expected_vocab_out = {(256,): 1, (1, 3): 1}
    expected_pair_freq_out = {(1, 3): 1}
    expected_pair_to_words_out = {(1, 3): {(1, 3)}}
    expected_merge = (b'\x01', b'\x02')

    # 断言输出是否正确
    assert vocab_out == expected_vocab_out
    assert pair_freq_out == expected_pair_freq_out
    assert pair_to_words_out == expected_pair_to_words_out
    assert merge == expected_merge

    # 检查内部状态是否更新
    assert tokenizer.vocab[256] == b'\x01\x02'
    assert tokenizer.inv_vocab[b'\x01\x02'] == 256
    assert tokenizer.next_token_id == 257


def test_train_epoch_no_pairs():
    tokenizer = BPETokenizer()
    tokenizer.next_token_id = 256

    # 单个 token，没有可合并的 pair
    vocab_in = {(1,): 1}
    pair_freq_in = {}
    pair_to_words_in = {}

    vocab_out, pair_freq_out, pair_to_words_out, merge = tokenizer.train_epoch(
        vocab_in, pair_freq_in, pair_to_words_in
    )

    assert vocab_out == vocab_in
    assert pair_freq_out == {}
    assert pair_to_words_out == {}
    assert merge is None

    # 确认状态未改变
    assert tokenizer.next_token_id == 256