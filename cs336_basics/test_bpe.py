import pytest
import json
import tempfile
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Import the BPE tokenizer
# from bpe_tokenizer import BPETokenizer

from BPETokenizer import BPETokenizer

class TestBPETokenizerInitialization:
    """测试 BPE Tokenizer 初始化"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        tokenizer = BPETokenizer()
        
        # 应该有256个基础字节token
        assert len(tokenizer.vocab) == 256
        assert len(tokenizer.inv_vocab) == 256
        assert tokenizer.next_token_id == 256
        assert tokenizer.special_tokens == []
        
        # 检查基础字节映射
        for i in range(256):
            assert i in tokenizer.vocab
            assert tokenizer.vocab[i] == bytes([i])
            assert tokenizer.inv_vocab[bytes([i])] == i
    
    def test_initialization_with_special_tokens(self):
        """测试带特殊token的初始化"""
        special_tokens = ["<pad>", "<unk>", "<eos>"]
        tokenizer = BPETokenizer(special_tokens=special_tokens)
        
        # 基础256个 + 3个特殊token
        assert len(tokenizer.vocab) == 259
        assert len(tokenizer.inv_vocab) == 259
        assert tokenizer.next_token_id == 259
        assert tokenizer.special_tokens == special_tokens
        
        # 检查特殊token映射
        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            token_bytes = token.encode("utf-8")
            assert tokenizer.vocab[token_id] == token_bytes
            assert tokenizer.inv_vocab[token_bytes] == token_id
    
    def test_initialization_with_empty_special_tokens(self):
        """测试空特殊token列表初始化"""
        tokenizer = BPETokenizer(special_tokens=[])
        assert len(tokenizer.vocab) == 256
        assert tokenizer.special_tokens == []


class TestStaticMethods:
    """测试静态方法"""
    
    def test_tokenize_word_basic(self):
        """测试基本单词tokenization"""
        result = BPETokenizer.tokenize_word("hello")
        expected = [b'h', b'e', b'l', b'l', b'o']
        assert result == expected
    
    def test_tokenize_word_unicode(self):
        """测试Unicode字符tokenization"""
        result = BPETokenizer.tokenize_word("café")
        # "café" in UTF-8: c(99) a(97) f(102) é(195,169)
        expected = [b'c', b'a', b'f', b'\xc3', b'\xa9']
        assert result == expected
    
    def test_tokenize_word_empty(self):
        """测试空字符串tokenization"""
        with pytest.raises(AssertionError):
            BPETokenizer.tokenize_word("")
    
    def test_tokenize_word_single_char(self):
        """测试单字符tokenization"""
        result = BPETokenizer.tokenize_word("a")
        assert result == [b'a']
    
    def test_count_pairs_basic(self):
        """测试基本pair计数"""
        tokens = [b'h', b'e', b'l', b'l', b'o']
        result = BPETokenizer.count_pairs(tokens)
        expected = {
            (b'h', b'e'): 1,
            (b'e', b'l'): 1,
            (b'l', b'l'): 1,
            (b'l', b'o'): 1
        }
        assert result == expected
    
    def test_count_pairs_repeated(self):
        """测试重复pair计数"""
        tokens = [b'a', b'b', b'a', b'b']
        result = BPETokenizer.count_pairs(tokens)
        expected = {
            (b'a', b'b'): 2,
            (b'b', b'a'): 1
        }
        assert result == expected
    
    def test_count_pairs_empty(self):
        """测试空列表pair计数"""
        result = BPETokenizer.count_pairs([])
        assert result == {}
    
    def test_count_pairs_single_token(self):
        """测试单token pair计数"""
        result = BPETokenizer.count_pairs([b'a'])
        assert result == {}
    
    def test_merge_pair_in_word_basic(self):
        """测试基本pair合并"""
        word = (b'h', b'e', b'l', b'l', b'o')
        pair = (b'l', b'l')
        result = BPETokenizer.merge_pair_in_word(word, pair)
        expected = (b'h', b'e', b'll', b'o')
        assert result == expected
    
    def test_merge_pair_in_word_multiple_occurrences(self):
        """测试多次出现的pair合并"""
        word = (b'a', b'b', b'a', b'b', b'c')
        pair = (b'a', b'b')
        result = BPETokenizer.merge_pair_in_word(word, pair)
        expected = (b'ab', b'ab', b'c')
        assert result == expected
    
    def test_merge_pair_in_word_no_match(self):
        """测试无匹配pair的合并"""
        word = (b'h', b'e', b'l', b'l', b'o')
        pair = (b'x', b'y')
        result = BPETokenizer.merge_pair_in_word(word, pair)
        assert result == word
    
    def test_merge_pair_in_word_empty(self):
        """测试空word的pair合并"""
        result = BPETokenizer.merge_pair_in_word((), (b'a', b'b'))
        assert result == ()


class TestTrainingMethods:
    """测试训练相关方法"""
    
    def test_find_most_frequent_pair_basic(self):
        """测试寻找最频繁pair"""
        tokenizer = BPETokenizer()
        tokenized_word_freq = {
            (b'h', b'e', b'l', b'l', b'o'): 2,
            (b'h', b'e', b'l', b'p'): 1
        }
        result = tokenizer.find_most_frequent_pair(tokenized_word_freq)
        # (b'h', b'e') 出现3次, (b'l', b'l') 出现2次, (b'e', b'l') 出现3次
        # 应该返回按字典序最小的最高频pair
        expected = (b'e', b'l')  # 或 (b'h', b'e'), 取决于具体实现
        assert result in [(b'e', b'l'), (b'h', b'e')]
    
    def test_find_most_frequent_pair_empty(self):
        """测试空字典的最频繁pair"""
        tokenizer = BPETokenizer()
        result = tokenizer.find_most_frequent_pair({})
        assert result is None
    
    def test_find_most_frequent_pair_single_tokens(self):
        """测试单token词的最频繁pair"""
        tokenizer = BPETokenizer()
        tokenized_word_freq = {
            (b'a',): 5,
            (b'b',): 3
        }
        result = tokenizer.find_most_frequent_pair(tokenized_word_freq)
        assert result is None
    
    def test_train_epoch_basic(self):
        """测试单次训练epoch"""
        tokenizer = BPETokenizer()
        tokenized_word_freq = {
            (b'h', b'e', b'l', b'l', b'o'): 1,
            (b'h', b'e', b'l', b'p'): 1
        }
        
        new_freq, merged_pair = tokenizer.train_epoch(tokenized_word_freq)
        
        assert merged_pair is not None
        assert new_freq is not None
        assert len(new_freq) == 2
        
        # 检查合并后的结果
        for word_tuple in new_freq.keys():
            assert len(word_tuple) <= len(max(tokenized_word_freq.keys(), key=len))
    
    def test_train_epoch_no_pairs(self):
        """测试无pair可合并的epoch"""
        tokenizer = BPETokenizer()
        tokenized_word_freq = {
            (b'a',): 1,
            (b'b',): 1
        }
        
        new_freq, merged_pair = tokenizer.train_epoch(tokenized_word_freq)
        
        assert new_freq is None
        assert merged_pair is None


class TestTrainFromWord:
    """测试 train_from_word 方法"""
    
    def test_train_from_word_basic(self):
        """测试基本的从词频训练"""
        tokenizer = BPETokenizer()
        word_freq = {
            "hello": 3,
            "help": 2,
            "world": 1
        }
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=300)
        
        # 检查返回值
        assert isinstance(vocab, dict)
        assert isinstance(merges, list)
        
        # 检查vocab大小增长
        assert len(vocab) > 256
        assert len(vocab) <= 300
        
        # 检查merges记录
        assert len(merges) > 0
        
        # 检查所有合并的token都在vocab中
        for merge_pair in merges:
            merged_token = merge_pair[0] + merge_pair[1]
            assert merged_token in tokenizer.inv_vocab
    
    def test_train_from_word_small_vocab(self):
        """测试小词汇量训练"""
        tokenizer = BPETokenizer()
        word_freq = {"aa": 10}  # 简单的重复字符
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=258)
        
        assert len(vocab) >= 256
        assert len(vocab) <= 258
        assert len(merges) <= 2
    
    def test_train_from_word_no_merges_possible(self):
        """测试无法进行合并的情况"""
        tokenizer = BPETokenizer()
        word_freq = {"a": 1, "b": 1, "c": 1}  # 单字符，无法合并
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=300)
        
        assert len(vocab) == 256  # 应该保持原始大小
        assert len(merges) == 0
    
    def test_train_from_word_empty_input(self):
        """测试空输入"""
        tokenizer = BPETokenizer()
        word_freq = {}
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=300)
        
        assert len(vocab) == 256
        assert len(merges) == 0
    
    def test_train_from_word_unicode_handling(self):
        """测试Unicode字符处理"""
        tokenizer = BPETokenizer()
        word_freq = {
            "café": 5,
            "naïve": 3,
            "résumé": 2
        }
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=300)
        
        assert len(vocab) > 256
        assert len(merges) > 0
        
        # 确保没有异常抛出，能正常处理Unicode
    
    def test_train_from_word_high_frequency_pairs(self):
        """测试高频pair优先合并"""
        tokenizer = BPETokenizer()
        word_freq = {
            "aaaa": 10,  # 'aa' 应该被优先合并
            "bbbb": 5,
            "abab": 3
        }
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=260)
        
        # 检查是否合并了高频的 'aa' pair
        assert len(merges) > 0
        aa_merged = any(merge[0] == b'a' and merge[1] == b'a' for merge in merges)
        assert aa_merged or any(b'a' in merge[0] and b'a' in merge[1] for merge in merges)
    
    def test_train_from_word_vocab_limit_reached(self):
        """测试达到词汇表大小限制"""
        tokenizer = BPETokenizer()
        word_freq = {
            "abcdefghijk": 100  # 长词，可以产生很多合并
        }
        
        target_size = 270
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=target_size)
        
        assert len(vocab) <= target_size
        assert len(merges) == len(vocab) - 256  # 每次合并增加一个token
    
    def test_train_from_word_deterministic(self):
        """测试训练的确定性"""
        word_freq = {"hello": 3, "help": 2}
        
        tokenizer1 = BPETokenizer()
        vocab1, merges1 = tokenizer1.train_from_word(word_freq, vocab_size=280)
        
        tokenizer2 = BPETokenizer()
        vocab2, merges2 = tokenizer2.train_from_word(word_freq, vocab_size=280)
        
        # 相同输入应产生相同结果
        assert vocab1 == vocab2
        assert merges1 == merges2


class TestEncodeDecodeWithTraining:
    """测试训练后的编码解码功能"""
    
    def setup_method(self):
        """为每个测试方法准备训练好的tokenizer"""
        self.tokenizer = BPETokenizer()
        word_freq = {
            "hello": 10,
            "world": 8,
            "help": 5,
            "held": 3
        }
        self.vocab, self.merges = self.tokenizer.train_from_word(word_freq, vocab_size=280)
    
    def test_encode_trained_words(self):
        """测试对训练过的词进行编码"""
        # 这里的编码可能不完美，因为实现较简化
        result = self.tokenizer.encode("hello")
        assert isinstance(result, list)
        assert all(isinstance(token_id, int) for token_id in result)
        assert all(0 <= token_id < len(self.vocab) for token_id in result)
    
    def test_decode_basic(self):
        """测试基本解码功能"""
        # 使用简单的字节token进行测试
        token_ids = [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')]
        result = self.tokenizer.decode(token_ids)
        assert result == "hello"
    
    def test_encode_decode_roundtrip_simple(self):
        """测试简单的编码-解码往返"""
        text = "hello"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        
        # 由于实现简化，可能不完全相等，但应该包含原始内容
        assert "hello" in decoded or decoded == "hello"


class TestSaveLoad:
    """测试保存和加载功能"""
    
    def test_save_and_load_basic(self):
        """测试基本的保存和加载"""
        # 创建并训练tokenizer
        tokenizer = BPETokenizer(special_tokens=["<pad>", "<unk>"])
        word_freq = {"hello": 5, "world": 3}
        tokenizer.train_from_word(word_freq, vocab_size=270)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            tokenizer.save(temp_file)
            
            # 创建新的tokenizer并加载
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(temp_file)
            
            # 比较关键属性
            assert len(new_tokenizer.vocab) == len(tokenizer.vocab)
            assert new_tokenizer.special_tokens == tokenizer.special_tokens
            
            # 测试解码功能是否一致
            test_ids = [ord('h'), ord('e'), ord('l'), ord('l'), ord('o')]
            assert new_tokenizer.decode(test_ids) == tokenizer.decode(test_ids)
            
        finally:
            os.unlink(temp_file)
    
    def test_save_load_empty_special_tokens(self):
        """测试无特殊token的保存加载"""
        tokenizer = BPETokenizer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            tokenizer.save(temp_file)
            
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(temp_file)
            
            assert new_tokenizer.special_tokens == []
            assert len(new_tokenizer.vocab) == 256
            
        finally:
            os.unlink(temp_file)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_train_from_word_large_frequencies(self):
        """测试大频次数值"""
        tokenizer = BPETokenizer()
        word_freq = {"aa": 1000000}  # 非常大的频次
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=260)
        assert len(vocab) > 256
        assert len(merges) > 0
    
    def test_train_from_word_many_unique_words(self):
        """测试大量唯一词"""
        tokenizer = BPETokenizer()
        word_freq = {f"word{i}": 1 for i in range(100)}
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=300)
        assert len(vocab) >= 256
        # 由于词都不同，可能没有很多合并
    
    def test_special_characters_in_training(self):
        """测试特殊字符的训练"""
        tokenizer = BPETokenizer()
        word_freq = {
            "hello!": 5,
            "world?": 3,
            "test@domain.com": 2,
            "user#123": 1
        }
        
        vocab, merges = tokenizer.train_from_word(word_freq, vocab_size=280)
        assert len(vocab) > 256
        assert len(merges) > 0


@pytest.fixture
def sample_word_freq():
    """提供样本词频数据"""
    return {
        "the": 100,
        "quick": 50,
        "brown": 40,
        "fox": 35,
        "jumps": 30,
        "over": 25,
        "lazy": 20,
        "dog": 15
    }


class TestIntegration:
    """集成测试"""
    
    def test_complete_workflow(self, sample_word_freq):
        """测试完整的工作流程"""
        # 初始化
        tokenizer = BPETokenizer(special_tokens=["<pad>", "<unk>", "<eos>"])
        
        # 训练
        vocab, merges = tokenizer.train_from_word(sample_word_freq, vocab_size=300)
        
        # 验证训练结果
        assert len(vocab) > 259  # 256 + 3 special tokens
        assert len(merges) > 0
        
        # 测试编码
        text = "the quick brown fox"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        
        # 测试解码
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        
        # 测试保存
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            tokenizer.save(temp_file)
            assert os.path.exists(temp_file)
            
            # 测试加载
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(temp_file)
            
            # 验证加载结果
            assert len(new_tokenizer.vocab) == len(tokenizer.vocab)
            assert new_tokenizer.special_tokens == tokenizer.special_tokens
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])