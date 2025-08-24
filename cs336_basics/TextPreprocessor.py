import os
import multiprocessing
from typing import BinaryIO, Dict, List, Optional, Union
from collections import defaultdict
from pathlib import Path
import regex as re


class TextPreprocessor:
    """
    文本预处理器，用于大文件的词频统计
    支持多进程处理和自定义分割标记
    """
    
    def __init__(
        self,
        num_processes: Optional[int] = None,
        split_token: str = "<|endoftext|>",
        chunk_read_size: int = 4096,
        pattern: Optional[str] = None
    ):
        """
        初始化文本预处理器
        
        Args:
            num_processes: 进程数，默认为CPU核心数
            split_token: 用于分割文件的特殊标记
            chunk_read_size: 读取文件时的缓冲区大小
            pattern: 自定义tokenization正则表达式
        """
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.split_token = split_token
        self.split_token_bytes = split_token.encode('utf-8')
        self.chunk_read_size = chunk_read_size
        
        # 默认的tokenization pattern，支持缩写、单词、数字、标点符号
        self.pattern = pattern or r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.word_freq = defaultdict(int)
        self.total_tokens = 0
        self.processed_chunks = 0
    
    def _pre_tokenize_chunk(self, chunk: str) -> Dict[str, int]:
        """
        对文本块进行tokenization并统计词频
        
        Args:
            chunk: 要处理的文本块
            
        Returns:
            词频统计字典
        """
        word_count = defaultdict(int)
        
        if not chunk:
            return word_count
            
        # 去除首尾空白字符
        chunk = chunk.strip()
        if not chunk:
            return word_count
        
        # 使用正则表达式进行tokenization
        pattern_iter = re.finditer(self.pattern, chunk)
        for match in pattern_iter:
            token = match.group()
            word_count[token] += 1
        
        return word_count
    
    def _find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int) -> List[int]:
        """
        找到文件的分块边界，确保在特殊标记处分割
        
        Args:
            file: 二进制文件对象
            desired_num_chunks: 期望的分块数量
            
        Returns:
            分块边界位置列表
        """
        # 获取文件总大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            return [0]
        
        # 计算大致的块大小
        chunk_size = file_size // desired_num_chunks
        
        # 初始边界猜测
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        # 调整边界到特殊标记位置
        for i in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[i]
            file.seek(initial_position)
            
            while True:
                mini_chunk = file.read(self.chunk_read_size)
                
                # 如果到了文件末尾
                if not mini_chunk:
                    chunk_boundaries[i] = file_size
                    break
                
                # 查找特殊标记
                found_at = mini_chunk.find(self.split_token_bytes)
                if found_at != -1:
                    chunk_boundaries[i] = initial_position + found_at
                    break
                
                initial_position += self.chunk_read_size
        
        # 确保边界唯一且有序
        return sorted(set(chunk_boundaries))
    
    def _read_file_chunks(self, file_path: Union[str, Path]) -> List[str]:
        """
        读取文件并分成多个文本块
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表
        """
        chunks = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")
        
        with open(file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, self.num_processes)
            
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                chunks.append(chunk_text)
        
        return chunks
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, int]:
        """
        处理文件并返回词频统计结果
        
        Args:
            file_path: 要处理的文件路径
            
        Returns:
            词频统计字典
        """
        # 重置统计信息
        self.word_freq = defaultdict(int)
        self.total_tokens = 0
        self.processed_chunks = 0
        
        # 读取文件分块
        chunks = self._read_file_chunks(file_path)
        self.processed_chunks = len(chunks)
        
        if not chunks:
            return dict(self.word_freq)
        
        # 多进程处理文本块
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            chunk_word_freqs = pool.map(self._pre_tokenize_chunk, chunks)
        
        # 合并结果
        for chunk_word_freq in chunk_word_freqs:
            for word, freq in chunk_word_freq.items():
                self.word_freq[word] += freq
                self.total_tokens += freq
        
        return dict(self.word_freq)
    
    def get_top_words(self, n: int = 10) -> List[tuple]:
        """
        获取出现频率最高的n个词
        
        Args:
            n: 返回的词数量
            
        Returns:
            (词, 频率)的元组列表，按频率降序排列
        """
        return sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_vocabulary_size(self) -> int:
        """获取词汇表大小（唯一token数量）"""
        return len(self.word_freq)
    
    def get_total_tokens(self) -> int:
        """获取总token数量"""
        return self.total_tokens
    
    def get_processing_info(self) -> Dict[str, Union[int, float]]:
        """
        获取处理信息统计
        
        Returns:
            包含处理统计信息的字典
        """
        return {
            "total_tokens": self.total_tokens,
            "vocabulary_size": self.get_vocabulary_size(),
            "processed_chunks": self.processed_chunks,
            "num_processes": self.num_processes,
            "avg_tokens_per_chunk": self.total_tokens / self.processed_chunks if self.processed_chunks > 0 else 0
        }
    
    def save_results(self, output_path: Union[str, Path], top_n: Optional[int] = None):
        """
        保存处理结果到文件
        
        Args:
            output_path: 输出文件路径
            top_n: 只保存频率最高的n个词，None表示保存全部
        """
        output_path = Path(output_path)
        
        if top_n:
            words_to_save = self.get_top_words(top_n)
        else:
            words_to_save = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Token\tFrequency\n")
            for word, freq in words_to_save:
                f.write(f"{repr(word)}\t{freq}\n")
        
        print(f"结果已保存到: {output_path}")
    
    def print_summary(self):
        """打印处理摘要"""
        info = self.get_processing_info()
        print("=== 文本预处理摘要 ===")
        print(f"总token数: {info['total_tokens']:,}")
        print(f"词汇表大小: {info['vocabulary_size']:,}")
        print(f"处理的文本块数: {info['processed_chunks']}")
        print(f"使用进程数: {info['num_processes']}")
        print(f"平均每块token数: {info['avg_tokens_per_chunk']:.1f}")
        
        if self.word_freq:
            print("\n=== 频率最高的10个token ===")
            for i, (token, freq) in enumerate(self.get_top_words(10), 1):
                print(f"{i:2d}. {repr(token):20} {freq:,}")


# 使用示例
if __name__ == "__main__":
    # 创建预处理器实例
    preprocessor = TextPreprocessor(
        num_processes=4,
        split_token="<|endoftext|>",
        chunk_read_size=4096
    )
    
    # 处理文件
    try:
        file_path = "../tests/fixtures/address.txt"  # 替换为你的文件路径
        word_freq = preprocessor.process_file(file_path)
        
        # 打印摘要
        preprocessor.print_summary()
        
        # 保存结果
        # preprocessor.save_results("word_frequency_results.txt", top_n=1000)
        
        # 获取处理信息
        info = preprocessor.get_processing_info()
        print(f"\n处理完成，词汇表大小: {info['vocabulary_size']}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")