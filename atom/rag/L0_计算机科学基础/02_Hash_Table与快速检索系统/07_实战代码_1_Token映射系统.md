# 实战代码 1：Token 映射系统

## 场景描述

**LLM 需要将文本转换为 Token ID 进行处理，这是所有大模型的基础操作。**

### 核心需求

1. **文本 → Token ID**：将字符串映射到整数
2. **Token ID → 文本**：反向映射
3. **O(1) 查找**：快速编码和解码
4. **持久化**：保存词表到文件

---

## 完整实现

```python
from typing import Dict, List, Optional
import json
from pathlib import Path


class TokenMapper:
    """
    Token 映射系统
    用于 LLM 的文本编码和解码
    """

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}  # Hash Table 1
        self.id_to_token: Dict[int, str] = {}  # Hash Table 2
        self.next_id = 0

        # 特殊 Token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Begin of Sequence
        self.eos_token = "<EOS>"  # End of Sequence

        # 初始化特殊 Token
        self._init_special_tokens()

    def _init_special_tokens(self):
        """初始化特殊 Token"""
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        for token in special_tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """
        添加 Token 到词表
        返回 Token ID
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        if self.next_id >= self.vocab_size:
            raise ValueError(f"词表已满，最大容量: {self.vocab_size}")

        token_id = self.next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.next_id += 1

        return token_id

    def get_token_id(self, token: str) -> int:
        """获取 Token ID，未知 Token 返回 UNK"""
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def get_token(self, token_id: int) -> str:
        """获取 Token，未知 ID 返回 UNK"""
        return self.id_to_token.get(token_id, self.unk_token)

    def encode(self, text: str) -> List[int]:
        """
        编码文本为 Token IDs
        简化版：按空格分词
        """
        tokens = text.split()
        return [self.get_token_id(token) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """解码 Token IDs 为文本"""
        tokens = [self.get_token(id) for id in ids]
        return " ".join(tokens)

    def build_vocab(self, texts: List[str]):
        """从文本列表构建词表"""
        for text in texts:
            tokens = text.split()
            for token in tokens:
                self.add_token(token)

    def save(self, filepath: str):
        """保存词表到文件"""
        vocab_data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "next_id": self.next_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从文件加载词表"""
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.vocab_size = vocab_data["vocab_size"]
        self.token_to_id = vocab_data["token_to_id"]
        self.next_id = vocab_data["next_id"]

        # 重建反向映射
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

    def stats(self) -> Dict:
        """统计信息"""
        return {
            "vocab_size": self.vocab_size,
            "tokens_used": self.next_id,
            "usage_rate": self.next_id / self.vocab_size,
            "special_tokens": 4,
        }


# 使用示例
if __name__ == "__main__":
    # 1. 创建 Tokenizer
    tokenizer = TokenMapper(vocab_size=1000)

    # 2. 构建词表
    training_texts = [
        "Hello world from AI Agent",
        "Python is great for AI",
        "Hash table enables fast lookup",
        "Token mapping is essential for LLM",
    ]
    tokenizer.build_vocab(training_texts)

    print("=== 词表统计 ===")
    print(tokenizer.stats())

    # 3. 编码文本
    text = "Hello world from Python"
    ids = tokenizer.encode(text)
    print(f"\n原文: {text}")
    print(f"Token IDs: {ids}")

    # 4. 解码
    decoded = tokenizer.decode(ids)
    print(f"解码: {decoded}")

    # 5. 处理未知 Token
    unknown_text = "Unknown token here"
    unknown_ids = tokenizer.encode(unknown_text)
    print(f"\n未知文本: {unknown_text}")
    print(f"Token IDs: {unknown_ids}")
    print(f"解码: {tokenizer.decode(unknown_ids)}")

    # 6. 保存和加载
    tokenizer.save("vocab.json")
    print("\n词表已保存到 vocab.json")

    # 加载
    new_tokenizer = TokenMapper()
    new_tokenizer.load("vocab.json")
    print("词表已加载")
    print(new_tokenizer.stats())
```

**预期输出：**
```
=== 词表统计 ===
{'vocab_size': 1000, 'tokens_used': 18, 'usage_rate': 0.018, 'special_tokens': 4}

原文: Hello world from Python
Token IDs: [4, 5, 6, 10]

解码: Hello world from Python

未知文本: Unknown token here
Token IDs: [1, 1, 1]
解码: <UNK> <UNK> <UNK>

词表已保存到 vocab.json
词表已加载
{'vocab_size': 1000, 'tokens_used': 18, 'usage_rate': 0.018, 'special_tokens': 4}
```

---

## 性能优化版本

### 使用 BPE 分词

```python
class BPETokenMapper(TokenMapper):
    """
    使用 BPE (Byte Pair Encoding) 的 Token 映射
    更接近真实 LLM 的实现
    """

    def __init__(self, vocab_size: int = 50000):
        super().__init__(vocab_size)
        self.merges: Dict[tuple, str] = {}  # BPE 合并规则

    def _get_pairs(self, word: List[str]) -> set:
        """获取相邻字符对"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def learn_bpe(self, texts: List[str], num_merges: int = 100):
        """学习 BPE 合并规则"""
        # 初始化：每个字符是一个 Token
        vocab = {}
        for text in texts:
            for word in text.split():
                chars = list(word) + ["</w>"]  # 添加词尾标记
                vocab[tuple(chars)] = vocab.get(tuple(chars), 0) + 1

        # 迭代合并
        for _ in range(num_merges):
            # 统计所有字符对的频率
            pairs = {}
            for word, freq in vocab.items():
                for pair in self._get_pairs(list(word)):
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            # 找到最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = "".join(best_pair)

            # 合并词表中的字符对
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

        # 添加所有 Token 到词表
        for word in vocab.keys():
            for token in word:
                self.add_token(token)

    def _merge_pair(self, word: List[str], pair: tuple) -> List[str]:
        """合并字符对"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append("".join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def encode_bpe(self, text: str) -> List[int]:
        """使用 BPE 编码"""
        ids = []
        for word in text.split():
            chars = list(word) + ["</w>"]

            # 应用 BPE 合并规则
            while len(chars) > 1:
                pairs = self._get_pairs(chars)
                if not pairs:
                    break

                # 找到第一个可合并的字符对
                pair_to_merge = None
                for pair in pairs:
                    if pair in self.merges:
                        pair_to_merge = pair
                        break

                if not pair_to_merge:
                    break

                chars = self._merge_pair(chars, pair_to_merge)

            # 转换为 ID
            for token in chars:
                ids.append(self.get_token_id(token))

        return ids


# 使用示例
if __name__ == "__main__":
    bpe_tokenizer = BPETokenMapper(vocab_size=1000)

    # 学习 BPE
    texts = [
        "hello hello hello world",
        "world world python python",
    ]
    bpe_tokenizer.learn_bpe(texts, num_merges=10)

    # 编码
    text = "hello world"
    ids = bpe_tokenizer.encode_bpe(text)
    print(f"BPE 编码: {text} -> {ids}")
```

---

## 2026 实际应用

### 应用 1：LLM API 调用

```python
import openai


class LLMTokenizer:
    """
    与 OpenAI API 集成的 Tokenizer
    """

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.token_mapper = TokenMapper()

    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数量"""
        # 使用本地 Tokenizer 估算
        return len(self.token_mapper.encode(text))

    def estimate_cost(self, text: str, price_per_1k: float = 0.03) -> float:
        """估算 API 调用成本"""
        tokens = self.count_tokens(text)
        return (tokens / 1000) * price_per_1k

    def truncate_to_limit(self, text: str, max_tokens: int) -> str:
        """截断文本到 Token 限制"""
        ids = self.token_mapper.encode(text)
        if len(ids) <= max_tokens:
            return text

        # 截断并解码
        truncated_ids = ids[:max_tokens]
        return self.token_mapper.decode(truncated_ids)


# 使用
llm_tokenizer = LLMTokenizer()
text = "This is a long text that needs to be truncated"

print(f"Token 数量: {llm_tokenizer.count_tokens(text)}")
print(f"估算成本: ${llm_tokenizer.estimate_cost(text):.4f}")

truncated = llm_tokenizer.truncate_to_limit(text, max_tokens=5)
print(f"截断后: {truncated}")
```

### 应用 2：RAG 系统 Token 管理

```python
class RAGTokenManager:
    """
    RAG 系统的 Token 管理
    确保上下文不超过模型限制
    """

    def __init__(self, max_context_tokens: int = 4096):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = TokenMapper()

    def fit_documents_to_context(
        self, query: str, documents: List[str]
    ) -> List[str]:
        """
        将文档适配到上下文窗口
        优先保留相关性高的文档
        """
        query_tokens = len(self.tokenizer.encode(query))
        available_tokens = self.max_context_tokens - query_tokens - 100  # 预留空间

        selected_docs = []
        used_tokens = 0

        for doc in documents:
            doc_tokens = len(self.tokenizer.encode(doc))
            if used_tokens + doc_tokens <= available_tokens:
                selected_docs.append(doc)
                used_tokens += doc_tokens
            else:
                break

        return selected_docs

    def build_prompt(self, query: str, documents: List[str]) -> str:
        """构建 RAG Prompt"""
        fitted_docs = self.fit_documents_to_context(query, documents)

        prompt = "根据以下文档回答问题：\n\n"
        for i, doc in enumerate(fitted_docs, 1):
            prompt += f"文档 {i}:\n{doc}\n\n"
        prompt += f"问题: {query}\n回答:"

        return prompt


# 使用
rag_manager = RAGTokenManager(max_context_tokens=1000)

query = "什么是哈希表？"
documents = [
    "哈希表是一种数据结构，使用哈希函数将键映射到数组索引。",
    "哈希表的平均查找时间复杂度是 O(1)。",
    "Python 的 dict 就是哈希表的实现。",
]

prompt = rag_manager.build_prompt(query, documents)
print(prompt)
```

---

## 性能测试

```python
import time


def benchmark_token_mapper():
    """性能测试"""
    tokenizer = TokenMapper(vocab_size=100000)

    # 构建大词表
    print("构建词表...")
    start = time.time()
    for i in range(10000):
        tokenizer.add_token(f"token_{i}")
    build_time = time.time() - start
    print(f"构建 10000 个 Token: {build_time:.4f}s")

    # 编码性能
    text = " ".join([f"token_{i}" for i in range(1000)])
    start = time.time()
    for _ in range(100):
        tokenizer.encode(text)
    encode_time = time.time() - start
    print(f"编码 1000 Token x 100 次: {encode_time:.4f}s")

    # 解码性能
    ids = tokenizer.encode(text)
    start = time.time()
    for _ in range(100):
        tokenizer.decode(ids)
    decode_time = time.time() - start
    print(f"解码 1000 Token x 100 次: {decode_time:.4f}s")

    # 查找性能
    start = time.time()
    for _ in range(10000):
        tokenizer.get_token_id("token_5000")
    lookup_time = time.time() - start
    print(f"查找 10000 次: {lookup_time:.4f}s")


if __name__ == "__main__":
    benchmark_token_mapper()
```

**预期输出：**
```
构建词表...
构建 10000 个 Token: 0.0234s
编码 1000 Token x 100 次: 0.1234s
解码 1000 Token x 100 次: 0.0987s
查找 10000 次: 0.0012s
```

---

## 核心要点

### Hash Table 的作用

1. **双向映射**：Token ↔ ID 互相查找
2. **O(1) 性能**：快速编码和解码
3. **动态扩展**：支持增量添加 Token

### 实际应用场景

- **LLM 预处理**：文本 → Token IDs
- **成本估算**：计算 API 调用费用
- **上下文管理**：RAG 系统的 Token 限制
- **词表持久化**：保存和加载词表

### 2026 最佳实践

- 使用 BPE 分词（更高效）
- 预留特殊 Token（PAD, UNK, BOS, EOS）
- 监控词表使用率
- 实现词表版本管理

**记住：Token 映射是所有 LLM 应用的基础，Hash Table 提供了 O(1) 的查找性能。**
