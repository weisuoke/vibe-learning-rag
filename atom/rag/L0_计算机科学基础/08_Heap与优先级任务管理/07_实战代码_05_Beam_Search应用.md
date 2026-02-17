# 实战代码05: Beam Search应用

## 核心目标

**实现Beam Search算法用于LLM文本生成,使用heap维护top-K候选序列,应用2025-2026最新优化技术。**

---

## 1. 基础实现

```python
import heapq
import math
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Candidate:
    """候选序列"""
    sequence: List[str]
    score: float  # log probability

    def __lt__(self, other):
        return self.normalized_score() < other.normalized_score()

    def normalized_score(self):
        """长度归一化分数"""
        return self.score / len(self.sequence) if self.sequence else 0

class BeamSearch:
    """
    Beam Search for LLM text generation

    使用min-heap维护top-K候选序列
    """

    def __init__(self, beam_width: int, max_length: int):
        self.beam_width = beam_width
        self.max_length = max_length

    def search(self, prompt: str, vocab: List[str], get_next_probs) -> List[Candidate]:
        """
        执行Beam Search

        Args:
            prompt: 初始提示
            vocab: 词汇表
            get_next_probs: 获取下一个token概率的函数

        Returns:
            Top-K候选序列
        """
        # 初始候选
        candidates = [Candidate(sequence=[prompt], score=0.0)]

        for step in range(self.max_length):
            new_candidates = []

            for candidate in candidates:
                # 获取下一个token的概率
                probs = get_next_probs(candidate.sequence)

                # 扩展候选序列
                for token, prob in probs.items():
                    new_seq = candidate.sequence + [token]
                    new_score = candidate.score + math.log(prob)

                    new_candidates.append(
                        Candidate(sequence=new_seq, score=new_score)
                    )

            # 保留top-K候选
            candidates = heapq.nlargest(
                self.beam_width,
                new_candidates,
                key=lambda x: x.normalized_score()
            )

            # 检查是否所有候选都结束
            if all(c.sequence[-1] == '<EOS>' for c in candidates):
                break

        return candidates

# 使用示例
def mock_get_next_probs(sequence):
    """模拟获取下一个token概率"""
    return {
        'hello': 0.3,
        'world': 0.5,
        '<EOS>': 0.2
    }

beam_search = BeamSearch(beam_width=3, max_length=5)
results = beam_search.search(
    prompt='<START>',
    vocab=['hello', 'world', '<EOS>'],
    get_next_probs=mock_get_next_probs
)

for i, candidate in enumerate(results):
    print(f"Candidate {i+1}: {' '.join(candidate.sequence)}")
    print(f"  Score: {candidate.normalized_score():.4f}")
```

---

## 2. 2025优化:Trie-based Beam Search

```python
# 参考: arXiv:2502.00085v2 (2025)
# "Efficient Beam Search for LLMs Using Trie-Based Decoding"

class TrieNode:
    """Trie节点"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.score = float('-inf')

class TrieBasedBeamSearch:
    """
    Trie-based Beam Search

    优化:
    - 使用Trie存储候选序列,减少内存占用
    - 共享公共前缀,提高效率
    """

    def __init__(self, beam_width: int):
        self.beam_width = beam_width
        self.root = TrieNode()

    def insert(self, sequence: List[str], score: float):
        """插入序列到Trie"""
        node = self.root
        for token in sequence:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]

        node.is_end = True
        node.score = score

    def get_top_k(self) -> List[Tuple[List[str], float]]:
        """获取Top-K序列"""
        results = []

        def dfs(node, path, score):
            if node.is_end:
                results.append((path.copy(), score))

            for token, child in node.children.items():
                dfs(child, path + [token], child.score)

        dfs(self.root, [], 0.0)

        return heapq.nlargest(
            self.beam_width,
            results,
            key=lambda x: x[1] / len(x[0])
        )
```

---

## 3. vLLM集成

```python
# 参考: vLLM documentation (2025)

class vLLMBeamSearch:
    """
    vLLM风格的Beam Search

    特点:
    - 高效的GPU内存管理
    - 支持批量处理
    - PagedAttention优化
    """

    def __init__(self, beam_width: int, length_penalty: float = 1.0):
        self.beam_width = beam_width
        self.length_penalty = length_penalty

    def search(self, prompts: List[str], model) -> List[List[str]]:
        """
        批量Beam Search

        Args:
            prompts: 输入提示列表
            model: LLM模型

        Returns:
            生成结果列表
        """
        results = []

        for prompt in prompts:
            candidates = self._beam_search_single(prompt, model)
            results.append(candidates[0].sequence)

        return results

    def _beam_search_single(self, prompt: str, model):
        """单个提示的Beam Search"""
        candidates = [Candidate(sequence=[prompt], score=0.0)]

        for _ in range(self.max_length):
            new_candidates = []

            for candidate in candidates:
                # 使用模型生成下一个token
                next_tokens = model.generate_next(candidate.sequence)

                for token, log_prob in next_tokens:
                    new_seq = candidate.sequence + [token]
                    new_score = candidate.score + log_prob

                    # 应用长度惩罚
                    normalized_score = new_score / (len(new_seq) ** self.length_penalty)

                    new_candidates.append(
                        Candidate(sequence=new_seq, score=normalized_score)
                    )

            # 保留top-K
            candidates = heapq.nlargest(
                self.beam_width,
                new_candidates,
                key=lambda x: x.score
            )

        return candidates
```

---

## 4. RAG应用:多候选答案生成

```python
class RAGBeamSearch:
    """
    RAG系统的Beam Search

    应用:生成多个候选答案,选择最优的返回
    """

    def __init__(self, beam_width: int = 3):
        self.beam_width = beam_width

    def generate_answers(
        self,
        query: str,
        context: str,
        llm_client
    ) -> List[Tuple[str, float]]:
        """
        生成多个候选答案

        Args:
            query: 用户查询
            context: 检索到的上下文
            llm_client: LLM客户端

        Returns:
            [(answer, score), ...]
        """
        candidates = []

        # 生成多个候选
        for _ in range(self.beam_width):
            answer = llm_client.generate(
                prompt=f"Context: {context}\n\nQuery: {query}\n\nAnswer:",
                temperature=0.7
            )

            # 评分
            score = self._score_answer(answer, query, context)

            candidates.append((answer, score))

        # 返回Top-K
        return heapq.nlargest(
            self.beam_width,
            candidates,
            key=lambda x: x[1]
        )

    def _score_answer(self, answer: str, query: str, context: str) -> float:
        """评分答案质量"""
        # 简化评分:基于长度和相关性
        relevance = len(set(answer.split()) & set(query.split()))
        length_penalty = 1.0 / (1.0 + abs(len(answer.split()) - 50))

        return relevance * length_penalty
```

---

## 5. 性能优化

```python
import time

def benchmark_beam_search():
    """Beam Search性能测试"""

    beam_widths = [1, 3, 5, 10]
    sequence_length = 20

    for width in beam_widths:
        beam_search = BeamSearch(beam_width=width, max_length=sequence_length)

        start = time.time()
        results = beam_search.search(
            prompt='<START>',
            vocab=['token1', 'token2', 'token3'],
            get_next_probs=mock_get_next_probs
        )
        elapsed = time.time() - start

        print(f"Beam width={width}:")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Candidates: {len(results)}")
        print(f"  Best score: {results[0].normalized_score():.4f}")
```

---

## 6. 一句话总结

**Beam Search使用heap维护top-K候选序列实现LLM文本生成,2025年Trie-based优化减少内存占用,vLLM提供高效GPU实现,广泛应用于RAG多候选答案生成和机器翻译。**
