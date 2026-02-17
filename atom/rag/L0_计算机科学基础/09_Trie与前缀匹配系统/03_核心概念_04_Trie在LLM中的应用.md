# Trie 在 LLM 中的应用

## 核心概念

**Trie 在 LLM 中的核心作用 = Token 前缀树管理 + Beam Search 优化 + 约束解码**

2025-2026 年，Trie 在 LLM 领域有了突破性应用，从传统的词表管理扩展到解码优化和约束生成。

---

## 1. Token 前缀树

### 1.1 什么是 Token 前缀树？

**定义：** 使用 Trie 存储 LLM 的 Token 词表，支持高效的前缀查询和 Token 匹配。

**背景：**
- LLM 的词表通常有 5 万 - 10 万个 Token
- Token 可能是字符、子词或完整单词
- 需要高效查询：给定前缀，找到所有可能的 Token

**示例：GPT-2 词表**
```python
# GPT-2 词表（部分）
tokens = [
    "hello",
    "world",
    "python",
    "pytorch",
    "programming",
    ...  # 50,257 个 Token
]

# 使用 Trie 存储
token_trie = Trie()
for token_id, token in enumerate(tokens):
    token_trie.insert(token, value=token_id)
```

---

### 1.2 Token 前缀树的优势

**优势 1：O(m) 查询复杂度**
```python
# 传统方案：遍历词表
def find_tokens_with_prefix(prefix, tokens):
    results = []
    for token in tokens:  # O(n)
        if token.startswith(prefix):  # O(m)
            results.append(token)
    return results
# 时间复杂度：O(n × m)

# Trie 方案
def find_tokens_with_prefix_trie(prefix, token_trie):
    return token_trie.get_words_with_prefix(prefix)
# 时间复杂度：O(m + k)，k 是结果数量
```

**优势 2：支持字符级前缀匹配**
```python
# 用户输入："py"
# Trie 快速找到所有以 "py" 开头的 Token
matches = token_trie.get_words_with_prefix("py")
# 返回：["python", "pytorch", "pydantic", ...]
```

**优势 3：内存效率**
- 共享前缀的 Token 共享路径
- 对于相似的 Token（如 "test", "testing", "tested"），节省空间

---

### 1.3 实现示例

```python
"""
Token 前缀树实现
演示：构建 Token Trie，支持前缀查询
"""

from transformers import GPT2Tokenizer

class TokenTrie:
    """Token 前缀树"""

    def __init__(self, tokenizer):
        self.trie = Trie()
        self.tokenizer = tokenizer
        self._build_trie()

    def _build_trie(self):
        """构建 Token Trie"""
        vocab = self.tokenizer.get_vocab()
        for token, token_id in vocab.items():
            # 清理 Token（去除特殊字符）
            clean_token = token.replace('Ġ', ' ').strip()
            if clean_token:
                self.trie.insert(clean_token, value=token_id)

    def get_tokens_with_prefix(self, prefix: str) -> list:
        """获取所有以 prefix 开头的 Token"""
        matches = self.trie.get_words_with_prefix(prefix)
        return [(token, self.trie._get_node(token).value) for token in matches]


# ===== 使用示例 =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
token_trie = TokenTrie(tokenizer)

# 查询前缀
prefix = "py"
tokens = token_trie.get_tokens_with_prefix(prefix)
print(f"以 '{prefix}' 开头的 Token: {tokens[:5]}")
# 输出：[('python', 12345), ('pytorch', 23456), ...]
```

---

## 2. Beam Search 优化

### 2.1 什么是 Beam Search？

**定义：** LLM 生成文本时，保留 top-k 个最可能的候选序列，逐步扩展。

**传统 Beam Search 问题：**
- 每一步需要计算所有可能的下一个 Token
- 词表大（5 万 - 10 万），计算量大
- 内存占用高

**Trie 优化方案（EMNLP 2025）：**
- 使用 Trie 存储候选序列
- 共享前缀的序列共享路径
- 减少内存占用和计算量

**来源：** "Efficient Beam Search for Large Language Models Using Trie-Based Decoding" (EMNLP 2025)
https://arxiv.org/abs/2502.00085

---

### 2.2 Trie-Based Beam Search 原理

**传统 Beam Search：**
```python
# 每个候选序列独立存储
candidates = [
    ["I", "love", "python"],
    ["I", "love", "pytorch"],
    ["I", "like", "python"],
]
# 内存：3 × 3 = 9 个 Token
```

**Trie-Based Beam Search：**
```
        root
         |
        "I"
       /   \
   "love"  "like"
    /  \      |
"python" "pytorch" "python"

# 内存：6 个节点（节省 33%）
```

**优势：**
- 共享前缀，减少内存占用
- 快速剪枝（删除低概率分支）
- 并行解码（多个候选序列同时扩展）

---

### 2.3 实现示例（简化版）

```python
"""
Trie-Based Beam Search 简化实现
演示：使用 Trie 优化 Beam Search
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TrieBeamSearch:
    """基于 Trie 的 Beam Search"""

    def __init__(self, model, tokenizer, beam_size=5):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.trie = Trie()

    def generate(self, prompt: str, max_length: int = 20):
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # 初始化 Trie
        self.trie.insert(prompt, value={"ids": input_ids, "score": 0.0})

        # Beam Search
        for _ in range(max_length):
            # 获取所有候选序列
            candidates = self._get_candidates()

            # 扩展每个候选序列
            new_candidates = []
            for seq, data in candidates:
                # 预测下一个 Token
                with torch.no_grad():
                    outputs = self.model(data["ids"])
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)

                # 获取 top-k Token
                top_k_probs, top_k_ids = torch.topk(probs, self.beam_size)

                # 扩展候选序列
                for prob, token_id in zip(top_k_probs[0], top_k_ids[0]):
                    new_seq = seq + self.tokenizer.decode([token_id])
                    new_ids = torch.cat([data["ids"], token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = data["score"] + torch.log(prob).item()
                    new_candidates.append((new_seq, new_ids, new_score))

            # 保留 top-k 候选序列
            new_candidates.sort(key=lambda x: x[2], reverse=True)
            new_candidates = new_candidates[:self.beam_size]

            # 更新 Trie
            self.trie = Trie()
            for seq, ids, score in new_candidates:
                self.trie.insert(seq, value={"ids": ids, "score": score})

        # 返回最佳序列
        best = max(self._get_candidates(), key=lambda x: x[1]["score"])
        return best[0]

    def _get_candidates(self):
        """获取所有候选序列"""
        candidates = []
        self._collect_candidates(self.trie.root, "", candidates)
        return candidates

    def _collect_candidates(self, node, path, candidates):
        """收集候选序列"""
        if node.is_end:
            candidates.append((path, node.value))
        for char, child in node.children.items():
            self._collect_candidates(child, path + char, candidates)


# ===== 使用示例 =====
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

beam_search = TrieBeamSearch(model, tokenizer, beam_size=5)
result = beam_search.generate("I love", max_length=10)
print(f"生成结果: {result}")
```

---

## 3. 约束解码（Constrained Decoding）

### 3.1 什么是约束解码？

**定义：** 在 LLM 生成过程中，限制只能生成符合特定规则的 Token。

**应用场景：**
1. **生成 JSON**：只允许生成合法的 JSON 结构
2. **代码补全**：只允许生成项目中存在的函数名
3. **领域生成**：只允许生成领域词汇（医疗、法律）

**传统方案问题：**
- 生成后验证：效率低，可能生成非法内容
- 重新采样：浪费计算资源

**Trie 方案：**
- 预先构建合法 Token 的 Trie
- 生成时，只采样 Trie 中存在的 Token
- 保证生成内容合法

---

### 3.2 字符级前缀条件约束

**核心思想：** 使用 Trie 存储所有合法前缀，生成时只采样合法前缀的下一个字符。

**来源：** "Solving Code Completion with Character Prefix Conditioning" (Medium 2025)
https://medium.com/@bridog314/solving-code-completion-with-character-prefix-conditioning-9321b394e2bf

**示例：代码补全**

```python
"""
约束解码：代码补全
演示：只生成项目中存在的函数名
"""

class ConstrainedDecoder:
    """约束解码器"""

    def __init__(self, model, tokenizer, allowed_tokens: list):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = Trie()

        # 构建合法 Token Trie
        for token in allowed_tokens:
            self.trie.insert(token)

    def generate(self, prompt: str, max_length: int = 20):
        """约束生成"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = prompt

        for _ in range(max_length):
            # 预测下一个 Token
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]

            # 过滤非法 Token
            allowed_token_ids = self._get_allowed_tokens(generated)
            mask = torch.full_like(logits, float('-inf'))
            mask[0, allowed_token_ids] = 0
            logits = logits + mask

            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 更新
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            generated += self.tokenizer.decode(next_token_id[0])

            # 检查是否完成
            if self.trie.search(generated):
                break

        return generated

    def _get_allowed_tokens(self, prefix: str):
        """获取合法的下一个 Token"""
        # 找到所有以 prefix 开头的合法 Token
        matches = self.trie.get_words_with_prefix(prefix)

        # 提取下一个字符
        next_chars = set()
        for match in matches:
            if len(match) > len(prefix):
                next_chars.add(match[len(prefix)])

        # 转换为 Token ID
        token_ids = []
        for char in next_chars:
            token_id = self.tokenizer.encode(char, add_special_tokens=False)[0]
            token_ids.append(token_id)

        return token_ids


# ===== 使用示例 =====
# 项目中存在的函数名
allowed_functions = [
    "calculate_sum",
    "calculate_average",
    "calculate_total",
    "process_data",
    "process_file",
]

decoder = ConstrainedDecoder(model, tokenizer, allowed_functions)

# 约束生成
prompt = "def calculate_"
result = decoder.generate(prompt, max_length=10)
print(f"生成结果: {result}")
# 输出：def calculate_sum 或 calculate_average 或 calculate_total
```

---

## 4. 领域感知解码

### 4.1 什么是领域感知解码？

**定义：** 在特定领域（医疗、法律）生成文本时，优先使用领域词汇。

**挑战：**
- 领域词汇持续更新
- 需要动态调整词表
- 保持生成质量

**Trie 方案（2026 最新）：**
- 构建领域词汇 Trie（持续更新）
- 解码时，优先采样 Trie 中的词汇
- 动态调整采样概率

**来源：** "Online Domain-aware LLM Decoding for Continual Domain Evolution" (2026)
https://arxiv.org/abs/2602.08088

---

### 4.2 实现示例

```python
"""
领域感知解码
演示：医疗领域文本生成
"""

class DomainAwareDecoder:
    """领域感知解码器"""

    def __init__(self, model, tokenizer, domain_vocab: list, boost_factor=2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.domain_trie = Trie()

        # 构建领域词汇 Trie
        for word in domain_vocab:
            self.domain_trie.insert(word)

    def generate(self, prompt: str, max_length: int = 50):
        """领域感知生成"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated = prompt

        for _ in range(max_length):
            # 预测下一个 Token
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]

            # 提升领域词汇的概率
            logits = self._boost_domain_tokens(logits, generated)

            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 更新
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            generated += self.tokenizer.decode(next_token_id[0])

        return generated

    def _boost_domain_tokens(self, logits, current_text):
        """提升领域词汇的概率"""
        # 获取当前文本的最后一个词
        words = current_text.split()
        if not words:
            return logits

        last_word = words[-1]

        # 查找以 last_word 开头的领域词汇
        matches = self.domain_trie.get_words_with_prefix(last_word)

        # 提升匹配词汇的概率
        for match in matches:
            # 获取下一个字符
            if len(match) > len(last_word):
                next_char = match[len(last_word)]
                token_id = self.tokenizer.encode(next_char, add_special_tokens=False)[0]
                logits[0, token_id] *= self.boost_factor

        return logits


# ===== 使用示例 =====
# 医疗领域词汇
medical_vocab = [
    "症状", "诊断", "治疗", "药物", "手术",
    "头痛", "发烧", "咳嗽", "胸痛",
    "高血压", "糖尿病", "冠心病",
]

decoder = DomainAwareDecoder(model, tokenizer, medical_vocab, boost_factor=2.0)

# 生成医疗文本
prompt = "患者主诉"
result = decoder.generate(prompt, max_length=30)
print(f"生成结果: {result}")
# 输出：患者主诉头痛、发烧，初步诊断为...（优先使用医疗词汇）
```

---

## 5. 语义前缀树

### 5.1 什么是语义前缀树？

**定义：** 基于 LLM 嵌入构建的 Trie，支持语义相似度检索。

**传统 Trie 问题：**
- 只支持字符级前缀匹配
- 无法处理语义相似但字符不同的查询

**语义前缀树方案（2025）：**
- 使用 LLM 嵌入表示 Token
- 构建语义 Trie（节点存储嵌入）
- 支持语义相似度检索

**来源：** "Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay" (2025)
https://arxiv.org/abs/2502.10151

---

### 5.2 应用场景

**场景 1：语义搜索**
```python
# 查询："机器学习"
# 传统 Trie：只返回以"机器学习"开头的结果
# 语义 Trie：返回语义相似的结果
# - "机器学习"
# - "深度学习"
# - "人工智能"
# - "神经网络"
```

**场景 2：AI Agent 路由**
```python
# 用户输入："帮我查天气"
# 传统 Trie：匹配关键词"查"、"天气"
# 语义 Trie：理解语义，匹配意图
# - "查询天气"
# - "天气预报"
# - "气象信息"
```

---

## 6. 性能对比

### 6.1 Beam Search 性能

| 方案 | 内存占用 | 解码速度 | 生成质量 |
|------|---------|---------|---------|
| 标准 Beam Search | 100% | 100% | 100% |
| Trie-Based Beam Search | 60% | 120% | 100% |

**数据来源：** EMNLP 2025 论文

---

### 6.2 约束解码性能

| 方案 | 合法率 | 生成速度 | 实现复杂度 |
|------|-------|---------|-----------|
| 后验证 | 70% | 50% | 低 |
| 重新采样 | 90% | 30% | 中 |
| Trie 约束解码 | 100% | 90% | 中 |

---

## 7. 总结

### 7.1 Trie 在 LLM 中的核心价值

1. **Token 管理**：高效存储和查询词表
2. **Beam Search 优化**：减少内存占用，提升解码速度
3. **约束解码**：保证生成内容合法
4. **领域感知**：优先使用领域词汇
5. **语义检索**：支持语义相似度匹配

### 7.2 未来方向

1. **多模态 Trie**：支持图像、音频 Token
2. **分布式 Trie**：大规模词表的分布式存储
3. **动态 Trie**：实时更新词表
4. **神经 Trie**：结合神经网络的 Trie

---

**版本**: v1.0
**最后更新**: 2026-02-14
**参考文献**:
- EMNLP 2025: "Efficient Beam Search for Large Language Models Using Trie-Based Decoding"
- 2026: "Online Domain-aware LLM Decoding for Continual Domain Evolution"
- 2025: "Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay"
- Medium 2025: "Solving Code Completion with Character Prefix Conditioning"

**下一步**: 学习 Trie 在 Agent 中的应用（`05_Trie在Agent中的应用.md`）
