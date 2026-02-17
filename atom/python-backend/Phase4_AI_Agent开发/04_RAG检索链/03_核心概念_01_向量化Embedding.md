# RAG检索链 - 核心概念1：向量化Embedding

> 将文本转换为数字向量，是 RAG 的基础技术

---

## 一句话定义

**Embedding 是将文本转换为高维向量的技术，相似的文本会被映射到向量空间中相近的位置，从而可以通过计算向量距离来判断文本的语义相似度。**

---

## 为什么需要 Embedding？

### 问题：计算机无法直接理解文本

```python
# 计算机看到的文本
text1 = "Python 是一门编程语言"
text2 = "Python 是一种程序设计语言"

# 问题：如何判断这两句话是否相似？
# 字符串比较？text1 == text2  # False（字面不同）
# 关键词匹配？有共同词"Python"，但无法量化相似度
```

### 解决方案：Embedding

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 转换为向量
vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)

# 计算相似度
from numpy import dot
from numpy.linalg import norm

similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
print(f"相似度: {similarity:.4f}")  # 0.9523（非常相似）
```

**关键洞察**：
- Embedding 把文本从"符号空间"映射到"向量空间"
- 在向量空间中，相似的文本距离近，不相似的文本距离远
- 可以用数学方法（余弦相似度）量化文本相似度

---

## Embedding 的工作原理

### 1. 从词向量到句向量

**词向量（Word Embedding）**：
```python
# 每个词映射到一个向量
"Python" → [0.2, 0.8, 0.3, ...]  # 300维向量
"编程"   → [0.3, 0.7, 0.4, ...]
"语言"   → [0.1, 0.6, 0.5, ...]
```

**句向量（Sentence Embedding）**：
```python
# 整个句子映射到一个向量
"Python 是一门编程语言" → [0.15, 0.72, 0.38, ...]  # 1536维向量

# 不是简单的词向量平均！
# 而是通过神经网络理解句子的整体语义
```

### 2. Embedding 模型的训练

**训练目标**：让语义相似的文本向量距离近

```python
# 训练数据示例
positive_pairs = [
    ("Python 是编程语言", "Python 是程序设计语言"),  # 相似
    ("快速排序算法", "快排的实现方法"),              # 相似
]

negative_pairs = [
    ("Python 是编程语言", "今天天气真好"),          # 不相似
    ("快速排序算法", "苹果很好吃"),                # 不相似
]

# 训练目标：
# - positive_pairs 的向量距离 → 最小化
# - negative_pairs 的向量距离 → 最大化
```

**常用训练方法**：
- **对比学习（Contrastive Learning）**：拉近相似样本，推远不相似样本
- **三元组损失（Triplet Loss）**：锚点、正样本、负样本
- **多任务学习**：同时训练多个任务（分类、相似度、问答）

### 3. Embedding 的维度

**维度的含义**：
```python
# 1536维向量示例
embedding = [
    0.023,   # 维度1：可能表示"技术"相关
    -0.145,  # 维度2：可能表示"抽象"程度
    0.678,   # 维度3：可能表示"编程"相关
    ...      # 其他1533个维度
]

# 注意：每个维度的具体含义是模型学习出来的，人类无法直接解释
```

**维度的权衡**：
| 维度 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **低维（128-384）** | 存储小、检索快 | 表达能力弱 | 简单文本、性能敏感 |
| **中维（768-1024）** | 平衡性能和效果 | 适中 | 大多数场景 |
| **高维（1536-3072）** | 表达能力强 | 存储大、检索慢 | 复杂语义、高精度要求 |

---

## Embedding 模型选择

### 1. OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# text-embedding-3-small（推荐）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # 维度：1536
    # 成本：$0.02 / 1M tokens
    # 速度：快
)

# text-embedding-3-large（高精度）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # 维度：3072
    # 成本：$0.13 / 1M tokens
    # 速度：较慢
)

# 使用
text = "Python 是一门编程语言"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")  # 1536
```

**优点**：
- ✅ 效果好，支持多语言
- ✅ API 调用简单
- ✅ 不需要本地部署

**缺点**：
- ❌ 需要联网，有延迟
- ❌ 有成本（虽然很低）
- ❌ 数据会发送到 OpenAI

### 2. Sentence Transformers（本地模型）

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用本地模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # 维度：384
    # 成本：免费
    # 速度：取决于硬件
)

# 使用
text = "Python 是一门编程语言"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")  # 384
```

**优点**：
- ✅ 免费，无 API 调用成本
- ✅ 数据不出本地，隐私安全
- ✅ 离线可用

**缺点**：
- ❌ 需要下载模型（几百MB）
- ❌ 需要 GPU 加速（否则很慢）
- ❌ 效果通常不如 OpenAI

### 3. 模型选择建议

```python
# 决策树
def choose_embedding_model():
    if 需要最高精度:
        return "OpenAI text-embedding-3-large"
    elif 有预算 and 需要联网:
        return "OpenAI text-embedding-3-small"
    elif 数据敏感 or 需要离线:
        return "Sentence Transformers"
    else:
        return "OpenAI text-embedding-3-small"  # 默认推荐
```

---

## Embedding 的关键技术

### 1. 文本预处理

```python
def preprocess_text(text: str) -> str:
    """预处理文本，提升 Embedding 质量"""
    # 1. 去除多余空白
    text = " ".join(text.split())

    # 2. 统一大小写（可选，取决于模型）
    # text = text.lower()  # 注意：OpenAI 模型不需要

    # 3. 去除特殊字符（可选）
    # import re
    # text = re.sub(r'[^\w\s]', '', text)

    # 4. 截断过长文本（避免超出模型限制）
    max_length = 8000  # OpenAI 限制 8191 tokens
    if len(text) > max_length:
        text = text[:max_length]

    return text

# 使用
raw_text = "  Python   是一门编程语言  "
clean_text = preprocess_text(raw_text)
vector = embeddings.embed_query(clean_text)
```

**注意事项**：
- OpenAI 模型对大小写敏感，不要随意转小写
- 不要过度清洗，保留语义信息
- 注意模型的 token 限制

### 2. 批量向量化

```python
# ❌ 低效：逐个向量化
texts = ["文本1", "文本2", "文本3", ...]
vectors = []
for text in texts:
    vector = embeddings.embed_query(text)  # 每次一个 API 调用
    vectors.append(vector)

# ✅ 高效：批量向量化
texts = ["文本1", "文本2", "文本3", ...]
vectors = embeddings.embed_documents(texts)  # 一次 API 调用

# 性能对比
# 逐个：100个文本 → 100次 API 调用 → 10秒
# 批量：100个文本 → 1次 API 调用 → 0.5秒
```

### 3. 向量缓存

```python
import hashlib
import json
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """从缓存获取向量"""
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def set(self, text: str, vector: list[float]):
        """保存向量到缓存"""
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        cache_file.write_text(json.dumps(vector))

# 使用
cache = EmbeddingCache()

def embed_with_cache(text: str) -> list[float]:
    # 先查缓存
    vector = cache.get(text)
    if vector:
        return vector

    # 缓存未命中，调用 API
    vector = embeddings.embed_query(text)
    cache.set(text, vector)
    return vector
```

---

## 手写简化版 Embedding

理解 Embedding 的原理，手写一个简化版：

```python
import numpy as np
from typing import List

class SimpleEmbedding:
    """简化版 Embedding（仅用于理解原理，实际请用专业模型）"""

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        # 随机初始化词向量矩阵
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_vectors = np.random.randn(vocab_size, embedding_dim)

        # 词汇表（词 → ID）
        self.vocab = {}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return text.lower().split()

    def _word_to_id(self, word: str) -> int:
        """词转ID"""
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) % self.vocab_size
        return self.vocab[word]

    def embed_query(self, text: str) -> np.ndarray:
        """文本转向量（简化版：词向量平均）"""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim)

        # 获取每个词的向量
        word_vecs = [
            self.word_vectors[self._word_to_id(word)]
            for word in tokens
        ]

        # 平均池化（实际模型会用更复杂的方法）
        sentence_vec = np.mean(word_vecs, axis=0)

        # 归一化
        sentence_vec = sentence_vec / np.linalg.norm(sentence_vec)

        return sentence_vec

# 使用
simple_emb = SimpleEmbedding()

text1 = "Python is a programming language"
text2 = "Python is a coding language"
text3 = "The weather is nice today"

vec1 = simple_emb.embed_query(text1)
vec2 = simple_emb.embed_query(text2)
vec3 = simple_emb.embed_query(text3)

# 计算相似度
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"text1 vs text2: {cosine_similarity(vec1, vec2):.4f}")  # 高相似度
print(f"text1 vs text3: {cosine_similarity(vec1, vec3):.4f}")  # 低相似度
```

**注意**：
- 这只是演示原理，实际效果很差
- 真实的 Embedding 模型使用 Transformer 等复杂架构
- 训练需要大量数据和计算资源

---

## 在 RAG 中的应用

### 1. 文档向量化

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 1. 加载文档
document = """
Python 是一门高级编程语言，由 Guido van Rossum 于 1991 年创建。
Python 的设计哲学强调代码的可读性和简洁的语法。
Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
"""

# 2. 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_text(document)

# 3. 向量化
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(chunks)

print(f"文档分成 {len(chunks)} 块")
print(f"每个向量维度: {len(vectors[0])}")
```

### 2. 问题向量化

```python
# 用户问题
question = "谁创建了 Python？"

# 向量化问题
question_vector = embeddings.embed_query(question)

# 计算与每个文档块的相似度
similarities = []
for i, doc_vector in enumerate(vectors):
    similarity = cosine_similarity(question_vector, doc_vector)
    similarities.append((i, similarity))

# 排序，找到最相关的块
similarities.sort(key=lambda x: x[1], reverse=True)

print("最相关的文档块:")
for i, sim in similarities[:2]:
    print(f"块 {i}: 相似度 {sim:.4f}")
    print(f"内容: {chunks[i]}\n")
```

### 3. 完整的 RAG 流程

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 1. 创建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(chunks, embeddings)

# 2. 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# 3. 提问
question = "谁创建了 Python？"
answer = qa_chain.invoke({"query": question})

print(f"问题: {question}")
print(f"答案: {answer['result']}")
# 输出: Guido van Rossum 于 1991 年创建了 Python。
```

---

## 常见问题

### Q1: Embedding 维度越高越好吗？

**A**: 不一定。

- **高维度**：表达能力强，但存储大、检索慢、容易过拟合
- **低维度**：存储小、检索快，但表达能力弱
- **最佳实践**：根据场景选择
  - 简单文本：384-768 维
  - 复杂语义：1536-3072 维

### Q2: 不同模型的向量可以混用吗？

**A**: 不可以。

```python
# ❌ 错误：混用不同模型的向量
vec1 = openai_embeddings.embed_query("Python")
vec2 = sentence_transformers_embeddings.embed_query("Java")
similarity = cosine_similarity(vec1, vec2)  # 结果无意义！

# ✅ 正确：使用同一个模型
vec1 = openai_embeddings.embed_query("Python")
vec2 = openai_embeddings.embed_query("Java")
similarity = cosine_similarity(vec1, vec2)  # 结果有意义
```

**原因**：
- 不同模型的向量空间不同
- 就像用米和英尺比较长度，单位不同

### Q3: 如何评估 Embedding 质量？

**A**: 使用相似度任务评估。

```python
# 准备测试数据
test_pairs = [
    ("Python 编程", "Python 程序设计", True),   # 相似
    ("Python 编程", "今天天气好", False),       # 不相似
    ("快速排序", "快排算法", True),             # 相似
    ("快速排序", "苹果好吃", False),            # 不相似
]

# 评估
correct = 0
for text1, text2, should_be_similar in test_pairs:
    vec1 = embeddings.embed_query(text1)
    vec2 = embeddings.embed_query(text2)
    similarity = cosine_similarity(vec1, vec2)

    is_similar = similarity > 0.7  # 阈值
    if is_similar == should_be_similar:
        correct += 1

accuracy = correct / len(test_pairs)
print(f"准确率: {accuracy:.2%}")
```

---

## 总结

**Embedding 的核心价值**：
1. 把文本转换为可计算的向量
2. 相似的文本 → 相似的向量
3. 支持语义检索（不是关键词匹配）

**在 RAG 中的作用**：
- 文档向量化：把知识库转换为向量
- 问题向量化：把用户问题转换为向量
- 相似度计算：找到最相关的文档

**最佳实践**：
- 使用专业模型（OpenAI 或 Sentence Transformers）
- 批量向量化提升性能
- 使用缓存减少 API 调用
- 根据场景选择合适的维度

---

**下一步**：学习 [03_核心概念_02_向量检索](./03_核心概念_02_向量检索.md)，了解如何高效检索向量。
