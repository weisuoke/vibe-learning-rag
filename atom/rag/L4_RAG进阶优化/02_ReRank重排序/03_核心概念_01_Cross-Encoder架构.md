# 03_核心概念_01_Cross-Encoder架构

## 什么是Cross-Encoder？

**定义：** Cross-Encoder是一种将query和document拼接后联合编码的深度神经网络架构，通过Transformer的自注意力机制实现query-document间的深度语义交互，输出相关性分数。

**核心特点：**
- 联合编码：query和doc作为单个输入序列
- 深度交互：每个token都与其他token做注意力计算
- 输出分数：直接输出相关性分数，不生成embedding

---

## Cross-Encoder vs Bi-Encoder

### 架构对比

**Bi-Encoder（双塔模型）：**

```python
# Bi-Encoder架构
class BiEncoder:
    def __init__(self):
        self.query_encoder = Transformer()  # Query塔
        self.doc_encoder = Transformer()    # Document塔

    def encode(self, query, doc):
        # 分别编码，无交互
        query_emb = self.query_encoder(query)  # [768]
        doc_emb = self.doc_encoder(doc)        # [768]
        return query_emb, doc_emb

    def score(self, query_emb, doc_emb):
        # 余弦相似度
        return cosine_similarity(query_emb, doc_emb)
```

**Cross-Encoder（交叉编码器）：**

```python
# Cross-Encoder架构
class CrossEncoder:
    def __init__(self):
        self.encoder = Transformer()  # 单个编码器
        self.classifier = nn.Linear(768, 1)  # 分类头

    def encode(self, query, doc):
        # 拼接输入，联合编码
        input_text = f"{query} [SEP] {doc}"
        tokens = tokenizer(input_text)  # [seq_len]

        # Transformer编码
        hidden_states = self.encoder(tokens)  # [seq_len, 768]

        # 提取[CLS] token表示
        cls_output = hidden_states[0]  # [768]

        # 输出相关性分数
        score = self.classifier(cls_output)  # [1]
        return score
```

### 关键差异表

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|--------------|
| **输入方式** | Query和Doc分别输入 | Query+Doc拼接输入 |
| **编码器数量** | 2个（双塔） | 1个（单塔） |
| **交互深度** | 无交互（后期点积） | 深度交互（注意力） |
| **输出** | 两个embedding向量 | 单个相关性分数 |
| **计算复杂度** | O(n) - 可预计算 | O(n²) - 必须实时 |
| **精度** | 中等 | 高（+15-48%） |
| **延迟** | 10-50ms | 200ms-2s |
| **适用场景** | 海量初筛 | 候选集精排 |

---

## Cross-Encoder的工作原理

### 1. 输入拼接

```python
# 输入格式
query = "什么是RAG？"
doc = "RAG是检索增强生成技术，结合了检索和生成"

# 拼接成单个序列
input_text = f"{query} [SEP] {doc}"
# 输出："什么是RAG？ [SEP] RAG是检索增强生成技术，结合了检索和生成"

# Tokenization
tokens = tokenizer(input_text)
# 输出：[CLS] 什么 是 RAG ？ [SEP] RAG 是 检索 增强 生成 技术 ， 结合 了 检索 和 生成 [SEP]
```

**特殊token说明：**
- `[CLS]`：分类token，用于提取整体表示
- `[SEP]`：分隔符，区分query和document
- `[PAD]`：填充token，对齐序列长度

### 2. Transformer编码

```python
# Transformer自注意力机制
class SelfAttention:
    def forward(self, tokens):
        # tokens: [seq_len, 768]

        # 计算Q, K, V
        Q = self.W_q(tokens)  # [seq_len, 768]
        K = self.W_k(tokens)  # [seq_len, 768]
        V = self.W_v(tokens)  # [seq_len, 768]

        # 注意力分数
        scores = Q @ K.T / sqrt(d_k)  # [seq_len, seq_len]
        attention = softmax(scores)   # [seq_len, seq_len]

        # 加权求和
        output = attention @ V  # [seq_len, 768]
        return output
```

**关键：** 每个token都与其他token计算注意力，实现深度交互

**注意力矩阵示例：**

```
        [CLS] 什么 是 RAG ？ [SEP] RAG 是 检索 增强 生成
[CLS]    0.1  0.05 0.05 0.1 0.05 0.05 0.2 0.1 0.15 0.1 0.05
什么     0.05 0.2  0.15 0.3 0.1  0.05 0.05 0.05 0.03 0.01 0.01
是       0.05 0.15 0.2  0.25 0.1  0.05 0.05 0.05 0.05 0.03 0.02
RAG      0.1  0.3  0.25 0.15 0.05 0.05 0.05 0.02 0.01 0.01 0.01
...
```

**解读：**
- "什么"对"RAG"的注意力权重高（0.3）：理解query的核心词
- "RAG"（query）对"RAG"（doc）的注意力权重高：匹配关键词
- "[CLS]"对"检索"、"增强"、"生成"的注意力权重高：捕捉核心语义

### 3. 分类头输出

```python
# 提取[CLS] token表示
cls_output = hidden_states[0]  # [768]

# 线性分类层
score = self.classifier(cls_output)  # [1]

# 可选：Sigmoid归一化
score = sigmoid(score)  # [0, 1]
```

**为什么用[CLS] token？**
- BERT预训练时，[CLS]被训练为整个序列的聚合表示
- 包含了query和document的全局语义信息
- 适合用于分类任务（相关/不相关）

---

## Cross-Encoder的优势

### 1. 深度语义理解

**示例1：否定语义**

```python
query = "不含糖的饮料"

# Bi-Encoder（词汇重叠）
doc1 = "含糖饮料"  # 词汇重叠高 → 高分 ❌
doc2 = "无糖可乐"  # 词汇重叠低 → 低分 ❌

# Cross-Encoder（理解否定）
doc1 = "含糖饮料"  # 理解"不含"的否定 → 低分 ✅
doc2 = "无糖可乐"  # 理解"不含"="无糖" → 高分 ✅
```

**示例2：因果关系**

```python
query = "为什么RAG能提升LLM准确性？"

# Bi-Encoder（表面匹配）
doc1 = "RAG是检索增强生成"  # 包含"RAG" → 高分 ❌
doc2 = "RAG通过检索相关文档提供上下文，减少幻觉"  # 解释因果 → 高分 ✅

# Cross-Encoder（理解因果）
doc1 = "RAG是检索增强生成"  # 没有回答"为什么" → 低分 ✅
doc2 = "RAG通过检索相关文档提供上下文，减少幻觉"  # 回答因果 → 高分 ✅
```

### 2. 细粒度匹配

**示例：条件匹配**

```python
query = "适合初学者的Python书籍"

# Bi-Encoder（粗粒度）
doc1 = "Python高级编程"  # 包含"Python" → 中等分 ❌
doc2 = "Python入门教程"  # 包含"Python"+"入门" → 高分 ✅

# Cross-Encoder（细粒度）
doc1 = "Python高级编程"  # 理解"高级"≠"初学者" → 低分 ✅
doc2 = "Python入门教程"  # 理解"入门"="初学者" → 高分 ✅
```

### 3. 上下文依赖

**示例：代词消歧**

```python
query = "它的主要功能是什么？"
context = "RAG是检索增强生成技术"

# Bi-Encoder（无上下文）
doc1 = "主要功能包括..."  # 不知道"它"指什么 → 中等分 ❌

# Cross-Encoder（有上下文）
# 如果将context拼接到query
input_text = f"{context} {query} [SEP] {doc1}"
# Cross-Encoder能理解"它"指"RAG" → 高分 ✅
```

---

## Cross-Encoder的劣势

### 1. 计算成本高

**问题：** 无法预计算document embedding

```python
# Bi-Encoder：可预计算
# 离线阶段
doc_embeddings = [encoder(doc) for doc in all_docs]  # 预计算
save(doc_embeddings)  # 保存

# 在线阶段
query_emb = encoder(query)  # 只需计算query
scores = [cosine(query_emb, doc_emb) for doc_emb in doc_embeddings]
# 延迟：10ms

# Cross-Encoder：必须实时计算
# 在线阶段
scores = [cross_encoder(query, doc) for doc in all_docs]
# 延迟：100万文档 × 0.2ms = 200秒 ❌
```

### 2. 延迟高

**2026年实测数据：**

| 文档数量 | Bi-Encoder | Cross-Encoder |
|---------|-----------|--------------|
| 10 | 5ms | 40ms |
| 50 | 10ms | 200ms |
| 100 | 20ms | 400ms |
| 1000 | 50ms | 4s |
| 100万 | 100ms | 55小时 ❌ |

### 3. 无法用于初检

**原因：**
- 计算复杂度O(n²)：每个query-doc对都要计算
- 无法利用向量索引（HNSW、IVF）加速
- 必须遍历所有文档

---

## 主流Cross-Encoder模型

### 1. BGE reranker系列（BAAI）

**2026年推荐模型：**

| 模型 | 参数量 | NDCG@10 | 延迟（50文档） | 特点 |
|------|--------|---------|---------------|------|
| bge-reranker-base | 278M | 0.82 | 150ms | 快速原型 |
| **bge-reranker-v2-m3** | **568M** | **0.85** | **200ms** | **生产推荐** |
| bge-reranker-large | 1.2B | 0.87 | 500ms | 高精度 |
| bge-reranker-v2-gemma | 2.5B | 0.88 | 1200ms | 研究实验 |

**使用示例：**

```python
from sentence_transformers import CrossEncoder

# 加载模型
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

# 计算分数
query = "什么是RAG？"
docs = ["RAG是检索增强生成", "今天天气很好"]
scores = reranker.predict([(query, doc) for doc in docs])
# 输出：[0.9876, 0.1234]
```

### 2. Cohere Rerank 4

**特点：**
- 企业级API服务
- 最高精度（NDCG@10: 0.90）
- 多语言支持（100+语言）
- Nimble快速版（延迟150ms）

**使用示例：**

```python
import cohere

client = cohere.Client(api_key="your_key")

# ReRank
results = client.rerank(
    query="什么是RAG？",
    documents=["RAG是检索增强生成", "今天天气很好"],
    top_n=5,
    model="rerank-4"
)

for result in results.results:
    print(f"分数: {result.relevance_score}, 文档: {result.document.text}")
```

### 3. ZeroEntropy zerank-1

**特点：**
- 最快延迟（60ms for 50文档）
- ELO-based训练方法
- 开源+商业许可
- NDCG@10: 0.87

**使用示例：**

```python
from zeroentropy import ZeRank

reranker = ZeRank('zerank-1')
scores = reranker.rerank(query, documents, top_k=5)
```

---

## Cross-Encoder训练方法

### 1. Pointwise训练

**目标：** 独立评分每个query-doc对

```python
# 训练数据
data = [
    (query1, doc1, 1),  # 相关
    (query1, doc2, 0),  # 不相关
    (query2, doc3, 1),  # 相关
]

# 损失函数
loss = BCELoss(predicted_score, ground_truth_label)
```

**优点：** 简单，易于标注
**缺点：** 忽略文档间的相对关系

### 2. Pairwise训练

**目标：** 学习相对排序

```python
# 训练数据
data = [
    (query1, doc_pos, doc_neg),  # doc_pos > doc_neg
    (query2, doc_pos, doc_neg),
]

# 损失函数
loss = max(0, margin - (score_pos - score_neg))
```

**优点：** 学习相对排序，更符合reranking任务
**缺点：** 需要成对标注

### 3. Listwise训练

**目标：** 整体优化排序质量

```python
# 训练数据
data = [
    (query1, [doc1, doc2, doc3], [3, 1, 2]),  # 相关性分数
]

# 损失函数（ListNet）
loss = -sum(y_true * log(softmax(y_pred)))
```

**优点：** 直接优化NDCG等排序指标
**缺点：** 计算复杂度高

---

## 实战优化技巧

### 1. 批处理优化

```python
# ❌ 逐个处理（慢）
scores = []
for doc in candidates:
    score = reranker.predict([(query, doc)])
    scores.append(score[0])
# 延迟：50文档 × 20ms = 1000ms

# ✅ 批处理（快）
scores = reranker.predict(
    [(query, doc) for doc in candidates],
    batch_size=16
)
# 延迟：200ms（5倍加速）
```

### 2. 最大长度截断

```python
# 问题：长文档超过max_length（512 tokens）

# 方案1：简单截断
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)

# 方案2：滑动窗口
def rerank_long_doc(query, doc, window_size=400, stride=200):
    chunks = split_with_overlap(doc, window_size, stride)
    scores = reranker.predict([(query, chunk) for chunk in chunks])
    return max(scores)  # 取最高分
```

### 3. GPU加速

```python
# CPU推理（慢）
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
# 50文档：2-3秒

# GPU推理（快）
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')
# 50文档：200ms（10倍加速）
```

---

## 关键要点速记

### 核心概念
1. Cross-Encoder：query+doc联合编码，深度交互
2. vs Bi-Encoder：联合编码 vs 分别编码
3. 输出：相关性分数，不是embedding

### 技术细节
4. 输入格式：`[CLS] query [SEP] doc [SEP]`
5. 核心机制：Transformer自注意力
6. 分类头：提取[CLS] token表示

### 优势
7. 深度语义理解：否定、因果、条件
8. 细粒度匹配：理解复杂语义关系
9. 精度提升：NDCG@10 +15-48%

### 劣势
10. 计算成本高：O(n²)，无法预计算
11. 延迟高：50文档200ms vs Bi-Encoder 10ms
12. 无法用于初检：必须两阶段策略

### 主流模型
13. BGE reranker-v2-m3：开源，568M，NDCG 0.85
14. Cohere Rerank 4：API，最高精度0.90
15. ZeroEntropy zerank-1：最快60ms

---

## 参考资料

### 核心论文
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - 2019
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) - 2020

### 技术文档
- [BGE Reranker v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Cohere Rerank Best Practices](https://docs.cohere.com/docs/reranking-best-practices)
- [Sentence-Transformers Cross-Encoders](https://www.sbert.net/docs/pretrained_cross-encoders.html)

### 研究报告
- [Ultimate Guide to Choosing the Best Reranking Model in 2026](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
- [Cross-Encoder Reranking Improves RAG Accuracy by 40%](https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study) - MIT, 2026

---

**版本：** v1.0 (2026年标准)
**最后更新：** 2026-02-16
**适用场景：** RAG开发、信息检索、搜索优化
