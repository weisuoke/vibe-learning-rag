# 核心概念5：上下文感知分块（Contextual Retrieval）⭐ NEW 2024-2025

**为每个 chunk 添加文档级上下文说明，减少 49-67% 检索失败，是 Anthropic 2024-2025 年推出的生产级优化方案。**

---

## 一句话定义

**上下文感知分块是在传统分块基础上，为每个 chunk 添加 50-100 token 的文档级上下文说明的方法，通过 LLM 生成包含章节位置、主题概括的上下文前缀，使孤立的 chunk 语义更清晰，单独使用减少 49% 检索失败，结合 reranking 减少 67% 检索失败。**

**研究来源**: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

## 核心原理

### 问题：传统分块的上下文丢失

```
原始文档：
"Python 编程教程
第一章：变量
变量是存储数据的容器。可以使用 = 赋值。

第二章：函数
函数是可重用的代码块。使用 def 定义。"

传统分块后：
Chunk 1: "变量是存储数据的容器。可以使用 = 赋值。"
Chunk 2: "函数是可重用的代码块。使用 def 定义。"

问题：
- Chunk 脱离了文档上下文
- 不知道这是 Python 教程的一部分
- 不知道属于哪个章节
- 语义模糊，检索效果差
```

### 解决方案：添加上下文

```
上下文感知分块后：
Chunk 1: 
"[上下文] 这是 Python 编程教程第一章关于变量的内容。

变量是存储数据的容器。可以使用 = 赋值。"

Chunk 2:
"[上下文] 这是 Python 编程教程第二章关于函数的内容。

函数是可重用的代码块。使用 def 定义。"

效果：
✅ 语义清晰
✅ 包含章节信息
✅ 检索准确率提升
```

---

## Anthropic 研究数据

### 效果提升

| 指标 | 传统分块 | +上下文感知 | +上下文+Rerank | 提升 |
|------|----------|------------|---------------|------|
| **检索失败率** | 基准 | -49% | -67% | 显著 |
| **Top-20 召回率** | 基准 | +5.7% | +10.2% | 显著 |
| **成本** | 低 | 中 | 中高 | 可接受 |

**研究来源**: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

### 关键发现

1. **单独使用效果显著**: 减少 49% 检索失败
2. **与 Reranking 协同**: 减少 67% 检索失败
3. **成本可控**: 使用 Claude Haiku 成本低
4. **通用性强**: 适用于各种文档类型

---

## Python 实现

### 基础实现

```python
from anthropic import Anthropic
from typing import List, Dict

client = Anthropic()

def add_context_to_chunk(
    chunk: str,
    document: str,
    chunk_index: int = 0,
    total_chunks: int = 0
) -> str:
    """
    为 chunk 添加上下文（Anthropic 方法）
    
    效果：减少 49% 检索失败
    """
    prompt = f"""
为以下文档片段生成 50-100 token 的上下文说明。

完整文档（前2000字符）：
{document[:2000]}

文档片段（第 {chunk_index+1}/{total_chunks} 块）：
{chunk}

要求：
1. 说明这个片段在文档中的位置（章节、主题）
2. 概括片段的核心内容
3. 保持简洁（50-100 tokens）

格式：
这是关于[主题]的[文档类型]，本片段位于[章节]，讨论[核心内容]。
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",  # 使用便宜的模型
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    
    context = response.content[0].text.strip()
    return f"{context}\n\n{chunk}"

# 使用示例
document = """
Python 编程基础教程

第一章：变量与数据类型
Python 是动态类型语言。变量不需要声明类型。
"""

chunk = "Python 是动态类型语言。变量不需要声明类型。"
contextual_chunk = add_context_to_chunk(chunk, document, 0, 5)

print("原始 chunk:")
print(chunk)
print("\n上下文感知 chunk:")
print(contextual_chunk)
```

### 生产级实现

```python
from anthropic import Anthropic
from typing import List, Dict
import asyncio

class ContextualChunker:
    """上下文感知分块器（生产级）"""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        context_length: int = 100
    ):
        self.client = Anthropic()
        self.model = model
        self.context_length = context_length
    
    def chunk_with_context(
        self,
        document: str,
        base_chunks: List[str]
    ) -> List[Dict]:
        """为基础分块添加上下文"""
        contextual_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            context = self._generate_context(
                chunk, document, i, len(base_chunks)
            )
            contextual_chunks.append({
                "original_chunk": chunk,
                "context": context,
                "final_text": f"{context}\n\n{chunk}",
                "index": i
            })
        
        return contextual_chunks
    
    def _generate_context(
        self,
        chunk: str,
        document: str,
        index: int,
        total: int
    ) -> str:
        """生成单个 chunk 的上下文"""
        prompt = self._build_prompt(chunk, document, index, total)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.context_length + 50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    def _build_prompt(
        self,
        chunk: str,
        document: str,
        index: int,
        total: int
    ) -> str:
        """构建上下文生成 prompt"""
        return f"""
为文档片段生成 {self.context_length} token 的上下文说明。

文档开头：
{document[:2000]}

片段位置：第 {index+1}/{total} 块
片段内容：
{chunk[:500]}

要求：
1. 说明片段在文档中的位置和主题
2. 保持简洁（约 {self.context_length} tokens）
3. 使用陈述句，不要使用"这是..."开头

示例格式：
在 Python 教程的变量章节中，本段介绍了动态类型的概念。
"""

# 使用示例
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 基础分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=77
)
base_chunks = splitter.split_text(document)

# 2. 添加上下文
chunker = ContextualChunker()
contextual_chunks = chunker.chunk_with_context(document, base_chunks)

# 3. 查看结果
for chunk_data in contextual_chunks[:2]:
    print(f"\n原始: {chunk_data['original_chunk'][:100]}...")
    print(f"上下文: {chunk_data['context']}")
    print(f"最终: {chunk_data['final_text'][:150]}...")
```

---

## 成本优化

### 模型选择

| 模型 | 价格（每百万 tokens） | 10个chunk成本 | 推荐场景 |
|------|---------------------|--------------|---------|
| **Claude 3 Haiku** | $0.25 (输入) / $1.25 (输出) | ~$0.02 | **推荐（性价比最高）** |
| Claude 3 Sonnet | $3 / $15 | ~$0.20 | 高质量要求 |
| GPT-4o-mini | $0.15 / $0.60 | ~$0.01 | 成本敏感 |

**推荐**: Claude 3 Haiku（Anthropic 官方推荐，性价比最高）

### 批量处理优化

```python
async def batch_add_context(
    chunks: List[str],
    document: str,
    batch_size: int = 10
) -> List[str]:
    """批量添加上下文（并发优化）"""
    
    async def process_one(chunk: str, index: int) -> str:
        return add_context_to_chunk(chunk, document, index, len(chunks))
    
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            process_one(chunk, i+j)
            for j, chunk in enumerate(batch)
        ])
        results.extend(batch_results)
    
    return results
```

### Prompt Caching（成本降低 90%）

```python
from anthropic import Anthropic

client = Anthropic()

def add_context_with_caching(
    chunk: str,
    document: str
) -> str:
    """使用 Prompt Caching 降低成本"""
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        system=[
            {
                "type": "text",
                "text": "你是文档上下文生成专家。",
            },
            {
                "type": "text",
                "text": f"完整文档：\n{document}",
                "cache_control": {"type": "ephemeral"}  # 缓存文档
            }
        ],
        messages=[{
            "role": "user",
            "content": f"为以下片段生成上下文：\n{chunk}"
        }]
    )
    
    return response.content[0].text.strip()
```

**效果**: 文档内容被缓存，后续 chunk 处理成本降低 90%

---

## 与 Reranking 结合

### 完整流程

```python
from typing import List, Dict

def contextual_retrieval_with_rerank(
    query: str,
    contextual_chunks: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    上下文感知检索 + Reranking
    
    效果：减少 67% 检索失败
    """
    # 1. 向量检索（使用上下文感知的 chunk）
    from openai import OpenAI
    client = OpenAI()
    
    # 获取查询向量
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    # 获取 chunk 向量（使用 final_text）
    chunk_embeddings = [
        client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["final_text"]
        ).data[0].embedding
        for chunk in contextual_chunks
    ]
    
    # 计算相似度
    import numpy as np
    similarities = [
        np.dot(query_embedding, chunk_emb)
        for chunk_emb in chunk_embeddings
    ]
    
    # 获取 top-20 候选
    top_20_indices = np.argsort(similarities)[-20:][::-1]
    candidates = [contextual_chunks[i] for i in top_20_indices]
    
    # 2. Reranking
    reranked = rerank_chunks(query, candidates, top_k)
    
    return reranked

def rerank_chunks(
    query: str,
    candidates: List[Dict],
    top_k: int
) -> List[Dict]:
    """使用 LLM 进行 reranking"""
    from anthropic import Anthropic
    client = Anthropic()
    
    # 构建 reranking prompt
    candidates_text = "\n\n".join([
        f"[{i+1}] {c['final_text'][:200]}..."
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""
查询：{query}

候选片段：
{candidates_text}

请按相关性排序，返回最相关的 {top_k} 个片段的编号（用逗号分隔）。
"""
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 解析排序结果
    rankings = [int(x.strip()) - 1 for x in response.content[0].text.split(",")]
    return [candidates[i] for i in rankings[:top_k]]
```

---

## 最佳实践

### 实践1：选择合适的基础分块策略

```python
# 推荐：递归字符分块 + 上下文感知
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # NVIDIA 2025 推荐
    chunk_overlap=77  # 15% overlap
)

base_chunks = splitter.split_text(document)
contextual_chunks = add_context_batch(base_chunks, document)
```

### 实践2：上下文长度优化

```python
# 根据文档类型调整上下文长度
def get_optimal_context_length(doc_type: str) -> int:
    """根据文档类型返回最优上下文长度"""
    if doc_type == "technical":
        return 100  # 技术文档需要更多上下文
    elif doc_type == "news":
        return 50   # 新闻文档上下文简短
    else:
        return 75   # 通用文档
```

### 实践3：质量验证

```python
def validate_context_quality(contextual_chunk: Dict) -> bool:
    """验证上下文质量"""
    context = contextual_chunk["context"]
    
    # 1. 长度检查
    if len(context.split()) < 30 or len(context.split()) > 120:
        return False
    
    # 2. 内容检查
    if not any(word in context.lower() for word in ["章节", "主题", "内容", "讨论"]):
        return False
    
    # 3. 格式检查
    if context.startswith("这是") or context.startswith("本段"):
        return False  # 避免模板化
    
    return True
```

---

## 常见问题

### Q1: 上下文感知适合所有场景吗？

**A**: ✅ **适合大多数生产场景**。

- 成本可控（使用 Haiku）
- 效果显著（减少 49-67% 失败）
- 实现简单（在现有分块基础上增强）

**不适合**: 成本极度敏感、实时处理场景

### Q2: 如何选择基础分块策略？

**A**: **推荐递归字符分块**。

- 上下文感知是增强层，不改变基础分块
- 递归分块 + 上下文感知 = 最佳组合
- 也可以与语义分块、代理式分块结合

### Q3: 上下文会增加多少存储成本？

**A**: **约增加 10-15%**。

- 每个 chunk 增加 50-100 tokens
- 原 chunk 500 tokens → 最终 600 tokens
- 存储成本增加可接受

### Q4: 与代理式分块如何选择？

**A**: 根据需求选择：

- **上下文感知**: 成本低、效果好、易实现（推荐）
- **代理式分块**: 质量最高、成本高、复杂度高
- **组合使用**: 代理式分块 + 上下文感知（最佳效果）

---

## 核心研究来源

**Anthropic 2024-2025**: [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- 减少 49% 检索失败（单独使用）
- 减少 67% 检索失败（结合 reranking）
- 使用 Claude 3 Haiku 成本可控

---

## 下一步学习

**同级概念：**
- ← [03_核心概念_04_代理式分块Agentic](./03_核心概念_04_代理式分块Agentic.md) - IBM 2025-2026

**实战代码：**
- → [07_实战代码_05_上下文感知实现](./07_实战代码_05_上下文感知实现.md) - 完整实现

**后续流程：**
- → [05_双重类比](./05_双重类比.md) - 用熟悉的概念理解分块
- → [06_反直觉点](./06_反直觉点.md) - 常见误区
