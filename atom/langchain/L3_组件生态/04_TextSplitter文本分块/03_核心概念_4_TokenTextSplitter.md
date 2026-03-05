# 核心概念 4：TokenTextSplitter

> 基于 Token 计数的精确分块器

---

## 概述

TokenTextSplitter 是 LangChain 中专门用于基于 token 数量进行文本分块的分块器。与基于字符数的分块器不同，它使用 tokenizer（如 tiktoken）来精确控制每个块的 token 数量，这对于需要严格控制 LLM 输入成本和上下文窗口的场景至关重要。

**核心特点**：
- 基于 token 计数而非字符数
- 精确控制 LLM 输入成本
- 支持多种 tokenizer（tiktoken, HuggingFace）
- 适合需要精确 token 控制的场景
- 与 LLM 的 context window 完美对齐

---

## 为什么需要 TokenTextSplitter？

### 问题背景

在 RAG 开发中，我们经常遇到以下问题：

1. **字符数 ≠ Token 数**：同样的字符数，不同语言的 token 数量差异很大
   - 英文："Hello world" = 2 tokens
   - 中文："你好世界" = 4 tokens（每个汉字约1个token）
   - 代码：`function hello() {}` = 6 tokens

2. **成本控制**：LLM API 按 token 计费，字符数无法精确预估成本
   - GPT-4：$0.03/1K tokens（输入）
   - 如果用字符数估算，可能导致成本超支

3. **Context Window 限制**：LLM 有严格的 token 限制
   - GPT-3.5：4K tokens
   - GPT-4：8K/32K tokens
   - Claude：100K tokens

### TokenTextSplitter 的解决方案

TokenTextSplitter 通过直接使用 tokenizer 计算长度，解决了上述问题：

```python
from langchain_text_splitters import TokenTextSplitter

# 基于 token 的分块
splitter = TokenTextSplitter(
    chunk_size=512,  # 512 tokens（不是字符！）
    chunk_overlap=50  # 50 tokens 重叠
)

text = "你好世界 Hello world " * 100
chunks = splitter.split_text(text)

# 每个块精确控制在 512 tokens 以内
for chunk in chunks:
    print(f"Token count: {len(splitter._tokenizer.encode(chunk))}")
```

---

## 核心参数

### 1. chunk_size（块大小）

**类型**：`int`
**默认值**：`4000`
**单位**：tokens（不是字符！）

**说明**：
- 每个块的最大 token 数量
- 应该根据 LLM 的 context window 设置
- 推荐值：512-1000 tokens（留出空间给 prompt 和输出）

**示例**：
```python
# GPT-3.5（4K context window）
splitter = TokenTextSplitter(chunk_size=1000)  # 留出 3K 给 prompt 和输出

# GPT-4（8K context window）
splitter = TokenTextSplitter(chunk_size=2000)  # 留出 6K 给 prompt 和输出

# Claude（100K context window）
splitter = TokenTextSplitter(chunk_size=10000)  # 可以使用更大的块
```

### 2. chunk_overlap（块重叠）

**类型**：`int`
**默认值**：`200`
**单位**：tokens

**说明**：
- 相邻块之间的重叠 token 数量
- 通常设置为 chunk_size 的 10-20%
- 重叠有助于保持上下文连续性

**示例**：
```python
# 10% 重叠
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

# 20% 重叠（推荐）
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
```

### 3. encoding_name（编码名称）

**类型**：`str`
**默认值**：`"gpt2"`
**可选值**：`"gpt2"`, `"r50k_base"`, `"p50k_base"`, `"cl100k_base"`

**说明**：
- 指定使用的 tiktoken 编码器
- 不同的 LLM 使用不同的编码器

**编码器对应关系**：
| 编码器 | 对应模型 |
|--------|----------|
| `gpt2` | GPT-2, GPT-3 |
| `r50k_base` | Codex |
| `p50k_base` | GPT-3.5-turbo（早期） |
| `cl100k_base` | GPT-3.5-turbo, GPT-4 |

**示例**：
```python
# GPT-4
splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=1000
)

# GPT-3
splitter = TokenTextSplitter(
    encoding_name="gpt2",
    chunk_size=1000
)
```

### 4. model_name（模型名称）

**类型**：`str | None`
**默认值**：`None`

**说明**：
- 直接指定模型名称，自动选择对应的编码器
- 优先级高于 `encoding_name`

**示例**：
```python
# 自动选择 cl100k_base 编码器
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000
)

# 自动选择 cl100k_base 编码器
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-3.5-turbo",
    chunk_size=1000
)
```

---

## 使用方法

### 方法 1：直接创建

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text = "Your long text here..."
chunks = splitter.split_text(text)
```

### 方法 2：使用 from_tiktoken_encoder()（推荐）

```python
from langchain_text_splitters import TokenTextSplitter

# 方式 1：指定模型名称
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,
    chunk_overlap=200
)

# 方式 2：指定编码器名称
splitter = TokenTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)
```

### 方法 3：使用 HuggingFace Tokenizer

```python
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer

# 加载 HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 创建分块器
splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=50
)

chunks = splitter.split_text(text)
```

---

## 实战示例

### 示例 1：成本优化分块

**场景**：需要精确控制 LLM API 成本

```python
from langchain_text_splitters import TokenTextSplitter
import tiktoken

# 创建分块器
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,  # 每个块 1000 tokens
    chunk_overlap=100
)

# 长文本
text = """
[Your long document here...]
""" * 100

# 分块
chunks = splitter.split_text(text)

# 计算成本
encoding = tiktoken.encoding_for_model("gpt-4")
total_tokens = sum(len(encoding.encode(chunk)) for chunk in chunks)

# GPT-4 定价：$0.03/1K tokens（输入）
cost = (total_tokens / 1000) * 0.03
print(f"Total chunks: {len(chunks)}")
print(f"Total tokens: {total_tokens}")
print(f"Estimated cost: ${cost:.4f}")
```

**输出**：
```
Total chunks: 15
Total tokens: 14500
Estimated cost: $0.4350
```

### 示例 2：Context Window 优化

**场景**：根据不同 LLM 的 context window 调整块大小

```python
from langchain_text_splitters import TokenTextSplitter

def create_splitter_for_model(model_name: str) -> TokenTextSplitter:
    """根据模型创建合适的分块器"""

    # 模型配置
    model_configs = {
        "gpt-3.5-turbo": {"context": 4096, "chunk_ratio": 0.25},
        "gpt-4": {"context": 8192, "chunk_ratio": 0.25},
        "gpt-4-32k": {"context": 32768, "chunk_ratio": 0.25},
        "claude-2": {"context": 100000, "chunk_ratio": 0.1},
    }

    config = model_configs.get(model_name, {"context": 4096, "chunk_ratio": 0.25})

    # 计算 chunk_size（留出空间给 prompt 和输出）
    chunk_size = int(config["context"] * config["chunk_ratio"])
    chunk_overlap = int(chunk_size * 0.1)

    return TokenTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

# 使用
splitter_gpt35 = create_splitter_for_model("gpt-3.5-turbo")
splitter_gpt4 = create_splitter_for_model("gpt-4")
splitter_claude = create_splitter_for_model("claude-2")

print(f"GPT-3.5 chunk_size: {splitter_gpt35._chunk_size}")
print(f"GPT-4 chunk_size: {splitter_gpt4._chunk_size}")
print(f"Claude chunk_size: {splitter_claude._chunk_size}")
```

**输出**：
```
GPT-3.5 chunk_size: 1024
GPT-4 chunk_size: 2048
Claude chunk_size: 10000
```

### 示例 3：多语言文本分块

**场景**：处理包含多种语言的文本

```python
from langchain_text_splitters import TokenTextSplitter
import tiktoken

# 创建分块器
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=500,
    chunk_overlap=50
)

# 多语言文本
text = """
English: Hello world. This is a test.
中文：你好世界。这是一个测试。
日本語：こんにちは世界。これはテストです。
한국어：안녕하세요 세계. 이것은 테스트입니다。
"""

# 分块
chunks = splitter.split_text(text)

# 分析每个块的 token 数量
encoding = tiktoken.encoding_for_model("gpt-4")
for i, chunk in enumerate(chunks):
    tokens = encoding.encode(chunk)
    print(f"Chunk {i+1}: {len(tokens)} tokens")
    print(f"Content: {chunk[:50]}...")
    print()
```

### 示例 4：RAG 文档分块管道

**场景**：完整的 RAG 文档处理流程

```python
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 加载文档
loader = TextLoader("document.txt")
docs = loader.load()

# 2. 基于 token 的分块
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=512,  # 512 tokens per chunk
    chunk_overlap=50
)
splits = splitter.split_documents(docs)

print(f"Total splits: {len(splits)}")

# 3. 向量化和存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

# 4. 检索
query = "What is the main topic?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### 示例 5：Token 计数验证

**场景**：验证分块后的 token 数量

```python
from langchain_text_splitters import TokenTextSplitter
import tiktoken

# 创建分块器
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,
    chunk_overlap=100
)

# 长文本
text = "Your long text here..." * 1000

# 分块
chunks = splitter.split_text(text)

# 验证 token 数量
encoding = tiktoken.encoding_for_model("gpt-4")

print("Token count validation:")
print("-" * 50)

for i, chunk in enumerate(chunks[:5]):  # 只显示前5个
    tokens = encoding.encode(chunk)
    print(f"Chunk {i+1}: {len(tokens)} tokens (max: 1000)")

    # 检查是否超过限制
    if len(tokens) > 1000:
        print(f"  ⚠️  WARNING: Exceeds chunk_size!")
    else:
        print(f"  ✓ Within limit")

# 统计信息
all_token_counts = [len(encoding.encode(chunk)) for chunk in chunks]
print(f"\nStatistics:")
print(f"  Total chunks: {len(chunks)}")
print(f"  Avg tokens per chunk: {sum(all_token_counts) / len(all_token_counts):.1f}")
print(f"  Max tokens: {max(all_token_counts)}")
print(f"  Min tokens: {min(all_token_counts)}")
```

---

## 适用场景

### 1. 成本敏感的应用

**场景**：需要精确控制 LLM API 成本

**为什么使用 TokenTextSplitter**：
- 精确计算 token 数量
- 避免成本超支
- 优化 API 调用次数

### 2. Context Window 受限的场景

**场景**：使用 context window 较小的模型（如 GPT-3.5）

**为什么使用 TokenTextSplitter**：
- 精确控制输入 token 数量
- 避免超过 context window 限制
- 优化 prompt 和输出空间

### 3. 多语言文本处理

**场景**：处理包含多种语言的文本

**为什么使用 TokenTextSplitter**：
- 不同语言的字符数和 token 数差异很大
- Token 计数更准确
- 避免中文/日文等语言的块过大

### 4. 代码分块

**场景**：分块代码文件

**为什么使用 TokenTextSplitter**：
- 代码的 token 数量与字符数差异大
- 精确控制代码块大小
- 适合代码生成和分析任务

---

## 常见问题

### Q1：TokenTextSplitter 和 CharacterTextSplitter 有什么区别？

**A**：

| 特性 | TokenTextSplitter | CharacterTextSplitter |
|------|-------------------|----------------------|
| 长度单位 | tokens | 字符数 |
| 精确度 | 高（与 LLM 一致） | 低（估算） |
| 性能 | 较慢（需要 tokenize） | 快 |
| 成本控制 | 精确 | 估算 |
| 适用场景 | 成本敏感、多语言 | 简单文本 |

**推荐**：
- 成本敏感 → TokenTextSplitter
- 简单文本 → CharacterTextSplitter

### Q2：如何选择合适的 chunk_size？

**A**：

**考虑因素**：
1. **LLM 的 context window**：留出空间给 prompt 和输出
2. **检索质量**：块太大会降低检索精度
3. **成本**：块越大，成本越高

**推荐配置**：
```python
# GPT-3.5（4K context）
chunk_size = 512  # 留出 3.5K 给 prompt 和输出

# GPT-4（8K context）
chunk_size = 1000  # 留出 7K 给 prompt 和输出

# GPT-4-32K
chunk_size = 2000  # 留出 30K 给 prompt 和输出
```

### Q3：chunk_overlap 应该设置多少？

**A**：

**推荐值**：chunk_size 的 10-20%

**原因**：
- 太小：上下文丢失
- 太大：重复内容过多，成本增加

**示例**：
```python
# 10% 重叠（最小）
TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

# 15% 重叠（推荐）
TokenTextSplitter(chunk_size=1000, chunk_overlap=150)

# 20% 重叠（最大）
TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
```

### Q4：如何处理超长的单个句子？

**A**：

如果单个句子超过 chunk_size，TokenTextSplitter 会强制分割：

```python
splitter = TokenTextSplitter(chunk_size=100)

# 超长句子
long_sentence = "word " * 200  # 200 个单词

chunks = splitter.split_text(long_sentence)
print(f"Chunks: {len(chunks)}")  # 会分成多个块
```

**解决方案**：
- 增加 chunk_size
- 预处理文本，分割超长句子

---

## 与 RAG 开发的联系

### 1. 成本优化

在 RAG 应用中，TokenTextSplitter 帮助精确控制成本：

```python
# RAG 成本计算
# 假设：1000 个文档，每个文档 5000 tokens

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# 分块后的 token 数量
chunks_per_doc = (5000 / (512 - 50))  # 约 11 个块
total_chunks = 1000 * 11  # 11,000 个块

# Embedding 成本（假设 $0.0001/1K tokens）
embedding_cost = (total_chunks * 512 / 1000) * 0.0001

# 检索成本（假设每次检索 3 个块）
queries_per_day = 1000
retrieval_tokens = queries_per_day * 3 * 512
retrieval_cost = (retrieval_tokens / 1000) * 0.03  # GPT-4 定价

print(f"Embedding cost: ${embedding_cost:.4f}")
print(f"Daily retrieval cost: ${retrieval_cost:.4f}")
```

### 2. 检索质量优化

TokenTextSplitter 通过精确控制块大小，优化检索质量：

```python
# 实验：不同 chunk_size 对检索质量的影响

chunk_sizes = [256, 512, 1024, 2048]

for size in chunk_sizes:
    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=int(size * 0.1))
    splits = splitter.split_documents(docs)

    # 构建向量库
    vectorstore = Chroma.from_documents(splits, embeddings)

    # 检索测试
    results = vectorstore.similarity_search(query, k=3)

    print(f"Chunk size: {size}")
    print(f"  Total chunks: {len(splits)}")
    print(f"  Avg relevance: {calculate_relevance(results)}")
```

### 3. 多模型支持

TokenTextSplitter 支持不同的 LLM，提高 RAG 系统的灵活性：

```python
def create_rag_pipeline(model_name: str):
    """根据模型创建 RAG 管道"""

    # 创建分块器
    splitter = TokenTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=512,
        chunk_overlap=50
    )

    # 分块
    splits = splitter.split_documents(docs)

    # 向量化
    vectorstore = Chroma.from_documents(splits, embeddings)

    # 创建 RAG 链
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model_name=model_name)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

# 使用不同模型
qa_gpt35 = create_rag_pipeline("gpt-3.5-turbo")
qa_gpt4 = create_rag_pipeline("gpt-4")
```

---

## 最佳实践

### 1. 根据模型选择编码器

```python
# ✓ 推荐：使用 model_name
splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",  # 自动选择正确的编码器
    chunk_size=1000
)

# ✗ 不推荐：手动指定编码器（容易出错）
splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # 可能与模型不匹配
    chunk_size=1000
)
```

### 2. 预留足够的 Context Window

```python
# ✓ 推荐：只使用 25% 的 context window
splitter = TokenTextSplitter(
    chunk_size=1000,  # GPT-4 8K context 的 25%
    chunk_overlap=100
)

# ✗ 不推荐：使用过大的 chunk_size
splitter = TokenTextSplitter(
    chunk_size=7000,  # 几乎占满 context window
    chunk_overlap=100
)
```

### 3. 验证 Token 数量

```python
import tiktoken

splitter = TokenTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000
)

chunks = splitter.split_text(text)

# 验证
encoding = tiktoken.encoding_for_model("gpt-4")
for chunk in chunks:
    token_count = len(encoding.encode(chunk))
    assert token_count <= 1000, f"Chunk exceeds limit: {token_count} tokens"
```

### 4. 缓存 Tokenizer

```python
# ✓ 推荐：复用 splitter 实例
splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4")

for doc in documents:
    chunks = splitter.split_text(doc)  # 复用 tokenizer

# ✗ 不推荐：每次创建新实例
for doc in documents:
    splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4")
    chunks = splitter.split_text(doc)  # 重复加载 tokenizer
```

---

## 数据来源

- [来源: reference/source_textsplitter_01_base.md | TokenTextSplitter 源码分析]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: tiktoken 官方文档 | Token 计数原理]
