# 核心概念 1：TextSplitter 基类架构

> TextSplitter 的核心架构和设计原理

---

## 概述

TextSplitter 是 LangChain 文本分块系统的基类，定义了所有分块器的核心接口和算法。理解基类架构是掌握 TextSplitter 的关键。

**核心地位**：
- 所有分块器的抽象基类
- 定义了统一的接口规范
- 实现了核心的 `_merge_splits()` 算法
- 提供了工厂方法支持

---

## 类继承关系

```python
BaseDocumentTransformer (LangChain Core)
    ↓
TextSplitter (抽象基类)
    ↓
├── CharacterTextSplitter
├── RecursiveCharacterTextSplitter
├── TokenTextSplitter
└── ... (其他专用分块器)
```

**设计模式**：
- **模板方法模式**：基类定义算法骨架，子类实现具体逻辑
- **策略模式**：`length_function` 参数允许替换长度计算策略

---

## 核心参数

### 1. chunk_size（块大小）

```python
chunk_size: int = 4000  # 默认值
```

**作用**：控制每个块的最大大小

**推荐配置**：
- 通用文本：1000 characters
- Token 优化：512 tokens
- 大文档：1500-2000 characters

**验证规则**：
```python
if chunk_size <= 0:
    raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
```

### 2. chunk_overlap（块重叠）

```python
chunk_overlap: int = 200  # 默认值
```

**作用**：在块之间保持重叠，避免切断上下文

**推荐配置**：
- 通常设置为 chunk_size 的 15-20%
- 最小值：100 characters
- 最大值：chunk_size 的 30%

**验证规则**：
```python
if chunk_overlap < 0:
    raise ValueError(f"chunk_overlap must be >= 0")
if chunk_overlap > chunk_size:
    raise ValueError("chunk_overlap must be smaller than chunk_size")
```

### 3. length_function（长度计算函数）

```python
length_function: Callable[[str], int] = len  # 默认值
```

**作用**：计算文本长度的函数

**可选值**：
- `len`：字符数（默认）
- `tiktoken.encode`：token 数
- 自定义函数

**示例**：
```python
# 使用 token 计数
def token_length(text: str) -> int:
    return len(tiktoken.encode(text))

splitter = TextSplitter(length_function=token_length)
```

### 4. keep_separator（保留分隔符）

```python
keep_separator: bool | Literal["start", "end"] = False
```

**作用**：是否保留分隔符及其位置

**可选值**：
- `False`：不保留分隔符
- `True` 或 `"start"`：分隔符附加到下一个块开头
- `"end"`：分隔符附加到前一个块末尾

### 5. add_start_index（添加起始索引）

```python
add_start_index: bool = False
```

**作用**：是否在元数据中添加块的起始索引

**用途**：便于溯源和调试

### 6. strip_whitespace（去除空白）

```python
strip_whitespace: bool = True
```

**作用**：是否去除文档首尾空白

---

## 核心方法

### 1. split_text()（抽象方法）

```python
@abstractmethod
def split_text(self, text: str) -> list[str]:
    """Split text into multiple components."""
```

**特点**：
- 抽象方法，子类必须实现
- 输入：文本字符串
- 输出：文本块列表

### 2. create_documents()

```python
def create_documents(
    self, texts: list[str], metadatas: list[dict] | None = None
) -> list[Document]:
```

**功能**：将文本列表转换为 Document 对象列表

**流程**：
1. 遍历文本列表
2. 调用 `split_text()` 分块
3. 为每个块创建 Document 对象
4. 可选添加 `start_index` 到元数据

**示例**：
```python
texts = ["长文本1", "长文本2"]
metadatas = [{"source": "doc1"}, {"source": "doc2"}]
docs = splitter.create_documents(texts, metadatas)
```

### 3. split_documents()

```python
def split_documents(self, documents: Iterable[Document]) -> list[Document]:
```

**功能**：分割 Document 对象列表

**流程**：
1. 提取文本和元数据
2. 调用 `create_documents()`
3. 返回分割后的 Document 列表

**示例**：
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt")
docs = loader.load()
splits = splitter.split_documents(docs)
```

### 4. _merge_splits()（核心算法）

```python
def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
```

**功能**：将小块合并成中等大小的块，同时处理重叠

**算法逻辑**：
```python
docs = []
current_doc: list[str] = []
total = 0

for d in splits:
    len_ = self._length_function(d)

    # 如果添加当前块会超过 chunk_size
    if total + len_ + separator_len > self._chunk_size:
        # 保存当前块
        if len(current_doc) > 0:
            docs.append(self._join_docs(current_doc, separator))

            # 处理重叠：移除开头的元素直到满足 chunk_overlap
            while total > self._chunk_overlap:
                total -= self._length_function(current_doc[0]) + separator_len
                current_doc = current_doc[1:]

    # 添加当前块
    current_doc.append(d)
    total += len_ + separator_len

# 保存最后一个块
if len(current_doc) > 0:
    docs.append(self._join_docs(current_doc, separator))

return docs
```

**关键点**：
- 使用滑动窗口维护 `chunk_overlap`
- 动态调整 `current_doc` 以保持重叠
- 处理分隔符长度

---

## 工厂方法

### 1. from_tiktoken_encoder()

```python
@classmethod
def from_tiktoken_encoder(
    cls,
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    **kwargs
) -> Self:
```

**功能**：使用 tiktoken 编码器计算长度

**示例**：
```python
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=512,
    chunk_overlap=100
)
```

### 2. from_huggingface_tokenizer()

```python
@classmethod
def from_huggingface_tokenizer(
    cls, tokenizer: PreTrainedTokenizerBase, **kwargs
) -> TextSplitter:
```

**功能**：使用 HuggingFace tokenizer 计算长度

**示例**：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512
)
```

---

## 实战示例

### 示例 1：基本使用

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建分块器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

# 分块文本
text = "很长的文本..."
chunks = splitter.split_text(text)

print(f"分块数量: {len(chunks)}")
print(f"第一块: {chunks[0][:100]}...")
```

### 示例 2：创建 Document 对象

```python
# 准备文本和元数据
texts = ["文档1内容...", "文档2内容..."]
metadatas = [
    {"source": "doc1.txt", "author": "Alice"},
    {"source": "doc2.txt", "author": "Bob"}
]

# 创建 Document 对象
docs = splitter.create_documents(texts, metadatas)

# 查看结果
for doc in docs[:3]:
    print(f"内容: {doc.page_content[:50]}...")
    print(f"元数据: {doc.metadata}")
    print("---")
```

### 示例 3：使用 token 计数

```python
# 使用 tiktoken 计数
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=512,  # 512 tokens
    chunk_overlap=100
)

# 分块
chunks = splitter.split_text(text)

# 验证 token 数量
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
for i, chunk in enumerate(chunks):
    token_count = len(enc.encode(chunk))
    print(f"块 {i}: {token_count} tokens")
```

---

## 设计决策

### 1. 为什么使用抽象基类？

**原因**：
- 强制子类实现 `split_text()` 方法
- 提供统一的接口规范
- 支持多态性

### 2. 为什么 _merge_splits() 在基类中实现？

**原因**：
- 合并算法是通用的，所有分块器都需要
- 避免代码重复
- 集中维护核心算法

### 3. 为什么提供工厂方法？

**原因**：
- 简化常见用例的配置
- 封装复杂的初始化逻辑
- 提供更好的用户体验

---

## 常见问题

### Q1: chunk_size 和 chunk_overlap 如何配置？

**A**:
- chunk_size: 根据 LLM 的 context window 调整（通常 1000 characters 或 512 tokens）
- chunk_overlap: 设置为 chunk_size 的 15-20%

### Q2: 如何自定义 length_function？

**A**:
```python
def custom_length(text: str) -> int:
    # 自定义长度计算逻辑
    return len(text.split())  # 按单词数计算

splitter = RecursiveCharacterTextSplitter(
    length_function=custom_length,
    chunk_size=100  # 100个单词
)
```

### Q3: add_start_index 有什么用？

**A**:
- 便于溯源：知道块在原始文档中的位置
- 便于调试：快速定位问题
- 便于可视化：展示块的分布

---

## 性能考虑

### 时间复杂度

- `split_text()`: O(n)，其中 n 是文本长度
- `_merge_splits()`: O(m)，其中 m 是小块数量
- 总体: O(n + m)

### 空间复杂度

- O(n)：需要存储所有块

### 优化建议

1. **使用合适的 chunk_size**：避免过小或过大
2. **缓存 tokenizer**：避免重复创建
3. **批处理**：一次处理多个文档

---

## 下一步

理解了 TextSplitter 基类架构后，建议学习：

1. **03_核心概念_2_CharacterTextSplitter.md** - 简单分块器
2. **03_核心概念_3_RecursiveCharacterTextSplitter.md** - 最常用的分块器
3. **03_核心概念_4_TokenTextSplitter.md** - Token 级别分块

---

**数据来源**：
- [来源: reference/source_textsplitter_01_base.md | TextSplitter 基类源码分析]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
