# 核心概念 3：RecursiveCharacterTextSplitter

> 最常用的文本分块器，LangChain 推荐的默认选择

---

## 概述

RecursiveCharacterTextSplitter 是 LangChain 中最常用的文本分块器，通过递归尝试多个分隔符来平衡语义完整性和块大小控制。

**核心特点**：
- 递归尝试多个分隔符
- 优先保留大的语义单元
- LangChain 推荐的默认分块器
- 支持20+种编程语言

**推荐指数**：⭐⭐⭐⭐⭐

---

## 核心参数

```python
class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: list[str] | None = None,  # 分隔符列表
        keep_separator: bool | Literal["start", "end"] = True,  # 保留分隔符
        is_separator_regex: bool = False,  # 是否为正则表达式
        **kwargs  # 继承自 TextSplitter 的参数
    ):
```

### 1. separators（分隔符列表）

**默认值**：
```python
["\n\n", "\n", " ", ""]
```

**优先级**：从大到小
1. `"\n\n"`：段落分隔符（最大语义单元）
2. `"\n"`：行分隔符
3. `" "`：单词分隔符
4. `""`：字符分隔符（最小单元）

### 2. keep_separator（保留分隔符）

**默认值**：`True`（等同于 `"start"`）

**作用**：保留分隔符以维持文档结构

---

## 递归算法

### 1. 算法流程

```python
def _split_text(self, text: str, separators: list[str]) -> list[str]:
    final_chunks = []
    separator = separators[-1]  # 默认使用最后一个
    new_separators = []

    # 找到第一个存在于文本中的分隔符
    for i, s_ in enumerate(separators):
        if re.search(s_, text):
            separator = s_
            new_separators = separators[i + 1:]  # 剩余分隔符
            break

    # 使用找到的分隔符分割
    splits = _split_text_with_regex(text, separator, keep_separator)

    # 递归处理
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)  # 满足大小，保留
        else:
            # 不满足大小，递归使用下一个分隔符
            if new_separators:
                other_info = self._split_text(s, new_separators)
                final_chunks.extend(other_info)
            else:
                final_chunks.append(s)  # 无更多分隔符，直接添加

    return final_chunks
```

### 2. 递归策略

**核心思想**：优先保留大的语义单元，只有在必要时才切割成更小的单元

**示例**：
```
文本: "段落1\n\n段落2\n\n很长的段落3..."

第1次尝试：使用 "\n\n" 分割
- "段落1" ✓ (< chunk_size)
- "段落2" ✓ (< chunk_size)
- "很长的段落3..." ✗ (> chunk_size)

第2次尝试：对"很长的段落3..."使用 "\n" 分割
- "行1" ✓
- "行2" ✓
- "很长的行3..." ✗ (> chunk_size)

第3次尝试：对"很长的行3..."使用 " " 分割
- "单词1 单词2 ..." ✓
- ...
```

---

## 推荐配置（2025-2026）

### 1. 通用配置

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 1000 characters
    chunk_overlap=200,  # 20% 重叠
    add_start_index=True  # 便于溯源
)
```

### 2. 基准测试最佳配置

```python
# 基于 Reddit 基准测试（69% 准确率）
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=512,  # 512 tokens
    chunk_overlap=100
)
```

### 3. RAG 应用配置

```python
# LangChain 官方推荐
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True
)
```

---

## 语言特定分块

### 1. from_language() 工厂方法

```python
@classmethod
def from_language(cls, language: Language, **kwargs) -> RecursiveCharacterTextSplitter:
    separators = cls.get_separators_for_language(language)
    return cls(separators=separators, is_separator_regex=True, **kwargs)
```

### 2. 支持的语言

**20+ 种语言**：
- Python, JavaScript, TypeScript
- Java, Kotlin, Scala
- C, C++, C#, Go, Rust
- Ruby, PHP, Perl, Lua
- HTML, Markdown, LaTeX
- 等

### 3. Python 示例

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

code = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass
"""

chunks = splitter.split_text(code)
```

**Python 分隔符优先级**：
```python
[
    "\nclass ",  # 类定义
    "\ndef ",    # 函数定义
    "\n\tdef ", # 缩进函数
    "\n\n",     # 段落
    "\n",       # 行
    " ",        # 单词
    ""          # 字符
]
```

---

## 实战示例

### 示例 1：基本使用

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
人工智能的发展需要大量数据。

数据是人工智能的基础。没有数据，AI 模型无法训练。

深度学习模型需要海量数据才能达到良好的性能。
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"块 {i}: {chunk}")
```

### 示例 2：RAG 文档分块

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loader = TextLoader("document.txt")
docs = loader.load()

# 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splits = splitter.split_documents(docs)

print(f"分块数量: {len(splits)}")
print(f"第一块元数据: {splits[0].metadata}")
```

### 示例 3：代码分块

```python
# Python 代码分块
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

# JavaScript 代码分块
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=500,
    chunk_overlap=50
)
```

---

## 与其他分块器对比

| 特性 | RecursiveCharacterTextSplitter | CharacterTextSplitter | TokenTextSplitter |
|------|-------------------------------|----------------------|-------------------|
| 分隔符数量 | 多个（递归） | 单个 | N/A（基于 token） |
| 语义完整性 | 高 | 低 | 中 |
| 性能 | 中 | 高 | 中 |
| 灵活性 | 高 | 低 | 中 |
| 代码支持 | ✓（20+ 语言） | ✗ | ✗ |
| 推荐场景 | 通用文本、代码 | 简单文本 | 成本优化 |

---

## 最佳实践

### 1. 选择合适的 chunk_size

```python
# 小文档（< 1000 字符）
splitter = RecursiveCharacterTextSplitter(chunk_size=500)

# 中等文档（1000-5000 字符）
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 大文档（> 5000 字符）
splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
```

### 2. 配置 chunk_overlap

```python
# 通常设置为 chunk_size 的 15-20%
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # 20%
)
```

### 3. 使用 add_start_index

```python
# 便于溯源和调试
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # 添加起始索引
)
```

### 4. 自定义分隔符

```python
# 自定义分隔符列表
custom_separators = [
    "\n## ",  # Markdown 二级标题
    "\n### ", # Markdown 三级标题
    "\n\n",   # 段落
    "\n",     # 行
    " ",      # 单词
    ""        # 字符
]

splitter = RecursiveCharacterTextSplitter(
    separators=custom_separators,
    chunk_size=1000,
    chunk_overlap=200
)
```

---

## 常见问题

### Q1: 为什么 RecursiveCharacterTextSplitter 是默认选择？

**A**: 因为它平衡了语义完整性和块大小控制：
- 优先保留大的语义单元（段落）
- 只有在必要时才切割成更小的单元
- 适用于大多数文本类型

### Q2: 如何处理代码文件？

**A**: 使用 `from_language()` 方法：
```python
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000
)
```

### Q3: 分块大小不均匀怎么办？

**A**: 这是正常的，因为递归算法优先保留语义完整性。如果需要更均匀的块，可以：
- 调整 chunk_size
- 使用 TokenTextSplitter

### Q4: 如何优化性能？

**A**:
- 使用合适的 chunk_size（避免过小）
- 减少分隔符数量
- 使用 CharacterTextSplitter（如果文本简单）

---

## 2025-2026 最佳实践总结

### 推荐配置

**来源**：Reddit 基准测试 + LangChain 官方文档 + GitHub 社区

1. **chunk_size**: 512 tokens（基准测试最佳）或 1000 characters（官方推荐）
2. **chunk_overlap**: 150-200 characters（15-20%）
3. **add_start_index**: True（便于溯源）
4. **keep_separator**: True（保留结构）

### 常见误区

1. ❌ 认为 RecursiveCharacterTextSplitter 比 CharacterTextSplitter 慢
   - 实际上性能差异很小，语义完整性提升显著

2. ❌ 认为所有文本都应该用相同的 chunk_size
   - 应该根据文档类型和 LLM 的 context window 调整

3. ❌ 认为 chunk_overlap 是浪费
   - chunk_overlap 是保持语义连续性的必要投资

---

## 下一步

理解了 RecursiveCharacterTextSplitter 后，建议学习：

1. **03_核心概念_4_TokenTextSplitter.md** - Token 级别分块
2. **07_实战代码_场景1_RAG文档分块管道.md** - 完整 RAG 流程
3. **07_实战代码_场景6_分块策略对比.md** - 性能对比

---

**数据来源**：
- [来源: reference/source_textsplitter_02_character.md | RecursiveCharacterTextSplitter 源码分析]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
- [来源: reference/search_textsplitter_01.md | 2025-2026 最佳实践和基准测试]
