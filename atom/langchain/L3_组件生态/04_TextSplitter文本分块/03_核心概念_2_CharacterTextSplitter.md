# 核心概念 2：CharacterTextSplitter

> 基于分隔符的简单文本分块器

---

## 概述

CharacterTextSplitter 是最简单的文本分块器，基于单个分隔符进行分块。适合结构简单、分隔符明确的文本。

**核心特点**：
- 基于单个分隔符分块
- 支持正则表达式分隔符
- 性能高效
- 配置简单

---

## 核心参数

```python
class CharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        separator: str = "\n\n",  # 分隔符（默认双换行）
        is_separator_regex: bool = False,  # 是否为正则表达式
        **kwargs  # 继承自 TextSplitter 的参数
    ):
```

### 1. separator（分隔符）

**作用**：指定分割文本的分隔符

**常用值**：
- `"\n\n"`：段落分隔符（默认）
- `"\n"`：行分隔符
- `". "`：句子分隔符
- 自定义字符串

### 2. is_separator_regex（正则表达式标志）

**作用**：指定 separator 是否为正则表达式

**示例**：
```python
# 字面量分隔符
splitter = CharacterTextSplitter(separator="\n\n")

# 正则表达式分隔符
splitter = CharacterTextSplitter(
    separator=r"\n#{1,6}\s",  # Markdown 标题
    is_separator_regex=True
)
```

---

## 分块算法

### 1. 基本流程

```python
def split_text(self, text: str) -> list[str]:
    # 1. 确定分割模式
    sep_pattern = (
        self._separator if self._is_separator_regex
        else re.escape(self._separator)
    )

    # 2. 使用正则表达式分割
    splits = _split_text_with_regex(
        text, sep_pattern, keep_separator=self._keep_separator
    )

    # 3. 处理零宽度断言
    is_lookaround = self._is_separator_regex and any(
        self._separator.startswith(p)
        for p in ("(?=", "(?<!", "(?<=", "(?!")
    )

    # 4. 决定合并分隔符
    merge_sep = "" if (self._keep_separator or is_lookaround) else self._separator

    # 5. 合并小块
    return self._merge_splits(splits, merge_sep)
```

### 2. 零宽度断言处理

**问题**：零宽度断言（如 `(?=pattern)`）不消耗字符

**解决**：检测零宽度断言，避免重新插入

```python
lookaround_prefixes = ("(?=", "(?<!", "(?<=", "(?!")
is_lookaround = any(self._separator.startswith(p) for p in lookaround_prefixes)
```

---

## 实战示例

### 示例 1：按段落分块

```python
from langchain_text_splitters import CharacterTextSplitter

text = """段落1内容...

段落2内容...

段落3内容..."""

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(text)
```

### 示例 2：按句子分块

```python
splitter = CharacterTextSplitter(
    separator=". ",
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
```

### 示例 3：使用正则表达式

```python
# Markdown 标题分块
splitter = CharacterTextSplitter(
    separator=r"\n#{1,6}\s",  # 匹配 # 到 ###### 标题
    is_separator_regex=True,
    keep_separator="start",
    chunk_size=1000
)
```

### 示例 4：保留分隔符

```python
# 分隔符附加到块末尾
splitter = CharacterTextSplitter(
    separator="\n",
    keep_separator="end",
    chunk_size=1000
)

# 分隔符附加到块开头
splitter = CharacterTextSplitter(
    separator="\n",
    keep_separator="start",
    chunk_size=1000
)
```

---

## 与 RecursiveCharacterTextSplitter 对比

| 特性 | CharacterTextSplitter | RecursiveCharacterTextSplitter |
|------|----------------------|-------------------------------|
| 分隔符数量 | 单个 | 多个（递归） |
| 语义完整性 | 低 | 高 |
| 性能 | 高 | 中 |
| 灵活性 | 低 | 高 |
| 适用场景 | 简单文本 | 复杂文本 |
| 配置复杂度 | 低 | 中 |

---

## 使用场景

### 适用场景

1. **结构简单的文本**：分隔符明确，无需递归
2. **性能要求高**：需要快速分块
3. **分隔符固定**：文本格式统一

### 不适用场景

1. **复杂文本**：需要多级分隔符
2. **语义完整性要求高**：需要保留更多上下文
3. **文本格式不统一**：分隔符不固定

---

## 常见问题

### Q1: 何时使用 CharacterTextSplitter？

**A**: 当文本结构简单、分隔符明确时使用。例如：
- 按段落分块的文章
- 按行分块的日志
- 格式统一的文档

### Q2: 如何处理多种分隔符？

**A**: 使用 RecursiveCharacterTextSplitter 或正则表达式：
```python
# 使用正则表达式匹配多种分隔符
splitter = CharacterTextSplitter(
    separator=r"(\n\n|\n|\.)",
    is_separator_regex=True
)
```

### Q3: 分隔符丢失怎么办？

**A**: 使用 `keep_separator` 参数：
```python
splitter = CharacterTextSplitter(
    separator="\n",
    keep_separator="start"  # 或 "end"
)
```

---

## 性能考虑

### 时间复杂度

- `split_text()`: O(n)，其中 n 是文本长度
- 比 RecursiveCharacterTextSplitter 快（无递归开销）

### 空间复杂度

- O(n)：需要存储所有块

### 优化建议

1. **使用字面量分隔符**：避免正则表达式开销
2. **合适的 chunk_size**：避免过小或过大
3. **批处理**：一次处理多个文档

---

## 下一步

理解了 CharacterTextSplitter 后，建议学习：

1. **03_核心概念_3_RecursiveCharacterTextSplitter.md** - 最常用的分块器
2. **03_核心概念_4_TokenTextSplitter.md** - Token 级别分块
3. **07_实战代码系列** - 实际应用场景

---

**数据来源**：
- [来源: reference/source_textsplitter_02_character.md | CharacterTextSplitter 源码分析]
- [来源: reference/context7_langchain_01.md | LangChain 官方文档]
