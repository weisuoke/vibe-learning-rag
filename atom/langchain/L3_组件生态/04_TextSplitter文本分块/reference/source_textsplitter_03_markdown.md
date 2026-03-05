---
type: source_code_analysis
source: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/markdown.py
analyzed_files:
  - sourcecode/langchain/libs/text-splitters/langchain_text_splitters/markdown.py
analyzed_at: 2026-02-25
knowledge_point: 04_TextSplitter文本分块
---

# 源码分析：Markdown 文本分块器

## 分析的文件
- `sourcecode/langchain/libs/text-splitters/langchain_text_splitters/markdown.py` - Markdown 专用分块器

## 关键发现

### 1. MarkdownTextSplitter 类

```python
class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""
```

**特点**：
- 继承自 `RecursiveCharacterTextSplitter`
- 使用 Markdown 特定的分隔符
- 简单的包装类

**实现**：
```python
def __init__(self, **kwargs: Any) -> None:
    separators = self.get_separators_for_language(Language.MARKDOWN)
    super().__init__(separators=separators, **kwargs)
```

**分隔符**：使用 `Language.MARKDOWN` 的预定义分隔符

### 2. MarkdownHeaderTextSplitter 类

```python
class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""
```

**特点**：
- 不继承自 `TextSplitter`（独立实现）
- 基于 Markdown 标题层级分块
- 保留标题层级信息到元数据

**核心参数**：
```python
def __init__(
    self,
    headers_to_split_on: list[tuple[str, str]],
    return_each_line: bool = False,
    strip_headers: bool = True,
    custom_header_patterns: dict[str, int] | None = None,
) -> None:
```

**参数说明**：
- `headers_to_split_on`: 要跟踪的标题列表，如 `[("#", "Header 1"), ("##", "Header 2")]`
- `return_each_line`: 是否逐行返回（带关联标题）
- `strip_headers`: 是否从块内容中去除标题
- `custom_header_patterns`: 自定义标题模式，如 `{"**": 1, "***": 2}`

### 3. 核心方法

#### 3.1 split_text()

```python
def split_text(self, text: str) -> list[Document]:
    """Split markdown file."""
    lines = text.split("\n")
    lines_with_metadata: list[LineType] = []
    current_content: list[str] = []
    current_metadata: dict[str, str] = {}
    header_stack: list[HeaderType] = []
    in_code_block = False
    opening_fence = ""
```

**算法流程**：
1. 按行分割文本
2. 跟踪当前标题栈（header_stack）
3. 检测代码块（避免在代码块中识别标题）
4. 识别标题行并更新元数据
5. 聚合内容到块中

#### 3.2 代码块检测

```python
if not in_code_block:
    if stripped_line.startswith("```") and stripped_line.count("```") == 1:
        in_code_block = True
        opening_fence = "```"
    elif stripped_line.startswith("~~~"):
        in_code_block = True
        opening_fence = "~~~"
elif stripped_line.startswith(opening_fence):
    in_code_block = False
    opening_fence = ""
```

**功能**：
- 检测代码块开始（``` 或 ~~~）
- 在代码块内不识别标题
- 检测代码块结束

#### 3.3 标题识别

```python
for sep, name in self.headers_to_split_on:
    is_standard_header = stripped_line.startswith(sep) and (
        len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
    )
    is_custom_header = self._is_custom_header(stripped_line, sep)

    if is_standard_header or is_custom_header:
        # 处理标题
```

**标准标题**：
- 以 `#`, `##`, `###` 等开头
- 标题后必须是空格或行尾

**自定义标题**：
- 支持自定义模式，如 `**Header**`
- 通过 `_is_custom_header()` 方法检测

#### 3.4 _is_custom_header()

```python
def _is_custom_header(self, line: str, sep: str) -> bool:
    if sep not in self.custom_header_patterns:
        return False

    escaped_sep = re.escape(sep)
    pattern = f"^{escaped_sep}(?!{escaped_sep})(.+?)(?<!{escaped_sep}){escaped_sep}$"

    match = re.match(pattern, line)
    if match:
        content = match.group(1).strip()
        if content and not all(c in sep for c in content.replace(" ", "")):
            return True
    return False
```

**功能**：
- 检测自定义标题模式
- 使用正则表达式匹配
- 确保内容不只是分隔符字符

#### 3.5 aggregate_lines_to_chunks()

```python
def aggregate_lines_to_chunks(self, lines: list[LineType]) -> list[Document]:
    aggregated_chunks: list[LineType] = []

    for line in lines:
        if (
            aggregated_chunks
            and aggregated_chunks[-1]["metadata"] == line["metadata"]
        ):
            # 相同元数据，合并内容
            aggregated_chunks[-1]["content"] += "  \n" + line["content"]
        elif (...):
            # 不同元数据但层级更深，合并并更新元数据
            aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            aggregated_chunks[-1]["metadata"] = line["metadata"]
        else:
            # 新块
            aggregated_chunks.append(line)

    return [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in aggregated_chunks
    ]
```

**功能**：
- 将具有相同元数据的行合并为块
- 处理标题层级变化
- 返回 Document 对象列表

### 4. 标题栈管理

**标题栈**：跟踪当前的标题层级

```python
header_stack: list[HeaderType] = []
```

**更新逻辑**：
1. 遇到新标题时，弹出所有更深层级的标题
2. 将新标题推入栈
3. 从栈构建当前元数据

**示例**：
```
# H1
## H2
### H3
## H2'
```

- 遇到 `# H1`：栈 = [H1]
- 遇到 `## H2`：栈 = [H1, H2]
- 遇到 `### H3`：栈 = [H1, H2, H3]
- 遇到 `## H2'`：弹出 H3，栈 = [H1, H2']

### 5. 元数据结构

**元数据格式**：
```python
{
    "Header 1": "Introduction",
    "Header 2": "Background",
    "Header 3": "Details"
}
```

**特点**：
- 键是标题名称（来自 `headers_to_split_on`）
- 值是标题文本内容
- 反映标题层级关系

### 6. 使用场景

#### 场景 1：技术文档分块

```python
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # 保留标题
)

docs = splitter.split_text(markdown_text)
```

#### 场景 2：博客文章分块

```python
headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=True  # 逐行返回
)
```

#### 场景 3：自定义标题模式

```python
headers_to_split_on = [
    ("**", "Bold Header"),
    ("***", "Bold Italic Header"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    custom_header_patterns={"**": 1, "***": 2}
)
```

### 7. 设计特点

#### 7.1 与 TextSplitter 的区别

| 特性 | MarkdownHeaderTextSplitter | TextSplitter |
|------|---------------------------|--------------|
| 继承关系 | 独立类 | 抽象基类 |
| 分块依据 | 标题层级 | 分隔符/token |
| 元数据 | 标题层级信息 | 可选 start_index |
| 输出 | Document 对象 | 文本列表或 Document |

#### 7.2 优势

- **语义完整性**：按标题分块保留文档结构
- **元数据丰富**：包含完整的标题层级信息
- **代码块感知**：不会在代码块中错误识别标题
- **灵活性**：支持自定义标题模式

#### 7.3 限制

- **不支持 chunk_size**：无法控制块大小
- **不支持 chunk_overlap**：块之间无重叠
- **依赖标题**：如果文档没有标题，效果不佳

### 8. 最佳实践

#### 8.1 与 RecursiveCharacterTextSplitter 组合

```python
# 第一步：按标题分块
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)
header_docs = header_splitter.split_text(markdown_text)

# 第二步：进一步分块（如果块太大）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_docs = text_splitter.split_documents(header_docs)
```

#### 8.2 保留标题信息

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")],
    strip_headers=False  # 保留标题在内容中
)
```

#### 8.3 处理嵌套标题

```python
# 支持多层级标题
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
]
```

### 9. 常见问题

#### 问题 1：代码块中的 # 被识别为标题

**原因**：未正确检测代码块

**解决**：源码已处理，通过 `in_code_block` 标志避免

#### 问题 2：块太大

**原因**：`MarkdownHeaderTextSplitter` 不支持 chunk_size

**解决**：与 `RecursiveCharacterTextSplitter` 组合使用

#### 问题 3：元数据丢失

**原因**：进一步分块时元数据未传递

**解决**：使用 `split_documents()` 而非 `split_text()`

### 10. 扩展点

#### 自定义标题识别

```python
class MyMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter):
    def _is_custom_header(self, line: str, sep: str) -> bool:
        # 自定义标题识别逻辑
        return custom_logic(line, sep)
```

#### 自定义元数据处理

```python
# 在 split_text() 后添加自定义元数据
docs = splitter.split_text(text)
for doc in docs:
    doc.metadata["custom_field"] = compute_value(doc)
```
