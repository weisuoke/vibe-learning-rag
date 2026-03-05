---
type: source_code_analysis
source: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/character.py
analyzed_files:
  - sourcecode/langchain/libs/text-splitters/langchain_text_splitters/character.py
analyzed_at: 2026-02-25
knowledge_point: 04_TextSplitter文本分块
---

# 源码分析：CharacterTextSplitter 和 RecursiveCharacterTextSplitter

## 分析的文件
- `sourcecode/langchain/libs/text-splitters/langchain_text_splitters/character.py` - 字符级文本分块器

## 关键发现

### 1. CharacterTextSplitter 类

```python
class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""
```

**特点**：
- 基于单个分隔符的简单分块器
- 支持正则表达式分隔符
- 处理零宽度断言（lookaround）

**核心参数**：
```python
def __init__(
    self,
    separator: str = "\n\n",
    is_separator_regex: bool = False,
    **kwargs: Any,
) -> None:
```

- `separator`: 分隔符（默认双换行符）
- `is_separator_regex`: 是否将分隔符视为正则表达式

**split_text() 实现**：

```python
def split_text(self, text: str) -> list[str]:
    # 1. 确定分割模式：原始正则或转义字面量
    sep_pattern = (
        self._separator if self._is_separator_regex else re.escape(self._separator)
    )

    # 2. 初始分割（如果需要保留分隔符）
    splits = _split_text_with_regex(
        text, sep_pattern, keep_separator=self._keep_separator
    )

    # 3. 检测零宽度断言，避免重新插入
    lookaround_prefixes = ("(?=", "(?<!", "(?<=", "(?!")
    is_lookaround = self._is_separator_regex and any(
        self._separator.startswith(p) for p in lookaround_prefixes
    )

    # 4. 决定合并分隔符
    merge_sep = ""
    if not (self._keep_separator or is_lookaround):
        merge_sep = self._separator

    # 5. 合并相邻分割并返回
    return self._merge_splits(splits, merge_sep)
```

**算法步骤**：
1. 确定分割模式（正则或字面量）
2. 使用正则表达式分割文本
3. 检测零宽度断言（如 `(?=pattern)`）
4. 决定是否重新插入分隔符
5. 调用 `_merge_splits()` 合并小块

**零宽度断言处理**：
- 零宽度断言（lookaround）不消耗字符
- 如果分隔符是零宽度断言，不应重新插入
- 支持的断言：`(?=`, `(?<!`, `(?<=`, `(?!`

### 2. _split_text_with_regex() 辅助函数

```python
def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: bool | Literal["start", "end"]
) -> list[str]:
```

**功能**：使用正则表达式分割文本，可选保留分隔符

**实现逻辑**：

```python
if separator:
    if keep_separator:
        # 使用括号保留分隔符
        splits_ = re.split(f"({separator})", text)

        if keep_separator == "end":
            # 将分隔符附加到前一个块的末尾
            splits = [splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)]
        else:  # "start"
            # 将分隔符附加到下一个块的开头
            splits = [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]

        # 处理奇数个元素的情况
        if len(splits_) % 2 == 0:
            splits += splits_[-1:]
        splits = (
            [*splits, splits_[-1]] if keep_separator == "end"
            else [splits_[0], *splits]
        )
    else:
        splits = re.split(separator, text)
else:
    splits = list(text)  # 按字符分割

return [s for s in splits if s]  # 过滤空字符串
```

**关键点**：
- 使用 `re.split(f"({separator})", text)` 保留分隔符
- `keep_separator="end"` 将分隔符附加到前一个块
- `keep_separator="start"` 将分隔符附加到下一个块
- 处理边界情况（奇数个元素）

### 3. RecursiveCharacterTextSplitter 类

```python
class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """
```

**特点**：
- 递归尝试多个分隔符
- 优先使用更大的分隔符（如段落、句子、单词）
- 最常用的分块器

**核心参数**：
```python
def __init__(
    self,
    separators: list[str] | None = None,
    keep_separator: bool | Literal["start", "end"] = True,
    is_separator_regex: bool = False,
    **kwargs: Any,
) -> None:
    super().__init__(keep_separator=keep_separator, **kwargs)
    self._separators = separators or ["\n\n", "\n", " ", ""]
    self._is_separator_regex = is_separator_regex
```

**默认分隔符优先级**：
```python
["\n\n", "\n", " ", ""]
```
1. 双换行符（段落）
2. 单换行符（行）
3. 空格（单词）
4. 空字符串（字符）

**_split_text() 核心算法**：

```python
def _split_text(self, text: str, separators: list[str]) -> list[str]:
    final_chunks = []
    separator = separators[-1]  # 默认使用最后一个分隔符
    new_separators = []

    # 找到第一个存在于文本中的分隔符
    for i, s_ in enumerate(separators):
        separator_ = s_ if self._is_separator_regex else re.escape(s_)
        if not s_:
            separator = s_
            break
        if re.search(separator_, text):
            separator = s_
            new_separators = separators[i + 1 :]  # 剩余的分隔符
            break

    # 使用找到的分隔符分割文本
    separator_ = separator if self._is_separator_regex else re.escape(separator)
    splits = _split_text_with_regex(
        text, separator_, keep_separator=self._keep_separator
    )

    # 合并和递归处理
    good_splits = []
    separator_ = "" if self._keep_separator else separator

    for s in splits:
        if self._length_function(s) < self._chunk_size:
            good_splits.append(s)  # 小于 chunk_size，保留
        else:
            # 大于 chunk_size，需要进一步处理
            if good_splits:
                merged_text = self._merge_splits(good_splits, separator_)
                final_chunks.extend(merged_text)
                good_splits = []

            if not new_separators:
                final_chunks.append(s)  # 没有更多分隔符，直接添加
            else:
                # 递归使用下一个分隔符
                other_info = self._split_text(s, new_separators)
                final_chunks.extend(other_info)

    if good_splits:
        merged_text = self._merge_splits(good_splits, separator_)
        final_chunks.extend(merged_text)

    return final_chunks
```

**算法流程**：
1. 遍历分隔符列表，找到第一个存在于文本中的分隔符
2. 使用该分隔符分割文本
3. 对每个分割块：
   - 如果小于 `chunk_size`，保留
   - 如果大于 `chunk_size`：
     - 先合并已保留的小块
     - 如果还有剩余分隔符，递归处理
     - 否则直接添加（即使超过 chunk_size）
4. 最后合并所有保留的小块

**递归策略**：
- 优先使用更大的分隔符（段落 → 行 → 单词 → 字符）
- 只有当前分隔符无法满足 chunk_size 时，才使用下一个分隔符
- 递归深度最多为分隔符数量

### 4. from_language() 工厂方法

```python
@classmethod
def from_language(
    cls, language: Language, **kwargs: Any
) -> RecursiveCharacterTextSplitter:
    """Return an instance of this class based on a specific language."""
    separators = cls.get_separators_for_language(language)
    return cls(separators=separators, is_separator_regex=True, **kwargs)
```

**功能**：为特定编程语言创建分块器

**支持的语言**：
- C/C++, Go, Java, Kotlin
- JavaScript, TypeScript
- Python, Ruby, Rust, Scala
- PHP, Perl, Lua, Haskell
- HTML, Markdown, LaTeX
- 等 20+ 种语言

### 5. get_separators_for_language() 语言特定分隔符

**C/C++ 示例**：
```python
if language in {Language.C, Language.CPP}:
    return [
        # 类定义
        "\nclass ",
        # 函数定义
        "\nvoid ", "\nint ", "\nfloat ", "\ndouble ",
        # 控制流
        "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ",
        # 通用分隔符
        "\n\n", "\n", " ", "",
    ]
```

**Go 示例**：
```python
if language == Language.GO:
    return [
        "\nfunc ", "\nvar ", "\nconst ", "\ntype ",
        "\nif ", "\nfor ", "\nswitch ", "\ncase ",
        "\n\n", "\n", " ", "",
    ]
```

**JavaScript 示例**：
```python
if language == Language.JS:
    return [
        "\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ",
        "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ", "\ndefault ",
        "\n\n", "\n", " ", "",
    ]
```

**设计原则**：
1. 优先按语法结构分割（类、函数、方法）
2. 其次按控制流分割（if、for、while）
3. 最后按通用分隔符分割（段落、行、单词、字符）

### 6. 设计模式和最佳实践

#### 6.1 策略模式
- `CharacterTextSplitter` 和 `RecursiveCharacterTextSplitter` 实现不同的分割策略
- 通过 `separator` 和 `separators` 参数配置策略

#### 6.2 递归模式
- `RecursiveCharacterTextSplitter` 使用递归处理大块
- 递归深度由分隔符列表长度决定

#### 6.3 正则表达式灵活性
- 支持字面量分隔符和正则表达式分隔符
- 自动转义字面量分隔符（`re.escape()`）
- 处理零宽度断言

#### 6.4 分隔符保留策略
- `keep_separator=False`: 不保留分隔符
- `keep_separator="start"`: 分隔符附加到下一个块开头
- `keep_separator="end"`: 分隔符附加到前一个块末尾

### 7. 使用场景

#### CharacterTextSplitter
**适用场景**：
- 文本结构简单，单一分隔符即可
- 需要精确控制分隔符处理
- 性能要求高（无递归开销）

**示例**：
```python
# 按段落分割
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# 使用正则表达式
splitter = CharacterTextSplitter(
    separator=r"\n#{1,6}\s",  # Markdown 标题
    is_separator_regex=True,
    keep_separator="start"
)
```

#### RecursiveCharacterTextSplitter
**适用场景**：
- 文本结构复杂，需要多级分割
- 希望保留语义完整性
- 通用文本分块（最常用）

**示例**：
```python
# 通用文本分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 代码分块
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)
```

### 8. 性能考虑

#### CharacterTextSplitter
- **时间复杂度**: O(n)，其中 n 是文本长度
- **空间复杂度**: O(n)
- **优点**: 简单高效，无递归开销
- **缺点**: 灵活性较低

#### RecursiveCharacterTextSplitter
- **时间复杂度**: O(n * m)，其中 m 是分隔符数量
- **空间复杂度**: O(n * d)，其中 d 是递归深度
- **优点**: 灵活性高，语义完整性好
- **缺点**: 递归开销，性能略低

### 9. 常见问题和解决方案

#### 问题 1：分块大小不均匀
**原因**：分隔符分布不均
**解决**：
- 调整 `chunk_size` 和 `chunk_overlap`
- 使用更细粒度的分隔符

#### 问题 2：代码块被切断
**原因**：使用通用分隔符
**解决**：
- 使用 `from_language()` 创建语言特定分块器
- 自定义分隔符列表

#### 问题 3：分隔符丢失
**原因**：`keep_separator=False`
**解决**：
- 设置 `keep_separator="start"` 或 `"end"`
- 根据需求选择分隔符位置

#### 问题 4：正则表达式不生效
**原因**：未设置 `is_separator_regex=True`
**解决**：
- 显式设置 `is_separator_regex=True`
- 确保正则表达式语法正确

### 10. 与其他分块器的对比

| 特性 | CharacterTextSplitter | RecursiveCharacterTextSplitter |
|------|----------------------|-------------------------------|
| 分隔符数量 | 单个 | 多个（递归） |
| 语义完整性 | 低 | 高 |
| 性能 | 高 | 中 |
| 灵活性 | 低 | 高 |
| 适用场景 | 简单文本 | 复杂文本、代码 |
| 代码支持 | 无 | 有（20+ 语言） |

### 11. 最佳实践

#### 1. 选择合适的分块器
- 简单文本 → `CharacterTextSplitter`
- 复杂文本 → `RecursiveCharacterTextSplitter`
- 代码 → `RecursiveCharacterTextSplitter.from_language()`

#### 2. 调整参数
- `chunk_size`: 根据 LLM 的 context window 调整
- `chunk_overlap`: 通常设置为 chunk_size 的 10-20%
- `keep_separator`: 根据是否需要保留结构信息

#### 3. 自定义分隔符
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

#### 4. 处理特殊格式
```python
# Markdown 标题保留
splitter = RecursiveCharacterTextSplitter(
    separators=[r"\n#{1,6}\s", "\n\n", "\n", " ", ""],
    is_separator_regex=True,
    keep_separator="start"
)
```

### 12. 扩展点

#### 自定义语言支持
```python
# 添加新语言支持
def get_custom_language_separators(language: str) -> list[str]:
    if language == "my_language":
        return [
            "\nfunction ",
            "\nclass ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    return ["\n\n", "\n", " ", ""]

# 使用自定义分隔符
splitter = RecursiveCharacterTextSplitter(
    separators=get_custom_language_separators("my_language"),
    is_separator_regex=True
)
```

#### 自定义分割逻辑
```python
class MyCharacterTextSplitter(CharacterTextSplitter):
    def split_text(self, text: str) -> list[str]:
        # 自定义分割逻辑
        splits = super().split_text(text)
        # 后处理
        return [s.strip() for s in splits if len(s) > 10]
```
