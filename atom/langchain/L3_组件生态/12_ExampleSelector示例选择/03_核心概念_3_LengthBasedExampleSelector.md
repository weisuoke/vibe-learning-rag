# 核心概念 3：LengthBasedExampleSelector

## 概念定义

**LengthBasedExampleSelector 是基于长度限制选择 Few-shot 示例的选择器,通过控制示例总长度不超过指定阈值,确保 Prompt 不会超过模型的 Token 限制。**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

这是 LangChain 中最简单直接的示例选择器,特别适合需要严格控制 Token 成本和避免超过 context window 的场景。

---

## 为什么需要长度限制选择?

### 问题场景

在 Few-shot learning 中,我们面临一个实际挑战:

**如何确保示例不会导致 Prompt 超过模型的 Token 限制?**

```python
# 假设你有很多示例
examples = [
    {"input": "很长的问题...", "output": "很长的答案..."},
    {"input": "另一个很长的问题...", "output": "另一个很长的答案..."},
    # ... 更多示例
]

# 用户输入也很长
user_input = "一个非常非常长的问题..." * 100

# 问题：
# ❌ 如果选择所有示例,可能超过 context window
# ❌ 如果只选择前 N 个,可能仍然超限
# ✅ 动态计算长度,确保不超限!
```

### 解决方案

LengthBasedExampleSelector 通过以下步骤解决这个问题:

1. **计算用户输入的长度**
2. **计算剩余可用长度** (max_length - 输入长度)
3. **按顺序选择示例**,直到达到长度限制
4. **返回符合长度限制的示例**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

---

## 核心架构

### 类定义

```python
class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """基于长度限制选择示例

    Attributes:
        examples: list[dict] - 示例列表
        example_prompt: PromptTemplate - 示例格式化模板
        get_text_length: Callable[[str], int] - 长度计算函数
        max_length: int - 最大长度限制 (默认 2048)
        example_text_lengths: list[int] - 每个示例的长度 (自动计算)
    """
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 关键特性

1. **简单直接**:不依赖向量化或复杂计算
2. **Token 可控**:严格控制 Prompt 总长度
3. **自定义长度函数**:支持不同的长度计算方式
4. **按顺序选择**:保持示例的原始顺序

---

## max_length 参数

### 参数说明

`max_length` 是 LengthBasedExampleSelector 的核心参数,控制 Prompt 的最大长度。

```python
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2048  # 最大长度限制
)
```

### 长度单位

默认情况下,长度单位是**单词数**(按空格和换行符分割):

```python
def _get_length_based(text: str) -> int:
    """默认长度计算函数"""
    return len(re.split(r"\n| ", text))  # 按空格和换行符分割
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### max_length 设置建议

| 模型 | Context Window | 推荐 max_length | 说明 |
|------|----------------|-----------------|------|
| **GPT-4o-mini** | 128K tokens | 2000-4000 | 留出足够空间给输出 |
| **GPT-4** | 8K tokens | 1500-2000 | 保守设置 |
| **GPT-3.5-turbo** | 4K tokens | 800-1500 | 严格限制 |
| **Claude 3** | 200K tokens | 5000-10000 | 可以更宽松 |

[来源: reference/search_example_selector_02.md]

**计算公式:**

```
max_length = context_window * 0.6 - expected_output_length
```

- 60% 用于输入 (包括示例)
- 40% 留给输出

---

## select_examples() 方法

### 核心逻辑

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    """根据长度限制选择示例

    Args:
        input_variables: 输入变量字典

    Returns:
        符合长度限制的示例列表
    """
    # 1. 计算输入的长度
    inputs = " ".join(input_variables.values())
    remaining_length = self.max_length - self.get_text_length(inputs)

    # 2. 按顺序选择示例,直到达到长度限制
    i = 0
    examples = []
    while remaining_length > 0 and i < len(self.examples):
        new_length = remaining_length - self.example_text_lengths[i]
        if new_length < 0:
            break  # 超过限制,停止选择
        examples.append(self.examples[i])
        remaining_length = new_length
        i += 1

    return examples
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 选择策略

**按顺序选择**:

```
示例列表: [ex1, ex2, ex3, ex4, ex5]
长度:     [100, 150, 200, 120, 180]

max_length = 500
输入长度 = 50
剩余长度 = 450

选择过程:
1. ex1 (100) → 剩余 350 ✅
2. ex2 (150) → 剩余 200 ✅
3. ex3 (200) → 剩余 0 ✅
4. ex4 (120) → 剩余 -120 ❌ 停止

最终选择: [ex1, ex2, ex3]
```

---

## 自定义长度计算函数

### 默认长度计算

默认按**单词数**计算:

```python
def _get_length_based(text: str) -> int:
    """按单词数计算长度"""
    return len(re.split(r"\n| ", text))
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 自定义长度函数

#### 1. 按字符数计算

```python
def get_char_length(text: str) -> int:
    """按字符数计算长度"""
    return len(text)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_char_length,
    max_length=5000  # 5000 个字符
)
```

#### 2. 按 Token 数计算 (推荐)

```python
import tiktoken

def get_token_length(text: str) -> int:
    """按 Token 数计算长度 (OpenAI)"""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=2000  # 2000 tokens
)
```

[来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]

#### 3. 按句子数计算

```python
def get_sentence_length(text: str) -> int:
    """按句子数计算长度"""
    return len(re.split(r'[.!?]+', text))

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_sentence_length,
    max_length=20  # 20 个句子
)
```

---

## 与 FewShotPromptTemplate 集成

### 完整示例

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_openai import ChatOpenAI

# 1. 定义示例
examples = [
    {"query": "How are you?", "answer": "I can't complain but sometimes I still do."},
    {"query": "What time is it?", "answer": "It's time to get a watch."},
    {"query": "What's the meaning of life?", "answer": "42, of course!"},
    {"query": "How's the weather?", "answer": "It's raining cats and dogs!"},
    {"query": "What's for dinner?", "answer": "Whatever you're cooking!"},
]

# 2. 定义示例模板
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="User: {query}\nAI: {answer}"
)

# 3. 创建长度限制选择器
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  # 限制总长度为 100 个单词
)

# 4. 创建 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="You are a witty AI assistant:",
    suffix="User: {query}\nAI:",
    input_variables=["query"]
)

# 5. 测试不同长度的输入
short_input = "Hi"
long_input = "Tell me a very long story about " * 20

print("=== Short Input ===")
print(few_shot_prompt.format(query=short_input))
# 输出: 包含更多示例

print("\n=== Long Input ===")
print(few_shot_prompt.format(query=long_input))
# 输出: 包含更少示例 (因为输入占用了更多空间)
```

[来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]

---

## 实际应用场景

### 场景 1: Token 限制严格的模型

```python
# GPT-3.5-turbo 只有 4K context window
import tiktoken

def get_token_length(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=1500  # 留出空间给输出
)
```

### 场景 2: 成本敏感的应用

```python
# 控制 Token 成本
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=500  # 严格限制,降低成本
)
```

### 场景 3: 动态调整示例数量

```python
def dynamic_max_length(input_text: str, base_max: int = 2000) -> int:
    """根据输入长度动态调整 max_length"""
    input_length = get_token_length(input_text)

    if input_length > 1000:
        return base_max - 500  # 输入长,减少示例空间
    elif input_length < 100:
        return base_max + 500  # 输入短,增加示例空间
    else:
        return base_max

# 使用
user_input = "..."
max_len = dynamic_max_length(user_input)
selector.max_length = max_len
selected = selector.select_examples({"query": user_input})
```

### 场景 4: 批量处理优化

```python
# 批量处理时,为每个输入动态调整示例
def process_batch(inputs: list[str], selector: LengthBasedExampleSelector):
    results = []
    for input_text in inputs:
        # 根据输入长度调整 max_length
        input_len = get_token_length(input_text)
        selector.max_length = 2000 - input_len

        # 选择示例
        selected = selector.select_examples({"query": input_text})
        results.append(selected)

    return results
```

---

## 初始化与示例长度计算

### 自动计算示例长度

在初始化时,LengthBasedExampleSelector 会自动计算每个示例的长度:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # 自动计算每个示例的长度
    if not self.example_text_lengths:
        string_examples = [
            self.example_prompt.format(**eg) for eg in self.examples
        ]
        self.example_text_lengths = [
            self.get_text_length(eg) for eg in string_examples
        ]
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 手动指定示例长度

```python
# 如果已知示例长度,可以手动指定避免重复计算
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2000,
    example_text_lengths=[100, 150, 200, 120, 180]  # 手动指定
)
```

---

## 与 SemanticSimilarityExampleSelector 对比

| 特性 | LengthBasedExampleSelector | SemanticSimilarityExampleSelector |
|------|---------------------------|-----------------------------------|
| **选择依据** | 长度限制 | 语义相似度 |
| **选择顺序** | 按原始顺序 | 按相似度排序 |
| **依赖** | 无 (只需长度函数) | 需要 Embeddings + VectorStore |
| **计算成本** | 低 (简单计算) | 高 (需要向量化) |
| **相关性** | 不考虑相关性 | 高度相关 |
| **Token 控制** | 精确控制 | 需要额外处理 |
| **适用场景** | Token 限制严格 | 需要高相关性 |

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/]

---

## 混合策略: 语义相似度 + 长度限制

### 问题

单独使用 SemanticSimilarityExampleSelector 可能导致超过 Token 限制:

```python
# ❌ 可能超限
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=10  # 可能太多
)
```

### 解决方案 1: 先语义后长度

```python
# 1. 先用语义相似度选择候选
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=20  # 选择更多候选
)

# 2. 从候选中用长度限制筛选
candidates = semantic_selector.select_examples({"input": user_input})

length_selector = LengthBasedExampleSelector(
    examples=candidates,
    example_prompt=example_prompt,
    max_length=2000
)

final_examples = length_selector.select_examples({"input": user_input})
```

### 解决方案 2: 社区扩展 (TypeScript)

```typescript
// 使用社区扩展项目
import { SemanticLengthExampleSelector } from '@whitesmith/langchain-semantic-length-example-selector';

const selector = new SemanticLengthExampleSelector({
  vectorStore: vectorStore,
  k: 6,  // 语义相似度选择 6 个
  maxLength: 2000  // 长度限制
});
```

[来源: reference/fetch_example_selector_10.md | https://github.com/whitesmith/langchain-semantic-length-example-selector]

---

## 性能优化

### 1. 缓存示例长度

```python
# 避免重复计算示例长度
class CachedLengthSelector(LengthBasedExampleSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 示例长度已在初始化时计算并缓存
        # 不需要额外操作
```

### 2. 预计算 Token 数

```python
import tiktoken

# 预计算所有示例的 Token 数
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

example_token_lengths = [
    len(encoding.encode(example_prompt.format(**eg)))
    for eg in examples
]

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=lambda text: len(encoding.encode(text)),
    max_length=2000,
    example_text_lengths=example_token_lengths  # 使用预计算的长度
)
```

### 3. 批量处理优化

```python
# 批量处理时,复用选择器
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2000
)

# 批量选择
results = [
    selector.select_examples({"input": inp})
    for inp in batch_inputs
]
```

---

## 常见误区

### 误区 1: 长度单位混淆 ❌

**错误观点:** "max_length=2000 表示 2000 个 Token"

**为什么错?**
- 默认是**单词数**,不是 Token 数
- 单词数 ≠ Token 数

**正确理解:**

```python
# ❌ 错误: 以为是 Token 数
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2000  # 实际是 2000 个单词
)

# ✅ 正确: 明确使用 Token 计数
import tiktoken

def get_token_length(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=2000  # 2000 tokens
)
```

[来源: reference/search_example_selector_02.md]

### 误区 2: 不考虑输出长度 ❌

**错误观点:** "max_length 设置为 context_window 大小"

**为什么错?**
- 需要留出空间给模型输出
- 可能导致输出被截断

**正确理解:**

```python
# ❌ 错误
max_length = 4096  # GPT-3.5-turbo 的 context window

# ✅ 正确
context_window = 4096
expected_output = 500
max_length = context_window - expected_output  # 3596
```

### 误区 3: 示例顺序不重要 ❌

**错误观点:** "示例顺序无所谓"

**为什么错?**
- LengthBasedExampleSelector **按顺序选择**
- 前面的示例优先被选中

**正确理解:**

```python
# ❌ 错误: 随机顺序
examples = random.shuffle(examples)

# ✅ 正确: 按重要性排序
examples = sorted(examples, key=lambda x: x['importance'], reverse=True)

selector = LengthBasedExampleSelector(
    examples=examples,  # 重要的示例在前面
    example_prompt=example_prompt,
    max_length=2000
)
```

---

## 高级用法

### 1. 动态调整 max_length

```python
class DynamicLengthSelector(LengthBasedExampleSelector):
    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        # 根据输入长度动态调整 max_length
        input_text = " ".join(input_variables.values())
        input_length = self.get_text_length(input_text)

        # 动态调整
        if input_length > 1000:
            self.max_length = 1500
        elif input_length < 100:
            self.max_length = 3000
        else:
            self.max_length = 2000

        return super().select_examples(input_variables)
```

### 2. 优先级加权选择

```python
# 为示例添加优先级
examples_with_priority = [
    {"input": "...", "output": "...", "priority": 10},
    {"input": "...", "output": "...", "priority": 5},
    {"input": "...", "output": "...", "priority": 8},
]

# 按优先级排序
sorted_examples = sorted(
    examples_with_priority,
    key=lambda x: x['priority'],
    reverse=True
)

# 使用排序后的示例
selector = LengthBasedExampleSelector(
    examples=sorted_examples,
    example_prompt=example_prompt,
    max_length=2000
)
```

### 3. 分组选择

```python
# 确保每个类别至少有一个示例
def select_with_diversity(
    examples: list[dict],
    categories: list[str],
    max_length: int
) -> list[dict]:
    """确保每个类别至少有一个示例"""
    selected = []
    remaining_length = max_length

    # 1. 每个类别选一个
    for category in categories:
        category_examples = [e for e in examples if e['category'] == category]
        if category_examples:
            selected.append(category_examples[0])
            remaining_length -= get_token_length(
                example_prompt.format(**category_examples[0])
            )

    # 2. 剩余空间填充其他示例
    for example in examples:
        if example not in selected:
            example_length = get_token_length(example_prompt.format(**example))
            if remaining_length >= example_length:
                selected.append(example)
                remaining_length -= example_length

    return selected
```

---

## 与 Prompt 工程结合

### 1. 动态 Prefix/Suffix

```python
def create_dynamic_prompt(input_length: int):
    """根据输入长度调整 prefix/suffix"""
    if input_length > 500:
        # 输入长,简化指令
        prefix = "Answer briefly:"
        suffix = "Q: {query}\nA:"
    else:
        # 输入短,详细指令
        prefix = "You are a helpful assistant. Answer the following question based on the examples:"
        suffix = "Question: {query}\nAnswer:"

    return FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"]
    )
```

### 2. 条件示例选择

```python
def conditional_selector(input_text: str, task_type: str):
    """根据任务类型选择不同的示例集"""
    if task_type == "classification":
        examples = classification_examples
        max_length = 1500
    elif task_type == "generation":
        examples = generation_examples
        max_length = 2500
    else:
        examples = general_examples
        max_length = 2000

    return LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=max_length
    )
```

---

## 总结

LengthBasedExampleSelector 是 LangChain 中最简单但非常实用的示例选择器:

**核心优势:**
1. 简单直接,无需复杂依赖
2. 精确控制 Token 使用
3. 支持自定义长度计算函数
4. 性能开销低

**最佳实践:**
- 使用 Token 计数而非单词数
- 留出足够空间给输出
- 按重要性排序示例
- 结合语义相似度使用

**适用场景:**
- Token 限制严格的模型
- 成本敏感的应用
- 需要精确控制 Prompt 长度
- 批量处理优化

**局限性:**
- 不考虑示例相关性
- 按顺序选择,可能错过相关示例
- 需要与其他选择器结合使用

[来源: 综合多个参考资料]

---

**参考资料:**
- [来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]
- [来源: reference/search_example_selector_02.md]
- [来源: reference/fetch_example_selector_02.md | https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates]
- [来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]
- [来源: reference/fetch_example_selector_10.md | https://github.com/whitesmith/langchain-semantic-length-example-selector]
