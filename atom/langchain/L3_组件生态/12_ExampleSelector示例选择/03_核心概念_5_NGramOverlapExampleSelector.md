# 核心概念 5：NGramOverlapExampleSelector

## 概念定义

**NGramOverlapExampleSelector 是基于 N-gram 重叠度选择 Few-shot 示例的轻量级选择器，通过计算输入与示例之间的 N-gram 重叠比例，选择文本相似度最高的示例，无需依赖向量化和 Embedding 模型。**

[来源: sourcecode/langchain/libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py]

这是 LangChain 中最轻量级的示例选择器，特别适合不需要语义理解、只关注文本表面相似度的场景。

---

## 为什么需要 N-gram 重叠选择？

### 问题场景

在 Few-shot learning 中，我们面临一个实际挑战：

**如何在不使用 Embedding 模型的情况下选择相似的示例？**

```python
# 场景：你没有 Embedding 模型或 API key
# 或者：你只需要简单的文本匹配

examples = [
    {"input": "How to install Python?", "output": "Use pip install..."},
    {"input": "How to setup Python?", "output": "Download from python.org..."},
    {"input": "How to debug code?", "output": "Use debugger..."},
]

# 用户问题
user_query = "How to install Python packages?"

# 问题：
# ❌ 使用 SemanticSimilarityExampleSelector？需要 Embedding 模型
# ❌ 使用 LengthBasedExampleSelector？不考虑相似度
# ✅ 使用 NGramOverlapExampleSelector？轻量级、无需 Embedding
```

### 解决方案

NGramOverlapExampleSelector 通过以下步骤解决这个问题：

1. **将文本分解为 N-gram**（如 bigram、trigram）
2. **计算 N-gram 重叠度**
3. **选择重叠度最高的示例**
4. **无需向量化或 Embedding**

[来源: sourcecode/langchain/libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py]

---

## 核心架构

### 类定义

```python
class NGramOverlapExampleSelector(BaseExampleSelector):
    """基于 N-gram 重叠度选择示例

    Attributes:
        examples: list[dict] - 示例列表
        example_prompt: PromptTemplate - 示例格式化模板
        threshold: float - 重叠度阈值（默认 -1.0，返回所有示例）
        ngram_size: int - N-gram 大小（默认 2，即 bigram）
    """
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py]

### 关键特性

1. **轻量级**：不依赖 Embedding 模型或向量存储
2. **快速**：纯文本匹配，计算速度快
3. **可配置**：支持自定义 N-gram 大小和阈值
4. **简单**：无需额外依赖或 API key

---

## N-gram 原理

### 什么是 N-gram？

**N-gram** 是文本中连续的 N 个词（或字符）的序列。

**示例：**

```python
text = "How to install Python"

# Unigram (N=1)
["How", "to", "install", "Python"]

# Bigram (N=2)
["How to", "to install", "install Python"]

# Trigram (N=3)
["How to install", "to install Python"]
```

### N-gram 重叠度计算

**公式：**

```
重叠度 = (共同 N-gram 数量) / (输入 N-gram 数量)
```

**示例：**

```python
# 输入
input_text = "How to install Python packages"
input_ngrams = ["How to", "to install", "install Python", "Python packages"]

# 示例
example_text = "How to install Python"
example_ngrams = ["How to", "to install", "install Python"]

# 共同 N-gram
common_ngrams = ["How to", "to install", "install Python"]  # 3个

# 重叠度
overlap = 3 / 4 = 0.75
```

[来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]

---

## select_examples() 方法

### 核心逻辑

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    """基于 N-gram 重叠度选择示例

    Args:
        input_variables: 输入变量字典

    Returns:
        选择的示例列表
    """
    # 1. 提取输入文本
    input_text = " ".join(input_variables.values())

    # 2. 生成输入的 N-gram
    input_ngrams = self._get_ngrams(input_text)

    # 3. 计算每个示例的重叠度
    overlaps = []
    for example in self.examples:
        example_text = self._example_to_text(example)
        example_ngrams = self._get_ngrams(example_text)
        overlap = self._calculate_overlap(input_ngrams, example_ngrams)
        overlaps.append((example, overlap))

    # 4. 过滤和排序
    filtered = [(ex, ov) for ex, ov in overlaps if ov >= self.threshold]
    sorted_examples = sorted(filtered, key=lambda x: x[1], reverse=True)

    # 5. 返回示例
    return [ex for ex, _ in sorted_examples]
```

[来源: sourcecode/langchain/libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py]

### 选择策略

**N-gram 重叠选择过程：**

```
1. 输入文本 → N-gram 分解
   "How to install Python packages"
   → ["How to", "to install", "install Python", "Python packages"]

2. 示例文本 → N-gram 分解
   Example 1: "How to install Python"
   → ["How to", "to install", "install Python"]

   Example 2: "How to debug code"
   → ["How to", "to debug", "debug code"]

3. 计算重叠度
   Example 1: 3/4 = 0.75
   Example 2: 1/4 = 0.25

4. 排序并返回
   [Example 1, Example 2]
```

---

## 实战代码示例

### 基础使用

```python
from langchain.prompts.example_selector import NGramOverlapExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 示例数据
examples = [
    {"input": "How to install Python?", "output": "Use pip install..."},
    {"input": "How to setup Python environment?", "output": "Create virtual env..."},
    {"input": "How to debug Python code?", "output": "Use pdb module..."},
    {"input": "How to test Python code?", "output": "Use pytest..."},
]

# 示例模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

# 创建 N-gram 选择器
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=-1.0,  # 返回所有示例（按重叠度排序）
    ngram_size=2     # 使用 bigram
)

# 选择示例
selected = selector.select_examples({"input": "How to install Python packages?"})

print(f"选择了 {len(selected)} 个示例：")
for ex in selected:
    print(f"  - {ex['input']}")

# 输出：
# 选择了 4 个示例：
#   - How to install Python?          # 重叠度最高
#   - How to setup Python environment? # 重叠度次高
#   - How to debug Python code?        # 重叠度较低
#   - How to test Python code?         # 重叠度最低
```

[来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]

### 使用阈值过滤

```python
# 创建 N-gram 选择器（带阈值）
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.5,   # 只返回重叠度 >= 0.5 的示例
    ngram_size=2
)

# 选择示例
selected = selector.select_examples({"input": "How to install Python packages?"})

print(f"选择了 {len(selected)} 个示例（重叠度 >= 0.5）：")
for ex in selected:
    print(f"  - {ex['input']}")

# 输出：
# 选择了 2 个示例（重叠度 >= 0.5）：
#   - How to install Python?          # 重叠度 0.75
#   - How to setup Python environment? # 重叠度 0.6
```

### 与 FewShotPromptTemplate 集成

```python
# 创建 Few-shot Prompt
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,  # 使用 N-gram 选择器
    example_prompt=example_prompt,
    prefix="Answer the question based on examples:",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# 格式化 Prompt
prompt = few_shot_prompt.format(input="How to install Python packages?")
print(prompt)
```

---

## 与其他选择器对比

### 核心区别

| 特性 | NGramOverlapExampleSelector | SemanticSimilarityExampleSelector | LengthBasedExampleSelector |
|------|----------------------------|-----------------------------------|----------------------------|
| **依赖** | 无 | Embedding 模型 + VectorStore | 无 |
| **相似度类型** | 文本表面相似度 | 语义相似度 | 不考虑相似度 |
| **计算速度** | 快 | 较慢（需要向量化） | 最快 |
| **准确性** | 中等 | 高 | 低 |
| **适用场景** | 轻量级、无 API key | 需要语义理解 | Token 限制 |

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/]

### 对比示例

```python
# 场景：用户问 "How to install Python packages?"

# NGramOverlapExampleSelector 可能选择：
# 1. "How to install Python?" (文本重叠度高)
# 2. "How to setup Python environment?" (包含 "Python")
# 优势：无需 Embedding，快速
# 劣势：无法理解语义（如 "setup" 和 "install" 的语义相似性）

# SemanticSimilarityExampleSelector 可能选择：
# 1. "How to install Python?" (语义相关)
# 2. "How to setup Python environment?" (语义相关)
# 优势：理解语义，更准确
# 劣势：需要 Embedding 模型，较慢

# LengthBasedExampleSelector 可能选择：
# 1. 前 N 个示例（不考虑相似度）
# 优势：最快
# 劣势：不考虑相关性
```

---

## 实际应用场景

### 1. 无 API key 场景

```python
# 场景：没有 OpenAI API key，无法使用 Embedding
examples = [
    {"q": "What is Python?", "a": "A programming language"},
    {"q": "What is JavaScript?", "a": "A web language"},
    {"q": "What is Java?", "a": "An OOP language"},
]

selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.3,
    ngram_size=2
)

# 用户问题
query = "What is Python programming?"

# N-gram 选择：
# 1. "What is Python?" (重叠度高)
# 2. "What is JavaScript?" (重叠度中等)
```

[来源: reference/fetch_example_selector_04.md | https://www.sandgarden.com/learn/few-shot-prompting]

### 2. 快速原型开发

```python
# 场景：快速开发原型，不想配置 Embedding
examples = [
    {"task": "sort list", "code": "sorted(list)"},
    {"task": "filter list", "code": "filter(lambda x: x > 0, list)"},
    {"task": "map list", "code": "map(lambda x: x * 2, list)"},
]

selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=-1.0,
    ngram_size=2
)

# 用户任务
task = "sort array"

# N-gram 选择：
# 1. "sort list" (重叠度高)
# 2. "filter list" (包含 "list")
```

### 3. 文本模板匹配

```python
# 场景：模板匹配场景，只需要文本相似度
examples = [
    {"template": "Hello {name}, welcome!", "type": "greeting"},
    {"template": "Goodbye {name}, see you!", "type": "farewell"},
    {"template": "Hi {name}, how are you?", "type": "greeting"},
]

selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.4,
    ngram_size=2
)

# 用户输入
user_input = "Hello John, welcome to our site!"

# N-gram 选择：
# 1. "Hello {name}, welcome!" (重叠度高)
# 2. "Hi {name}, how are you?" (包含 "Hello" 的变体)
```

---

## 最佳实践

### 1. ngram_size 参数调优

```python
# 经验法则
ngram_size = 2  # Bigram（默认，适合大多数场景）
ngram_size = 3  # Trigram（更精确，但可能过于严格）
ngram_size = 1  # Unigram（更宽松，但可能不够精确）

# 示例
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.5,
    ngram_size=2  # 推荐使用 bigram
)
```

[来源: reference/search_example_selector_01.md]

### 2. threshold 参数设置

```python
# 阈值设置建议
threshold = -1.0  # 返回所有示例（按重叠度排序）
threshold = 0.3   # 宽松（返回大部分相关示例）
threshold = 0.5   # 标准（返回中等相关示例）
threshold = 0.7   # 严格（只返回高度相关示例）

# 示例
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.5,  # 标准配置
    ngram_size=2
)
```

### 3. 示例库设计

```python
# ✅ 好的示例库：文本多样化
examples = [
    {"input": "How to install Python?", "output": "..."},
    {"input": "How to debug code?", "output": "..."},
    {"input": "What is Python?", "output": "..."},
]

# ❌ 不好的示例库：文本过于相似
examples = [
    {"input": "How to install Python?", "output": "..."},
    {"input": "How to install Python packages?", "output": "..."},
    {"input": "How to install Python modules?", "output": "..."},
]
```

---

## 常见误区

### 误区 1：N-gram 选择器可以理解语义 ❌

**为什么错？**
- N-gram 只计算文本表面相似度
- 无法理解同义词或近义词
- 例如："install" 和 "setup" 在语义上相似，但 N-gram 无法识别

**正确理解：**
```python
# ❌ 错误期望
# 输入："How to setup Python?"
# 期望选择："How to install Python?" (语义相似)
# 实际选择：可能选择其他包含 "setup" 的示例

# ✅ 正确理解
# N-gram 只匹配文本表面，不理解语义
# 如果需要语义理解，使用 SemanticSimilarityExampleSelector
```

[来源: reference/search_example_selector_02.md]

### 误区 2：N-gram 选择器总是比语义相似度快 ❌

**为什么错？**
- 对于小规模示例库，差异不明显
- 对于大规模示例库，N-gram 计算也需要时间
- 语义相似度选择器可以使用向量索引加速

**正确理解：**
```python
# 小规模示例库（< 100 个）
# N-gram 和语义相似度速度差异不大

# 大规模示例库（> 1000 个）
# N-gram 可能更快，但语义相似度可以使用 FAISS 等索引加速
```

### 误区 3：threshold 越高越好 ❌

**为什么错？**
- threshold 过高可能导致没有示例被选中
- threshold 过低可能返回不相关的示例
- 需要根据实际场景调整

**正确理解：**
```python
# ❌ 不推荐
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.9,  # 过高，可能没有示例
    ngram_size=2
)

# ✅ 推荐
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=0.5,  # 合理
    ngram_size=2
)
```

---

## 双重类比

### 前端开发类比

**N-gram 选择器 = 字符串模糊匹配**

```javascript
// 前端模糊搜索
const searchResults = [
  { title: "How to install Python", score: 0.75 },
  { title: "How to setup Python", score: 0.6 },
  { title: "How to debug code", score: 0.25 },
];

// 基于文本相似度排序（类似 N-gram）
const sortedResults = searchResults.sort((a, b) => b.score - a.score);

// 类似于 N-gram 的 threshold
const filteredResults = sortedResults.filter(r => r.score >= 0.5);
```

### 日常生活类比

**N-gram 选择器 = 图书馆关键词搜索**

```
场景：在图书馆搜索书籍

语义相似度选择（SemanticSimilarityExampleSelector）：
- 搜索："编程入门"
- 结果：《Python 入门》、《编程基础》、《代码学习》
- 特点：理解语义，找到相关书籍

N-gram 选择（NGramOverlapExampleSelector）：
- 搜索："编程入门"
- 结果：《编程入门指南》、《入门编程》、《编程》
- 特点：匹配关键词，快速但不理解语义

长度选择（LengthBasedExampleSelector）：
- 搜索："编程入门"
- 结果：前 N 本书（不考虑相关性）
- 特点：最快，但不考虑相关性
```

---

## 性能优化

### 1. 缓存 N-gram

```python
# 预计算示例的 N-gram
class CachedNGramSelector(NGramOverlapExampleSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 预计算所有示例的 N-gram
        self.example_ngrams = [
            self._get_ngrams(self._example_to_text(ex))
            for ex in self.examples
        ]

    def select_examples(self, input_variables):
        input_text = " ".join(input_variables.values())
        input_ngrams = self._get_ngrams(input_text)

        # 使用缓存的 N-gram
        overlaps = [
            (ex, self._calculate_overlap(input_ngrams, ex_ngrams))
            for ex, ex_ngrams in zip(self.examples, self.example_ngrams)
        ]

        # 过滤和排序
        filtered = [(ex, ov) for ex, ov in overlaps if ov >= self.threshold]
        sorted_examples = sorted(filtered, key=lambda x: x[1], reverse=True)

        return [ex for ex, _ in sorted_examples]
```

### 2. 并行计算

```python
from concurrent.futures import ThreadPoolExecutor

def calculate_overlap_parallel(examples, input_ngrams, threshold):
    """并行计算重叠度"""
    with ThreadPoolExecutor() as executor:
        overlaps = list(executor.map(
            lambda ex: calculate_single_overlap(ex, input_ngrams),
            examples
        ))

    # 过滤和排序
    filtered = [(ex, ov) for ex, ov in zip(examples, overlaps) if ov >= threshold]
    sorted_examples = sorted(filtered, key=lambda x: x[1], reverse=True)

    return [ex for ex, _ in sorted_examples]
```

---

## 总结

NGramOverlapExampleSelector 是 LangChain 中最轻量级的示例选择器：

**核心特点：**
1. **轻量级**：不依赖 Embedding 模型或向量存储
2. **快速**：纯文本匹配，计算速度快
3. **简单**：无需额外依赖或 API key
4. **文本相似度**：基于 N-gram 重叠度选择示例

**适用场景：**
- 无 API key 或 Embedding 模型的场景
- 快速原型开发
- 文本模板匹配
- 只需要文本表面相似度的场景

**关键参数：**
- `ngram_size`：N-gram 大小（默认 2，即 bigram）
- `threshold`：重叠度阈值（默认 -1.0，返回所有示例）

**与其他选择器的关系：**
- 比 SemanticSimilarityExampleSelector 更轻量，但不理解语义
- 比 LengthBasedExampleSelector 更智能，考虑文本相似度
- 适合作为 SemanticSimilarityExampleSelector 的轻量级替代方案

**何时使用：**
- ✅ 没有 Embedding 模型或 API key
- ✅ 需要快速原型开发
- ✅ 只需要文本表面相似度
- ❌ 需要语义理解（使用 SemanticSimilarityExampleSelector）
- ❌ 需要多样性（使用 MaxMarginalRelevanceExampleSelector）

[来源: sourcecode/langchain/libs/langchain/langchain_classic/prompts/example_selector/ngram_overlap.py | reference/fetch_example_selector_06.md | reference/search_example_selector_01.md]
