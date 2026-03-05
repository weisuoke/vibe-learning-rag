# 核心概念 4：MaxMarginalRelevanceExampleSelector

## 概念定义

**MaxMarginalRelevanceExampleSelector 是基于最大边际相关性（MMR）算法选择 Few-shot 示例的选择器，通过平衡相关性和多样性，避免选择过于相似的示例，提高 Few-shot learning 的效果。**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

这是 LangChain 中用于提高示例多样性的高级选择器，特别适合需要避免示例冗余、提高 Prompt 质量的场景。

---

## 为什么需要 MMR 选择？

### 问题场景

在 Few-shot learning 中，我们面临一个关键挑战：

**如何避免选择过于相似的示例？**

```python
# 假设你有很多相似的示例
examples = [
    {"input": "What is Python?", "output": "A programming language"},
    {"input": "What is Python programming?", "output": "A programming language"},
    {"input": "Explain Python", "output": "A programming language"},
    {"input": "What is JavaScript?", "output": "A programming language"},
    # ... 更多示例
]

# 用户问题
user_query = "What is Python?"

# 问题：使用语义相似度选择
# ❌ 可能选择：前3个示例（都是关于 Python 的，过于相似）
# ✅ 理想选择：1个 Python + 1个 JavaScript + 1个其他（多样性更高）
```

### 解决方案

MaxMarginalRelevanceExampleSelector 通过 MMR 算法解决这个问题：

1. **先获取更多候选**（fetch_k 个）
2. **计算相关性和多样性**
3. **平衡选择**（返回 k 个）
4. **避免冗余**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

---

## 核心架构

### 类定义

```python
class MaxMarginalRelevanceExampleSelector(_VectorStoreExampleSelector):
    """基于最大边际相关性选择示例

    论文：https://arxiv.org/pdf/2211.13892.pdf

    Attributes:
        vectorstore: VectorStore - 向量存储后端
        k: int - 最终返回的示例数量（默认 4）
        fetch_k: int - 先获取的候选数量（默认 20）
        example_keys: list[str] | None - 过滤示例的键
        input_keys: list[str] | None - 过滤输入的键
        vectorstore_kwargs: dict[str, Any] | None - 向量存储额外参数
    """
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 关键特性

1. **MMR 算法**：平衡相关性和多样性
2. **两阶段选择**：先获取 fetch_k 个候选，再选择 k 个
3. **VectorStore 集成**：支持多种向量数据库
4. **异步支持**：提供 `aselect_examples()` 异步方法

---

## MMR 算法原理

### 什么是 MMR？

**Maximum Marginal Relevance（最大边际相关性）** 是一种平衡相关性和多样性的算法。

**核心思想：**
- 选择与查询相关的示例
- 同时避免选择彼此相似的示例
- 最大化边际相关性（新增示例的价值）

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

### MMR 公式

```
MMR = λ * Sim(D, Q) - (1-λ) * max Sim(D, Di)
```

其中：
- `Sim(D, Q)`：文档 D 与查询 Q 的相似度（相关性）
- `max Sim(D, Di)`：文档 D 与已选文档的最大相似度（多样性惩罚）
- `λ`：平衡参数（0-1 之间，通常为 0.5）

**解释：**
- λ = 1：只考虑相关性（退化为语义相似度选择）
- λ = 0：只考虑多样性（可能选择不相关的示例）
- λ = 0.5：平衡相关性和多样性

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## fetch_k 参数

### 参数说明

`fetch_k` 是 MMR 选择器的核心参数，控制第一阶段获取的候选数量。

```python
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=4,        # 最终返回 4 个示例
    fetch_k=20  # 先获取 20 个候选
)
```

### 两阶段选择流程

```
阶段1：获取候选
示例库（100个）→ 语义相似度排序 → Top-20 候选

阶段2：MMR 选择
Top-20 候选 → MMR 算法 → Top-4 最终示例
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### fetch_k 设置建议

| 场景 | k | fetch_k | 说明 |
|------|---|---------|------|
| **高多样性** | 3 | 15-20 | 从更多候选中选择 |
| **平衡** | 4 | 12-16 | 标准配置 |
| **高相关性** | 5 | 8-10 | 候选较少，更关注相关性 |

**经验法则：**
```
fetch_k = k * 3 到 k * 5
```

[来源: reference/search_example_selector_02.md]

---

## select_examples() 方法

### 核心逻辑

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    """使用 MMR 算法选择示例

    Args:
        input_variables: 输入变量字典

    Returns:
        选择的示例列表
    """
    # 使用 MMR 算法选择示例
    example_docs = self.vectorstore.max_marginal_relevance_search(
        self._example_to_text(input_variables, self.input_keys),
        k=self.k,
        fetch_k=self.fetch_k,  # 先获取更多候选
    )
    return self._documents_to_examples(example_docs)
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 选择策略

**MMR 选择过程：**

```
1. 获取 fetch_k 个最相似的候选
   示例库 → 语义相似度排序 → Top-20

2. 初始化已选集合
   selected = []

3. 迭代选择 k 个示例
   for i in range(k):
       计算每个候选的 MMR 分数
       选择 MMR 分数最高的候选
       添加到 selected
       从候选中移除

4. 返回 selected
```

---

## 实战代码示例

### 基础使用

```python
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 示例数据
examples = [
    {"input": "What is Python?", "output": "A programming language"},
    {"input": "What is Python programming?", "output": "A high-level language"},
    {"input": "Explain Python", "output": "An interpreted language"},
    {"input": "What is JavaScript?", "output": "A web programming language"},
    {"input": "What is Java?", "output": "An object-oriented language"},
    {"input": "What is C++?", "output": "A compiled language"},
]

# 创建 MMR 选择器
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3,        # 返回 3 个示例
    fetch_k=6   # 先获取 6 个候选
)

# 选择示例
selected = selector.select_examples({"input": "What is Python?"})

print(f"选择了 {len(selected)} 个示例：")
for ex in selected:
    print(f"  - {ex['input']}")

# 输出：
# 选择了 3 个示例：
#   - What is Python?          # 最相关
#   - What is JavaScript?      # 多样性（不同语言）
#   - What is Java?            # 多样性（不同语言）
```

[来源: reference/fetch_example_selector_01.md | https://medium.com/donato-story/exploring-few-shot-prompts-with-langchain-852f27ea4e1d]

### 与 FewShotPromptTemplate 集成

```python
# 示例模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

# 创建 Few-shot Prompt
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,  # 使用 MMR 选择器
    example_prompt=example_prompt,
    prefix="Answer the question based on examples:",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# 格式化 Prompt
prompt = few_shot_prompt.format(input="What is Python?")
print(prompt)
```

[来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]

---

## 与 SemanticSimilarityExampleSelector 对比

### 核心区别

| 特性 | SemanticSimilarityExampleSelector | MaxMarginalRelevanceExampleSelector |
|------|-----------------------------------|-------------------------------------|
| **选择策略** | 只考虑相关性 | 平衡相关性和多样性 |
| **参数** | k | k + fetch_k |
| **算法** | 余弦相似度 | MMR 算法 |
| **示例多样性** | 可能重复 | 避免重复 |
| **适用场景** | 需要高度相关的示例 | 需要多样化的示例 |

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py]

### 对比示例

```python
# 场景：用户问 "What is Python?"

# SemanticSimilarityExampleSelector 可能选择：
# 1. "What is Python?" (相似度 0.99)
# 2. "What is Python programming?" (相似度 0.95)
# 3. "Explain Python" (相似度 0.92)
# 问题：3个示例都是关于 Python 的，过于相似

# MaxMarginalRelevanceExampleSelector 可能选择：
# 1. "What is Python?" (相关性高)
# 2. "What is JavaScript?" (多样性：不同语言)
# 3. "What is Java?" (多样性：不同语言)
# 优势：既相关又多样
```

---

## 实际应用场景

### 1. 问答系统

```python
# 场景：技术问答系统
examples = [
    {"q": "How to install Python?", "a": "Use pip install..."},
    {"q": "How to setup Python?", "a": "Download from python.org..."},
    {"q": "How to install Node.js?", "a": "Use npm install..."},
    {"q": "How to debug Python?", "a": "Use pdb module..."},
]

selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2,
    fetch_k=4
)

# 用户问题
query = "How to install Python?"

# MMR 选择：
# 1. "How to install Python?" (最相关)
# 2. "How to debug Python?" (多样性：不同操作)
# 而不是：
# 1. "How to install Python?"
# 2. "How to setup Python?" (过于相似)
```

[来源: reference/fetch_example_selector_05.md | https://www.swarnendu.de/blog/langchain-best-practices]

### 2. 代码生成

```python
# 场景：代码生成示例选择
examples = [
    {"task": "sort list", "code": "sorted(list)"},
    {"task": "sort array", "code": "sorted(array)"},
    {"task": "filter list", "code": "filter(lambda x: x > 0, list)"},
    {"task": "map list", "code": "map(lambda x: x * 2, list)"},
]

selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3,
    fetch_k=4
)

# 用户任务
task = "sort list"

# MMR 选择：
# 1. "sort list" (最相关)
# 2. "filter list" (多样性：不同操作)
# 3. "map list" (多样性：不同操作)
```

[来源: reference/fetch_example_selector_04.md | https://www.sandgarden.com/learn/few-shot-prompting]

### 3. RAG 系统中的 Few-shot 优化

```python
# 场景：RAG 系统中的 Few-shot learning
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 创建 MMR 选择器
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=qa_examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=3,
    fetch_k=10
)

# 创建 Few-shot Prompt
few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Answer based on context and examples:",
    suffix="Context: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

# 创建 RAG 链
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | few_shot_prompt
    | ChatOpenAI()
)

# 使用
result = chain.invoke("What is Python?")
```

[来源: reference/fetch_example_selector_05.md | https://www.swarnendu.de/blog/langchain-best-practices]

---

## 最佳实践

### 1. fetch_k 参数调优

```python
# 经验法则
fetch_k = k * 3  # 标准配置
fetch_k = k * 5  # 高多样性场景
fetch_k = k * 2  # 高相关性场景

# 示例
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=4,
    fetch_k=12  # 4 * 3
)
```

[来源: reference/search_example_selector_02.md]

### 2. 示例库设计

```python
# ✅ 好的示例库：多样化
examples = [
    {"input": "Python basics", "output": "..."},
    {"input": "JavaScript basics", "output": "..."},
    {"input": "Python advanced", "output": "..."},
    {"input": "Java basics", "output": "..."},
]

# ❌ 不好的示例库：过于相似
examples = [
    {"input": "Python basics", "output": "..."},
    {"input": "Python fundamentals", "output": "..."},
    {"input": "Python introduction", "output": "..."},
    {"input": "Python tutorial", "output": "..."},
]
```

### 3. 性能优化

```python
# 缓存向量存储
from langchain_chroma import Chroma

# 持久化向量存储
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

selector = MaxMarginalRelevanceExampleSelector(
    vectorstore=vectorstore,
    k=4,
    fetch_k=12
)
```

[来源: reference/search_example_selector_02.md]

---

## 常见误区

### 误区 1：fetch_k 越大越好 ❌

**为什么错？**
- fetch_k 过大会增加计算成本
- 候选过多可能引入噪音
- 通常 k * 3 到 k * 5 就足够

**正确理解：**
```python
# ❌ 不推荐
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=4,
    fetch_k=100  # 过大
)

# ✅ 推荐
selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=4,
    fetch_k=12  # 合理
)
```

[来源: reference/search_example_selector_02.md]

### 误区 2：MMR 总是比语义相似度好 ❌

**为什么错？**
- 如果需要高度相关的示例，语义相似度更好
- MMR 适合需要多样性的场景
- 根据实际需求选择

**正确理解：**
```python
# 场景1：需要高度相关的示例
# 使用 SemanticSimilarityExampleSelector

# 场景2：需要多样化的示例
# 使用 MaxMarginalRelevanceExampleSelector
```

### 误区 3：不考虑示例库质量 ❌

**为什么错？**
- MMR 算法依赖示例库的多样性
- 如果示例库本身就很相似，MMR 也无法提高多样性

**正确理解：**
```python
# ✅ 好的示例库：覆盖多个主题
examples = [
    {"topic": "Python", "content": "..."},
    {"topic": "JavaScript", "content": "..."},
    {"topic": "Database", "content": "..."},
]

# ❌ 不好的示例库：只有一个主题
examples = [
    {"topic": "Python", "content": "..."},
    {"topic": "Python", "content": "..."},
    {"topic": "Python", "content": "..."},
]
```

---

## 双重类比

### 前端开发类比

**MMR 选择器 = 搜索结果去重 + 多样化推荐**

```javascript
// 前端搜索结果去重
const searchResults = [
  { title: "Python tutorial", score: 0.95 },
  { title: "Python guide", score: 0.93 },
  { title: "JavaScript tutorial", score: 0.85 },
];

// 去重逻辑：避免选择过于相似的结果
const diverseResults = deduplicateResults(searchResults, threshold=0.9);

// 类似于 MMR 的 fetch_k 和 k
// fetch_k = 获取更多候选
// k = 最终返回的数量
```

### 日常生活类比

**MMR 选择器 = 餐厅推荐系统**

```
场景：推荐餐厅

语义相似度选择：
- 用户喜欢川菜
- 推荐：川菜馆A、川菜馆B、川菜馆C
- 问题：都是川菜，缺乏多样性

MMR 选择：
- 用户喜欢川菜
- 推荐：川菜馆A（最相关）、粤菜馆B（多样性）、日料馆C（多样性）
- 优势：既满足偏好，又提供多样选择
```

---

## 总结

MaxMarginalRelevanceExampleSelector 是 LangChain 中用于提高示例多样性的高级选择器：

**核心特点：**
1. **MMR 算法**：平衡相关性和多样性
2. **两阶段选择**：fetch_k 候选 → k 最终示例
3. **避免冗余**：不选择过于相似的示例
4. **提高质量**：提供更多样化的 Few-shot 示例

**适用场景：**
- 需要多样化示例的 Few-shot learning
- 避免示例冗余的场景
- 提高 Prompt 质量的应用

**关键参数：**
- `k`：最终返回的示例数量
- `fetch_k`：先获取的候选数量（通常为 k * 3 到 k * 5）

**与其他选择器的关系：**
- 比 SemanticSimilarityExampleSelector 更注重多样性
- 比 LengthBasedExampleSelector 更智能
- 适合与 FewShotPromptTemplate 集成使用

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/semantic_similarity.py | reference/context7_langchain_01.md | reference/search_example_selector_02.md]
