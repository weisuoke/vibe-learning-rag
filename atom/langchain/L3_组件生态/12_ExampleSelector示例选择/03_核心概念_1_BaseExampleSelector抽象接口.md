# 核心概念 1：BaseExampleSelector 抽象接口

## 一句话定义

**BaseExampleSelector 是 LangChain 中定义示例选择器统一接口的抽象基类，通过两个核心方法（add_example 和 select_examples）实现示例的添加和动态选择，支持同步和异步操作。**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/base.py]

---

## 为什么需要抽象接口？

### 问题：多种选择策略，如何统一？

在 AI Agent 开发中，不同场景需要不同的示例选择策略：

- **问答系统**：需要语义相似度选择
- **Token 限制场景**：需要长度限制选择
- **多样性需求**：需要 MMR 选择
- **轻量级场景**：需要 N-gram 重叠选择

如果每种策略都有不同的接口，代码会变得混乱：

```python
# ❌ 没有统一接口的情况
semantic_selector.find_similar(query, k=3)
length_selector.select_by_length(query, max_length=100)
mmr_selector.get_diverse_examples(query, k=5, fetch_k=20)
```

**解决方案：定义统一的抽象接口**

```python
# ✅ 统一接口
selector.select_examples(input_variables)
```

[来源: reference/source_example_selector_01.md | LangChain 源码分析]

---

## BaseExampleSelector 源码分析

### 完整源码

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.runnables import run_in_executor

class BaseExampleSelector(ABC):
    """示例选择器的抽象基类

    定义了所有示例选择器必须实现的接口。
    """

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """添加新示例到存储

        Args:
            example: 示例字典，键值对形式

        Returns:
            Any: 实现类可以返回任何类型
        """
        pass

    async def aadd_example(self, example: Dict[str, str]) -> Any:
        """异步添加示例

        默认实现：使用 run_in_executor 包装同步方法

        Args:
            example: 示例字典

        Returns:
            Any: 与 add_example 相同的返回类型
        """
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
        """根据输入选择示例

        Args:
            input_variables: 输入变量字典

        Returns:
            List[Dict]: 选中的示例列表
        """
        pass

    async def aselect_examples(
        self, input_variables: Dict[str, str]
    ) -> List[Dict]:
        """异步选择示例

        默认实现：使用 run_in_executor 包装同步方法

        Args:
            input_variables: 输入变量字典

        Returns:
            List[Dict]: 选中的示例列表
        """
        return await run_in_executor(
            None, self.select_examples, input_variables
        )
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/base.py]

---

## 核心方法详解

### 方法 1：add_example() - 添加示例

**作用：** 将新示例添加到选择器的存储中

**签名：**
```python
def add_example(self, example: Dict[str, str]) -> Any
```

**参数：**
- `example`: 示例字典，通常包含 `input` 和 `output` 键

**返回值：**
- `Any`: 实现类可以返回任何类型（通常返回 None 或存储 ID）

**使用场景：**
1. **初始化时批量添加示例**
2. **运行时动态添加新示例**
3. **从用户反馈中学习新示例**

**代码示例：**

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建选择器
selector = SemanticSimilarityExampleSelector.from_examples(
    [],  # 初始为空
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

# 添加示例
selector.add_example({"input": "2+2", "output": "4"})
selector.add_example({"input": "2+3", "output": "5"})
selector.add_example({"input": "What is Python?", "output": "A programming language"})

# 选择示例
selected = selector.select_examples({"input": "What is 5+5?"})
print(selected)
# 输出：[{"input": "2+2", "output": "4"}, {"input": "2+3", "output": "5"}]
```

[来源: reference/fetch_example_selector_01.md | Medium 教程]

---

### 方法 2：select_examples() - 选择示例

**作用：** 根据输入变量选择最相关的示例

**签名：**
```python
def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]
```

**参数：**
- `input_variables`: 输入变量字典，通常包含用户查询

**返回值：**
- `List[Dict]`: 选中的示例列表

**选择策略：**
不同的实现类有不同的选择策略：

| 实现类 | 选择策略 | 适用场景 |
|--------|---------|---------|
| SemanticSimilarityExampleSelector | 语义相似度 | 问答系统、分类任务 |
| LengthBasedExampleSelector | 长度限制 | Token 限制场景 |
| MaxMarginalRelevanceExampleSelector | MMR（多样性） | 需要多样化示例 |
| NGramOverlapExampleSelector | N-gram 重叠 | 轻量级文本相似度 |

[来源: reference/source_example_selector_01.md | LangChain 源码分析]

**代码示例：**

```python
# 场景 1：语义相似度选择
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[
        {"input": "How to cook pasta?", "output": "Boil water, add pasta..."},
        {"input": "How to fix a car?", "output": "Check the engine..."},
        {"input": "How to bake bread?", "output": "Mix flour, water..."},
    ],
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2
)

# 查询：烹饪相关
selected = semantic_selector.select_examples({"input": "How to make pizza?"})
print(selected)
# 输出：[{"input": "How to cook pasta?", ...}, {"input": "How to bake bread?", ...}]

# 场景 2：长度限制选择
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

length_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  # 限制总长度
)

selected = length_selector.select_examples({"input": "Short query"})
# 自动选择适合长度限制的示例
```

[来源: reference/fetch_example_selector_06.md | AI Mind 教程]

---

## 异步支持机制

### 为什么需要异步？

**问题：** 向量搜索可能很慢，特别是在大规模数据集上。

**同步操作的问题：**
```python
# ❌ 同步操作会阻塞主线程
examples = selector.select_examples(input_variables)  # 可能需要 1-2 秒
# 在这期间，整个应用被阻塞
```

**异步操作的优势：**
```python
# ✅ 异步操作不阻塞主线程
examples = await selector.aselect_examples(input_variables)
# 可以同时处理其他请求
```

[来源: reference/source_example_selector_01.md | LangChain 源码分析]

---

### 异步方法实现

**aadd_example() - 异步添加示例**

```python
async def aadd_example(self, example: Dict[str, str]) -> Any:
    """异步添加示例

    默认实现：使用 run_in_executor 包装同步方法
    """
    return await run_in_executor(None, self.add_example, example)
```

**aselect_examples() - 异步选择示例**

```python
async def aselect_examples(
    self, input_variables: Dict[str, str]
) -> List[Dict]:
    """异步选择示例

    默认实现：使用 run_in_executor 包装同步方法
    """
    return await run_in_executor(
        None, self.select_examples, input_variables
    )
```

**run_in_executor 的作用：**
- 将同步方法包装为异步方法
- 在线程池中执行同步方法
- 不阻塞事件循环

[来源: reference/source_example_selector_01.md | LangChain 源码分析]

---

### 异步使用示例

```python
import asyncio
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

async def main():
    # 创建选择器
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=[
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
        ],
        embeddings=OpenAIEmbeddings(),
        vectorstore_cls=Chroma,
        k=2
    )

    # 异步添加示例
    await selector.aadd_example({"input": "3+3", "output": "6"})

    # 异步选择示例
    selected = await selector.aselect_examples({"input": "What is 5+5?"})
    print(selected)

    # 并发处理多个查询
    queries = [
        {"input": "What is 1+1?"},
        {"input": "What is 2+2?"},
        {"input": "What is 3+3?"},
    ]

    results = await asyncio.gather(*[
        selector.aselect_examples(query) for query in queries
    ])

    for query, result in zip(queries, results):
        print(f"Query: {query['input']}")
        print(f"Selected: {result}")
        print()

# 运行异步代码
asyncio.run(main())
```

**输出示例：**
```
Query: What is 1+1?
Selected: [{'input': '2+2', 'output': '4'}, {'input': '2+3', 'output': '5'}]

Query: What is 2+2?
Selected: [{'input': '2+2', 'output': '4'}, {'input': '3+3', 'output': '6'}]

Query: What is 3+3?
Selected: [{'input': '3+3', 'output': '6'}, {'input': '2+3', 'output': '5'}]
```

---

## 双重类比

### 类比 1：BaseExampleSelector = Express 中间件接口

**前端类比：**

在 Express.js 中，中间件有统一的接口：

```javascript
// Express 中间件接口
function middleware(req, res, next) {
    // 处理请求
    next();
}

// 不同的中间件实现
function logger(req, res, next) {
    console.log(req.url);
    next();
}

function auth(req, res, next) {
    if (req.headers.authorization) {
        next();
    } else {
        res.status(401).send('Unauthorized');
    }
}
```

**LangChain 对应：**

```python
# BaseExampleSelector 接口
class BaseExampleSelector(ABC):
    @abstractmethod
    def select_examples(self, input_variables):
        pass

# 不同的选择器实现
class SemanticSimilarityExampleSelector(BaseExampleSelector):
    def select_examples(self, input_variables):
        # 语义相似度选择
        pass

class LengthBasedExampleSelector(BaseExampleSelector):
    def select_examples(self, input_variables):
        # 长度限制选择
        pass
```

**相似点：**
- 统一的接口定义
- 多种具体实现
- 可以互换使用

[来源: CLAUDE_LANGCHAIN.md | LangChain 类比对照表]

---

### 类比 2：add_example() = 数据库 INSERT 操作

**日常生活类比：**

就像往图书馆添加新书：
- **add_example()**：把新书放到书架上
- **select_examples()**：根据读者需求找到最合适的书

**前端类比：**

```javascript
// 数据库操作
db.insert({ title: "Book 1", author: "Author 1" });
db.insert({ title: "Book 2", author: "Author 2" });

// 查询
const results = db.find({ author: "Author 1" });
```

**LangChain 对应：**

```python
# 添加示例
selector.add_example({"input": "Question 1", "output": "Answer 1"})
selector.add_example({"input": "Question 2", "output": "Answer 2"})

# 选择示例
selected = selector.select_examples({"input": "Question 3"})
```

---

### 类比 3：异步支持 = Promise/async-await

**前端类比：**

```javascript
// 同步操作（阻塞）
const data = fetchDataSync();  // 阻塞 2 秒
console.log(data);

// 异步操作（非阻塞）
const data = await fetchDataAsync();  // 不阻塞
console.log(data);
```

**LangChain 对应：**

```python
# 同步操作（阻塞）
examples = selector.select_examples(input_variables)  # 阻塞 1-2 秒

# 异步操作（非阻塞）
examples = await selector.aselect_examples(input_variables)  # 不阻塞
```

**日常生活类比：**

- **同步**：排队买咖啡，前面的人不买完你不能买
- **异步**：点外卖，下单后可以继续做其他事情

---

## 在 AI Agent 开发中的应用

### 应用 1：动态 Few-shot Prompt

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

# 创建示例选择器
selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[
        {"input": "How to cook pasta?", "output": "Boil water..."},
        {"input": "How to fix a car?", "output": "Check engine..."},
        {"input": "How to bake bread?", "output": "Mix flour..."},
    ],
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2
)

# 创建示例模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

# 创建 Few-shot Prompt
prompt = FewShotPromptTemplate(
    example_selector=selector,  # 使用选择器
    example_prompt=example_prompt,
    prefix="You are a helpful assistant.",
    suffix="Q: {input}\nA:",
    input_variables=["input"]
)

# 使用
llm = ChatOpenAI()
chain = prompt | llm

# 查询 1：烹饪相关
response = chain.invoke({"input": "How to make pizza?"})
# 自动选择烹饪相关的示例

# 查询 2：修理相关
response = chain.invoke({"input": "How to fix a bike?"})
# 自动选择修理相关的示例
```

[来源: reference/fetch_example_selector_01.md | Medium 教程]

---

### 应用 2：RAG 系统中的 Few-shot 优化

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建 RAG 系统
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 创建示例选择器（用于优化 Prompt）
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[
        {"query": "What is Python?", "answer": "Python is a programming language..."},
        {"query": "How to use loops?", "answer": "Loops allow you to repeat..."},
    ],
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2
)

# 创建 Few-shot Prompt
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["query", "answer"],
        template="Q: {query}\nA: {answer}"
    ),
    prefix="Answer the question based on the context and examples.",
    suffix="Context: {context}\n\nQ: {question}\nA:",
    input_variables=["context", "question"]
)

# 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 使用
response = qa_chain.invoke({"query": "What is machine learning?"})
```

[来源: reference/fetch_example_selector_05.md | Swarnendu 博客]

---

### 应用 3：多模态 Agent（2025-2026 新趋势）

```python
# 2025-2026 年，多模态 Agent 成为趋势
# ExampleSelector 可以选择包含图像的示例

from langchain_google_genai import ChatGoogleGenerativeAI

# 创建多模态示例选择器
multimodal_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=[
        {
            "input": "Describe this image",
            "output": "This is a cat",
            "image_url": "https://example.com/cat.jpg"
        },
        {
            "input": "What's in this picture?",
            "output": "This is a dog",
            "image_url": "https://example.com/dog.jpg"
        },
    ],
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2
)

# 使用 Gemini 2.5（支持多模态）
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# 选择多模态示例
selected = multimodal_selector.select_examples({"input": "Identify this animal"})
# 返回包含图像的示例
```

[来源: CLAUDE_LANGCHAIN.md | 2025-2026 新增场景]

---

## 设计模式分析

### 模式 1：策略模式（Strategy Pattern）

**定义：** 定义一系列算法，把它们一个个封装起来，并且使它们可以互换。

**在 BaseExampleSelector 中的体现：**

```python
# 抽象策略
class BaseExampleSelector(ABC):
    @abstractmethod
    def select_examples(self, input_variables):
        pass

# 具体策略 1：语义相似度
class SemanticSimilarityExampleSelector(BaseExampleSelector):
    def select_examples(self, input_variables):
        # 语义相似度算法
        pass

# 具体策略 2：长度限制
class LengthBasedExampleSelector(BaseExampleSelector):
    def select_examples(self, input_variables):
        # 长度限制算法
        pass

# 使用策略
def use_selector(selector: BaseExampleSelector, input_variables):
    return selector.select_examples(input_variables)

# 可以互换使用
use_selector(SemanticSimilarityExampleSelector(...), input_variables)
use_selector(LengthBasedExampleSelector(...), input_variables)
```

[来源: reference/source_example_selector_01.md | LangChain 源码分析]

---

### 模式 2：模板方法模式（Template Method Pattern）

**定义：** 定义一个操作中的算法骨架，将一些步骤延迟到子类中。

**在 BaseExampleSelector 中的体现：**

```python
class BaseExampleSelector(ABC):
    # 模板方法（已实现）
    async def aadd_example(self, example):
        return await run_in_executor(None, self.add_example, example)

    # 抽象方法（子类实现）
    @abstractmethod
    def add_example(self, example):
        pass
```

**优势：**
- 子类只需实现同步方法
- 异步方法自动生成
- 减少重复代码

---

## 实际应用最佳实践

### 最佳实践 1：选择合适的选择器

| 场景 | 推荐选择器 | 原因 |
|------|-----------|------|
| 问答系统 | SemanticSimilarityExampleSelector | 语义相关性最重要 |
| Token 限制 | LengthBasedExampleSelector | 必须控制长度 |
| 多样性需求 | MaxMarginalRelevanceExampleSelector | 避免示例过于相似 |
| 轻量级场景 | NGramOverlapExampleSelector | 不需要 embeddings |

[来源: reference/search_example_selector_02.md | 2025-2026 最佳实践]

---

### 最佳实践 2：缓存 Embeddings

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    """缓存 embeddings 以提高性能"""
    return embeddings.embed_query(text)

# 使用缓存的 embeddings
class CachedSemanticSelector(SemanticSimilarityExampleSelector):
    def _get_embedding(self, text):
        return get_embedding(text)
```

**优势：**
- 减少重复计算
- 降低 API 调用成本
- 提高响应速度

[来源: reference/search_example_selector_02.md | 性能优化]

---

### 最佳实践 3：混合策略（2025-2026 新趋势）

```python
# 结合语义相似度和长度限制
class HybridExampleSelector(BaseExampleSelector):
    def __init__(self, semantic_selector, length_selector):
        self.semantic_selector = semantic_selector
        self.length_selector = length_selector

    def select_examples(self, input_variables):
        # 先用语义相似度选择
        candidates = self.semantic_selector.select_examples(input_variables)

        # 再用长度限制过滤
        filtered = self.length_selector.select_examples(input_variables)

        # 返回交集
        return [ex for ex in candidates if ex in filtered]
```

[来源: reference/fetch_example_selector_10.md | 社区扩展项目]

---

## 总结

### BaseExampleSelector 的核心价值

1. **统一接口**：所有选择器都实现相同的接口
2. **策略模式**：可以轻松切换不同的选择策略
3. **异步支持**：默认提供异步方法，提高性能
4. **易于扩展**：可以自定义选择器实现

### 两个核心方法

- **add_example()**：添加示例到存储
- **select_examples()**：根据输入选择示例

### 异步支持机制

- **aadd_example()**：异步添加示例
- **aselect_examples()**：异步选择示例
- 使用 `run_in_executor` 包装同步方法

### 在 AI Agent 开发中的应用

- 动态 Few-shot Prompt
- RAG 系统优化
- 多模态 Agent（2025-2026 新趋势）

---

## 下一步学习

1. **深入学习具体实现**：
   - SemanticSimilarityExampleSelector
   - LengthBasedExampleSelector
   - MaxMarginalRelevanceExampleSelector

2. **实战练习**：
   - 实现自定义选择器
   - 构建混合策略选择器
   - 优化 RAG 系统的 Few-shot Prompt

3. **性能优化**：
   - 缓存 embeddings
   - 批量处理
   - 异步并发

---

**参考资料：**
- [来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/base.py]
- [来源: reference/source_example_selector_01.md | LangChain 源码分析]
- [来源: reference/fetch_example_selector_01.md | Medium 教程]
- [来源: reference/fetch_example_selector_05.md | Swarnendu 博客 - LangChain Best Practices 2025]
- [来源: reference/fetch_example_selector_10.md | 社区扩展项目]
- [来源: CLAUDE_LANGCHAIN.md | LangChain 特定配置]
