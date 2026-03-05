---
type: context7_documentation
library: LangChain
version: main (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: ExampleSelector示例选择
context7_query: ExampleSelector semantic similarity few-shot learning prompt template
---

# Context7 文档：LangChain ExampleSelector

## 文档来源
- 库名称：LangChain
- 版本：main (最后更新：2026-02-17)
- 官方文档链接：https://docs.langchain.com
- Context7 库 ID：/websites/langchain
- 总代码片段数：26,795
- 信任评分：10/10
- 基准评分：83/100

---

## 关键信息提取

### 1. 语义搜索在示例选择中的应用

**来源：** LangSmith 优化分类器文档
**链接：** https://docs.langchain.com/langsmith/optimize-classifier

#### 核心概念
将语义搜索集成到主题分类器中，用于动态选择最相关的 Few-shot 示例。

#### 实现示例
```python
ls_client = Client()

def create_example_string(examples):
    final_strings = []
    for e in examples:
        final_strings.append(f"Input: {e.inputs['topic']}\n> {e.outputs['output']}")
    return "\n\n".join(final_strings)

client = openai.Client()

available_topics = [
    "bug",
    "improvement",
    "new_feature",
    "documentation",
    "integration",
]

prompt_template = """Classify the type of the issue as one of {topics}.

Here are some examples:
{examples}

Begin!
Issue: {text}
>"""

@traceable(
    run_type="chain",
    name="Classifier",
)
def topic_classifier(topic: str):
    # 从数据集中获取示例
    examples = list(ls_client.list_examples(dataset_name="classifier-github-issues"))

    # 使用语义相似度选择最相关的示例
    examples = find_similar(examples, topic)

    # 格式化示例
    example_string = create_example_string(examples)

    return client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt_template.format(
                    topics=','.join(available_topics),
                    text=topic,
                    examples=example_string,
                )
            }
        ],
    ).choices[0].message.content
```

**关键点：**
- 在格式化 Prompt 之前，动态选择最相关的示例
- 基于语义相似度提高分类器性能
- 提供更有针对性的 Few-shot 示例

---

### 2. 语义搜索实现（find_similar 函数）

**来源：** LangSmith 优化分类器文档
**链接：** https://docs.langchain.com/langsmith/optimize-classifier

#### 实现代码
```python
import numpy as np

def find_similar(examples, topic, k=5):
    """使用 OpenAI embeddings 查找最相似的示例

    Args:
        examples: 示例列表
        topic: 查询主题
        k: 返回的示例数量（默认 5）

    Returns:
        最相似的 k 个示例
    """
    # 提取所有示例的主题 + 查询主题
    inputs = [e.inputs['topic'] for e in examples] + [topic]

    # 生成 embeddings
    vectors = client.embeddings.create(
        input=inputs,
        model="text-embedding-3-small"
    )
    vectors = [e.embedding for e in vectors.data]
    vectors = np.array(vectors)

    # 计算余弦相似度并排序
    args = np.argsort(-vectors.dot(vectors[-1])[:-1])[:k]

    # 返回最相似的示例
    examples = [examples[i] for i in args]
    return examples
```

**关键技术：**
- 使用 OpenAI `text-embedding-3-small` 模型生成 embeddings
- 计算余弦相似度（通过向量点积）
- 返回 top-k 最相似的示例
- 适用于大型示例数据集

---

### 3. LangSmith 中的 Few-Shot 示例占位符

**来源：** LangSmith Prompt 模板格式文档
**链接：** https://docs.langchain.com/langsmith/prompt-template-format

#### Mustache 模板示例
```mustache
{{!-- Template --}}
You are a sentiment classifier.

{{few_shot_examples}}

Now classify this text:
Text: {{text}}
Sentiment:
```

**特点：**
- 使用 `{{few_shot_examples}}` 占位符动态注入示例
- Few-shot 示例在 LangSmith UI 中单独配置
- 保持 Prompt 模板清晰和模块化
- 帮助 LLM 理解格式期望、边缘案例和任务细节

---

### 4. 为什么使用语义相似度选择示例

**来源：** LangSmith 优化分类器文档
**链接：** https://docs.langchain.com/langsmith/optimize-classifier

#### 核心观点

**问题：**
随着示例数据集的增长，使用所有可用示例会变得低效。

**解决方案：**
使用语义相似度只选择最相关的示例。

**实现步骤：**
1. 为输入主题和所有可用示例计算 embeddings
2. 使用向量相似度识别 k 个最相似的示例
3. 通过过滤示例减少 Prompt 中的噪音
4. 保持高分类准确性

**优势：**
- 减少 Prompt 长度
- 提高相关性
- 降低成本（更少的 tokens）
- 保持或提高准确性

---

### 5. Few-Shot Prompting 与情景记忆

**来源：** LangChain JavaScript 概念文档（Memory）
**链接：** https://docs.langchain.com/oss/javascript/concepts/memory

#### 核心概念

**情景记忆（Episodic Memory）：**
在实践中，情景记忆通常通过 Few-shot 示例提示实现，代理从过去的序列中学习以正确执行任务。

**关键观点：**
- 有时"展示"比"告诉"更容易
- LLM 从示例中学习效果很好
- Few-shot learning 让你通过输入-输出示例"编程" LLM
- 展示预期行为

**挑战：**
虽然可以使用各种最佳实践生成 Few-shot 示例，但**挑战在于根据用户输入选择最相关的示例**。

**这正是 ExampleSelector 要解决的问题！**

---

## 与 LangChain ExampleSelector 的关系

### 1. 设计理念
- **动态选择**：根据输入动态选择最相关的示例
- **语义相似度**：使用 embeddings 和向量相似度
- **可扩展性**：适用于大型示例数据集

### 2. 实际应用场景
- **分类任务**：如 GitHub issue 分类
- **情感分析**：动态选择相关的情感示例
- **问答系统**：选择相似的问答对
- **代码生成**：选择相似的代码示例

### 3. 与源码的对应关系

**Context7 文档中的 `find_similar` 函数** 对应 **LangChain 源码中的 `SemanticSimilarityExampleSelector`**：

| Context7 示例 | LangChain 源码 |
|---------------|----------------|
| `find_similar(examples, topic, k=5)` | `SemanticSimilarityExampleSelector.select_examples()` |
| 手动计算 embeddings | 使用 `Embeddings` 抽象 |
| 手动计算余弦相似度 | 使用 `VectorStore.similarity_search()` |
| 返回 top-k 示例 | `k` 参数控制返回数量 |

**LangChain 的优势：**
- 更高层次的抽象
- 支持多种向量存储后端（Chroma、FAISS 等）
- 统一的 `BaseExampleSelector` 接口
- 支持异步操作
- 更易于集成到 LCEL 链中

---

## 最佳实践总结

### 1. 何时使用语义相似度选择
- ✅ 示例数据集较大（> 10 个示例）
- ✅ 需要动态选择最相关的示例
- ✅ 希望减少 Prompt 长度和成本
- ✅ 任务需要上下文相关的示例

### 2. 何时使用固定示例
- ✅ 示例数量很少（< 5 个）
- ✅ 所有示例都高度相关
- ✅ 不需要动态选择
- ✅ 简单的任务

### 3. 实现建议
- 使用 `text-embedding-3-small` 模型（成本低、速度快）
- 选择合适的 k 值（通常 3-5 个示例）
- 缓存 embeddings 以提高性能
- 使用向量数据库（如 Chroma、FAISS）存储示例

---

## 参考论文

**Max Marginal Relevance (MMR)：**
- 论文：https://arxiv.org/pdf/2211.13892.pdf
- 用于提高示例多样性
- 平衡相关性和多样性

---

## 总结

Context7 文档主要展示了：
1. **语义搜索在示例选择中的实际应用**
2. **手动实现 `find_similar` 函数的方法**
3. **LangSmith 中的 Few-shot 示例管理**
4. **为什么需要动态选择示例**

这些内容与 LangChain 的 `SemanticSimilarityExampleSelector` 源码高度相关，提供了实际应用场景和最佳实践。
