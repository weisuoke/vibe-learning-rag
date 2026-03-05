# 核心概念 6：FewShotPromptTemplate 集成

## 1. 【30字核心】

**FewShotPromptTemplate 是 LangChain 中将 ExampleSelector 与 Prompt 模板集成的桥梁，实现动态示例选择与格式化的统一管理。**

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/base.py | reference/fetch_example_selector_02.md]

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题。

### FewShotPromptTemplate 集成的第一性原理

#### 1. 最基础的定义

**FewShotPromptTemplate = 示例选择器 + 示例格式化 + Prompt 组装**

仅此而已！它是一个组合器，将三个独立的部分组合成一个完整的 Prompt 生成流程。

#### 2. 为什么需要 FewShotPromptTemplate？

**核心问题：如何在 Prompt 中动态插入最相关的示例？**

在 Few-shot learning 中，我们需要：
- 从大量示例中选择最相关的几个
- 将选中的示例格式化为统一的文本
- 将格式化后的示例插入到 Prompt 模板中

如果没有 FewShotPromptTemplate，我们需要手动完成这三个步骤，代码会变得冗长且难以维护。

#### 3. FewShotPromptTemplate 的三层价值

##### 价值1：统一接口

将示例选择、格式化、Prompt 组装三个步骤封装成一个统一的接口，简化开发流程。

```python
# 没有 FewShotPromptTemplate（手动实现）
selector = SemanticSimilarityExampleSelector(...)
selected_examples = selector.select_examples({"input": query})
formatted_examples = "\n\n".join([format_example(ex) for ex in selected_examples])
final_prompt = f"{prefix}\n\n{formatted_examples}\n\n{suffix.format(input=query)}"

# 使用 FewShotPromptTemplate（一行搞定）
prompt = few_shot_prompt_template.format(input=query)
```

[来源: reference/fetch_example_selector_06.md | LangChain in Chains #6: Example Selectors]

##### 价值2：声明式配置

使用声明式的方式配置 Prompt 结构，而不是命令式的代码拼接。

##### 价值3：与 LCEL 无缝集成

FewShotPromptTemplate 实现了 Runnable 接口，可以直接用于 LCEL 链式调用。

```python
chain = few_shot_prompt_template | llm | parser
```

[来源: reference/context7_langchain_01.md | LangChain 官方文档]

#### 4. 从第一性原理推导 AI Agent 应用

**推理链：**
```
1. AI Agent 需要根据不同的输入给出不同的响应
   ↓
2. Few-shot learning 可以通过示例引导模型行为
   ↓
3. 示例的相关性直接影响模型的输出质量
   ↓
4. 需要动态选择最相关的示例
   ↓
5. FewShotPromptTemplate 提供了统一的动态示例选择与格式化机制
   ↓
6. AI Agent 可以根据用户输入动态调整 Prompt 中的示例
```

[来源: reference/fetch_example_selector_05.md | LangChain Best Practices 2025]

#### 5. 一句话总结第一性原理

**FewShotPromptTemplate 是示例选择、格式化、Prompt 组装的统一抽象，通过声明式配置简化 Few-shot learning 的实现。**

---

## 3. 【核心概念】

### 核心概念1：静态示例 vs 动态示例

**静态示例：固定的示例列表，所有查询使用相同的示例。**

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 定义示例格式化模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# 创建 FewShotPromptTemplate（静态示例）
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,  # 固定示例
    example_prompt=example_prompt,
    prefix="You are a math tutor. Here are some examples:",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"]
)

print(few_shot_prompt.format(query="3+3"))
```

**动态示例：使用 ExampleSelector 根据输入动态选择最相关的示例。**

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

# 创建 FewShotPromptTemplate（动态示例）
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 动态选择
    example_prompt=example_prompt,
    prefix="You are a math tutor. Here are some examples:",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"]
)

print(dynamic_prompt.format(query="What is 5+5?"))
```

**在 AI Agent 开发中的应用：**
- 静态示例适合任务明确、示例数量少的场景
- 动态示例适合示例库较大、需要根据输入选择最相关示例的场景

[来源: reference/fetch_example_selector_01.md | Exploring Few-Shot Prompts with LangChain]

---

### 核心概念2：FewShotPromptTemplate 的组成部分

**FewShotPromptTemplate 由以下部分组成：**

1. **examples 或 example_selector**：示例来源
   - `examples`: 固定的示例列表
   - `example_selector`: 动态示例选择器

2. **example_prompt**：示例格式化模板
   - 定义每个示例如何格式化为文本

3. **prefix**：Prompt 前缀
   - 通常包含任务描述和指令

4. **suffix**：Prompt 后缀
   - 通常包含用户输入和输出指示符

5. **input_variables**：输入变量列表
   - 定义 Prompt 需要的输入参数

6. **example_separator**：示例分隔符
   - 默认为 `"\n\n"`

```python
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                    # 或 example_selector
    example_prompt=example_prompt,        # 示例格式化模板
    prefix="Task description...",         # 前缀
    suffix="Input: {query}\nOutput:",     # 后缀
    input_variables=["query"],            # 输入变量
    example_separator="\n\n"              # 分隔符
)
```

**生成的 Prompt 结构：**
```
{prefix}

{example_1}

{example_2}

{suffix}
```

[来源: reference/fetch_example_selector_02.md | Pinecone LangChain Prompt Templates]

---

### 核心概念3：LangSmith 中的 {{few_shot_examples}} 占位符

**LangSmith 提供了专门的占位符来管理 Few-shot 示例。**

在 LangSmith 的 Prompt 模板中，可以使用 `{{few_shot_examples}}` 占位符：

```mustache
{{!-- Template --}}
You are a sentiment classifier.

{{few_shot_examples}}

Now classify this text:
Text: {{text}}
Sentiment:
```

**特点：**
- Few-shot 示例在 LangSmith UI 中单独配置
- 保持 Prompt 模板清晰和模块化
- 帮助 LLM 理解格式期望、边缘案例和任务细节

**与 FewShotPromptTemplate 的关系：**
- LangSmith 的 `{{few_shot_examples}}` 是 UI 层面的抽象
- FewShotPromptTemplate 是代码层面的实现
- 两者都旨在简化 Few-shot 示例的管理

[来源: reference/context7_langchain_01.md | LangSmith Prompt Template Format]

---

## 4. 【最小可用】

掌握以下内容，就能开始使用 FewShotPromptTemplate：

### 4.1 基础静态示例

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"query": "How are you?", "answer": "I'm doing well, thanks!"},
    {"query": "What's your name?", "answer": "I'm Claude."},
]

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="Q: {query}\nA: {answer}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a friendly assistant:",
    suffix="Q: {query}\nA:",
    input_variables=["query"]
)
```

### 4.2 动态示例选择

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="You are a friendly assistant:",
    suffix="Q: {query}\nA:",
    input_variables=["query"]
)
```

### 4.3 与 LCEL 集成

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
chain = dynamic_prompt | llm

response = chain.invoke({"query": "What's the weather like?"})
```

**这些知识足以：**
- 创建基础的 Few-shot Prompt
- 实现动态示例选择
- 集成到 LCEL 链中

[来源: reference/fetch_example_selector_03.md | LangChain Cookbook]

---

## 5. 【双重类比】

### 类比1：FewShotPromptTemplate

**前端类比：** React 的高阶组件（HOC）

FewShotPromptTemplate 就像一个 HOC，它接收示例选择器和格式化模板作为输入，返回一个增强的 Prompt 生成器。

```javascript
// React HOC
const withExamples = (Component) => (props) => {
  const examples = selectExamples(props.input);
  return <Component {...props} examples={examples} />;
};

// FewShotPromptTemplate
const prompt = FewShotPromptTemplate({
  example_selector: selector,
  example_prompt: template
});
```

**日常生活类比：** 餐厅的菜单模板

FewShotPromptTemplate 就像餐厅的菜单模板：
- **prefix**：菜单标题（"今日特色"）
- **examples**：菜品列表（根据顾客口味动态推荐）
- **suffix**：点餐提示（"请选择您的菜品"）

---

### 类比2：静态示例 vs 动态示例

**前端类比：** 静态路由 vs 动态路由

```javascript
// 静态路由（固定页面）
<Route path="/about" component={AboutPage} />

// 动态路由（根据参数加载）
<Route path="/user/:id" component={UserPage} />
```

**日常生活类比：** 固定菜单 vs 个性化推荐

- 静态示例：快餐店的固定套餐
- 动态示例：智能推荐系统根据你的口味推荐菜品

---

### 类比3：example_prompt

**前端类比：** 列表项渲染模板

```javascript
// React 列表渲染
{items.map(item => (
  <div key={item.id}>
    <h3>{item.title}</h3>
    <p>{item.description}</p>
  </div>
))}
```

**日常生活类比：** 名片模板

example_prompt 就像名片模板，定义了每个示例的展示格式。

---

## 6. 【反直觉点】

### 误区1：示例越多越好 ❌

**为什么错？**
- 过多示例会增加 Prompt 长度，消耗更多 tokens
- 可能引入噪音，降低模型的理解能力
- 超过模型的 context window 限制

**为什么人们容易这样错？**
人们直觉认为"更多信息 = 更好的结果"，但在 Few-shot learning 中，质量比数量更重要。

**正确理解：**
```python
# ❌ 错误：使用所有示例
selector = SemanticSimilarityExampleSelector(k=20)  # 太多了

# ✅ 正确：选择 3-5 个最相关的示例
selector = SemanticSimilarityExampleSelector(k=3)  # 刚刚好
```

[来源: reference/search_example_selector_02.md | 2025-2026 最佳实践]

---

### 误区2：FewShotPromptTemplate 只能用于简单任务 ❌

**为什么错？**
FewShotPromptTemplate 可以处理复杂的任务，包括：
- 多步推理（Chain-of-Thought）
- 代码生成
- 结构化输出

**正确理解：**
```python
# 复杂任务示例：代码生成
examples = [
    {
        "description": "Create a function to calculate factorial",
        "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
    }
]

example_prompt = PromptTemplate(
    input_variables=["description", "code"],
    template="Description: {description}\n\nCode:\n```python\n{code}\n```"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Python expert. Generate code based on the description:",
    suffix="Description: {description}\n\nCode:\n```python\n",
    input_variables=["description"]
)
```

---

### 误区3：静态示例和动态示例不能混用 ❌

**为什么错？**
虽然 FewShotPromptTemplate 只接受 `examples` 或 `example_selector` 之一，但可以通过自定义 ExampleSelector 实现混合策略。

**正确理解：**
```python
class HybridExampleSelector(BaseExampleSelector):
    def __init__(self, static_examples, dynamic_selector):
        self.static_examples = static_examples
        self.dynamic_selector = dynamic_selector
    
    def select_examples(self, input_variables):
        # 先返回静态示例
        examples = self.static_examples.copy()
        # 再添加动态选择的示例
        examples.extend(self.dynamic_selector.select_examples(input_variables))
        return examples
```

[来源: reference/fetch_example_selector_10.md | langchain-semantic-length-example-selector]

---

## 7. 【实战代码】

```python
"""
FewShotPromptTemplate 集成实战
演示：静态示例、动态示例、与 LCEL 集成
"""

import os
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

# ===== 1. 静态示例 =====
print("=== 静态示例 ===")

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

static_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the opposite of the word:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

print(static_prompt.format(word="big"))
print()

# ===== 2. 动态示例选择 =====
print("=== 动态示例选择 ===")

# 扩展示例库
extended_examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
    {"input": "fast", "output": "slow"},
    {"input": "bright", "output": "dark"},
    {"input": "loud", "output": "quiet"},
]

# 创建语义相似度选择器
selector = SemanticSimilarityExampleSelector.from_examples(
    extended_examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2  # 只选择 2 个最相关的示例
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Give the opposite of the word:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

print(dynamic_prompt.format(word="noisy"))
print()

# ===== 3. 与 LCEL 集成 =====
print("=== 与 LCEL 集成 ===")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = dynamic_prompt | llm

response = chain.invoke({"word": "expensive"})
print(f"Input: expensive")
print(f"Output: {response.content}")
print()

# ===== 4. AI Agent 场景：问答系统 =====
print("=== AI Agent 场景：问答系统 ===")

qa_examples = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare wrote Romeo and Juliet."
    },
    {
        "question": "What is the largest planet?",
        "answer": "Jupiter is the largest planet in our solar system."
    },
]

qa_example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}"
)

qa_selector = SemanticSimilarityExampleSelector.from_examples(
    qa_examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2
)

qa_prompt = FewShotPromptTemplate(
    example_selector=qa_selector,
    example_prompt=qa_example_prompt,
    prefix="You are a knowledgeable assistant. Answer questions based on the examples:",
    suffix="Q: {question}\nA:",
    input_variables=["question"]
)

qa_chain = qa_prompt | llm

question = "What is the smallest planet?"
response = qa_chain.invoke({"question": question})
print(f"Q: {question}")
print(f"A: {response.content}")
```

**运行输出示例：**
```
=== 静态示例 ===
Give the opposite of the word:

Input: happy
Output: sad

Input: tall
Output: short

Input: hot
Output: cold

Input: big
Output:

=== 动态示例选择 ===
Give the opposite of the word:

Input: loud
Output: quiet

Input: bright
Output: dark

Input: noisy
Output:

=== 与 LCEL 集成 ===
Input: expensive
Output: cheap

=== AI Agent 场景：问答系统 ===
Q: What is the smallest planet?
A: Mercury is the smallest planet in our solar system.
```

[来源: reference/fetch_example_selector_03.md | LangChain Cookbook]

---

## 8. 【面试必问】

### 问题："FewShotPromptTemplate 和直接拼接字符串有什么区别？"

**普通回答（❌ 不出彩）：**
"FewShotPromptTemplate 可以动态选择示例，而字符串拼接是固定的。"

**出彩回答（✅ 推荐）：**

> **FewShotPromptTemplate 有三层优势：**
>
> 1. **抽象层面**：提供了统一的接口，将示例选择、格式化、Prompt 组装三个步骤封装成一个声明式的配置，代码更清晰、更易维护。
>
> 2. **功能层面**：支持动态示例选择（通过 ExampleSelector），可以根据输入自动选择最相关的示例，而字符串拼接只能使用固定示例。
>
> 3. **集成层面**：实现了 Runnable 接口，可以无缝集成到 LCEL 链中，支持流式输出、批处理、异步调用等高级特性。
>
> **与字符串拼接的对比**：
> - 字符串拼接：`f"{prefix}\n\n{examples}\n\n{suffix}"`（命令式）
> - FewShotPromptTemplate：声明式配置，支持动态选择和 LCEL 集成
>
> **在实际工作中的应用**：在构建 AI Agent 时，我们使用 FewShotPromptTemplate 结合 SemanticSimilarityExampleSelector，根据用户查询动态选择最相关的示例，显著提高了模型的响应质量。

**为什么这个回答出彩？**
1. ✅ 从抽象、功能、集成三个层面全面对比
2. ✅ 提供了具体的代码示例
3. ✅ 联系实际工作场景

[来源: reference/fetch_example_selector_05.md | LangChain Best Practices 2025]

---

## 9. 【化骨绵掌】

### 卡片1：FewShotPromptTemplate 的本质

**一句话：** FewShotPromptTemplate 是示例选择、格式化、Prompt 组装的统一抽象。

**举例：**
```python
prompt = FewShotPromptTemplate(
    example_selector=selector,  # 选择
    example_prompt=template,    # 格式化
    prefix="...",               # 组装
    suffix="..."
)
```

**应用：** 在 AI Agent 中，用于动态生成包含相关示例的 Prompt。

---

### 卡片2：静态示例的适用场景

**一句话：** 静态示例适合示例数量少、任务明确的场景。

**举例：**
- 数学计算（示例：2+2=4, 3+3=6）
- 格式转换（示例：JSON → YAML）

**应用：** 快速原型开发、简单任务。

---

### 卡片3：动态示例的优势

**一句话：** 动态示例根据输入选择最相关的示例，提高模型理解能力。

**举例：**
```python
selector = SemanticSimilarityExampleSelector(k=3)
prompt = FewShotPromptTemplate(example_selector=selector, ...)
```

**应用：** 大型示例库、复杂任务、需要上下文相关的场景。

---

### 卡片4：example_prompt 的作用

**一句话：** example_prompt 定义每个示例的格式化模板。

**举例：**
```python
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)
```

**应用：** 统一示例格式，提高模型的理解能力。

---

### 卡片5：prefix 和 suffix 的设计

**一句话：** prefix 包含任务描述，suffix 包含用户输入和输出指示符。

**举例：**
```python
prefix = "You are a helpful assistant:"
suffix = "Q: {query}\nA:"
```

**应用：** 引导模型理解任务和输出格式。

---

### 卡片6：与 LCEL 的集成

**一句话：** FewShotPromptTemplate 实现了 Runnable 接口，可以直接用于 LCEL 链。

**举例：**
```python
chain = prompt | llm | parser
```

**应用：** 构建复杂的 AI Agent 工作流。

---

### 卡片7：LangSmith 的 {{few_shot_examples}} 占位符

**一句话：** LangSmith 提供了 UI 层面的 Few-shot 示例管理。

**举例：**
```mustache
{{few_shot_examples}}
```

**应用：** 在 LangSmith 中可视化管理示例。

---

### 卡片8：示例数量的选择

**一句话：** 通常选择 3-5 个示例效果最好。

**举例：**
```python
selector = SemanticSimilarityExampleSelector(k=3)
```

**应用：** 平衡相关性和 token 消耗。

---

### 卡片9：混合策略

**一句话：** 可以通过自定义 ExampleSelector 实现静态 + 动态的混合策略。

**举例：**
```python
class HybridSelector(BaseExampleSelector):
    def select_examples(self, input_variables):
        return static_examples + dynamic_examples
```

**应用：** 既保证核心示例，又提供动态相关示例。

---

### 卡片10：最佳实践

**一句话：** 使用语义相似度选择器 + 长度限制，确保既相关又不超过 token 限制。

**举例：**
```python
selector = SemanticSimilarityExampleSelector(k=5)
# 结合 LengthBasedExampleSelector 控制长度
```

**应用：** 生产环境中的 Few-shot learning。

[来源: reference/search_example_selector_02.md | 2025-2026 最新趋势]

---

## 10. 【一句话总结】

**FewShotPromptTemplate 是 LangChain 中将 ExampleSelector 与 Prompt 模板集成的统一抽象，通过声明式配置实现动态示例选择、格式化和 Prompt 组装，在 AI Agent 开发中用于根据输入动态生成包含最相关示例的 Prompt。**

---

**版本：** v1.0
**最后更新：** 2026-02-26
**维护者：** Claude Code
