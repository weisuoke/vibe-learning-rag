# Template Composition 高级模式

> 深入理解 LangChain PromptTemplate 的组合机制与高级应用模式

---

## 概述

**Template Composition（模板组合）** 是 LangChain PromptTemplate 的核心高级特性，允许通过 `+` 操作符将多个模板组合成一个新模板。这种组合机制不仅简化了复杂提示词的构建，还提供了强大的模块化和复用能力。

**核心价值：**
- 模块化提示词管理
- 提高代码复用性
- 简化复杂提示词构建
- 支持动态组合策略

---

## 源码实现原理

### `__add__` 方法实现

LangChain 通过重载 `__add__` 方法实现模板组合功能。

**源码位置：** `langchain_core/prompts/prompt.py:142-184`

```python
def __add__(self, other: Any) -> PromptTemplate:
    """Override the `+` operator to allow for combining prompt templates."""
    if isinstance(other, PromptTemplate):
        # 检查模板格式一致性
        if self.template_format != other.template_format:
            msg = "Cannot add templates of different formats"
            raise ValueError(msg)

        # 合并 input_variables（取并集）
        input_variables = list(
            set(self.input_variables) | set(other.input_variables)
        )

        # 拼接模板字符串
        template = self.template + other.template

        # 合并验证标志（AND 逻辑）
        validate_template = self.validate_template and other.validate_template

        # 合并 partial_variables（检查冲突）
        partial_variables = dict(self.partial_variables.items())
        for k, v in other.partial_variables.items():
            if k in partial_variables:
                msg = "Cannot have same variable partialed twice."
                raise ValueError(msg)
            partial_variables[k] = v

        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            partial_variables=partial_variables,
            template_format=self.template_format,
            validate_template=validate_template,
        )

    # 支持 PromptTemplate + str 的组合
    if isinstance(other, str):
        prompt = PromptTemplate.from_template(
            other,
            template_format=self.template_format,
        )
        return self + prompt
```

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

---

## 核心特性详解

### 1. 变量自动合并

组合时会自动合并两个模板的 `input_variables`，采用**并集**策略。

**示例：**

```python
from langchain_core.prompts import PromptTemplate

# 模板 1：包含变量 {topic}
template1 = PromptTemplate.from_template("Write an article about {topic}.")

# 模板 2：包含变量 {style}
template2 = PromptTemplate.from_template(" Use {style} writing style.")

# 组合后：包含变量 {topic} 和 {style}
combined = template1 + template2

print(f"Input variables: {combined.input_variables}")
# 输出: Input variables: ['topic', 'style']

print(combined.format(topic="AI", style="academic"))
# 输出: Write an article about AI. Use academic writing style.
```

**关键点：**
- 自动去重：相同变量只保留一个
- 顺序无关：`template1 + template2` 和 `template2 + template1` 的变量集合相同
- 模板字符串按顺序拼接

---

### 2. Partial Variables 冲突检查

组合时会检查 `partial_variables` 是否存在冲突，如果同一变量在两个模板中都被 partial，会抛出异常。

**冲突示例：**

```python
from langchain_core.prompts import PromptTemplate

# 模板 1：partial 变量 date
template1 = PromptTemplate(
    template="Today is {date}. ",
    input_variables=[],
    partial_variables={"date": "2026-02-26"}
)

# 模板 2：也 partial 变量 date（冲突！）
template2 = PromptTemplate(
    template="The date is {date}.",
    input_variables=[],
    partial_variables={"date": "2026-03-01"}
)

try:
    combined = template1 + template2
except ValueError as e:
    print(f"Error: {e}")
    # 输出: Error: Cannot have same variable partialed twice.
```

**正确做法：**

```python
# 方案 1：只在一个模板中 partial
template1 = PromptTemplate(
    template="Today is {date}. ",
    input_variables=[],
    partial_variables={"date": "2026-02-26"}
)

template2 = PromptTemplate.from_template("The date is {date}.")

combined = template1 + template2
print(combined.format())
# 输出: Today is 2026-02-26. The date is 2026-02-26.

# 方案 2：使用不同的变量名
template1 = PromptTemplate(
    template="Today is {today}. ",
    input_variables=[],
    partial_variables={"today": "2026-02-26"}
)

template2 = PromptTemplate(
    template="Tomorrow is {tomorrow}.",
    input_variables=[],
    partial_variables={"tomorrow": "2026-02-27"}
)

combined = template1 + template2
print(combined.format())
# 输出: Today is 2026-02-26. Tomorrow is 2026-02-27.
```

---

### 3. Template Format 一致性要求

组合的两个模板必须使用相同的 `template_format`，否则会抛出异常。

**错误示例：**

```python
from langchain_core.prompts import PromptTemplate

# f-string 格式
template1 = PromptTemplate.from_template("Hello {name}.")

# mustache 格式
template2 = PromptTemplate.from_template(
    "Welcome {{user}}.",
    template_format="mustache"
)

try:
    combined = template1 + template2
except ValueError as e:
    print(f"Error: {e}")
    # 输出: Error: Cannot add templates of different formats
```

**正确做法：**

```python
# 统一使用 f-string 格式
template1 = PromptTemplate.from_template("Hello {name}.")
template2 = PromptTemplate.from_template("Welcome {user}.")

combined = template1 + template2
print(combined.format(name="Alice", user="Bob"))
# 输出: Hello Alice.Welcome Bob.

# 或统一使用 mustache 格式
template1 = PromptTemplate.from_template(
    "Hello {{name}}.",
    template_format="mustache"
)
template2 = PromptTemplate.from_template(
    "Welcome {{user}}.",
    template_format="mustache"
)

combined = template1 + template2
print(combined.format(name="Alice", user="Bob"))
# 输出: Hello Alice.Welcome Bob.
```

---

### 4. 支持字符串直接组合

可以直接将字符串与 PromptTemplate 组合，字符串会自动转换为 PromptTemplate。

**示例：**

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Hello {name}.")

# 直接加字符串
combined = template + " How are you today?"

print(combined.format(name="Alice"))
# 输出: Hello Alice. How are you today?

# 字符串在前也可以（需要先转换）
prefix = PromptTemplate.from_template("Greeting: ")
combined = prefix + template

print(combined.format(name="Bob"))
# 输出: Greeting: Hello Bob.
```

---

## 高级组合模式

### 模式 1：分层组合（Layered Composition）

将提示词分为多个层次，逐层组合。

```python
from langchain_core.prompts import PromptTemplate

# 第 1 层：系统角色
system_layer = PromptTemplate.from_template(
    "You are a {role}. "
)

# 第 2 层：任务描述
task_layer = PromptTemplate.from_template(
    "Your task is to {task}. "
)

# 第 3 层：输出格式
format_layer = PromptTemplate.from_template(
    "Output format: {format}."
)

# 组合所有层
full_prompt = system_layer + task_layer + format_layer

print(full_prompt.format(
    role="helpful assistant",
    task="answer questions",
    format="JSON"
))
# 输出: You are a helpful assistant. Your task is to answer questions. Output format: JSON.
```

**应用场景：**
- Agent 系统提示词构建
- RAG 系统多阶段提示词
- 复杂工作流提示词管理

---

### 模式 2：条件组合（Conditional Composition）

根据条件动态选择组合不同的模板。

```python
from langchain_core.prompts import PromptTemplate

def build_prompt(include_examples: bool = False, include_constraints: bool = False):
    """根据条件动态构建提示词"""

    # 基础模板
    base = PromptTemplate.from_template("Task: {task}\n\n")

    # 可选：示例
    if include_examples:
        examples = PromptTemplate.from_template(
            "Examples:\n{examples}\n\n"
        )
        base = base + examples

    # 可选：约束条件
    if include_constraints:
        constraints = PromptTemplate.from_template(
            "Constraints:\n{constraints}\n\n"
        )
        base = base + constraints

    # 输出指令
    output = PromptTemplate.from_template("Output: ")

    return base + output

# 场景 1：只有任务
prompt1 = build_prompt()
print(prompt1.format(task="Summarize the text"))

# 场景 2：任务 + 示例
prompt2 = build_prompt(include_examples=True)
print(prompt2.format(
    task="Summarize the text",
    examples="Example 1: ...\nExample 2: ..."
))

# 场景 3：任务 + 示例 + 约束
prompt3 = build_prompt(include_examples=True, include_constraints=True)
print(prompt3.format(
    task="Summarize the text",
    examples="Example 1: ...",
    constraints="- Max 100 words\n- Use bullet points"
))
```

**应用场景：**
- Few-shot vs Zero-shot 切换
- 调试模式 vs 生产模式
- 不同用户权限的提示词

---

### 模式 3：模板库复用（Template Library Reuse）

构建可复用的模板库，通过组合实现快速开发。

```python
from langchain_core.prompts import PromptTemplate

class PromptLibrary:
    """可复用的提示词模板库"""

    # 角色模板
    ROLES = {
        "assistant": PromptTemplate.from_template("You are a helpful assistant. "),
        "expert": PromptTemplate.from_template("You are an expert in {domain}. "),
        "teacher": PromptTemplate.from_template("You are a patient teacher. "),
    }

    # 任务模板
    TASKS = {
        "summarize": PromptTemplate.from_template("Summarize the following text: {text}"),
        "translate": PromptTemplate.from_template("Translate {text} to {language}"),
        "explain": PromptTemplate.from_template("Explain {concept} in simple terms"),
    }

    # 格式模板
    FORMATS = {
        "json": PromptTemplate.from_template("\n\nOutput as JSON."),
        "markdown": PromptTemplate.from_template("\n\nOutput as Markdown."),
        "bullet": PromptTemplate.from_template("\n\nOutput as bullet points."),
    }

    @classmethod
    def build(cls, role: str, task: str, format: str = None, **kwargs):
        """快速构建提示词"""
        prompt = cls.ROLES[role] + cls.TASKS[task]

        if format:
            prompt = prompt + cls.FORMATS[format]

        return prompt

# 使用示例
prompt = PromptLibrary.build(
    role="expert",
    task="explain",
    format="bullet",
    domain="machine learning"
)

print(prompt.format(concept="gradient descent"))
# 输出: You are an expert in machine learning. Explain gradient descent in simple terms
#       Output as bullet points.
```

**应用场景：**
- 企业级提示词管理
- 多项目共享模板
- 快速原型开发

---

### 模式 4：链式组合（Chained Composition）

将多个模板按顺序链式组合，形成完整的对话流程。

```python
from langchain_core.prompts import PromptTemplate

class ConversationBuilder:
    """对话流程构建器"""

    def __init__(self):
        self.prompt = PromptTemplate.from_template("")

    def add_greeting(self, name: str = None):
        """添加问候语"""
        if name:
            greeting = PromptTemplate(
                template=f"Hello {name}! ",
                input_variables=[]
            )
        else:
            greeting = PromptTemplate.from_template("Hello {name}! ")

        self.prompt = self.prompt + greeting
        return self

    def add_context(self):
        """添加上下文"""
        context = PromptTemplate.from_template(
            "Context: {context}\n\n"
        )
        self.prompt = self.prompt + context
        return self

    def add_question(self):
        """添加问题"""
        question = PromptTemplate.from_template(
            "Question: {question}\n\n"
        )
        self.prompt = self.prompt + question
        return self

    def add_instruction(self):
        """添加指令"""
        instruction = PromptTemplate.from_template(
            "Please {instruction}."
        )
        self.prompt = self.prompt + instruction
        return self

    def build(self):
        """构建最终提示词"""
        return self.prompt

# 使用链式调用
prompt = (ConversationBuilder()
    .add_greeting()
    .add_context()
    .add_question()
    .add_instruction()
    .build())

print(prompt.format(
    name="Alice",
    context="You are helping with a coding problem",
    question="How to reverse a string in Python?",
    instruction="provide a code example"
))
```

**应用场景：**
- 多轮对话构建
- 工作流步骤组合
- 动态对话生成

---

## 与 PipelinePromptTemplate 对比

### PipelinePromptTemplate（旧方式）

```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts import PromptTemplate

# 定义子模板
intro_template = PromptTemplate.from_template("You are a {role}.")
task_template = PromptTemplate.from_template("Task: {task}")

# 使用 PipelinePromptTemplate
pipeline = PipelinePromptTemplate(
    final_prompt=PromptTemplate.from_template("{intro}\n{task}"),
    pipeline_prompts=[
        ("intro", intro_template),
        ("task", task_template),
    ]
)

print(pipeline.format(role="assistant", task="help users"))
```

### Template Composition（推荐方式）

```python
from langchain_core.prompts import PromptTemplate

# 直接使用 + 操作符
intro_template = PromptTemplate.from_template("You are a {role}.\n")
task_template = PromptTemplate.from_template("Task: {task}")

combined = intro_template + task_template

print(combined.format(role="assistant", task="help users"))
```

**对比总结：**

| 特性 | PipelinePromptTemplate | Template Composition |
|------|------------------------|----------------------|
| 语法简洁性 | 较复杂，需要定义 pipeline | 简洁，直接使用 `+` |
| 可读性 | 中等 | 高 |
| 灵活性 | 中等 | 高（支持动态组合） |
| 性能 | 略慢（多层嵌套） | 快（直接拼接） |
| 推荐度 | 不推荐（已过时） | 推荐 |

[来源: reference/search_composition_01.md | 社区最佳实践]

---

## 最佳实践

### 1. 使用有意义的变量名

```python
# ❌ 不好：变量名不清晰
template = PromptTemplate.from_template("Do {x} with {y}")

# ✅ 好：变量名清晰
template = PromptTemplate.from_template("Do {action} with {target}")
```

### 2. 避免 Partial Variables 冲突

```python
# ❌ 不好：可能冲突
template1 = PromptTemplate(
    template="{date}: {event}",
    input_variables=["event"],
    partial_variables={"date": "2026-02-26"}
)

template2 = PromptTemplate(
    template="{date}: {task}",
    input_variables=["task"],
    partial_variables={"date": "2026-03-01"}  # 冲突！
)

# ✅ 好：使用不同变量名
template1 = PromptTemplate(
    template="{event_date}: {event}",
    input_variables=["event"],
    partial_variables={"event_date": "2026-02-26"}
)

template2 = PromptTemplate(
    template="{task_date}: {task}",
    input_variables=["task"],
    partial_variables={"task_date": "2026-03-01"}
)
```

### 3. 统一 Template Format

```python
# ❌ 不好：混用格式
template1 = PromptTemplate.from_template("Hello {name}")  # f-string
template2 = PromptTemplate.from_template("Hi {{user}}", template_format="mustache")

# ✅ 好：统一格式
template1 = PromptTemplate.from_template("Hello {name}")
template2 = PromptTemplate.from_template("Hi {user}")
```

### 4. 模块化设计

```python
# ✅ 好：将提示词拆分为可复用的模块
class PromptModules:
    SYSTEM = PromptTemplate.from_template("System: {system_message}\n\n")
    USER = PromptTemplate.from_template("User: {user_input}\n\n")
    ASSISTANT = PromptTemplate.from_template("Assistant: ")

    @classmethod
    def build_conversation(cls, include_system=True):
        if include_system:
            return cls.SYSTEM + cls.USER + cls.ASSISTANT
        return cls.USER + cls.ASSISTANT
```

---

## 常见陷阱

### 陷阱 1：忘记添加分隔符

```python
# ❌ 问题：模板直接拼接，没有分隔
template1 = PromptTemplate.from_template("Hello {name}.")
template2 = PromptTemplate.from_template("How are you?")

combined = template1 + template2
print(combined.format(name="Alice"))
# 输出: Hello Alice.How are you?  # 缺少空格

# ✅ 解决：添加分隔符
template1 = PromptTemplate.from_template("Hello {name}. ")  # 注意末尾空格
template2 = PromptTemplate.from_template("How are you?")

combined = template1 + template2
print(combined.format(name="Alice"))
# 输出: Hello Alice. How are you?
```

### 陷阱 2：变量名冲突但含义不同

```python
# ❌ 问题：两个模板都有 {name}，但含义不同
template1 = PromptTemplate.from_template("Author: {name}")
template2 = PromptTemplate.from_template("Reviewer: {name}")

combined = template1 + template2
print(combined.format(name="Alice"))
# 输出: Author: AliceReviewer: Alice  # 两个 name 都是 Alice

# ✅ 解决：使用不同的变量名
template1 = PromptTemplate.from_template("Author: {author_name}")
template2 = PromptTemplate.from_template("Reviewer: {reviewer_name}")

combined = template1 + template2
print(combined.format(author_name="Alice", reviewer_name="Bob"))
# 输出: Author: AliceReviewer: Bob
```

### 陷阱 3：过度组合导致可读性下降

```python
# ❌ 问题：过度组合，难以维护
prompt = (template1 + template2 + template3 + template4 +
          template5 + template6 + template7 + template8)

# ✅ 解决：使用中间变量或函数
def build_complex_prompt():
    header = template1 + template2
    body = template3 + template4 + template5
    footer = template6 + template7 + template8
    return header + body + footer

prompt = build_complex_prompt()
```

---

## 实战案例：RAG 系统提示词管理

```python
from langchain_core.prompts import PromptTemplate
from typing import Optional

class RAGPromptManager:
    """RAG 系统提示词管理器"""

    # 基础模板
    SYSTEM_ROLE = PromptTemplate.from_template(
        "You are a helpful AI assistant with access to a knowledge base. "
    )

    CONTEXT_INTRO = PromptTemplate.from_template(
        "Here is the relevant context from the knowledge base:\n\n{context}\n\n"
    )

    QUESTION = PromptTemplate.from_template(
        "User question: {question}\n\n"
    )

    INSTRUCTION = PromptTemplate.from_template(
        "Please answer the question based on the context provided. "
        "If the context doesn't contain enough information, say so."
    )

    @classmethod
    def build_rag_prompt(
        cls,
        include_system: bool = True,
        include_instruction: bool = True,
        custom_instruction: Optional[str] = None
    ) -> PromptTemplate:
        """构建 RAG 提示词"""

        prompt = PromptTemplate.from_template("")

        if include_system:
            prompt = prompt + cls.SYSTEM_ROLE

        prompt = prompt + cls.CONTEXT_INTRO + cls.QUESTION

        if include_instruction:
            if custom_instruction:
                instruction = PromptTemplate.from_template(custom_instruction)
                prompt = prompt + instruction
            else:
                prompt = prompt + cls.INSTRUCTION

        return prompt

# 使用示例
rag_prompt = RAGPromptManager.build_rag_prompt()

formatted = rag_prompt.format(
    context="LangChain is a framework for developing applications powered by LLMs.",
    question="What is LangChain?"
)

print(formatted)
```

---

## 双重类比

### 前端类比：React 组件组合

```javascript
// React 组件组合
const Header = () => <header>Header</header>;
const Content = () => <main>Content</main>;
const Footer = () => <footer>Footer</footer>;

const Page = () => (
  <>
    <Header />
    <Content />
    <Footer />
  </>
);
```

```python
# LangChain 模板组合
header = PromptTemplate.from_template("Header: {title}\n")
content = PromptTemplate.from_template("Content: {body}\n")
footer = PromptTemplate.from_template("Footer: {note}")

page = header + content + footer
```

### 日常生活类比：乐高积木

就像用乐高积木搭建模型：
- 每个模板 = 一块积木
- `+` 操作符 = 拼接积木
- 变量 = 积木的连接点
- 最终提示词 = 完整的模型

---

## 总结

Template Composition 是 LangChain 提示词工程的核心特性，通过简单的 `+` 操作符实现强大的模块化能力。

**关键要点：**
1. 自动合并变量（并集）
2. 检查 partial_variables 冲突
3. 要求 template_format 一致
4. 支持字符串直接组合
5. 优于 PipelinePromptTemplate

**最佳实践：**
- 使用清晰的变量名
- 避免变量冲突
- 统一模板格式
- 模块化设计
- 添加适当的分隔符

**应用场景：**
- Agent 系统提示词
- RAG 系统多阶段提示词
- 复杂工作流管理
- 企业级提示词库

---

**参考资料：**
- [LangChain 源码分析](reference/source_prompttemplate_01.md)
- [社区最佳实践](reference/search_composition_01.md)
- [官方文档](reference/context7_langchain_03.md)
