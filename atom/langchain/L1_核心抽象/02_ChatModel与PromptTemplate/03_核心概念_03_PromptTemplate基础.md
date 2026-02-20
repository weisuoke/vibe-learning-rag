# 核心概念 03: PromptTemplate 基础

> **本文目标**: 深入理解 PromptTemplate 的设计理念、核心功能和使用方法

---

## 概述

**PromptTemplate** 是 LangChain 中用于构建可复用提示词的模板系统。它不仅仅是字符串格式化，而是一个完整的 Runnable 组件，支持变量替换、类型验证和 LCEL 组合。

**核心定位**: PromptTemplate 是变量到消息的生成器，是提示词工程化的基础设施。

---

## 1. PromptTemplate 的本质

### 1.1 函数式定义

**PromptTemplate 本质上是一个函数**:

```python
PromptTemplate: Dict[str, Any] → List[Message]
```

用 Python 类型注解表示:

```python
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

class PromptTemplate:
    def __call__(self, variables: Dict[str, Any]) -> List[BaseMessage]:
        """接收变量字典，返回消息列表"""
        pass
```

**这个定义揭示了三个关键点**:

1. **输入是字典**: 变量名到值的映射
2. **输出是消息列表**: 不是字符串，而是结构化消息
3. **是纯函数**: 相同输入总是产生相同输出

### 1.2 与字符串格式化的区别

```python
# 字符串格式化 - 简单替换
template_str = "你是{role}，请回答：{question}"
result = template_str.format(role="助手", question="你好")
# 输出: "你是助手，请回答：你好" (字符串)

# PromptTemplate - 生成消息
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("你是{role}，请回答：{question}")
result = template.invoke({"role": "助手", "question": "你好"})
# 输出: StringPromptValue (可转换为消息)
```

**对比表**:

| 维度 | 字符串格式化 | PromptTemplate |
|------|--------------|----------------|
| **输入** | 位置参数或关键字参数 | 字典 |
| **输出** | 字符串 | PromptValue (可转换为消息) |
| **类型验证** | 无 | 有（缺少变量会报错） |
| **Runnable** | 否 | 是（支持 invoke/batch/stream） |
| **LCEL 组合** | 否 | 是（可用 \| 操作符） |
| **部分应用** | 否 | 是（支持 partial） |
| **可序列化** | 否 | 是（可保存和加载） |

---

## 2. PromptTemplate 的核心类

### 2.1 类型层次结构

```python
from langchain_core.prompts import (
    BasePromptTemplate,      # 抽象基类
    PromptTemplate,          # 字符串模板
    ChatPromptTemplate,      # 对话模板（推荐）
    FewShotPromptTemplate,   # Few-shot 模板
    PipelinePromptTemplate,  # 管道模板
)

# 继承关系
BasePromptTemplate (抽象基类)
├── PromptTemplate           # 生成单个字符串
├── ChatPromptTemplate       # 生成消息列表（推荐）
├── FewShotPromptTemplate    # 包含示例的模板
└── PipelinePromptTemplate   # 组合多个模板
```

### 2.2 PromptTemplate - 字符串模板

**用途**: 生成单个字符串提示词（适用于 LLM，不适用于 ChatModel）

```python
from langchain_core.prompts import PromptTemplate

# 创建模板
template = PromptTemplate.from_template(
    "你是{role}，请回答：{question}"
)

# 使用模板
result = template.invoke({"role": "助手", "question": "你好"})
print(result.to_string())
# 输出: "你是助手，请回答：你好"

# 与 LLM 组合
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
chain = template | llm
response = chain.invoke({"role": "助手", "question": "你好"})
```

**注意**: PromptTemplate 生成字符串，适用于 LLM（字符串输入），不适用于 ChatModel（消息列表输入）。

### 2.3 ChatPromptTemplate - 对话模板（推荐）

**用途**: 生成消息列表（适用于 ChatModel）

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建模板
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 使用模板
messages = template.invoke({"role": "助手", "question": "你好"})
print(messages)
# 输出: [SystemMessage(content="你是助手"), HumanMessage(content="你好")]

# 与 ChatModel 组合
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
chain = template | model
response = chain.invoke({"role": "助手", "question": "你好"})
```

**推荐**: 现代应用优先使用 ChatPromptTemplate。

---

## 3. 变量替换机制

### 3.1 基础变量替换

**语法**: 使用 `{variable_name}` 占位符

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 提供所有变量
result = template.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})
```

### 3.2 变量验证

**PromptTemplate 会验证变量是否完整**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# ❌ 错误：缺少变量
try:
    template.invoke({"role": "助手"})  # 缺少 question
except KeyError as e:
    print(f"错误: {e}")  # KeyError: 'question'

# ✅ 正确：提供所有变量
template.invoke({"role": "助手", "question": "你好"})
```

**这是 PromptTemplate 相比字符串格式化的重要优势**：编译时发现错误，而不是运行时。

### 3.3 变量类型

**PromptTemplate 支持多种类型的变量**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "问题：{question}\n上下文：{context}\n选项：{options}")
])

result = template.invoke({
    "role": "助手",
    "question": "Python 和 JavaScript 有什么区别？",
    "context": "这是一个编程语言对比问题",
    "options": ["语法", "性能", "生态"]  # 列表会自动转换为字符串
})
```

**自动类型转换**:

```python
# 数字
template.invoke({"age": 25})  # 自动转换为 "25"

# 列表
template.invoke({"items": [1, 2, 3]})  # 自动转换为 "[1, 2, 3]"

# 字典
template.invoke({"data": {"key": "value"}})  # 自动转换为 "{'key': 'value'}"

# None
template.invoke({"optional": None})  # 自动转换为 "None"
```

### 3.4 格式化选项

**f-string 格式（推荐）**:

```python
# 默认使用 f-string 格式
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
```

**jinja2 格式（不推荐）**:

```python
# 使用 jinja2 格式（更强大但有安全风险）
template = ChatPromptTemplate.from_messages([
    ("system", "你是{{ role }}"),
    ("human", "{{ question }}")
], template_format="jinja2")

# jinja2 支持更复杂的逻辑
template = ChatPromptTemplate.from_messages([
    ("system", "你是{{ role }}"),
    ("human", """
    {% if context %}
    上下文：{{ context }}
    {% endif %}
    问题：{{ question }}
    """)
], template_format="jinja2")
```

**安全警告（CVE-2025-65106）**: jinja2 格式存在模板注入风险，推荐使用 f-string 格式。

---

## 4. 部分应用（Partial）

### 4.1 什么是部分应用？

**部分应用**允许你预先填充部分变量，生成新的模板。

```python
from langchain_core.prompts import ChatPromptTemplate

# 原始模板
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，专长是{skill}"),
    ("human", "{question}")
])

# 部分应用：预填充 role
partial_template = template.partial(role="Python专家")

# 使用部分应用的模板（只需提供剩余变量）
result = partial_template.invoke({
    "skill": "代码优化",
    "question": "如何优化这段代码？"
})
```

### 4.2 使用场景

**场景1: 固定角色，动态问题**

```python
# 创建专门的助手模板
python_expert = template.partial(role="Python专家", skill="代码优化")
js_expert = template.partial(role="JavaScript专家", skill="前端开发")

# 使用时只需提供问题
python_response = python_expert.invoke({"question": "如何优化Python代码？"})
js_response = js_expert.invoke({"question": "如何优化JavaScript代码？"})
```

**场景2: 动态时间戳**

```python
from datetime import datetime

template = ChatPromptTemplate.from_messages([
    ("system", "当前时间：{current_time}"),
    ("human", "{question}")
])

# 部分应用：动态生成时间
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

partial_template = template.partial(current_time=get_current_time)

# 每次调用都会获取最新时间
result = partial_template.invoke({"question": "现在几点？"})
```

**场景3: 环境配置**

```python
import os

template = ChatPromptTemplate.from_messages([
    ("system", "环境：{environment}，版本：{version}"),
    ("human", "{question}")
])

# 部分应用：从环境变量读取
partial_template = template.partial(
    environment=os.getenv("ENV", "development"),
    version=os.getenv("VERSION", "1.0.0")
)
```

### 4.3 部分应用的链式调用

```python
# 逐步部分应用
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，专长是{skill}，风格是{style}"),
    ("human", "{question}")
])

# 第一步：固定角色
step1 = template.partial(role="Python专家")

# 第二步：固定专长
step2 = step1.partial(skill="代码优化")

# 第三步：固定风格
step3 = step2.partial(style="简洁明了")

# 最终只需提供问题
result = step3.invoke({"question": "如何优化代码？"})
```

---

## 5. 模板组合

### 5.1 消息级别组合

**直接组合多个消息**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "问题1：{q1}"),
    ("ai", "回答1：{a1}"),
    ("human", "问题2：{q2}")
])

result = template.invoke({
    "role": "助手",
    "q1": "什么是Python？",
    "a1": "Python是一种编程语言",
    "q2": "它有什么特点？"
})
```

### 5.2 模板级别组合

**组合多个模板**:

```python
from langchain_core.prompts import ChatPromptTemplate

# 系统模板
system_template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}")
])

# 用户模板
user_template = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# 组合模板
combined_template = system_template + user_template

result = combined_template.invoke({
    "role": "助手",
    "question": "你好"
})
```

### 5.3 PipelinePromptTemplate

**管道式组合**:

```python
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

# 第一步：生成上下文
context_template = PromptTemplate.from_template(
    "主题：{topic}\n背景：{background}"
)

# 第二步：生成最终提示
final_template = PromptTemplate.from_template(
    "{context}\n\n问题：{question}"
)

# 组合成管道
pipeline = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
        ("context", context_template)
    ]
)

result = pipeline.invoke({
    "topic": "Python",
    "background": "编程语言",
    "question": "什么是装饰器？"
})
```

---

## 6. 模板序列化

### 6.1 保存模板

**保存为 JSON**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 保存为 JSON
template_json = template.to_json()
print(template_json)

# 保存到文件
with open("template.json", "w") as f:
    f.write(template_json)
```

**保存为 YAML**:

```python
# 保存为 YAML
template_yaml = template.to_yaml()
print(template_yaml)

# 保存到文件
with open("template.yaml", "w") as f:
    f.write(template_yaml)
```

### 6.2 加载模板

**从 JSON 加载**:

```python
from langchain_core.prompts import load_prompt

# 从文件加载
template = load_prompt("template.json")

# 从字符串加载
import json
template_dict = json.loads(template_json)
template = ChatPromptTemplate.from_dict(template_dict)
```

**从 YAML 加载**:

```python
# 从文件加载
template = load_prompt("template.yaml")
```

### 6.3 版本控制

**模板版本管理**:

```yaml
# prompts/assistant_v1.yaml
_type: chat_prompt
input_variables:
  - role
  - question
messages:
  - type: system
    content: "你是{role}"
  - type: human
    content: "{question}"
```

```yaml
# prompts/assistant_v2.yaml
_type: chat_prompt
input_variables:
  - role
  - skill
  - question
messages:
  - type: system
    content: "你是{role}，专长是{skill}"
  - type: human
    content: "{question}"
```

```python
# 加载不同版本
v1 = load_prompt("prompts/assistant_v1.yaml")
v2 = load_prompt("prompts/assistant_v2.yaml")

# A/B 测试
if use_v2:
    template = v2
else:
    template = v1
```

---

## 7. 安全最佳实践

### 7.1 模板注入风险（CVE-2025-65106）

**问题**: 允许用户定义模板可能导致信息泄露

```python
# ❌ 危险：直接使用用户输入作为模板
user_template = request.get("template")  # 用户提供
template = ChatPromptTemplate.from_template(user_template)
# 攻击者可以注入: "{__import__('os').environ}"
```

**解决方案1: 只允许变量值**

```python
# ✅ 安全：模板固定，只允许变量值
SAFE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{user_input}")
])

user_input = request.get("question")  # 只接受变量值
response = (SAFE_TEMPLATE | model).invoke({"user_input": user_input})
```

**解决方案2: 白名单模板**

```python
# ✅ 安全：预定义模板白名单
ALLOWED_TEMPLATES = {
    "assistant": ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ]),
    "expert": ChatPromptTemplate.from_messages([
        ("system", "你是专家"),
        ("human", "{question}")
    ])
}

template_name = request.get("template_name")
if template_name not in ALLOWED_TEMPLATES:
    raise ValueError("Invalid template")

template = ALLOWED_TEMPLATES[template_name]
```

### 7.2 输入验证

**验证变量值**:

```python
from pydantic import BaseModel, Field

class TemplateInput(BaseModel):
    role: str = Field(max_length=50)
    question: str = Field(max_length=500)

# 验证输入
try:
    validated_input = TemplateInput(**user_input)
    result = template.invoke(validated_input.dict())
except ValidationError as e:
    print(f"输入验证失败: {e}")
```

### 7.3 使用 f-string 而非 jinja2

```python
# ✅ 推荐：f-string 格式（安全）
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# ❌ 不推荐：jinja2 格式（有风险）
template = ChatPromptTemplate.from_messages([
    ("system", "你是{{ role }}"),
    ("human", "{{ question }}")
], template_format="jinja2")
```

---

## 8. 实际应用模式

### 8.1 基础模板模式

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

chain = template | model
response = chain.invoke({
    "role": "Python专家",
    "question": "什么是装饰器？"
})
```

### 8.2 可复用模板模式

```python
# 定义一次，多处使用
ASSISTANT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "你是友好的助手"),
    ("human", "{question}")
])

# 场景1：问答
qa_chain = ASSISTANT_TEMPLATE | model
qa_response = qa_chain.invoke({"question": "Python是什么？"})

# 场景2：翻译
translate_chain = ASSISTANT_TEMPLATE | model
translate_response = translate_chain.invoke({"question": "翻译：Hello"})
```

### 8.3 模板工厂模式

```python
def create_expert_template(domain: str, style: str) -> ChatPromptTemplate:
    """创建专家模板"""
    return ChatPromptTemplate.from_messages([
        ("system", f"你是{domain}专家，回答风格是{style}"),
        ("human", "{{question}}")
    ])

# 创建不同的专家
python_expert = create_expert_template("Python", "简洁专业")
js_expert = create_expert_template("JavaScript", "详细友好")

# 使用
python_chain = python_expert | model
js_chain = js_expert | model
```

### 8.4 模板继承模式

```python
# 基础模板
base_template = ChatPromptTemplate.from_messages([
    ("system", "你是助手")
])

# 扩展模板1：添加用户问题
qa_template = base_template + ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# 扩展模板2：添加上下文
context_template = base_template + ChatPromptTemplate.from_messages([
    ("human", "上下文：{context}\n问题：{question}")
])
```

---

## 9. 性能优化

### 9.1 模板缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_template(template_name: str) -> ChatPromptTemplate:
    """缓存模板，避免重复创建"""
    return load_prompt(f"prompts/{template_name}.yaml")

# 使用缓存的模板
template = get_template("assistant")  # 第一次：从文件加载
template = get_template("assistant")  # 第二次：从缓存返回
```

### 9.2 批量格式化

```python
# 批量生成消息
inputs = [
    {"role": "助手", "question": "问题1"},
    {"role": "助手", "question": "问题2"},
    {"role": "助手", "question": "问题3"}
]

# 使用 batch 方法
messages_list = template.batch(inputs)

# 批量调用模型
responses = model.batch(messages_list)
```

### 9.3 避免重复创建

```python
# ❌ 不好：每次都创建新模板
def ask(question: str):
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ])
    return (template | model).invoke({"question": question})

# ✅ 好：复用模板
TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{question}")
])
CHAIN = TEMPLATE | model

def ask(question: str):
    return CHAIN.invoke({"question": question})
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 理解 PromptTemplate 的函数式定义
- [ ] 区分 PromptTemplate 和 ChatPromptTemplate
- [ ] 使用变量替换机制
- [ ] 理解变量验证的价值
- [ ] 使用部分应用（partial）
- [ ] 组合多个模板
- [ ] 序列化和加载模板
- [ ] 了解模板注入风险（CVE-2025-65106）
- [ ] 实现基础的模板模式
- [ ] 优化模板性能

---

**下一步**: 阅读 `03_核心概念_04_ChatPromptTemplate.md` 深入学习对话模板的高级用法
