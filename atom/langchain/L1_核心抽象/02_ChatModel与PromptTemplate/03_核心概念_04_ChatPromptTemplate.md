# 核心概念 04: ChatPromptTemplate

> **本文目标**: 深入掌握 ChatPromptTemplate 的高级用法和最佳实践

---

## 概述

**ChatPromptTemplate** 是 LangChain 中最常用的模板类型，专为 ChatModel 设计。它生成消息列表而非字符串，是构建现代对话应用的标准选择。

**核心定位**: ChatPromptTemplate 是消息列表的生成器，是 ChatModel 的最佳搭档。

---

## 1. ChatPromptTemplate 的本质

### 1.1 函数式定义

**ChatPromptTemplate 本质上是一个函数**:

```python
ChatPromptTemplate: Dict[str, Any] → List[BaseMessage]
```

用 Python 类型注解表示:

```python
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

class ChatPromptTemplate:
    def __call__(self, variables: Dict[str, Any]) -> List[BaseMessage]:
        """接收变量字典，返回消息列表"""
        pass
```

### 1.2 与 PromptTemplate 的区别

```python
# PromptTemplate - 生成字符串
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("你是{role}")
result = template.invoke({"role": "助手"})
# 输出: StringPromptValue → "你是助手"

# ChatPromptTemplate - 生成消息列表
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}")
])
result = template.invoke({"role": "助手"})
# 输出: ChatPromptValue → [SystemMessage(content="你是助手")]
```

**对比表**:

| 维度 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| **输出类型** | StringPromptValue | ChatPromptValue |
| **转换为** | 字符串 | 消息列表 |
| **适用于** | LLM | ChatModel |
| **推荐度** | 不推荐（旧） | 推荐（新） |

---

## 2. 创建 ChatPromptTemplate

### 2.1 from_messages() - 推荐方式

**最常用的创建方式**:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
```

**支持的消息类型简写**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "..."),    # SystemMessage
    ("human", "..."),     # HumanMessage
    ("ai", "..."),        # AIMessage
    ("assistant", "..."), # AIMessage (别名)
    ("user", "..."),      # HumanMessage (别名)
])
```

### 2.2 使用消息对象

**直接使用消息类**:

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是{role}"),
    HumanMessage(content="{question}")
])
```

### 2.3 from_template() - 单消息模板

**快速创建单消息模板**:

```python
# 创建单个 HumanMessage 的模板
template = ChatPromptTemplate.from_template("{question}")

# 等价于
template = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])
```

### 2.4 混合使用

**混合字符串和消息对象**:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
```

---

## 3. MessagesPlaceholder - 动态消息注入

### 3.1 什么是 MessagesPlaceholder？

**MessagesPlaceholder** 是一个特殊的占位符，用于注入不定数量的消息。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),  # 动态插槽
    ("human", "{question}")
])
```

**核心特性**:

1. **不定长度**: 可以注入任意数量的消息
2. **类型灵活**: 可以注入任何类型的消息
3. **可选性**: 可以设置为可选（optional=True）

### 3.2 基础用法

**注入对话历史**:

```python
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 使用时注入历史
history = [
    HumanMessage(content="我叫张三"),
    AIMessage(content="你好张三！"),
    HumanMessage(content="我喜欢Python"),
    AIMessage(content="Python很棒！")
]

messages = template.invoke({
    "history": history,
    "question": "我叫什么名字？"
})

# 生成的消息列表:
# [
#   SystemMessage(content="你是助手"),
#   HumanMessage(content="我叫张三"),
#   AIMessage(content="你好张三！"),
#   HumanMessage(content="我喜欢Python"),
#   AIMessage(content="Python很棒！"),
#   HumanMessage(content="我叫什么名字？")
# ]
```

### 3.3 可选的 MessagesPlaceholder

**设置为可选**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history", optional=True),  # 可选
    ("human", "{question}")
])

# 可以不提供 history
messages = template.invoke({"question": "你好"})
# 生成: [SystemMessage(...), HumanMessage(...)]

# 也可以提供 history
messages = template.invoke({
    "history": [HumanMessage("历史消息")],
    "question": "你好"
})
# 生成: [SystemMessage(...), HumanMessage("历史消息"), HumanMessage(...)]
```

### 3.4 多个 MessagesPlaceholder

**在一个模板中使用多个占位符**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="examples"),      # Few-shot 示例
    MessagesPlaceholder(variable_name="history"),       # 对话历史
    MessagesPlaceholder(variable_name="tool_results"),  # 工具结果
    ("human", "{question}")
])

messages = template.invoke({
    "examples": [
        HumanMessage("示例问题"),
        AIMessage("示例回答")
    ],
    "history": [
        HumanMessage("历史问题"),
        AIMessage("历史回答")
    ],
    "tool_results": [
        ToolMessage(content="工具结果", tool_call_id="call_123")
    ],
    "question": "新问题"
})
```

### 3.5 实际应用场景

**场景1: 对话历史管理**

```python
class ConversationManager:
    def __init__(self, model):
        self.model = model
        self.template = ChatPromptTemplate.from_messages([
            ("system", "你是助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        self.history = []

    def chat(self, question: str) -> str:
        # 生成消息
        chain = self.template | self.model
        response = chain.invoke({
            "history": self.history,
            "question": question
        })

        # 更新历史
        self.history.append(HumanMessage(content=question))
        self.history.append(response)

        return response.content
```

**场景2: Few-shot 学习**

```python
# 定义示例
examples = [
    HumanMessage(content="这个产品很好"),
    AIMessage(content="正面"),
    HumanMessage(content="这个产品很差"),
    AIMessage(content="负面")
]

template = ChatPromptTemplate.from_messages([
    ("system", "你是情感分类助手"),
    MessagesPlaceholder(variable_name="examples"),
    ("human", "{text}")
])

# 使用示例
chain = template | model
result = chain.invoke({
    "examples": examples,
    "text": "这个产品还不错"
})
```

**场景3: Agent 工具调用**

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手，可以调用工具"),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="tool_results", optional=True),
    ("human", "{question}")
])

# 第一轮：用户提问
messages = template.invoke({
    "history": [],
    "question": "北京的天气怎么样？"
})

# 第二轮：注入工具结果
messages = template.invoke({
    "history": [
        HumanMessage("北京的天气怎么样？"),
        AIMessage("我需要查询天气", tool_calls=[...])
    ],
    "tool_results": [
        ToolMessage(content="晴天，25度", tool_call_id="call_123")
    ],
    "question": ""  # 不需要新问题
})
```

---

## 4. 模板组合

### 4.1 使用 + 操作符

**组合多个模板**:

```python
# 系统模板
system_template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}")
])

# 用户模板
user_template = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# 组合
combined = system_template + user_template

messages = combined.invoke({
    "role": "助手",
    "question": "你好"
})
# 生成: [SystemMessage(...), HumanMessage(...)]
```

### 4.2 动态组合

**根据条件组合模板**:

```python
base_template = ChatPromptTemplate.from_messages([
    ("system", "你是助手")
])

# 根据需要添加上下文
if has_context:
    context_template = ChatPromptTemplate.from_messages([
        ("human", "上下文：{context}")
    ])
    template = base_template + context_template
else:
    template = base_template

# 添加用户问题
question_template = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])
template = template + question_template
```

### 4.3 模板继承

**创建模板层次结构**:

```python
# 基础模板
base = ChatPromptTemplate.from_messages([
    ("system", "你是助手")
])

# 专家模板（继承基础）
expert = base + ChatPromptTemplate.from_messages([
    ("system", "你的专长是{skill}")
])

# 对话模板（继承专家）
conversation = expert + ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
```

---

## 5. 高级特性

### 5.1 部分应用（Partial）

**预填充部分变量**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，专长是{skill}"),
    ("human", "{question}")
])

# 部分应用：固定角色和专长
partial_template = template.partial(
    role="Python专家",
    skill="代码优化"
)

# 使用时只需提供问题
messages = partial_template.invoke({"question": "如何优化代码？"})
```

**动态部分应用**:

```python
from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

template = ChatPromptTemplate.from_messages([
    ("system", "当前时间：{current_time}"),
    ("human", "{question}")
])

# 部分应用：动态时间
partial_template = template.partial(current_time=get_current_time)

# 每次调用都会获取最新时间
messages = partial_template.invoke({"question": "现在几点？"})
```

### 5.2 模板格式化选项

**f-string 格式（默认）**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
```

**jinja2 格式（不推荐）**:

```python
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

**安全警告**: jinja2 格式存在模板注入风险（CVE-2025-65106），推荐使用 f-string。

### 5.3 输入变量验证

**自动验证输入变量**:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 查看需要的变量
print(template.input_variables)  # ['role', 'question']

# 缺少变量会报错
try:
    template.invoke({"role": "助手"})  # 缺少 question
except KeyError as e:
    print(f"错误: {e}")
```

### 5.4 输出解析器集成

**与输出解析器组合**:

```python
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{question}")
])

# 组合模板、模型和解析器
chain = template | model | StrOutputParser()

# 直接返回字符串
result = chain.invoke({"question": "你好"})
print(type(result))  # <class 'str'>
```

---

## 6. 实际应用模式

### 6.1 基础对话模式

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

### 6.2 多轮对话模式

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

class ChatBot:
    def __init__(self, model):
        self.model = model
        self.template = template
        self.history = []

    def chat(self, question: str) -> str:
        chain = self.template | self.model
        response = chain.invoke({
            "history": self.history,
            "question": question
        })

        self.history.append(HumanMessage(content=question))
        self.history.append(response)

        return response.content
```

### 6.3 Few-shot 学习模式

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{task}助手"),
    MessagesPlaceholder(variable_name="examples"),
    ("human", "{input}")
])

# 情感分类示例
sentiment_examples = [
    HumanMessage("这个产品很好"),
    AIMessage("正面"),
    HumanMessage("这个产品很差"),
    AIMessage("负面")
]

chain = template | model
result = chain.invoke({
    "task": "情感分类",
    "examples": sentiment_examples,
    "input": "这个产品还不错"
})
```

### 6.4 上下文增强模式

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手，根据提供的上下文回答问题"),
    ("human", "上下文：{context}"),
    ("human", "问题：{question}")
])

# RAG 应用
def rag_query(question: str, documents: List[str]) -> str:
    context = "\n\n".join(documents)
    chain = template | model
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return response.content
```

### 6.5 角色扮演模式

```python
template = ChatPromptTemplate.from_messages([
    ("system", """你是{character}，具有以下特征：
    - 性格：{personality}
    - 说话风格：{style}
    - 专长：{expertise}
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}")
])

# 创建不同角色
sherlock = template.partial(
    character="夏洛克·福尔摩斯",
    personality="冷静、理性、观察力敏锐",
    style="简洁、逻辑性强",
    expertise="推理和侦探"
)

tony_stark = template.partial(
    character="托尼·斯塔克",
    personality="自信、幽默、天才",
    style="风趣、略带讽刺",
    expertise="科技和工程"
)
```

---

## 7. 性能优化

### 7.1 模板复用

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

### 7.2 批量处理

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{question}")
])

# 批量生成消息
inputs = [
    {"question": "问题1"},
    {"question": "问题2"},
    {"question": "问题3"}
]

# 使用 batch
chain = template | model
responses = chain.batch(inputs)
```

### 7.3 历史长度管理

```python
class ChatBot:
    MAX_HISTORY = 10  # 限制历史长度

    def __init__(self, model):
        self.model = model
        self.template = ChatPromptTemplate.from_messages([
            ("system", "你是助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        self.history = []

    def chat(self, question: str) -> str:
        # 限制历史长度
        history = self.history[-self.MAX_HISTORY:]

        chain = self.template | self.model
        response = chain.invoke({
            "history": history,
            "question": question
        })

        self.history.append(HumanMessage(content=question))
        self.history.append(response)

        return response.content
```

---

## 8. 最佳实践

### 8.1 模板设计原则

**✅ 推荐做法**:

```python
# 1. 清晰的角色定义
ChatPromptTemplate.from_messages([
    ("system", "你是Python专家，回答要简洁专业"),
    ("human", "{question}")
])

# 2. 使用 MessagesPlaceholder 管理历史
ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 3. 变量命名清晰
ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "问题：{user_question}\n上下文：{context}")
])
```

**❌ 不推荐做法**:

```python
# 1. 角色定义模糊
ChatPromptTemplate.from_messages([
    ("system", "你是助手"),  # 太笼统
    ("human", "{question}")
])

# 2. 硬编码历史
ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "历史问题1"),
    ("ai", "历史回答1"),
    ("human", "{question}")  # 不灵活
])

# 3. 变量命名不清
ChatPromptTemplate.from_messages([
    ("system", "你是{x}"),
    ("human", "{y}")  # x, y 是什么？
])
```

### 8.2 安全实践

**防止模板注入**:

```python
# ❌ 危险：允许用户定义模板
user_template = request.get("template")
template = ChatPromptTemplate.from_template(user_template)

# ✅ 安全：只允许预定义模板
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

### 8.3 测试实践

**单元测试模板**:

```python
import pytest
from langchain_core.prompts import ChatPromptTemplate

def test_template_variables():
    """测试模板变量"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是{role}"),
        ("human", "{question}")
    ])

    assert template.input_variables == ["role", "question"]

def test_template_output():
    """测试模板输出"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是{role}"),
        ("human", "{question}")
    ])

    messages = template.invoke({
        "role": "助手",
        "question": "你好"
    })

    assert len(messages) == 2
    assert messages[0].content == "你是助手"
    assert messages[1].content == "你好"

def test_template_missing_variable():
    """测试缺少变量"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是{role}"),
        ("human", "{question}")
    ])

    with pytest.raises(KeyError):
        template.invoke({"role": "助手"})  # 缺少 question
```

---

## 9. 2025-2026 新特性

### 9.1 结构化输出集成

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手，回答要包含答案和置信度"),
    ("human", "{question}")
])

# 与结构化输出组合
structured_model = model.with_structured_output(Answer)
chain = template | structured_model

result = chain.invoke({"question": "Python是什么？"})
print(result.answer)      # "Python是一种编程语言"
print(result.confidence)  # 0.95
```

### 9.2 多模态模板

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是图像分析助手"),
    ("human", [
        {"type": "text", "text": "{question}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ])
])

result = template.invoke({
    "question": "这张图片是什么？",
    "image_url": "https://example.com/image.jpg"
})
```

### 9.3 LangSmith 集成

```python
from langsmith import traceable

@traceable
def chat_with_template(question: str) -> str:
    """可追踪的对话函数"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是助手"),
        ("human", "{question}")
    ])

    chain = template | model
    response = chain.invoke({"question": question})
    return response.content

# 自动追踪到 LangSmith
result = chat_with_template("你好")
```

---

## 检查清单

完成本节学习后，你应该能够：

- [ ] 理解 ChatPromptTemplate 的本质
- [ ] 使用 from_messages() 创建模板
- [ ] 使用 MessagesPlaceholder 管理动态消息
- [ ] 组合多个模板
- [ ] 使用部分应用（partial）
- [ ] 实现多轮对话管理
- [ ] 实现 Few-shot 学习
- [ ] 避免模板注入风险
- [ ] 编写模板单元测试
- [ ] 应用 2025-2026 新特性

---

**下一步**: 阅读 `03_核心概念_05_消息格式化.md` 学习消息格式化的高级技巧
