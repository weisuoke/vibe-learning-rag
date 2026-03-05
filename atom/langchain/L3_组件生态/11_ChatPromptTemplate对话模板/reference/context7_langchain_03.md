---
type: context7_documentation
library: langchain
version: v0.2
fetched_at: 2026-02-26
knowledge_point: 11_ChatPromptTemplate对话模板
context7_query: ChatPromptTemplate partial variables template composition multimodal images
---

# Context7 文档：LangChain ChatPromptTemplate 高级特性

## 文档来源
- 库名称：LangChain
- 版本：v0.2
- 官方文档链接：https://python.langchain.com/v0.2/

## 关键信息提取

### 1. Example: Partial Prompt Templating with ChatPromptTemplate

**来源**: https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate

**核心内容**: 使用 partial 方法预填充变量

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant named {name}."),
        ("human", "Hi I'm {user}"),
        ("ai", "Hi there, {user}, I'm {name}."),
        ("human", "{input}"),
    ]
)
template2 = template.partial(user="Lucy", name="R2D2")

template2.format_messages(input="hello")
```

**关键特性**:
- 预填充常用变量（如 name, user）
- 减少重复传递参数
- 创建可复用的模板

---

### 2. Create Partial Chat Prompt Template (Python)

**来源**: https://python.langchain.com/v0.2/api_reference/_modules/langchain_core/prompts/chat

**核心内容**: 返回预填充部分变量的新模板

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant named {name}.",),
    ("human", "Hi I'm {user}",),
    ("ai", "Hi there, {user}, I'm {name}.",),
    ("human", "{input}",),
])

partial_template = template.partial(name="Assistant")
# Now you can use partial_template and only need to provide the 'user' and 'input' variables.
```

**关键特性**:
- 固定部分变量
- 简化后续调用
- 创建特定场景的模板

---

### 3. Create and Partially Fill ChatPromptTemplate in Python

**来源**: https://python.langchain.com/v0.2/api_reference/langchain/agents/langchain.agents.schema.AgentScratchPadChatPromptTemplate

**核心内容**: 从消息列表创建模板并预填充变量

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant named {name}."),
        ("human", "Hi I'm {user}"),
        ("ai", "Hi there, {user}, I'm {name}."),
        ("human", "{input}"),
    ]
)
template2 = template.partial(user="Lucy", name="R2D2")

print(template2.format_messages(input="hello"))
```

**关键特性**:
- 支持多个变量预填充
- 保持模板结构不变
- 返回新的模板实例

---

### 4. Create Partial Chat Prompt Template

**来源**: https://python.langchain.com/v0.2/api_reference/core/prompts/langchain_core.prompts.structured.StructuredPrompt

**核心内容**: 创建可复用的预填充模板

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant named {name}."),
        ("human", "Hi I'm {user}"),
    ]
)

template2 = template.partial(user="Lucy", name="R2D2")

formatted_messages = template2.format_messages()
```

**关键特性**:
- 预填充所有变量
- 无需传递参数即可格式化
- 适用于固定场景

---

### 5. ChatPromptTemplate - partial API

**来源**: https://python.langchain.com/v0.2/api_reference/_modules/langchain_core/prompts/chat

**核心内容**: partial 方法的 API 文档

**API 规范**:
```
## POST /chat/prompt/partial

### Description
Returns a new ChatPromptTemplate with some input variables pre-filled. This is useful for creating reusable templates where certain parameters are fixed.

### Method
POST

### Endpoint
/chat/prompt/partial

### Parameters
#### Request Body
- **kwargs** (Any) - Required - Keyword arguments to use for filling in template variables. These should be a subset of the input variables of the original template.

### Request Example
```json
{
  "name": "Assistant"
}
```

### Response
#### Success Response (200)
- **ChatPromptTemplate** (object) - A new ChatPromptTemplate instance with partial variables set.

#### Response Example
```json
{
  "messages": [
    {
      "type": "SystemMessage",
      "content": "You are an AI assistant named Assistant."
    },
    {
      "type": "HumanMessage",
      "content": "Hi I'm {user}"
    },
    {
      "type": "AIMessage",
      "content": "Hi there, {user}, I'm Assistant."
    },
    {
      "type": "HumanMessage",
      "content": "{input}"
    }
  ],
  "partial_variables": {
    "name": "Assistant"
  }
}
```
```

**关键特性**:
- 返回新的 ChatPromptTemplate 实例
- 保留 partial_variables 信息
- 不修改原始模板

---

## Partial Variables 核心概念

### 1. 什么是 Partial Variables

**定义**: 预填充模板变量，避免每次调用时重复传递相同的值

**使用场景**:
- 共享系统设置（如 AI 名称、语言）
- 固定的上下文信息（如公司名称、产品名称）
- 减少重复代码

### 2. Partial Variables 的工作原理

**原理**:
1. 调用 `template.partial(**kwargs)` 创建新模板
2. 新模板的 `partial_variables` 包含预填充的值
3. 调用 `format_messages()` 时自动合并 partial_variables 和用户传递的变量

**示例**:
```python
# 原始模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are {name}."),
    ("human", "{input}")
])

# 预填充 name
template2 = template.partial(name="Assistant")

# 只需传递 input
template2.format_messages(input="Hello")
# 等价于
template.format_messages(name="Assistant", input="Hello")
```

### 3. Partial Variables 的优势

**优势**:
1. **减少重复**: 避免每次调用时传递相同的参数
2. **提高可读性**: 代码更简洁
3. **创建特定场景模板**: 为不同场景创建预配置的模板
4. **支持函数动态值**: 可以传递函数（如 `lambda: datetime.now()`）

### 4. Partial Variables 的限制

**限制**:
1. **兼容性问题**: 某些版本的 ChatPromptTemplate 可能有兼容性问题（参考 GitHub Issue #17560）
2. **不支持图片列表模板**: 图片模板不支持 partial_variables
3. **变量冲突**: 如果 partial_variables 和用户传递的变量有冲突，会抛出错误

---

## 实际应用场景

### 场景 1: 多语言模板管理

```python
# 基础模板
base_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Language: {language}"),
    ("human", "{input}")
])

# 中文模板
zh_template = base_template.partial(language="Chinese")

# 英文模板
en_template = base_template.partial(language="English")

# 使用
zh_template.format_messages(input="你好")
en_template.format_messages(input="Hello")
```

### 场景 2: 动态日期注入

```python
from datetime import datetime

template = ChatPromptTemplate.from_messages([
    ("system", "Today is {date}. You are a helpful assistant."),
    ("human", "{input}")
])

# 使用函数动态值
template_with_date = template.partial(date=lambda: datetime.now().strftime("%Y-%m-%d"))

# 每次调用时自动获取当前日期
template_with_date.format_messages(input="What's the date?")
```

### 场景 3: RAG 系统模板管理

```python
# 基础 RAG 模板
rag_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer based on the context."),
    ("human", "Context: {context}\nQuestion: {question}")
])

# 客服模板
customer_service_template = rag_template.partial(role="customer service assistant")

# 技术支持模板
tech_support_template = rag_template.partial(role="technical support specialist")

# 使用
customer_service_template.format_messages(
    context="...",
    question="How do I reset my password?"
)
```

### 场景 4: Agent 系统提示词设计

```python
# Agent 基础模板
agent_template = ChatPromptTemplate.from_messages([
    ("system", "You are {agent_name}. Tools: {tools}"),
    ("human", "{input}")
])

# 搜索 Agent
search_agent = agent_template.partial(
    agent_name="Search Agent",
    tools="web_search, wikipedia"
)

# 计算 Agent
calc_agent = agent_template.partial(
    agent_name="Calculator Agent",
    tools="calculator, math_solver"
)
```

---

## 与 PromptTemplate 的对比

| 特性 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| Partial Variables 支持 | ✅ 完全支持 | ⚠️ 部分版本有兼容性问题 |
| 使用方式 | `template.partial(**kwargs)` | `template.partial(**kwargs)` |
| 函数动态值 | ✅ 支持 | ✅ 支持 |
| 返回类型 | PromptTemplate | ChatPromptTemplate |
| 兼容性 | ✅ 稳定 | ⚠️ 需要测试 |

---

## 最佳实践

1. **优先使用字符串固定值**: 对于不变的值（如 AI 名称），使用字符串
2. **函数用于动态值**: 对于需要动态计算的值（如日期），使用函数
3. **测试兼容性**: 在使用前测试 ChatPromptTemplate 的 partial_variables 是否正常工作
4. **避免过度使用**: 只对真正需要共享的变量使用 partial
5. **命名空间前缀**: 使用前缀避免变量冲突（如 `system_date`, `user_date`）

---

## 常见问题

### Q1: ChatPromptTemplate 的 partial_variables 有兼容性问题吗？

**A**: 是的，某些版本中存在兼容性问题（参考 GitHub Issue #17560）。建议在使用前测试，或者使用 PromptTemplate 组合的方式来实现类似功能。

### Q2: 如何使用函数动态值？

**A**: 传递一个无参数的 lambda 函数：
```python
template.partial(date=lambda: datetime.now().strftime("%Y-%m-%d"))
```

### Q3: partial_variables 和直接传递变量有什么区别？

**A**:
- **partial_variables**: 在模板创建时固定，后续调用不需要传递
- **直接传递**: 每次调用时都需要传递

### Q4: 可以多次调用 partial 吗？

**A**: 可以，每次调用 partial 都会返回一个新的模板实例，可以链式调用：
```python
template.partial(name="Alice").partial(role="Assistant")
```

---

## 下一步调研方向

基于官方文档，需要进一步调研:
1. **模板组合**: 使用 `+` 操作符组合模板
2. **多模态支持**: 图片消息的实际使用
3. **社区实践**: Partial Variables 的实际应用案例
4. **兼容性问题**: GitHub Issue #17560 的详细情况
5. **性能优化**: 大规模模板管理的最佳实践
