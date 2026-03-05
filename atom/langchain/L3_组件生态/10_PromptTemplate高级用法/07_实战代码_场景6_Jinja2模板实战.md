# 实战代码 - 场景6：Jinja2模板实战

> **本文档展示 Jinja2 模板格式在 LangChain PromptTemplate 中的高级用法与安全实践**

[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py | LangChain 源码分析]

---

## 场景概述

Jinja2 是一个功能强大的模板引擎，支持变量、过滤器、循环、条件判断等复杂逻辑。在 LangChain 中，Jinja2 模板提供了最强大的功能，但也需要特别注意安全性。

**适用场景**：
- 需要复杂逻辑处理的提示词
- 需要使用过滤器转换数据
- 需要循环和条件判断
- 高级 RAG 系统的动态提示词生成

**安全警告**：
- ⚠️ 不要接受来自不可信源的 Jinja2 模板
- ⚠️ 可能导致任意 Python 代码执行
- ⚠️ 必须使用 SandboxedEnvironment

[来源: reference/source_prompttemplate_01.md:134-141]

---

## 双重类比

### 前端类比
**Jinja2 模板 = Vue/React 模板语法**
- 都支持变量插值 `{{ variable }}`
- 都支持条件渲染 `{% if %}`
- 都支持循环 `{% for %}`
- 都支持过滤器/管道 `{{ value | filter }}`

### 日常生活类比
**Jinja2 模板 = 智能表单生成器**
- `{{ name }}` 就像"自动填写姓名"
- `{% if premium %}` 就像"如果是会员，显示特权内容"
- `{% for item in items %}` 就像"逐项列出购物清单"
- `{{ price | round(2) }}` 就像"价格保留两位小数"

---

## 核心特性

### 1. 基础变量替换

```python
from langchain_core.prompts import PromptTemplate

# 创建 Jinja2 模板
template = "Hello {{ name }}, welcome to {{ platform }}!"
prompt = PromptTemplate.from_template(
    template,
    template_format="jinja2"
)

# 格式化
result = prompt.format(name="Alice", platform="LangChain")
print(result)
# 输出: Hello Alice, welcome to LangChain!
```

[来源: reference/source_prompttemplate_01.md:133-136]

### 2. 条件判断

Jinja2 支持 if/elif/else 条件判断：

```python
from langchain_core.prompts import PromptTemplate

# 条件判断模板
template = """
{% if user_type == "premium" %}
Welcome, Premium Member {{ name }}!
You have access to all features.
{% elif user_type == "standard" %}
Welcome, {{ name }}!
You have access to standard features.
{% else %}
Welcome, Guest!
Please sign up to access more features.
{% endif %}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="jinja2"
)

# 场景1：Premium 用户
print("=== Premium 用户 ===")
print(prompt.format(user_type="premium", name="Alice"))

# 场景2：Standard 用户
print("\n=== Standard 用户 ===")
print(prompt.format(user_type="standard", name="Bob"))

# 场景3：Guest 用户
print("\n=== Guest 用户 ===")
print(prompt.format(user_type="guest", name=""))
```

**输出**：
```
=== Premium 用户 ===

Welcome, Premium Member Alice!
You have access to all features.


=== Standard 用户 ===

Welcome, Bob!
You have access to standard features.


=== Guest 用户 ===

Welcome, Guest!
Please sign up to access more features.
```

### 3. 循环遍历

Jinja2 支持 for 循环遍历列表和字典：

```python
from langchain_core.prompts import PromptTemplate

# 循环遍历模板
template = """
Available Products:
{% for product in products %}
{{ loop.index }}. {{ product.name }} - ${{ product.price }}
   {% if product.in_stock %}✓ In Stock{% else %}✗ Out of Stock{% endif %}
{% endfor %}

Total: {{ products | length }} products
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="jinja2"
)

products = [
    {"name": "Laptop", "price": 999.99, "in_stock": True},
    {"name": "Mouse", "price": 29.99, "in_stock": True},
    {"name": "Keyboard", "price": 79.99, "in_stock": False}
]

result = prompt.format(products=products)
print(result)
```

**输出**：
```
Available Products:

1. Laptop - $999.99
   ✓ In Stock

2. Mouse - $29.99
   ✓ In Stock

3. Keyboard - $79.99
   ✗ Out of Stock


Total: 3 products
```

### 4. 过滤器（Filters）

Jinja2 提供了丰富的内置过滤器：

```python
from langchain_core.prompts import PromptTemplate

# 过滤器模板
template = """
Original: {{ text }}
Upper: {{ text | upper }}
Lower: {{ text | lower }}
Title: {{ text | title }}
Length: {{ text | length }}
First 10 chars: {{ text[:10] }}

Numbers: {{ numbers }}
Sum: {{ numbers | sum }}
Max: {{ numbers | max }}
Min: {{ numbers | min }}

Date: {{ date }}
Default: {{ missing_value | default("N/A") }}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="jinja2"
)

result = prompt.format(
    text="hello world",
    numbers=[1, 2, 3, 4, 5],
    date="2026-02-26"
)
print(result)
```

**输出**：
```
Original: hello world
Upper: HELLO WORLD
Lower: hello world
Title: Hello World
Length: 11
First 10 chars: hello worl

Numbers: [1, 2, 3, 4, 5]
Sum: 15
Max: 5
Min: 1

Date: 2026-02-26
Default: N/A
```

---

## 实战场景1：RAG 系统动态提示词生成

在 RAG 系统中，使用 Jinja2 生成复杂的动态提示词。

```python
"""
RAG 系统动态提示词生成
演示：使用 Jinja2 处理复杂的 RAG 场景
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ===== 1. 定义 RAG 提示词模板 =====
print("=== RAG 动态提示词模板 ===\n")

rag_template = """
You are a helpful AI assistant. Answer the user's question based on the provided context.

{% if context %}
Context Information:
{% for doc in context %}
---
Document {{ loop.index }}: {{ doc.title }}
Source: {{ doc.source }}
Relevance Score: {{ doc.score | round(2) }}

{{ doc.content }}
---
{% endfor %}
{% else %}
No context available. Please answer based on your general knowledge.
{% endif %}

{% if conversation_history %}
Previous Conversation:
{% for msg in conversation_history %}
{{ msg.role | upper }}: {{ msg.content }}
{% endfor %}
{% endif %}

User Question: {{ question }}

{% if answer_format == "bullet_points" %}
Please provide your answer in bullet points.
{% elif answer_format == "detailed" %}
Please provide a detailed explanation.
{% elif answer_format == "concise" %}
Please provide a concise answer (max 2 sentences).
{% endif %}

{% if language != "en" %}
Please answer in {{ language }}.
{% endif %}

Answer:
"""

prompt = PromptTemplate.from_template(
    rag_template,
    template_format="jinja2"
)

# ===== 2. 准备 RAG 数据 =====
context_docs = [
    {
        "title": "LangChain Introduction",
        "source": "https://docs.langchain.com/intro",
        "score": 0.95,
        "content": "LangChain is a framework for developing applications powered by language models."
    },
    {
        "title": "LCEL Guide",
        "source": "https://docs.langchain.com/lcel",
        "score": 0.87,
        "content": "LCEL (LangChain Expression Language) is a declarative way to compose chains."
    }
]

conversation_history = [
    {"role": "user", "content": "What is LangChain?"},
    {"role": "assistant", "content": "LangChain is a framework for building AI applications."}
]

# ===== 3. 场景1：详细回答 + 英文 =====
print("=== 场景1：详细回答 + 英文 ===\n")
formatted_prompt = prompt.format(
    context=context_docs,
    conversation_history=conversation_history,
    question="How does LCEL work?",
    answer_format="detailed",
    language="en"
)
print(formatted_prompt)

# ===== 4. 场景2：简洁回答 + 中文 =====
print("\n=== 场景2：简洁回答 + 中文 ===\n")
formatted_prompt = prompt.format(
    context=context_docs,
    conversation_history=None,
    question="什么是 LCEL？",
    answer_format="concise",
    language="Chinese"
)
print(formatted_prompt)

# ===== 5. 场景3：无上下文 =====
print("\n=== 场景3：无上下文 ===\n")
formatted_prompt = prompt.format(
    context=None,
    conversation_history=None,
    question="What is AI?",
    answer_format="bullet_points",
    language="en"
)
print(formatted_prompt)
```

[来源: reference/search_composition_01.md | 动态提示实践]

---

## 实战场景2：安全的 Jinja2 模板使用

演示如何安全地使用 Jinja2 模板，避免安全风险。

```python
"""
安全的 Jinja2 模板使用
演示：SandboxedEnvironment 的使用
"""

from jinja2 import Environment, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

# ===== 1. 不安全的模板（仅用于演示，不要在生产环境使用）=====
print("=== 不安全的模板示例（不要使用）===\n")

# ❌ 危险：可能执行任意代码
dangerous_template = """
{{ ''.__class__.__mro__[1].__subclasses__() }}
"""

# 不要运行这个！仅用于说明危险性

# ===== 2. 安全的模板（使用 SandboxedEnvironment）=====
print("=== 安全的模板（推荐）===\n")

# ✅ 安全：使用 SandboxedEnvironment
safe_env = SandboxedEnvironment(undefined=StrictUndefined)

safe_template_str = """
User: {{ user.name }}
Email: {{ user.email }}
Role: {{ user.role }}

{% if user.is_admin %}
Admin Panel: Accessible
{% else %}
Admin Panel: Not Accessible
{% endif %}
"""

safe_template = safe_env.from_string(safe_template_str)

user_data = {
    "user": {
        "name": "Alice",
        "email": "alice@example.com",
        "role": "Developer",
        "is_admin": False
    }
}

result = safe_template.render(**user_data)
print(result)

# ===== 3. LangChain 中的安全使用 =====
print("\n=== LangChain 中的安全使用 ===\n")

from langchain_core.prompts import PromptTemplate

# LangChain 内部使用 SandboxedEnvironment
safe_prompt_template = """
{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
"""

prompt = PromptTemplate.from_template(
    safe_prompt_template,
    template_format="jinja2"
)

items = [
    {"name": "Item 1", "value": 100},
    {"name": "Item 2", "value": 200}
]

result = prompt.format(items=items)
print(result)

# ===== 4. 输入验证 =====
print("\n=== 输入验证 ===\n")

def validate_template_input(data):
    """验证模板输入数据"""
    # 检查数据类型
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    # 检查是否包含危险字符
    dangerous_patterns = ["__class__", "__mro__", "__subclasses__", "eval", "exec"]

    def check_value(value):
        if isinstance(value, str):
            for pattern in dangerous_patterns:
                if pattern in value:
                    raise ValueError(f"Dangerous pattern detected: {pattern}")
        elif isinstance(value, dict):
            for v in value.values():
                check_value(v)
        elif isinstance(value, list):
            for v in value:
                check_value(v)

    check_value(data)
    return True

# 测试验证
try:
    safe_data = {"name": "Alice", "role": "Developer"}
    validate_template_input(safe_data)
    print("✓ 安全数据验证通过")
except ValueError as e:
    print(f"✗ 验证失败: {e}")

try:
    dangerous_data = {"name": "__class__"}
    validate_template_input(dangerous_data)
    print("✓ 数据验证通过")
except ValueError as e:
    print(f"✗ 验证失败（预期）: {e}")
```

[来源: reference/source_prompttemplate_01.md:134-141]

---

## 实战场景3：Agent 系统提示词模板

使用 Jinja2 为 Agent 系统生成复杂的提示词。

```python
"""
Agent 系统提示词模板
演示：使用 Jinja2 处理 Agent 场景
"""

from langchain_core.prompts import PromptTemplate

# ===== 1. 定义 Agent 提示词模板 =====
print("=== Agent 提示词模板 ===\n")

agent_template = """
You are an AI agent with access to the following tools:

{% for tool in tools %}
{{ loop.index }}. {{ tool.name }}
   Description: {{ tool.description }}
   Parameters: {{ tool.parameters | join(", ") }}
{% endfor %}

{% if task_history %}
Previous Tasks:
{% for task in task_history %}
- Task {{ loop.index }}: {{ task.description }}
  Status: {{ task.status }}
  {% if task.result %}Result: {{ task.result }}{% endif %}
{% endfor %}
{% endif %}

Current Task: {{ current_task }}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if max_steps %}
Maximum Steps: {{ max_steps }}
{% endif %}

Please think step by step and use the appropriate tools to complete the task.

Response Format:
1. Thought: [Your reasoning]
2. Action: [Tool name]
3. Action Input: [Tool parameters]
4. Observation: [Tool output]
5. Final Answer: [Your conclusion]
"""

prompt = PromptTemplate.from_template(
    agent_template,
    template_format="jinja2"
)

# ===== 2. 准备 Agent 数据 =====
tools = [
    {
        "name": "search",
        "description": "Search the web for information",
        "parameters": ["query"]
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": ["expression"]
    },
    {
        "name": "database_query",
        "description": "Query the database",
        "parameters": ["sql"]
    }
]

task_history = [
    {
        "description": "Search for LangChain documentation",
        "status": "completed",
        "result": "Found official documentation"
    },
    {
        "description": "Calculate total cost",
        "status": "completed",
        "result": "Total: $1,234.56"
    }
]

# ===== 3. 格式化并输出 =====
formatted_prompt = prompt.format(
    tools=tools,
    task_history=task_history,
    current_task="Find the latest version of LangChain and calculate the upgrade cost",
    constraints=[
        "Must use official sources only",
        "Budget limit: $5,000",
        "Complete within 5 steps"
    ],
    max_steps=5
)

print(formatted_prompt)
```

[来源: reference/search_composition_01.md | Agent 系统应用]

---

## 关键要点总结

### Jinja2 模板的优势

1. **强大的逻辑支持**：if/elif/else、for 循环、过滤器
2. **丰富的内置过滤器**：upper、lower、length、sum、max、min 等
3. **灵活的数据处理**：支持复杂的数据结构和转换
4. **可读性强**：语法清晰，易于维护

### 安全注意事项

- ⚠️ **永远不要接受不可信的模板**：可能导致代码执行
- ✅ **使用 SandboxedEnvironment**：LangChain 内部已使用
- ✅ **验证输入数据**：检查危险模式和字符
- ✅ **限制模板功能**：只使用必要的功能

[来源: reference/source_prompttemplate_01.md:134-141]

### 适用场景

- ✅ 需要复杂逻辑的提示词（条件、循环）
- ✅ 需要数据转换的场景（过滤器）
- ✅ Agent 系统的动态提示词
- ✅ 高级 RAG 系统的提示词管理

### 不适用场景

- ❌ 简单的字符串替换（f-string 更简单）
- ❌ 不可信的用户输入（安全风险）
- ❌ 性能敏感的场景（Jinja2 较慢）

---

## 与其他模板格式对比

| 特性 | f-string | Mustache | Jinja2 |
|------|----------|----------|--------|
| 条件判断 | ❌ | ✅ (Section) | ✅ (if/elif/else) |
| 循环 | ❌ | ✅ (Section) | ✅ (for/while) |
| 过滤器 | ❌ | ❌ | ✅ |
| 复杂逻辑 | ❌ | ❌ | ✅ |
| 安全性 | ✅ | ✅ | ⚠️ (需要 Sandbox) |
| 性能 | 最快 | 中等 | 较慢 |
| 学习曲线 | 最简单 | 简单 | 中等 |

---

## 最佳实践

### 1. 安全第一

```python
# ✅ 推荐：验证输入
def safe_format(template, data):
    validate_template_input(data)
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    return prompt.format(**data)

# ❌ 不推荐：直接使用未验证的输入
# prompt.format(**untrusted_data)
```

### 2. 使用过滤器简化逻辑

```python
# ✅ 推荐：使用过滤器
template = "Total: {{ prices | sum }}"

# ❌ 不推荐：在数据层计算
# total = sum(prices)
# template = "Total: {{ total }}"
```

### 3. 保持模板简洁

```python
# ✅ 推荐：简洁的模板
template = """
{% for item in items %}
- {{ item.name }}
{% endfor %}
"""

# ❌ 不推荐：过于复杂的模板
# 复杂逻辑应该在数据层处理
```

---

## 常见问题

### Q1: Jinja2 模板如何避免安全风险？

**A**:
1. 使用 LangChain 的 PromptTemplate（内部使用 SandboxedEnvironment）
2. 验证所有输入数据
3. 不接受不可信的模板字符串

### Q2: Jinja2 和 Mustache 如何选择？

**A**:
- 需要过滤器和复杂逻辑 → Jinja2
- 简单条件和循环 → Mustache
- 简单替换 → f-string

### Q3: Jinja2 模板性能如何优化？

**A**:
1. 缓存编译后的模板
2. 减少模板复杂度
3. 在数据层预处理数据
4. 考虑使用更简单的模板格式

---

## 参考资源

- [LangChain 源码 - PromptTemplate](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/prompt.py)
- [Jinja2 官方文档](https://jinja.palletsprojects.com/)
- [Jinja2 安全指南](https://jinja.palletsprojects.com/en/3.1.x/sandbox/)
- [LangChain 官方文档 - Prompt Templates](https://docs.langchain.com/)

---

**文档版本**: v1.0
**最后更新**: 2026-02-26
**知识点**: PromptTemplate高级用法
**场景**: Jinja2模板实战
