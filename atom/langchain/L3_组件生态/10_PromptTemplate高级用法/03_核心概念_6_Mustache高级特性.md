# 核心概念6：Mustache 高级特性

## 概述

Mustache 是 LangChain PromptTemplate 支持的三种模板格式之一，相比默认的 f-string 格式，Mustache 提供了更强大的嵌套变量、条件渲染和无转义输出等高级特性。本文深入解析 Mustache 模板的高级用法、Schema 生成机制以及在 AI Agent 开发中的实际应用。

**核心特性：**
- 嵌套变量访问（`{{obj.bar}}`）
- Section 变量（`{{#foo}} {{bar}} {{/foo}}`）
- 无转义输出（`{{{foo}}}`）
- 自动 Schema 生成（Pydantic 模型）
- 不支持模板验证（设计限制）

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:80-84, string.py:158-194]**

---

## 1. Mustache 模板格式基础

### 1.1 为什么选择 Mustache？

**f-string vs Mustache 对比：**

| 特性 | f-string | Mustache |
|------|----------|----------|
| 语法 | `{variable}` | `{{variable}}` |
| 嵌套访问 | ❌ 不支持 | ✅ `{{obj.field}}` |
| 条件渲染 | ❌ 不支持 | ✅ `{{#condition}}...{{/condition}}` |
| 无转义 | ❌ 不支持 | ✅ `{{{html}}}` |
| 模板验证 | ✅ 支持 | ❌ 不支持 |
| 性能 | 更快 | 稍慢 |

**适用场景：**
- 需要访问嵌套对象属性
- 需要条件渲染（Section 变量）
- 需要输出 HTML/JSON 等特殊字符
- 需要自动生成 Schema

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:102-114]**

### 1.2 创建 Mustache 模板

**基础用法：**

```python
from langchain_core.prompts import PromptTemplate

# 创建 Mustache 模板
template = "This is a {{foo}} test."
prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 格式化
result = prompt.format(foo="bar")
print(result)  # "This is a bar test."
```

**关键点：**
- 使用 `template_format="mustache"` 指定格式
- 变量使用双花括号 `{{variable}}`
- 自动提取变量名到 `input_variables`

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:122-125]**

---

## 2. 嵌套变量访问

### 2.1 对象属性访问

Mustache 支持使用点号访问嵌套对象的属性，这是 f-string 无法实现的强大特性。

**语法：** `{{object.property}}`

**示例：**

```python
from langchain_core.prompts import PromptTemplate

# 创建嵌套变量模板
template = """
User Profile:
- Name: {{user.name}}
- Email: {{user.email}}
- Role: {{user.role}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 传递嵌套对象
user_data = {
    "user": {
        "name": "Alice",
        "email": "alice@example.com",
        "role": "Admin"
    }
}

result = prompt.format(**user_data)
print(result)
```

**输出：**
```
User Profile:
- Name: Alice
- Email: alice@example.com
- Role: Admin
```

### 2.2 多层嵌套访问

**支持任意深度的嵌套：**

```python
template = """
Company: {{company.name}}
Department: {{company.department.name}}
Manager: {{company.department.manager.name}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

data = {
    "company": {
        "name": "TechCorp",
        "department": {
            "name": "Engineering",
            "manager": {
                "name": "Bob"
            }
        }
    }
}

result = prompt.format(**data)
```

**在 AI Agent 中的应用：**
- 访问复杂的上下文对象
- 处理结构化的知识库数据
- 渲染多层级的配置信息

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/string.py:158-194]**

---

## 3. Section 变量（条件渲染）

### 3.1 Section 基础语法

Section 变量允许根据变量的真假值进行条件渲染，类似于模板引擎中的 `if` 语句。

**语法：** `{{#variable}} content {{/variable}}`

**行为规则：**
- 如果 `variable` 为真值（非空、非 False、非 0），渲染 content
- 如果 `variable` 为假值，跳过 content
- 如果 `variable` 是列表，遍历列表并渲染每个元素

**示例 1：条件渲染**

```python
from langchain_core.prompts import PromptTemplate

template = """
{{#is_premium}}
Welcome, Premium User! You have access to advanced features.
{{/is_premium}}

{{#is_trial}}
You are on a trial plan. Upgrade to unlock more features.
{{/is_trial}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# Premium 用户
result1 = prompt.format(is_premium=True, is_trial=False)
print(result1)

# Trial 用户
result2 = prompt.format(is_premium=False, is_trial=True)
print(result2)
```

### 3.2 列表遍历

**Section 变量可以遍历列表：**

```python
template = """
Available Tools:
{{#tools}}
- {{name}}: {{description}}
{{/tools}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

data = {
    "tools": [
        {"name": "Calculator", "description": "Perform math calculations"},
        {"name": "Search", "description": "Search the web"},
        {"name": "Database", "description": "Query database"}
    ]
}

result = prompt.format(**data)
print(result)
```

**输出：**
```
Available Tools:
- Calculator: Perform math calculations
- Search: Search the web
- Database: Query database
```

### 3.3 Inverted Section（反向条件）

**语法：** `{{^variable}} content {{/variable}}`

**行为：** 当 `variable` 为假值时渲染 content

```python
template = """
{{#has_results}}
Found {{count}} results.
{{/has_results}}

{{^has_results}}
No results found. Try a different query.
{{/has_results}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 有结果
result1 = prompt.format(has_results=True, count=5)
print(result1)  # "Found 5 results."

# 无结果
result2 = prompt.format(has_results=False, count=0)
print(result2)  # "No results found. Try a different query."
```

**在 AI Agent 中的应用：**
- 根据检索结果动态调整 Prompt
- 根据用户权限显示不同内容
- 根据上下文长度决定是否包含历史记录

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/string.py:238-250]**

---

## 4. 无转义输出（No Escape）

### 4.1 HTML/JSON 输出

默认情况下，Mustache 会转义特殊字符（如 `<`, `>`, `&`）以防止 XSS 攻击。但在某些场景下，我们需要输出原始内容。

**语法：** `{{{variable}}}` （三个花括号）

**示例：**

```python
from langchain_core.prompts import PromptTemplate

template = """
Escaped: {{html_content}}
Unescaped: {{{html_content}}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

data = {
    "html_content": "<strong>Bold Text</strong>"
}

result = prompt.format(**data)
print(result)
```

**输出：**
```
Escaped: &lt;strong&gt;Bold Text&lt;/strong&gt;
Unescaped: <strong>Bold Text</strong>
```

### 4.2 JSON 输出场景

**在 AI Agent 中输出 JSON 示例：**

```python
template = """
Generate a response in the following JSON format:

{{{json_schema}}}

Your response:
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

json_schema = '''{
  "name": "string",
  "age": "number",
  "skills": ["string"]
}'''

result = prompt.format(json_schema=json_schema)
print(result)
```

**应用场景：**
- 在 Prompt 中嵌入 JSON Schema
- 输出 HTML 格式的示例
- 保留代码片段中的特殊字符

---

## 5. Mustache Schema 生成

### 5.1 自动生成 Pydantic 模型

LangChain 提供了 `mustache_schema()` 函数，可以从 Mustache 模板自动生成 Pydantic 模型，用于类型验证和 Schema 生成。

**源码分析：**

```python
# 来源: langchain_core/prompts/string.py:158-194
def mustache_schema(template: str) -> type[BaseModel]:
    """Get the variables from a mustache template.

    Args:
        template: The template string.

    Returns:
        The variables from the template as a Pydantic model.
    """
    fields = {}
    prefix: tuple[str, ...] = ()
    section_stack: list[tuple[str, ...]] = []

    # 解析 Mustache 模板的 token
    for type_, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type_ == "end":
            if section_stack:
                prefix = section_stack.pop()
        elif type_ in {"section", "inverted section"}:
            section_stack.append(prefix)
            prefix += tuple(key.split("."))
            fields[prefix] = False  # Section 变量
        elif type_ in {"variable", "no escape"}:
            fields[prefix + tuple(key.split("."))] = True  # 普通变量

    # 构建嵌套的 Pydantic 模型
    # ... (省略后续代码)
```

**关键特性：**
- 自动识别嵌套变量（`obj.field`）
- 区分 Section 变量和普通变量
- 生成类型安全的 Pydantic 模型

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/string.py:158-194]**

### 5.2 Schema 生成示例

```python
from langchain_core.prompts.string import mustache_schema

template = """
User: {{user.name}}
Email: {{user.email}}

{{#tools}}
Tool: {{name}}
{{/tools}}
"""

# 生成 Schema
schema = mustache_schema(template)
print(schema.schema())
```

**生成的 Schema（简化版）：**
```json
{
  "type": "object",
  "properties": {
    "user": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"}
      }
    },
    "tools": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"}
        }
      }
    }
  }
}
```

**应用场景：**
- 自动生成 API 文档
- 验证输入数据的完整性
- 为 LLM 提供结构化的输入 Schema

---

## 6. Mustache 的限制与注意事项

### 6.1 不支持模板验证

**源码限制：**

```python
# 来源: langchain_core/prompts/prompt.py:102-114
if values.get("validate_template"):
    if values["template_format"] == "mustache":
        msg = "Mustache templates cannot be validated."
        raise ValueError(msg)
```

**原因：**
- Mustache 的动态特性（Section 变量、嵌套访问）难以静态验证
- 变量可能在运行时动态生成
- 嵌套结构的复杂性

**解决方案：**
- 使用 `mustache_schema()` 生成 Schema 进行运行时验证
- 编写单元测试覆盖所有变量路径
- 使用 Pydantic 模型验证输入数据

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:102-114]**

### 6.2 性能考虑

**Mustache vs f-string 性能对比：**

| 操作 | f-string | Mustache |
|------|----------|----------|
| 简单变量替换 | ~1x | ~2-3x |
| 嵌套访问 | N/A | ~3-5x |
| Section 遍历 | N/A | ~5-10x |

**优化建议：**
- 简单场景优先使用 f-string
- 复杂嵌套场景使用 Mustache
- 避免在循环中频繁渲染 Mustache 模板
- 考虑缓存渲染结果

### 6.3 与其他模板格式的组合限制

**不能混用不同格式：**

```python
# ❌ 错误：不能组合不同格式的模板
prompt1 = PromptTemplate.from_template("Say {foo}", template_format="f-string")
prompt2 = PromptTemplate.from_template("Say {{bar}}", template_format="mustache")

# 这会抛出 ValueError
combined = prompt1 + prompt2  # ValueError: Cannot add templates of different formats
```

**[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:142-184]**

---

## 7. 实战应用场景

### 7.1 RAG 系统中的动态 Prompt

**场景：** 根据检索结果数量动态调整 Prompt

```python
from langchain_core.prompts import PromptTemplate

template = """
You are a helpful assistant. Answer the user's question based on the following context.

{{#has_context}}
Context:
{{#context_chunks}}
- {{content}}
{{/context_chunks}}
{{/has_context}}

{{^has_context}}
No relevant context found. Please answer based on your general knowledge.
{{/has_context}}

Question: {{question}}
Answer:
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 有检索结果
data_with_context = {
    "has_context": True,
    "context_chunks": [
        {"content": "LangChain is a framework for building LLM applications."},
        {"content": "It provides tools for prompt management, chains, and agents."}
    ],
    "question": "What is LangChain?"
}

result1 = prompt.format(**data_with_context)

# 无检索结果
data_without_context = {
    "has_context": False,
    "context_chunks": [],
    "question": "What is LangChain?"
}

result2 = prompt.format(**data_without_context)
```

### 7.2 Agent 工具调用模板

**场景：** 根据可用工具动态生成 Prompt

```python
template = """
You have access to the following tools:

{{#tools}}
{{name}}: {{description}}
Input Schema: {{{input_schema}}}
{{/tools}}

{{^tools}}
No tools available. Please answer directly.
{{/tools}}

User Query: {{query}}

Think step by step and decide which tool to use.
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

data = {
    "tools": [
        {
            "name": "search",
            "description": "Search the web for information",
            "input_schema": '{"query": "string"}'
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "input_schema": '{"expression": "string"}'
        }
    ],
    "query": "What is 25 * 37?"
}

result = prompt.format(**data)
```

### 7.3 多语言模板管理

**场景：** 根据用户语言动态选择内容

```python
template = """
{{#lang.en}}
Welcome! How can I help you today?
{{/lang.en}}

{{#lang.zh}}
欢迎！今天我能帮您什么？
{{/lang.zh}}

{{#lang.es}}
¡Bienvenido! ¿Cómo puedo ayudarte hoy?
{{/lang.es}}

User: {{user_input}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 中文用户
data_zh = {
    "lang": {"en": False, "zh": True, "es": False},
    "user_input": "你好"
}

result = prompt.format(**data_zh)
```

---

## 8. 与 LCEL 的集成

### 8.1 在 LCEL 链中使用 Mustache

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 创建 Mustache 模板
prompt = PromptTemplate.from_template(
    """
    {{#system_message}}
    System: {{content}}
    {{/system_message}}

    User: {{user_input}}
    """,
    template_format="mustache"
)

# 构建 LCEL 链
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

# 调用
result = chain.invoke({
    "system_message": {"content": "You are a helpful assistant."},
    "user_input": "Hello!"
})
```

### 8.2 与 RunnableParallel 组合

```python
from langchain_core.runnables import RunnableParallel

# 并行处理多个模板
parallel_chain = RunnableParallel(
    summary=prompt_summary | llm,
    details=prompt_details | llm
)

result = parallel_chain.invoke(data)
```

---

## 9. 最佳实践

### 9.1 何时使用 Mustache

**✅ 推荐使用：**
- 需要访问嵌套对象属性
- 需要条件渲染（Section 变量）
- 需要遍历列表
- 需要输出 HTML/JSON 等特殊字符
- 需要自动生成 Schema

**❌ 不推荐使用：**
- 简单的变量替换（用 f-string 更快）
- 需要模板验证（Mustache 不支持）
- 性能敏感的场景（f-string 更快）

### 9.2 代码组织建议

```python
# 将复杂的 Mustache 模板存储在单独的文件中
# templates/agent_prompt.mustache

from langchain_core.prompts import PromptTemplate

def load_mustache_template(file_path: str) -> PromptTemplate:
    """从文件加载 Mustache 模板"""
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return PromptTemplate.from_template(
        template,
        template_format="mustache"
    )

# 使用
prompt = load_mustache_template("templates/agent_prompt.mustache")
```

### 9.3 测试策略

```python
import pytest
from langchain_core.prompts import PromptTemplate

def test_mustache_nested_access():
    """测试嵌套变量访问"""
    template = "{{user.name}}"
    prompt = PromptTemplate.from_template(template, template_format="mustache")

    result = prompt.format(user={"name": "Alice"})
    assert result == "Alice"

def test_mustache_section():
    """测试 Section 变量"""
    template = "{{#show}}Content{{/show}}"
    prompt = PromptTemplate.from_template(template, template_format="mustache")

    result1 = prompt.format(show=True)
    assert "Content" in result1

    result2 = prompt.format(show=False)
    assert "Content" not in result2
```

---

## 10. 总结

### 核心要点

1. **嵌套访问**：`{{obj.field}}` 支持任意深度的嵌套
2. **条件渲染**：`{{#var}}...{{/var}}` 根据真假值渲染
3. **列表遍历**：Section 变量自动遍历列表
4. **无转义**：`{{{var}}}` 输出原始内容
5. **Schema 生成**：自动生成 Pydantic 模型
6. **不支持验证**：Mustache 模板无法静态验证

### 类比总结

| Mustache 特性 | 前端类比 | 日常生活类比 |
|---------------|----------|--------------|
| 嵌套访问 | `obj.prop.subprop` | 文件夹路径 `/folder/subfolder/file` |
| Section 变量 | `v-if` / `*ngIf` | 根据天气决定是否带伞 |
| 列表遍历 | `v-for` / `*ngFor` | 遍历购物清单逐项购买 |
| 无转义 | `v-html` / `[innerHTML]` | 原样复制粘贴（不修改格式） |
| Schema 生成 | TypeScript 类型推断 | 根据表格自动生成数据字典 |

### 下一步学习

- **核心概念7**：Template Validation - 模板验证机制
- **实战代码5**：Mustache 模板实战 - 完整的 RAG 应用示例
- **化骨绵掌**：深入理解 Mustache 解析器的实现原理

---

**文档版本：** v1.0
**最后更新：** 2026-02-26
**知识点：** PromptTemplate高级用法 - Mustache高级特性
**层级：** L3_组件生态
