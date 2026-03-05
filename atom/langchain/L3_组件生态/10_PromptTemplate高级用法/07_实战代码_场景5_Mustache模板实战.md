# 实战代码 - 场景5：Mustache模板实战

> **本文档展示 Mustache 模板格式在 LangChain PromptTemplate 中的高级用法**

[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py | LangChain 源码分析]

---

## 场景概述

Mustache 是一种无逻辑的模板语言，支持嵌套变量、Section 变量和 Schema 生成。在 LangChain 中，Mustache 模板提供了比 f-string 更强大的功能，特别适合处理复杂的嵌套数据结构。

**适用场景**：
- 需要处理嵌套对象的提示词
- 需要条件渲染的场景（Section 变量）
- 需要自动生成 Schema 验证的场景
- RAG 系统中的复杂元数据处理

---

## 双重类比

### 前端类比
**Mustache 模板 = Handlebars 模板引擎**
- 都使用 `{{}}` 语法
- 都支持嵌套对象访问
- 都支持条件渲染（Section）
- 都是无逻辑模板（Logic-less）

### 日常生活类比
**Mustache 模板 = 填空题模板**
- `{{name}}` 就像"请填写您的姓名：______"
- `{{user.email}}` 就像"请填写您的邮箱（在个人信息中）：______"
- `{{#items}} ... {{/items}}` 就像"如果有物品清单，请逐项填写"

---

## 核心特性

### 1. 基础变量替换

```python
from langchain_core.prompts import PromptTemplate

# 创建 Mustache 模板
template = "Hello {{name}}, welcome to {{platform}}!"
prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 格式化
result = prompt.format(name="Alice", platform="LangChain")
print(result)
# 输出: Hello Alice, welcome to LangChain!
```

[来源: reference/source_prompttemplate_01.md:122-125]

### 2. 嵌套变量访问

Mustache 支持使用点号访问嵌套对象：

```python
from langchain_core.prompts import PromptTemplate

# 嵌套变量模板
template = """
User Profile:
- Name: {{user.name}}
- Email: {{user.email}}
- Role: {{user.role}}
- Department: {{user.department.name}}
- Manager: {{user.department.manager}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 使用嵌套数据
user_data = {
    "user": {
        "name": "Alice",
        "email": "alice@example.com",
        "role": "Engineer",
        "department": {
            "name": "AI Research",
            "manager": "Bob"
        }
    }
}

result = prompt.format(**user_data)
print(result)
```

**输出**：
```
User Profile:
- Name: Alice
- Email: alice@example.com
- Role: Engineer
- Department: AI Research
- Manager: Bob
```

[来源: reference/source_prompttemplate_01.md:127-130]

### 3. Section 变量（条件渲染）

Section 变量允许根据数据是否存在来条件渲染内容：

```python
from langchain_core.prompts import PromptTemplate

# Section 变量模板
template = """
{{#user}}
Welcome back, {{name}}!
Your last login: {{last_login}}
{{/user}}

{{#items}}
Available items:
{{#items}}
- {{name}}: ${{price}}
{{/items}}
{{/items}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

# 场景1：用户已登录，有商品
data1 = {
    "user": {
        "name": "Alice",
        "last_login": "2026-02-25"
    },
    "items": [
        {"name": "Book", "price": 29.99},
        {"name": "Pen", "price": 2.99}
    ]
}

print("=== 场景1：完整数据 ===")
print(prompt.format(**data1))

# 场景2：用户未登录，无商品
data2 = {}

print("\n=== 场景2：空数据 ===")
print(prompt.format(**data2))
```

**输出**：
```
=== 场景1：完整数据 ===

Welcome back, Alice!
Your last login: 2026-02-25


Available items:

- Book: $29.99

- Pen: $2.99


=== 场景2：空数据 ===


```

[来源: reference/source_prompttemplate_01.md:127-130]

### 4. No Escape（不转义 HTML）

使用三个花括号 `{{{variable}}}` 可以避免 HTML 转义：

```python
from langchain_core.prompts import PromptTemplate

# No escape 模板
template = """
Escaped: {{html_content}}
Not escaped: {{{html_content}}}
"""

prompt = PromptTemplate.from_template(
    template,
    template_format="mustache"
)

result = prompt.format(html_content="<b>Bold Text</b>")
print(result)
```

**输出**：
```
Escaped: &lt;b&gt;Bold Text&lt;/b&gt;
Not escaped: <b>Bold Text</b>
```

[来源: reference/source_prompttemplate_01.md:127-130]

---

## 实战场景1：RAG 系统文档元数据模板

在 RAG 系统中，我们经常需要处理复杂的文档元数据。Mustache 模板非常适合这种场景。

```python
"""
RAG 系统文档元数据模板实战
演示：使用 Mustache 处理复杂的文档元数据
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime

# ===== 1. 定义文档元数据模板 =====
print("=== RAG 文档元数据模板 ===\n")

metadata_template = """
Based on the following document metadata, answer the user's question.

Document Information:
- Title: {{document.title}}
- Author: {{document.author}}
- Published: {{document.published_date}}
- Category: {{document.category}}
{{#document.tags}}
- Tags: {{#document.tags}}{{.}}, {{/document.tags}}
{{/document.tags}}

{{#document.source}}
Source Information:
- URL: {{document.source.url}}
- Type: {{document.source.type}}
- Last Updated: {{document.source.last_updated}}
{{/document.source}}

Content Summary:
{{document.summary}}

{{#document.sections}}
Key Sections:
{{#document.sections}}
- {{title}}: {{description}}
{{/document.sections}}
{{/document.sections}}

User Question: {{question}}

Please provide a detailed answer based on the document metadata above.
"""

prompt = PromptTemplate.from_template(
    metadata_template,
    template_format="mustache"
)

# ===== 2. 准备文档元数据 =====
document_metadata = {
    "document": {
        "title": "Introduction to LangChain",
        "author": "LangChain Team",
        "published_date": "2026-01-15",
        "category": "AI Development",
        "tags": ["LangChain", "AI", "RAG", "Prompt Engineering"],
        "source": {
            "url": "https://docs.langchain.com/intro",
            "type": "Official Documentation",
            "last_updated": "2026-02-20"
        },
        "summary": "This document provides a comprehensive introduction to LangChain, covering core concepts, LCEL expressions, and practical examples.",
        "sections": [
            {
                "title": "Core Abstractions",
                "description": "Runnable interface and LCEL basics"
            },
            {
                "title": "Components",
                "description": "PromptTemplate, ChatModel, OutputParser"
            },
            {
                "title": "Advanced Features",
                "description": "Agent systems and memory management"
            }
        ]
    },
    "question": "What are the main topics covered in this document?"
}

# ===== 3. 格式化并输出 =====
formatted_prompt = prompt.format(**document_metadata)
print(formatted_prompt)

# ===== 4. 与 LLM 集成 =====
print("\n=== 与 LLM 集成 ===\n")

# 注意：需要设置 OPENAI_API_KEY 环境变量
# from dotenv import load_dotenv
# load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# chain = prompt | llm
# response = chain.invoke(document_metadata)
# print(response.content)
```

**运行输出**：
```
=== RAG 文档元数据模板 ===

Based on the following document metadata, answer the user's question.

Document Information:
- Title: Introduction to LangChain
- Author: LangChain Team
- Published: 2026-01-15
- Category: AI Development

- Tags: LangChain, AI, RAG, Prompt Engineering,


Source Information:
- URL: https://docs.langchain.com/intro
- Type: Official Documentation
- Last Updated: 2026-02-20


Content Summary:
This document provides a comprehensive introduction to LangChain, covering core concepts, LCEL expressions, and practical examples.


Key Sections:

- Core Abstractions: Runnable interface and LCEL basics

- Components: PromptTemplate, ChatModel, OutputParser

- Advanced Features: Agent systems and memory management


User Question: What are the main topics covered in this document?

Please provide a detailed answer based on the document metadata above.
```

[来源: reference/search_composition_01.md | 动态提示实践]

---

## 实战场景2：Mustache Schema 生成与验证

Mustache 模板的一个强大特性是可以自动生成 Pydantic Schema，用于类型验证。

```python
"""
Mustache Schema 生成与验证
演示：自动生成 Pydantic 模型进行类型验证
"""

from langchain_core.prompts.string import mustache_schema
from pydantic import ValidationError

# ===== 1. 定义复杂的 Mustache 模板 =====
print("=== Mustache Schema 生成 ===\n")

complex_template = """
User: {{user.name}}
Email: {{user.email}}
{{#user.preferences}}
Preferences:
  Theme: {{theme}}
  Language: {{language}}
  {{#notifications}}
  Notifications: {{type}} - {{enabled}}
  {{/notifications}}
{{/user.preferences}}
"""

# ===== 2. 生成 Pydantic Schema =====
schema = mustache_schema(complex_template)
print(f"Generated Schema: {schema}")
print(f"Schema fields: {schema.model_fields}")

# ===== 3. 验证数据 =====
print("\n=== 数据验证 ===\n")

# 有效数据
valid_data = {
    "user": {
        "name": "Alice",
        "email": "alice@example.com",
        "preferences": {
            "theme": "dark",
            "language": "en",
            "notifications": [
                {"type": "email", "enabled": True},
                {"type": "sms", "enabled": False}
            ]
        }
    }
}

try:
    validated = schema(**valid_data)
    print("✓ 数据验证成功")
    print(f"Validated data: {validated}")
except ValidationError as e:
    print(f"✗ 数据验证失败: {e}")

# 无效数据（缺少必需字段）
invalid_data = {
    "user": {
        "name": "Bob"
        # 缺少 email 字段
    }
}

print("\n=== 无效数据测试 ===\n")
try:
    validated = schema(**invalid_data)
    print("✓ 数据验证成功")
except ValidationError as e:
    print(f"✗ 数据验证失败（预期）: {e}")
```

[来源: reference/source_prompttemplate_01.md:223-256]

---

## 实战场景3：多语言 RAG 系统模板

使用 Mustache 模板处理多语言 RAG 系统的提示词。

```python
"""
多语言 RAG 系统模板
演示：使用 Mustache 处理多语言场景
"""

from langchain_core.prompts import PromptTemplate

# ===== 1. 定义多语言模板 =====
print("=== 多语言 RAG 模板 ===\n")

multilang_template = """
{{#language.zh}}
基于以下文档回答问题：

文档标题：{{document.title}}
文档内容：{{document.content}}

用户问题：{{question}}

请用中文回答。
{{/language.zh}}

{{#language.en}}
Answer the question based on the following document:

Document Title: {{document.title}}
Document Content: {{document.content}}

User Question: {{question}}

Please answer in English.
{{/language.en}}

{{#language.ja}}
以下の文書に基づいて質問に答えてください：

文書タイトル：{{document.title}}
文書内容：{{document.content}}

ユーザーの質問：{{question}}

日本語で答えてください。
{{/language.ja}}
"""

prompt = PromptTemplate.from_template(
    multilang_template,
    template_format="mustache"
)

# ===== 2. 中文场景 =====
print("=== 中文场景 ===\n")

zh_data = {
    "language": {"zh": True},
    "document": {
        "title": "LangChain 入门指南",
        "content": "LangChain 是一个用于构建 AI 应用的框架..."
    },
    "question": "什么是 LangChain？"
}

print(prompt.format(**zh_data))

# ===== 3. 英文场景 =====
print("\n=== 英文场景 ===\n")

en_data = {
    "language": {"en": True},
    "document": {
        "title": "LangChain Introduction",
        "content": "LangChain is a framework for building AI applications..."
    },
    "question": "What is LangChain?"
}

print(prompt.format(**en_data))

# ===== 4. 日文场景 =====
print("\n=== 日文场景 ===\n")

ja_data = {
    "language": {"ja": True},
    "document": {
        "title": "LangChain入門ガイド",
        "content": "LangChainはAIアプリケーションを構築するためのフレームワークです..."
    },
    "question": "LangChainとは何ですか？"
}

print(prompt.format(**ja_data))
```

[来源: reference/search_composition_01.md | 动态提示与多语言支持]

---

## 关键要点总结

### Mustache 模板的优势

1. **嵌套对象支持**：使用 `{{obj.property}}` 访问嵌套数据
2. **条件渲染**：使用 `{{#section}} ... {{/section}}` 实现条件逻辑
3. **Schema 生成**：自动生成 Pydantic 模型进行类型验证
4. **无逻辑设计**：保持模板简洁，逻辑在数据层处理

### 适用场景

- ✅ 复杂的嵌套数据结构（如 RAG 文档元数据）
- ✅ 需要条件渲染的场景（如多语言支持）
- ✅ 需要类型验证的场景（Schema 生成）
- ✅ 需要避免 HTML 转义的场景（No escape）

### 不适用场景

- ❌ 简单的字符串替换（f-string 更简单）
- ❌ 需要复杂逻辑的场景（考虑 Jinja2）
- ❌ 需要模板验证的场景（Mustache 不支持验证）

[来源: reference/source_prompttemplate_01.md:154-173]

---

## 与其他模板格式对比

| 特性 | f-string | Mustache | Jinja2 |
|------|----------|----------|--------|
| 语法 | `{var}` | `{{var}}` | `{{ var }}` |
| 嵌套对象 | ❌ | ✅ | ✅ |
| 条件渲染 | ❌ | ✅ (Section) | ✅ (if/else) |
| 循环 | ❌ | ✅ (Section) | ✅ (for) |
| Schema 生成 | ❌ | ✅ | ❌ |
| 模板验证 | ✅ | ❌ | ✅ |
| 安全性 | ✅ | ✅ | ⚠️ (需要 Sandbox) |
| 性能 | 最快 | 中等 | 较慢 |

---

## 最佳实践

### 1. 数据结构设计

```python
# ✅ 推荐：清晰的嵌套结构
data = {
    "user": {
        "profile": {
            "name": "Alice",
            "email": "alice@example.com"
        },
        "settings": {
            "theme": "dark"
        }
    }
}

# ❌ 不推荐：扁平结构（无法利用 Mustache 优势）
data = {
    "user_name": "Alice",
    "user_email": "alice@example.com",
    "user_theme": "dark"
}
```

### 2. Section 变量使用

```python
# ✅ 推荐：使用 Section 进行条件渲染
template = """
{{#user}}
Welcome, {{name}}!
{{/user}}
{{^user}}
Please log in.
{{/user}}
"""

# ❌ 不推荐：在数据层处理条件逻辑
# 应该让模板处理条件渲染
```

### 3. Schema 验证

```python
# ✅ 推荐：使用 Schema 验证数据
from langchain_core.prompts.string import mustache_schema

schema = mustache_schema(template)
validated_data = schema(**data)  # 自动验证

# ❌ 不推荐：手动验证数据
# 容易遗漏字段或类型错误
```

---

## 常见问题

### Q1: Mustache 模板不支持验证怎么办？

**A**: 使用 `mustache_schema` 生成 Pydantic 模型进行验证：

```python
from langchain_core.prompts.string import mustache_schema

schema = mustache_schema(template)
validated_data = schema(**data)
```

[来源: reference/source_prompttemplate_01.md:154-173]

### Q2: 如何处理列表数据？

**A**: 使用 Section 变量：

```python
template = """
{{#items}}
- {{name}}: {{price}}
{{/items}}
"""
```

### Q3: Mustache 和 Jinja2 如何选择？

**A**:
- 简单嵌套 + 条件渲染 → Mustache
- 复杂逻辑 + 过滤器 → Jinja2
- 简单替换 → f-string

---

## 参考资源

- [LangChain 源码 - PromptTemplate](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/prompt.py)
- [Mustache 官方文档](https://mustache.github.io/)
- [LangChain 官方文档 - Prompt Templates](https://docs.langchain.com/)

---

**文档版本**: v1.0
**最后更新**: 2026-02-26
**知识点**: PromptTemplate高级用法
**场景**: Mustache模板实战
