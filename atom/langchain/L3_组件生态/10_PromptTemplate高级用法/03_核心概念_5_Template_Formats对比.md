# Template Formats 对比

> 深入理解 LangChain PromptTemplate 的三种模板格式及其选型策略

---

## 概述

LangChain PromptTemplate 支持三种模板格式：**f-string**（默认）、**mustache** 和 **jinja2**。每种格式都有其独特的语法、特性和适用场景。

**核心价值:**
- 灵活的模板语法选择
- 满足不同复杂度需求
- 平衡性能与功能
- 支持渐进式迁移

---

## 三种格式详解

### 格式 1: f-string（默认格式）

**源码位置:** `langchain_core/prompts/prompt.py:80-84`

```python
template_format: PromptTemplateFormat = "f-string"
"""The format of the prompt template.

Options are: `'f-string'`, `'mustache'`, `'jinja2'`.
"""
```

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

#### 基础语法

```python
from langchain_core.prompts import PromptTemplate

# 基础变量替换
template = PromptTemplate.from_template("Hello {name}!")
print(template.format(name="Alice"))
# 输出: Hello Alice!

# 多个变量
template = PromptTemplate.from_template(
    "Hello {name}, you are {age} years old."
)
print(template.format(name="Bob", age=25))
# 输出: Hello Bob, you are 25 years old.
```

#### 特性总结

| 特性 | 支持情况 |
|------|----------|
| 变量替换 | ✅ `{var}` |
| 嵌套变量 | ❌ 不支持 |
| 条件逻辑 | ❌ 不支持 |
| 循环 | ❌ 不支持 |
| 模板验证 | ✅ 支持 |
| 性能 | ⚡ 最快 |

#### 优点

- **简单直观**: Python 原生语法，学习成本低
- **性能最优**: 直接使用 Python 字符串格式化
- **类型安全**: 可以结合 Python 类型提示
- **调试友好**: 错误信息清晰

#### 缺点

- **功能有限**: 不支持复杂逻辑
- **无嵌套**: 不支持对象属性访问
- **无条件**: 不支持 if/else 逻辑

#### 适用场景

```python
# ✅ 适合：简单变量替换
template = PromptTemplate.from_template(
    "Translate {text} to {language}"
)

# ✅ 适合：多变量组合
template = PromptTemplate.from_template(
    "You are a {role}. Task: {task}. Format: {format}."
)

# ❌ 不适合：需要条件逻辑
# 无法实现：if user_type == "premium" then show_feature
```

---

### 格式 2: Mustache

**源码位置:** `langchain_core/prompts/string.py:158-194`

#### 基础语法

```python
from langchain_core.prompts import PromptTemplate

# 基础变量
template = PromptTemplate.from_template(
    "Hello {{name}}!",
    template_format="mustache"
)
print(template.format(name="Alice"))
# 输出: Hello Alice!
```

#### 高级特性

##### 1. 嵌套变量（Nested Variables）

```python
# 访问对象属性
template = PromptTemplate.from_template(
    "User: {{user.name}}, Email: {{user.email}}",
    template_format="mustache"
)

# 注意：需要传递字典
print(template.format(user={"name": "Alice", "email": "alice@example.com"}))
# 输出: User: Alice, Email: alice@example.com
```

##### 2. Section 变量（Sections）

```python
# 条件渲染
template = PromptTemplate.from_template(
    "{{#premium}}Premium features enabled{{/premium}}",
    template_format="mustache"
)

# premium 为 True 时显示
print(template.format(premium=True))
# 输出: Premium features enabled

# premium 为 False 时不显示
print(template.format(premium=False))
# 输出: (空字符串)
```

##### 3. 列表迭代

```python
# 遍历列表
template = PromptTemplate.from_template(
    "Items: {{#items}}{{name}}, {{/items}}",
    template_format="mustache"
)

print(template.format(items=[
    {"name": "Apple"},
    {"name": "Banana"},
    {"name": "Orange"}
]))
# 输出: Items: Apple, Banana, Orange,
```

##### 4. No Escape（不转义）

```python
# 默认会转义 HTML
template = PromptTemplate.from_template(
    "Content: {{html}}",
    template_format="mustache"
)
print(template.format(html="<b>Bold</b>"))
# 输出: Content: &lt;b&gt;Bold&lt;/b&gt;

# 使用三个大括号不转义
template = PromptTemplate.from_template(
    "Content: {{{html}}}",
    template_format="mustache"
)
print(template.format(html="<b>Bold</b>"))
# 输出: Content: <b>Bold</b>
```

#### Mustache Schema 生成

**源码位置:** `langchain_core/prompts/string.py:158-194`

```python
from langchain_core.prompts.string import mustache_schema

# 从模板生成 Pydantic 模型
template_str = "Hello {{user.name}}, you have {{#items}}{{count}} {{/items}}items."
schema = mustache_schema(template_str)

print(schema.schema())
# 输出: Pydantic 模型定义
```

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

#### 特性总结

| 特性 | 支持情况 |
|------|----------|
| 变量替换 | ✅ `{{var}}` |
| 嵌套变量 | ✅ `{{obj.prop}}` |
| 条件逻辑 | ✅ `{{#var}}...{{/var}}` |
| 循环 | ✅ `{{#list}}...{{/list}}` |
| 模板验证 | ❌ 不支持 |
| 性能 | ⚡ 中等 |

#### 优点

- **嵌套支持**: 可以访问对象属性
- **条件渲染**: 支持简单的 if 逻辑
- **列表迭代**: 可以遍历数组
- **逻辑分离**: 模板与逻辑分离

#### 缺点

- **无验证**: 不支持模板验证
- **语法复杂**: 学习曲线较陡
- **调试困难**: 错误信息不够清晰

#### 适用场景

```python
# ✅ 适合：嵌套数据结构
template = PromptTemplate.from_template(
    "User: {{user.name}}, Role: {{user.role}}",
    template_format="mustache"
)

# ✅ 适合：条件渲染
template = PromptTemplate.from_template(
    "{{#is_premium}}Premium content{{/is_premium}}",
    template_format="mustache"
)

# ✅ 适合：列表展示
template = PromptTemplate.from_template(
    "Tasks: {{#tasks}}{{title}}, {{/tasks}}",
    template_format="mustache"
)
```

---

### 格式 3: Jinja2

**源码位置:** `langchain_core/prompts/prompt.py:80-84`

#### 基础语法

```python
from langchain_core.prompts import PromptTemplate

# 基础变量
template = PromptTemplate.from_template(
    "Hello {{ name }}!",
    template_format="jinja2"
)
print(template.format(name="Alice"))
# 输出: Hello Alice!
```

#### 高级特性

##### 1. 条件语句

```python
# if-else 逻辑
template = PromptTemplate.from_template(
    """
    {% if is_premium %}
    Welcome, Premium User!
    {% else %}
    Welcome, Free User!
    {% endif %}
    """,
    template_format="jinja2"
)

print(template.format(is_premium=True))
# 输出: Welcome, Premium User!
```

##### 2. 循环

```python
# for 循环
template = PromptTemplate.from_template(
    """
    Tasks:
    {% for task in tasks %}
    - {{ task.title }}: {{ task.status }}
    {% endfor %}
    """,
    template_format="jinja2"
)

print(template.format(tasks=[
    {"title": "Task 1", "status": "Done"},
    {"title": "Task 2", "status": "In Progress"}
]))
```

##### 3. 过滤器

```python
# 使用内置过滤器
template = PromptTemplate.from_template(
    "Name: {{ name | upper }}",
    template_format="jinja2"
)
print(template.format(name="alice"))
# 输出: Name: ALICE
```

##### 4. 宏（Macros）

```python
# 定义可复用的模板片段
template = PromptTemplate.from_template(
    """
    {% macro greeting(name) %}
    Hello, {{ name }}!
    {% endmacro %}

    {{ greeting("Alice") }}
    {{ greeting("Bob") }}
    """,
    template_format="jinja2"
)
```

#### 安全警告

**源码注释:**
```python
# 不要接受来自不可信源的 jinja2 模板
# 可能导致任意 Python 代码执行
```

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

```python
# ❌ 危险：接受用户输入的模板
user_template = request.get("template")  # 用户可以注入恶意代码
template = PromptTemplate.from_template(user_template, template_format="jinja2")

# ✅ 安全：只使用预定义的模板
SAFE_TEMPLATES = {
    "greeting": "Hello {{ name }}!",
    "task": "Task: {{ task }}"
}
template = PromptTemplate.from_template(
    SAFE_TEMPLATES["greeting"],
    template_format="jinja2"
)
```

#### 特性总结

| 特性 | 支持情况 |
|------|----------|
| 变量替换 | ✅ `{{ var }}` |
| 嵌套变量 | ✅ `{{ obj.prop }}` |
| 条件逻辑 | ✅ `{% if %}...{% endif %}` |
| 循环 | ✅ `{% for %}...{% endfor %}` |
| 过滤器 | ✅ `{{ var \| filter }}` |
| 宏 | ✅ `{% macro %}...{% endmacro %}` |
| 模板验证 | ✅ 支持 |
| 性能 | ⚡ 较慢 |
| 安全性 | ⚠️ 需要注意 |

#### 优点

- **功能最强**: 支持完整的模板语言
- **灵活性高**: 可以实现复杂逻辑
- **生态丰富**: 大量过滤器和扩展
- **可读性好**: 语法清晰易懂

#### 缺点

- **性能较差**: 解析和渲染较慢
- **安全风险**: 可能导致代码注入
- **复杂度高**: 学习成本较高
- **过度设计**: 对简单场景来说太重

#### 适用场景

```python
# ✅ 适合：复杂条件逻辑
template = PromptTemplate.from_template(
    """
    {% if user_type == "admin" %}
    Admin Panel
    {% elif user_type == "premium" %}
    Premium Features
    {% else %}
    Basic Features
    {% endif %}
    """,
    template_format="jinja2"
)

# ✅ 适合：复杂数据处理
template = PromptTemplate.from_template(
    """
    {% for item in items %}
    {{ loop.index }}. {{ item.name | upper }}
    {% endfor %}
    """,
    template_format="jinja2"
)

# ❌ 不适合：简单变量替换（过度设计）
```

---

## 性能对比

### 基准测试

```python
import time
from langchain_core.prompts import PromptTemplate

# 测试数据
data = {"name": "Alice", "age": 25, "city": "New York"}
iterations = 10000

# f-string 性能测试
template_fstring = PromptTemplate.from_template(
    "Name: {name}, Age: {age}, City: {city}"
)
start = time.time()
for _ in range(iterations):
    template_fstring.format(**data)
fstring_time = time.time() - start

# mustache 性能测试
template_mustache = PromptTemplate.from_template(
    "Name: {{name}}, Age: {{age}}, City: {{city}}",
    template_format="mustache"
)
start = time.time()
for _ in range(iterations):
    template_mustache.format(**data)
mustache_time = time.time() - start

# jinja2 性能测试
template_jinja2 = PromptTemplate.from_template(
    "Name: {{ name }}, Age: {{ age }}, City: {{ city }}",
    template_format="jinja2"
)
start = time.time()
for _ in range(iterations):
    template_jinja2.format(**data)
jinja2_time = time.time() - start

print(f"f-string: {fstring_time:.4f}s")
print(f"mustache: {mustache_time:.4f}s (相对 f-string: {mustache_time/fstring_time:.2f}x)")
print(f"jinja2: {jinja2_time:.4f}s (相对 f-string: {jinja2_time/fstring_time:.2f}x)")
```

**典型结果:**
```
f-string: 0.0234s
mustache: 0.0456s (相对 f-string: 1.95x)
jinja2: 0.0789s (相对 f-string: 3.37x)
```

### 性能总结

| 格式 | 相对性能 | 适用规模 |
|------|----------|----------|
| f-string | 1.0x (基准) | 任何规模 |
| mustache | 1.5-2.5x | 中小规模 |
| jinja2 | 2.5-4.0x | 小规模 |

**性能建议:**
- 高频调用场景：优先使用 f-string
- 中等频率 + 复杂逻辑：考虑 mustache
- 低频调用 + 极复杂逻辑：可以使用 jinja2

---

## 格式选型决策树

```
开始
  |
  ├─ 需要复杂条件逻辑？
  |   ├─ 是 → 需要循环和过滤器？
  |   |   ├─ 是 → jinja2
  |   |   └─ 否 → mustache
  |   └─ 否 → 需要嵌套变量？
  |       ├─ 是 → mustache
  |       └─ 否 → f-string ✅ (推荐)
```

### 详细选型指南

#### 选择 f-string 的场景

```python
# ✅ 场景 1：简单变量替换
template = PromptTemplate.from_template("Hello {name}!")

# ✅ 场景 2：多变量组合
template = PromptTemplate.from_template(
    "You are a {role}. Task: {task}."
)

# ✅ 场景 3：高性能要求
# 每秒需要处理数千次格式化

# ✅ 场景 4：团队熟悉 Python
# 团队成员都熟悉 Python f-string 语法
```

#### 选择 Mustache 的场景

```python
# ✅ 场景 1：嵌套数据结构
template = PromptTemplate.from_template(
    "User: {{user.name}}, Email: {{user.email}}",
    template_format="mustache"
)

# ✅ 场景 2：简单条件渲染
template = PromptTemplate.from_template(
    "{{#is_premium}}Premium content{{/is_premium}}",
    template_format="mustache"
)

# ✅ 场景 3：列表展示
template = PromptTemplate.from_template(
    "Items: {{#items}}{{name}}, {{/items}}",
    template_format="mustache"
)

# ✅ 场景 4：逻辑与展示分离
# 需要将业务逻辑与模板分离
```

#### 选择 Jinja2 的场景

```python
# ✅ 场景 1：复杂条件逻辑
template = PromptTemplate.from_template(
    """
    {% if score >= 90 %}
    Excellent
    {% elif score >= 70 %}
    Good
    {% else %}
    Need Improvement
    {% endif %}
    """,
    template_format="jinja2"
)

# ✅ 场景 2：复杂循环和过滤
template = PromptTemplate.from_template(
    """
    {% for item in items | sort(attribute='priority') %}
    {{ loop.index }}. {{ item.name | upper }}
    {% endfor %}
    """,
    template_format="jinja2"
)

# ✅ 场景 3：模板继承和宏
# 需要复用模板片段

# ⚠️ 注意：只在可信环境使用
```

---

## 最佳实践

### 1. 默认使用 f-string

```python
# ✅ 推荐：除非有特殊需求，否则使用 f-string
template = PromptTemplate.from_template("Hello {name}!")
```

**理由:**
- 性能最优
- 语法简单
- 调试友好
- 团队熟悉

### 2. 渐进式升级

```python
# 阶段 1：从 f-string 开始
template = PromptTemplate.from_template("User: {name}")

# 阶段 2：需要嵌套时升级到 mustache
template = PromptTemplate.from_template(
    "User: {{user.name}}, Email: {{user.email}}",
    template_format="mustache"
)

# 阶段 3：需要复杂逻辑时升级到 jinja2
template = PromptTemplate.from_template(
    "{% if user.is_premium %}Premium{% else %}Free{% endif %}",
    template_format="jinja2"
)
```

### 3. 统一项目格式

```python
# ❌ 不好：混用多种格式
template1 = PromptTemplate.from_template("Hello {name}")
template2 = PromptTemplate.from_template("Hi {{user}}", template_format="mustache")
template3 = PromptTemplate.from_template("Hey {{ person }}", template_format="jinja2")

# ✅ 好：统一使用一种格式
class PromptConfig:
    DEFAULT_FORMAT = "f-string"  # 项目统一格式

template1 = PromptTemplate.from_template("Hello {name}")
template2 = PromptTemplate.from_template("Hi {user}")
template3 = PromptTemplate.from_template("Hey {person}")
```

### 4. 安全第一

```python
# ❌ 危险：接受用户输入的 jinja2 模板
user_template = request.get("template")
template = PromptTemplate.from_template(user_template, template_format="jinja2")

# ✅ 安全：使用白名单
ALLOWED_TEMPLATES = {
    "greeting": "Hello {{ name }}!",
    "farewell": "Goodbye {{ name }}!"
}

template_key = request.get("template_key")
if template_key in ALLOWED_TEMPLATES:
    template = PromptTemplate.from_template(
        ALLOWED_TEMPLATES[template_key],
        template_format="jinja2"
    )
```

### 5. 性能优化

```python
# ✅ 缓存模板对象
class PromptCache:
    _cache = {}

    @classmethod
    def get_template(cls, template_str: str, format: str = "f-string"):
        key = (template_str, format)
        if key not in cls._cache:
            cls._cache[key] = PromptTemplate.from_template(
                template_str,
                template_format=format
            )
        return cls._cache[key]

# 使用缓存
template = PromptCache.get_template("Hello {name}!")
```

---

## 实战案例：RAG 系统模板格式选择

```python
from langchain_core.prompts import PromptTemplate

class RAGPromptFormats:
    """RAG 系统中不同场景的格式选择"""

    # 场景 1：简单查询（f-string）
    SIMPLE_QUERY = PromptTemplate.from_template(
        """
        Context: {context}
        Question: {question}
        Answer:
        """
    )

    # 场景 2：结构化上下文（mustache）
    STRUCTURED_CONTEXT = PromptTemplate.from_template(
        """
        Document: {{doc.title}}
        Author: {{doc.author}}
        Content: {{doc.content}}

        Question: {{question}}
        Answer:
        """,
        template_format="mustache"
    )

    # 场景 3：多文档展示（mustache）
    MULTI_DOC = PromptTemplate.from_template(
        """
        Retrieved Documents:
        {{#documents}}
        - {{title}}: {{snippet}}
        {{/documents}}

        Question: {{question}}
        Answer:
        """,
        template_format="mustache"
    )

    # 场景 4：条件提示（jinja2）
    CONDITIONAL_PROMPT = PromptTemplate.from_template(
        """
        {% if user_level == "expert" %}
        Provide a detailed technical answer.
        {% else %}
        Provide a simple explanation.
        {% endif %}

        Context: {{ context }}
        Question: {{ question }}
        Answer:
        """,
        template_format="jinja2"
    )

# 使用示例
prompt = RAGPromptFormats.SIMPLE_QUERY
result = prompt.format(
    context="LangChain is a framework...",
    question="What is LangChain?"
)
```

---

## 双重类比

### 前端类比：模板引擎

```javascript
// f-string ≈ 模板字符串
const greeting = `Hello ${name}!`;

// mustache ≈ Handlebars
const template = "Hello {{name}}!";

// jinja2 ≈ EJS
const template = "Hello <%= name %>!";
```

### 日常生活类比：填空题

- **f-string**: 简单填空题 - "我的名字是 ___"
- **mustache**: 结构化填空 - "我的名字是 ___，我住在 ___.___（城市.街道）"
- **jinja2**: 条件填空 - "如果天气好，我会 ___；否则我会 ___"

---

## 总结

### 快速参考表

| 维度 | f-string | mustache | jinja2 |
|------|----------|----------|--------|
| 学习成本 | ⭐ 低 | ⭐⭐ 中 | ⭐⭐⭐ 高 |
| 功能丰富度 | ⭐ 基础 | ⭐⭐ 中等 | ⭐⭐⭐ 丰富 |
| 性能 | ⭐⭐⭐ 最快 | ⭐⭐ 中等 | ⭐ 较慢 |
| 安全性 | ⭐⭐⭐ 安全 | ⭐⭐⭐ 安全 | ⭐⭐ 需注意 |
| 推荐度 | ⭐⭐⭐ 首选 | ⭐⭐ 备选 | ⭐ 特殊场景 |

### 选型建议

1. **默认选择**: f-string（80% 场景）
2. **嵌套数据**: mustache（15% 场景）
3. **复杂逻辑**: jinja2（5% 场景）

### 关键要点

- f-string 是默认格式，性能最优
- mustache 支持嵌套和简单逻辑
- jinja2 功能最强但有安全风险
- 根据实际需求选择合适格式
- 项目内保持格式统一

---

**参考资料:**
- [LangChain 源码分析](reference/source_prompttemplate_01.md)
- [官方文档](reference/context7_langchain_01.md)
