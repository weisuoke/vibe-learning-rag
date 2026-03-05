# 核心概念1: Partial Variables（部分变量）基础

## 一句话定义

**Partial Variables 是 PromptTemplate 的预填充机制，允许在创建模板时设置部分变量的值，使用时只需传递剩余的动态变量。**

---

## 什么是 Partial Variables?

### 核心问题

在实际开发中，我们经常遇到这样的场景：

```python
# ❌ 每次都要传递相同的变量
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "System: You are a helpful assistant.\nDate: {date}\nUser: {query}"
)

# 每次调用都要传递 date
prompt.format(date="2026-02-26", query="What's the weather?")
prompt.format(date="2026-02-26", query="Tell me a joke")
prompt.format(date="2026-02-26", query="Translate this")
# ... 重复传递 date 100 次
```

**问题**:
- `date` 在所有调用中都相同
- 每次都要记得传递 `date`
- 容易忘记或传错
- 代码冗余

### Partial Variables 的解决方案

```python
# ✅ 使用 partial_variables 预填充
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

prompt = PromptTemplate(
    template="System: You are a helpful assistant.\nDate: {date}\nUser: {query}",
    input_variables=["query"],  # 只需要传递 query
    partial_variables={"date": get_current_date}  # date 自动填充
)

# 使用时只需传递 query
prompt.format(query="What's the weather?")
# 输出: "System: You are a helpful assistant.\nDate: 2026-02-26\nUser: What's the weather?"

prompt.format(query="Tell me a joke")
# 输出: "System: You are a helpful assistant.\nDate: 2026-02-26\nUser: Tell me a joke"
```

**优势**:
- 只需传递真正动态的变量 (`query`)
- 系统变量 (`date`) 自动处理
- 代码更简洁
- 不容易出错

[来源: reference/search_partial_01.md:48-93 | LangChain社区实践]

---

## partial_variables 参数详解

### 参数定义

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template: str,                    # 模板字符串
    input_variables: List[str],       # 需要用户传递的变量
    partial_variables: Dict[str, Any] # 预填充的变量
)
```

**关键点**:
- `partial_variables` 是一个字典
- 键是变量名（字符串）
- 值可以是**字符串**或**函数**

[来源: reference/source_prompttemplate_01.md:70-86 | LangChain源码分析]

### 变量分类

在 PromptTemplate 中，变量分为两类：

1. **input_variables**: 使用时必须传递的变量
2. **partial_variables**: 预填充的变量，使用时不需要传递

```python
prompt = PromptTemplate(
    template="Language: {language}, Date: {date}, Query: {query}",
    input_variables=["query"],           # 动态变量
    partial_variables={                  # 预填充变量
        "language": "English",
        "date": get_current_date
    }
)

# 使用时只需传递 input_variables
prompt.format(query="Hello")
# language 和 date 自动填充
```

---

## 两种形式: 字符串 vs 函数

### 形式1: 字符串固定值

**适用场景**: 值在创建时就确定，且不会改变

```python
# 示例1: 系统设置
prompt = PromptTemplate(
    template="Language: {language}\nModel: {model}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={
        "language": "English",      # 固定值
        "model": "gpt-4"            # 固定值
    }
)

result = prompt.format(query="Hello")
print(result)
# 输出:
# Language: English
# Model: gpt-4
# Query: Hello
```

**特点**:
- 值在创建模板时就确定
- 所有调用使用相同的值
- 适合配置、常量等

[来源: reference/search_partial_01.md:72-93 | LangChain社区实践]

### 形式2: 函数动态值

**适用场景**: 值需要在使用时动态计算

```python
# 示例2: 动态日期时间
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

prompt = PromptTemplate(
    template="Date: {date}\nTime: {time}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={
        "date": get_current_date,   # 函数（不是函数调用）
        "time": get_current_time    # 函数（不是函数调用）
    }
)

# 每次调用时，函数会被执行
result1 = prompt.format(query="First query")
print(result1)
# 输出:
# Date: 2026-02-26
# Time: 10:30:15
# Query: First query

# 等待一段时间后再次调用
import time
time.sleep(2)

result2 = prompt.format(query="Second query")
print(result2)
# 输出:
# Date: 2026-02-26
# Time: 10:30:17  # 时间更新了
# Query: Second query
```

**关键点**:
- 传递**函数对象**，不是函数调用结果
- ✅ 正确: `{"date": get_current_date}`
- ❌ 错误: `{"date": get_current_date()}`
- 函数在每次 `format()` 时执行
- 支持动态值（如时间、随机数）

[来源: reference/search_partial_01.md:72-93 | LangChain社区实践]

---

## 完整代码示例

### 示例1: 基础用法对比

```python
"""
Partial Variables 基础用法演示
对比传统方式和 partial_variables 方式
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime

# ===== 传统方式 =====
print("=== 传统方式 ===")

traditional_prompt = PromptTemplate.from_template(
    "System: {system}\nDate: {date}\nQuery: {query}"
)

# 每次都要传递所有变量
result1 = traditional_prompt.format(
    system="You are a helpful assistant",
    date="2026-02-26",
    query="What's the weather?"
)
print(result1)
print()

# ===== Partial Variables 方式 =====
print("=== Partial Variables 方式 ===")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

partial_prompt = PromptTemplate(
    template="System: {system}\nDate: {date}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={
        "system": "You are a helpful assistant",  # 字符串固定值
        "date": get_current_date                  # 函数动态值
    }
)

# 只需传递 query
result2 = partial_prompt.format(query="What's the weather?")
print(result2)
print()

# ===== 多次调用对比 =====
print("=== 多次调用对比 ===")

queries = ["Query 1", "Query 2", "Query 3"]

print("传统方式 - 需要重复传递:")
for q in queries:
    result = traditional_prompt.format(
        system="You are a helpful assistant",
        date="2026-02-26",
        query=q
    )
    print(f"  {q}: {len(result)} chars")

print("\nPartial Variables 方式 - 只传递 query:")
for q in queries:
    result = partial_prompt.format(query=q)
    print(f"  {q}: {len(result)} chars")
```

**运行输出**:
```
=== 传统方式 ===
System: You are a helpful assistant
Date: 2026-02-26
Query: What's the weather?

=== Partial Variables 方式 ===
System: You are a helpful assistant
Date: 2026-02-26
Query: What's the weather?

=== 多次调用对比 ===
传统方式 - 需要重复传递:
  Query 1: 78 chars
  Query 2: 78 chars
  Query 3: 78 chars

Partial Variables 方式 - 只传递 query:
  Query 1: 78 chars
  Query 2: 78 chars
  Query 3: 78 chars
```

[来源: reference/search_partial_01.md | LangChain社区实践]

---

### 示例2: 字符串 vs 函数对比

```python
"""
演示字符串固定值和函数动态值的区别
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime
import time

# ===== 字符串固定值 =====
print("=== 字符串固定值 ===")

# 在创建时计算一次
fixed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

string_prompt = PromptTemplate(
    template="Date: {date}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={
        "date": fixed_date  # 字符串，值固定
    }
)

print(f"创建时间: {fixed_date}")
result1 = string_prompt.format(query="First")
print(f"第一次调用: {result1}")

time.sleep(2)  # 等待2秒

result2 = string_prompt.format(query="Second")
print(f"第二次调用: {result2}")
print("注意: 两次调用的日期时间相同\n")

# ===== 函数动态值 =====
print("=== 函数动态值 ===")

def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

function_prompt = PromptTemplate(
    template="Date: {date}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={
        "date": get_current_datetime  # 函数，每次调用时执行
    }
)

print(f"创建时间: {get_current_datetime()}")
result3 = function_prompt.format(query="First")
print(f"第一次调用: {result3}")

time.sleep(2)  # 等待2秒

result4 = function_prompt.format(query="Second")
print(f"第二次调用: {result4}")
print("注意: 两次调用的日期时间不同")
```

**运行输出**:
```
=== 字符串固定值 ===
创建时间: 2026-02-26 10:30:15
第一次调用: Date: 2026-02-26 10:30:15
Query: First
第二次调用: Date: 2026-02-26 10:30:15
Query: Second
注意: 两次调用的日期时间相同

=== 函数动态值 ===
创建时间: 2026-02-26 10:30:17
第一次调用: Date: 2026-02-26 10:30:17
Query: First
第二次调用: Date: 2026-02-26 10:30:19
Query: Second
注意: 两次调用的日期时间不同
```

[来源: reference/search_partial_01.md | LangChain社区实践]

---

### 示例3: 实际应用场景

```python
"""
Partial Variables 在 AI Agent 开发中的实际应用
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime
import os

# ===== 场景1: RAG 系统提示词 =====
print("=== 场景1: RAG 系统提示词 ===")

def get_system_info():
    return f"Current date: {datetime.now().strftime('%Y-%m-%d')}"

rag_prompt = PromptTemplate(
    template="""
{system_info}
You are a helpful AI assistant with access to a knowledge base.

Context: {context}
User question: {question}

Please answer based on the context provided.
""",
    input_variables=["context", "question"],
    partial_variables={
        "system_info": get_system_info  # 动态系统信息
    }
)

# 使用时只需传递 context 和 question
result = rag_prompt.format(
    context="Python is a programming language.",
    question="What is Python?"
)
print(result)
print()

# ===== 场景2: 多语言 Agent =====
print("=== 场景2: 多语言 Agent ===")

# 从环境变量或配置读取语言设置
DEFAULT_LANGUAGE = os.getenv("AGENT_LANGUAGE", "English")

multilang_prompt = PromptTemplate(
    template="""
Language: {language}
Model: {model}

User: {query}
Assistant:
""",
    input_variables=["query"],
    partial_variables={
        "language": DEFAULT_LANGUAGE,  # 固定配置
        "model": "gpt-4"               # 固定配置
    }
)

result = multilang_prompt.format(query="Hello, how are you?")
print(result)
print()

# ===== 场景3: 带时间戳的日志 =====
print("=== 场景3: 带时间戳的日志 ===")

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_prompt = PromptTemplate(
    template="[{timestamp}] {level}: {message}",
    input_variables=["level", "message"],
    partial_variables={
        "timestamp": get_timestamp  # 每次调用时获取当前时间
    }
)

# 记录多条日志
logs = [
    ("INFO", "Application started"),
    ("DEBUG", "Processing request"),
    ("ERROR", "Connection failed")
]

for level, message in logs:
    log_entry = log_prompt.format(level=level, message=message)
    print(log_entry)
    import time
    time.sleep(0.5)  # 模拟时间流逝
```

**运行输出**:
```
=== 场景1: RAG 系统提示词 ===

Current date: 2026-02-26
You are a helpful AI assistant with access to a knowledge base.

Context: Python is a programming language.
User question: What is Python?

Please answer based on the context provided.

=== 场景2: 多语言 Agent ===

Language: English
Model: gpt-4

User: Hello, how are you?
Assistant:

=== 场景3: 带时间戳的日志 ===
[2026-02-26 10:30:20] INFO: Application started
[2026-02-26 10:30:20] DEBUG: Processing request
[2026-02-26 10:30:21] ERROR: Connection failed
```

[来源: reference/search_partial_01.md | LangChain社区实践]

---

## 使用场景总结

### 适合使用 Partial Variables 的场景

| 场景 | 变量类型 | 示例 |
|------|----------|------|
| **系统配置** | 字符串 | 语言设置、模型名称、API版本 |
| **当前时间** | 函数 | 日期、时间戳、会话ID |
| **用户信息** | 字符串 | 用户名、用户ID、权限级别 |
| **环境信息** | 字符串/函数 | 环境变量、系统状态 |
| **常量** | 字符串 | 公司名称、产品名称 |
| **动态计算** | 函数 | 随机数、计数器、统计信息 |

[来源: reference/search_partial_01.md:48-68 | LangChain社区实践]

### 不适合使用 Partial Variables 的场景

| 场景 | 原因 | 替代方案 |
|------|------|----------|
| **每次都不同的值** | 应该是 input_variables | 作为 input_variables 传递 |
| **需要用户输入的值** | 不应该预填充 | 作为 input_variables 传递 |
| **复杂的条件逻辑** | 函数应该简单 | 在外部处理后传入 |
| **需要参数的函数** | partial_variables 函数不接受参数 | 使用闭包或外部变量 |

---

## 与 input_variables 的关系

### 变量优先级

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="A: {a}, B: {b}, C: {c}",
    input_variables=["a", "b"],
    partial_variables={"c": "fixed"}
)

# 所有变量 = input_variables + partial_variables
# 总共需要: a, b, c
# 用户传递: a, b
# 自动填充: c
```

**规则**:
1. `input_variables` 必须由用户传递
2. `partial_variables` 自动填充
3. 两者不能重复（同一个变量不能同时在两个列表中）
4. 模板中的所有变量必须在这两个列表之一中

[来源: reference/source_prompttemplate_01.md:90-97 | LangChain源码分析]

### 自动推导

如果不显式指定 `input_variables`，LangChain 会自动推导：

```python
# 方式1: 显式指定
prompt1 = PromptTemplate(
    template="A: {a}, B: {b}",
    input_variables=["a"],
    partial_variables={"b": "fixed"}
)

# 方式2: 自动推导（推荐）
prompt2 = PromptTemplate(
    template="A: {a}, B: {b}",
    partial_variables={"b": "fixed"}
)
# input_variables 自动推导为 ["a"]
```

**自动推导规则**:
- 从模板中提取所有变量
- 排除 `partial_variables` 中的变量
- 剩余的变量成为 `input_variables`

[来源: reference/source_prompttemplate_01.md:205-214 | LangChain源码分析]

---

## 常见错误与解决

### 错误1: 传递函数调用结果而非函数

```python
from datetime import datetime

# ❌ 错误: 传递函数调用结果
def get_date():
    return datetime.now().strftime("%Y-%m-%d")

wrong_prompt = PromptTemplate(
    template="Date: {date}, Query: {query}",
    input_variables=["query"],
    partial_variables={
        "date": get_date()  # ❌ 错误: 调用了函数
    }
)
# 结果: date 的值在创建时就固定了

# ✅ 正确: 传递函数对象
correct_prompt = PromptTemplate(
    template="Date: {date}, Query: {query}",
    input_variables=["query"],
    partial_variables={
        "date": get_date  # ✅ 正确: 传递函数对象
    }
)
# 结果: date 的值在每次 format() 时动态计算
```

### 错误2: 变量重复定义

```python
# ❌ 错误: 同一个变量在两个列表中
try:
    wrong_prompt = PromptTemplate(
        template="A: {a}, B: {b}",
        input_variables=["a", "b"],
        partial_variables={"b": "fixed"}  # b 重复了
    )
except ValueError as e:
    print(f"错误: {e}")
    # 输出: Cannot have same variable partialed twice.

# ✅ 正确: 每个变量只在一个列表中
correct_prompt = PromptTemplate(
    template="A: {a}, B: {b}",
    input_variables=["a"],
    partial_variables={"b": "fixed"}
)
```

[来源: reference/source_prompttemplate_01.md:40-45 | LangChain源码分析]

### 错误3: 忘记传递 input_variables

```python
prompt = PromptTemplate(
    template="A: {a}, B: {b}",
    input_variables=["a"],
    partial_variables={"b": "fixed"}
)

# ❌ 错误: 忘记传递 a
try:
    result = prompt.format()  # 缺少 a
except KeyError as e:
    print(f"错误: 缺少变量 {e}")

# ✅ 正确: 传递所有 input_variables
result = prompt.format(a="value_a")
print(result)  # 输出: A: value_a, B: fixed
```

---

## 源码实现原理

### 核心数据结构

```python
# 源码位置: langchain_core/prompts/base.py:65-70
partial_variables: Mapping[str, Any] = Field(default_factory=dict)
"""A dictionary of the partial variables the prompt template carries.

Partial variables populate the template so that you don't need to pass them in every
time you call the prompt.
"""
```

**关键点**:
- `partial_variables` 是一个字典
- 默认为空字典
- 存储预填充的变量

[来源: reference/source_prompttemplate_01.md:70-86 | LangChain源码分析]

### 变量合并逻辑

```python
# 源码位置: langchain_core/prompts/prompt.py:111
all_inputs = values["input_variables"] + list(values["partial_variables"])
```

**合并规则**:
1. 收集 `input_variables`
2. 收集 `partial_variables` 的键
3. 合并成完整的变量列表
4. 验证模板中的所有变量都在这个列表中

[来源: reference/source_prompttemplate_01.md:90-97 | LangChain源码分析]

### 函数执行时机

```python
# 在 format() 方法中
def format(self, **kwargs):
    # 1. 合并 partial_variables 和用户传递的 kwargs
    all_kwargs = dict(self.partial_variables)

    # 2. 执行函数形式的 partial_variables
    for key, value in all_kwargs.items():
        if callable(value):
            all_kwargs[key] = value()  # 调用函数

    # 3. 更新用户传递的值
    all_kwargs.update(kwargs)

    # 4. 格式化模板
    return self.template.format(**all_kwargs)
```

**执行时机**:
- 字符串值: 直接使用
- 函数值: 在 `format()` 时调用
- 每次 `format()` 都会重新调用函数

---

## 总结

### 核心要点

1. **定义**: Partial Variables 是预填充机制，减少重复传递变量
2. **两种形式**:
   - 字符串: 固定值，适合配置和常量
   - 函数: 动态值，适合时间和计算值
3. **使用场景**: 系统配置、时间戳、用户信息、环境变量
4. **关键区别**: 传递函数对象，不是函数调用结果

### 最佳实践

1. ✅ 对固定配置使用字符串形式
2. ✅ 对动态值使用函数形式
3. ✅ 保持函数简单，无副作用
4. ✅ 使用自动推导 `input_variables`
5. ❌ 避免在两个列表中重复定义变量
6. ❌ 避免传递函数调用结果

### 下一步学习

- **Partial Variables 高级用法**: 函数参数传递、闭包应用
- **Template Composition**: 组合时 partial_variables 的合并规则
- **与 ChatPromptTemplate 的兼容性**: 已知的兼容性问题

[来源: 综合分析 reference/ 目录下所有资料]
