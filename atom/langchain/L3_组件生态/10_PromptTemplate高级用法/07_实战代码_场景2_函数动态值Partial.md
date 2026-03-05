# 实战代码 - 场景2:函数动态值 Partial

> 本文档展示如何使用函数动态值的 Partial Variables 实现延迟计算和运行时动态值注入

---

## 场景概述

**核心问题:** 某些变量的值需要在模板格式化时动态计算(如当前时间、随机数、系统状态),而不是在模板创建时固定。

**解决方案:** 使用函数作为 `partial_variables` 的值,实现延迟计算(Lazy Evaluation)。

**适用场景:**
- 当前日期时间
- 动态系统状态
- 运行时计算的值
- 需要每次调用都更新的变量

[来源: reference/search_partial_01.md | LangChain 官方文档]

---

## 双重类比

### 前端类比:Getter 函数

```javascript
// 前端:使用 getter 动态计算值
const config = {
    company: "Acme Inc",
    get timestamp() {
        return new Date().toISOString();
    }
};

// 每次访问都会重新计算
console.log(config.timestamp);  // 2026-02-26T10:00:00.000Z
// 等待1秒
console.log(config.timestamp);  // 2026-02-26T10:00:01.000Z
```

**LangChain 对应:**
```python
from datetime import datetime

def get_timestamp():
    return datetime.now().isoformat()

prompt = PromptTemplate(
    template="Time: {timestamp}, Query: {query}",
    input_variables=["query"],
    partial_variables={"timestamp": get_timestamp}
)

# 每次格式化都会调用函数
prompt.format(query="test1")  # Time: 2026-02-26T10:00:00...
# 等待1秒
prompt.format(query="test2")  # Time: 2026-02-26T10:00:01...
```

### 日常生活类比:实时天气预报

想象你在看天气预报:
- **固定值方式:** 早上看到的天气预报,全天都显示同样的内容
- **函数动态值方式:** 每次刷新都获取最新的天气数据

---

## 核心实现原理

### 1. 延迟计算机制

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# 定义动态值函数
def get_current_date():
    """每次调用都返回当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

# 方式1:构造函数中指定(传递函数对象,不是函数调用结果)
prompt = PromptTemplate(
    template="Date: {date}, Question: {question}",
    input_variables=["question"],
    partial_variables={"date": get_current_date}  # 注意:不是 get_current_date()
)

# 方式2:使用 partial() 方法
prompt = PromptTemplate.from_template("Date: {date}, Question: {question}")
prompt_partial = prompt.partial(date=get_current_date)
```

[来源: reference/search_partial_01.md:76-93]

### 2. 函数调用时机

**关键理解:**
- 函数在模板**格式化时**调用,而不是模板**创建时**
- 每次调用 `format()` 都会重新执行函数
- 这就是"延迟计算"(Lazy Evaluation)的含义

```python
from datetime import datetime
import time

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

prompt = PromptTemplate(
    template="Time: {time}, Message: {message}",
    input_variables=["message"],
    partial_variables={"time": get_timestamp}
)

# 第一次格式化
print(prompt.format(message="First"))   # Time: 10:00:00, Message: First

time.sleep(2)  # 等待2秒

# 第二次格式化(时间会更新)
print(prompt.format(message="Second"))  # Time: 10:00:02, Message: Second
```

---

## 完整实战代码

### 场景1:日志系统时间戳

```python
"""
场景1:日志系统时间戳
演示:使用函数动态值为每条日志添加精确时间戳
"""

import os
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ===== 1. 定义时间戳函数 =====
print("=== 1. 定义时间戳函数 ===")

def get_timestamp() -> str:
    """返回当前时间戳(ISO 8601 格式)"""
    return datetime.now().isoformat()

def get_date() -> str:
    """返回当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

def get_time() -> str:
    """返回当前时间"""
    return datetime.now().strftime("%H:%M:%S")

print("✅ 时间函数定义完成")
print()

# ===== 2. 创建带动态时间戳的日志模板 =====
print("=== 2. 创建带动态时间戳的日志模板 ===")

log_template = PromptTemplate(
    template="""[{timestamp}] {level}: {message}

Context: {context}
""",
    input_variables=["level", "message", "context"],
    partial_variables={"timestamp": get_timestamp}
)

print("✅ 日志模板创建成功")
print(f"需要传递的变量: {log_template.input_variables}")
print(f"动态变量: {list(log_template.partial_variables.keys())}")
print()

# ===== 3. 生成多条日志(观察时间戳变化) =====
print("=== 3. 生成多条日志 ===")

import time

logs = [
    {"level": "INFO", "message": "System started", "context": "Initialization"},
    {"level": "DEBUG", "message": "Loading configuration", "context": "Config module"},
    {"level": "WARNING", "message": "High memory usage", "context": "Memory monitor"},
]

for log_data in logs:
    log_entry = log_template.format(**log_data)
    print(log_entry)
    time.sleep(0.5)  # 模拟时间流逝

# ===== 4. 对比:固定时间 vs 动态时间 =====
print("=== 4. 对比:固定时间 vs 动态时间 ===")

# 固定时间(在创建时确定)
fixed_time = datetime.now().isoformat()
fixed_template = PromptTemplate(
    template="[{timestamp}] {message}",
    input_variables=["message"],
    partial_variables={"timestamp": fixed_time}  # 字符串固定值
)

# 动态时间(每次格式化时计算)
dynamic_template = PromptTemplate(
    template="[{timestamp}] {message}",
    input_variables=["message"],
    partial_variables={"timestamp": get_timestamp}  # 函数动态值
)

print("固定时间模板:")
print(fixed_template.format(message="Message 1"))
time.sleep(1)
print(fixed_template.format(message="Message 2"))
print()

print("动态时间模板:")
print(dynamic_template.format(message="Message 1"))
time.sleep(1)
print(dynamic_template.format(message="Message 2"))
print()

# ===== 5. 与 LLM 集成:带时间戳的对话 =====
print("=== 5. 与 LLM 集成:带时间戳的对话 ===")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

conversation_template = PromptTemplate(
    template="""Conversation at {timestamp}

User: {user_message}

Please provide a helpful response.
""",
    input_variables=["user_message"],
    partial_variables={"timestamp": get_timestamp}
)

chain = conversation_template | llm

# 模拟多轮对话
messages = [
    "What time is it?",
    "Tell me a joke",
]

for msg in messages:
    print(f"User: {msg}")
    response = chain.invoke({"user_message": msg})
    print(f"Assistant: {response.content[:100]}...")
    print()
    time.sleep(1)

print("✅ 场景1演示完成!")
```

**运行输出示例:**
```
=== 3. 生成多条日志 ===
[2026-02-26T10:00:00.123456] INFO: System started

Context: Initialization

[2026-02-26T10:00:00.623456] DEBUG: Loading configuration

Context: Config module

[2026-02-26T10:00:01.123456] WARNING: High memory usage

Context: Memory monitor

=== 4. 对比:固定时间 vs 动态时间 ===
固定时间模板:
[2026-02-26T10:00:00.000000] Message 1
[2026-02-26T10:00:00.000000] Message 2

动态时间模板:
[2026-02-26T10:00:00.000000] Message 1
[2026-02-26T10:00:01.000000] Message 2
```

---

### 场景2:RAG 系统动态上下文

```python
"""
场景2:RAG 系统动态上下文
演示:为 RAG 查询注入动态系统状态和统计信息
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime
import random

# ===== 1. 定义动态上下文函数 =====
print("=== 1. 定义动态上下文函数 ===")

# 模拟文档数据库
class DocumentDatabase:
    def __init__(self):
        self.total_docs = 1000
        self.queries_today = 0

    def get_stats(self) -> Dict[str, any]:
        """获取实时统计信息"""
        self.queries_today += 1
        return {
            "total_docs": self.total_docs,
            "queries_today": self.queries_today,
            "avg_response_time": round(random.uniform(0.1, 0.5), 2)
        }

db = DocumentDatabase()

def get_system_stats() -> str:
    """返回系统统计信息"""
    stats = db.get_stats()
    return f"Total Docs: {stats['total_docs']}, Queries Today: {stats['queries_today']}, Avg Response: {stats['avg_response_time']}s"

def get_current_datetime() -> str:
    """返回当前日期时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("✅ 动态上下文函数定义完成")
print()

# ===== 2. 创建 RAG 查询模板 =====
print("=== 2. 创建 RAG 查询模板 ===")

rag_template = PromptTemplate(
    template="""RAG System Query
Timestamp: {datetime}
System Stats: {stats}

User Query: {query}

Retrieved Context:
{context}

Please provide a comprehensive answer based on the retrieved context.
""",
    input_variables=["query", "context"],
    partial_variables={
        "datetime": get_current_datetime,
        "stats": get_system_stats
    }
)

print("✅ RAG 模板创建成功")
print()

# ===== 3. 模拟多次查询 =====
print("=== 3. 模拟多次查询 ===")

queries = [
    {
        "query": "What is machine learning?",
        "context": "Machine learning is a subset of AI..."
    },
    {
        "query": "How does neural network work?",
        "context": "Neural networks are computing systems..."
    },
    {
        "query": "What is deep learning?",
        "context": "Deep learning is a type of machine learning..."
    }
]

for i, query_data in enumerate(queries, 1):
    print(f"查询 {i}:")
    prompt = rag_template.format(**query_data)
    print(prompt[:200] + "...")
    print()
    time.sleep(0.5)

print("✅ 场景2演示完成!")
```

---

### 场景3:A/B 测试与实验追踪

```python
"""
场景3:A/B 测试与实验追踪
演示:为每次请求生成唯一的实验 ID 和追踪信息
"""

from langchain_core.prompts import PromptTemplate
import uuid
import random

# ===== 1. 定义实验追踪函数 =====
print("=== 1. 定义实验追踪函数 ===")

def generate_request_id() -> str:
    """生成唯一的请求 ID"""
    return str(uuid.uuid4())[:8]

def get_experiment_variant() -> str:
    """随机分配实验变体"""
    variants = ["control", "variant_a", "variant_b"]
    return random.choice(variants)

def get_session_info() -> str:
    """获取会话信息"""
    return f"session_{random.randint(1000, 9999)}"

print("✅ 实验追踪函数定义完成")
print()

# ===== 2. 创建实验追踪模板 =====
print("=== 2. 创建实验追踪模板 ===")

experiment_template = PromptTemplate(
    template="""Experiment Tracking
Request ID: {request_id}
Variant: {variant}
Session: {session}
Timestamp: {timestamp}

User Query: {query}

[Variant-specific instructions based on {variant}]
Please provide a response optimized for this experiment variant.
""",
    input_variables=["query"],
    partial_variables={
        "request_id": generate_request_id,
        "variant": get_experiment_variant,
        "session": get_session_info,
        "timestamp": get_timestamp
    }
)

print("✅ 实验追踪模板创建成功")
print()

# ===== 3. 模拟多次请求(观察动态值变化) =====
print("=== 3. 模拟多次请求 ===")

for i in range(5):
    print(f"请求 {i+1}:")
    prompt = experiment_template.format(query="What is AI?")
    # 只显示前几行
    lines = prompt.split('\n')[:6]
    print('\n'.join(lines))
    print()

print("✅ 场景3演示完成!")
```

---

### 场景4:带参数的动态函数

```python
"""
场景4:带参数的动态函数
演示:使用闭包或 lambda 创建带参数的动态函数
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta

# ===== 1. 使用闭包创建参数化函数 =====
print("=== 1. 使用闭包创建参数化函数 ===")

def create_date_formatter(format_string: str):
    """创建一个返回指定格式日期的函数"""
    def get_formatted_date():
        return datetime.now().strftime(format_string)
    return get_formatted_date

# 创建不同格式的日期函数
get_iso_date = create_date_formatter("%Y-%m-%d")
get_us_date = create_date_formatter("%m/%d/%Y")
get_full_datetime = create_date_formatter("%Y-%m-%d %H:%M:%S")

print("✅ 参数化函数创建完成")
print()

# ===== 2. 创建多语言日期模板 =====
print("=== 2. 创建多语言日期模板 ===")

# 英文模板(ISO 格式)
en_template = PromptTemplate(
    template="Date: {date}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={"date": get_iso_date}
)

# 美国模板(MM/DD/YYYY 格式)
us_template = PromptTemplate(
    template="Date: {date}\nQuery: {query}",
    input_variables=["query"],
    partial_variables={"date": get_us_date}
)

print("英文模板:")
print(en_template.format(query="test"))
print()

print("美国模板:")
print(us_template.format(query="test"))
print()

# ===== 3. 使用 lambda 创建简单动态函数 =====
print("=== 3. 使用 lambda 创建简单动态函数 ===")

# Lambda 函数:返回当前小时
get_hour = lambda: datetime.now().hour

# Lambda 函数:返回问候语
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

greeting_template = PromptTemplate(
    template="{greeting}! Current hour: {hour}\n\nUser: {message}",
    input_variables=["message"],
    partial_variables={
        "greeting": get_greeting,
        "hour": get_hour
    }
)

print(greeting_template.format(message="Hello!"))
print()

# ===== 4. 动态计算相对时间 =====
print("=== 4. 动态计算相对时间 ===")

def get_tomorrow():
    """返回明天的日期"""
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

def get_next_week():
    """返回下周的日期"""
    return (datetime.now() + timedelta(weeks=1)).strftime("%Y-%m-%d")

reminder_template = PromptTemplate(
    template="""Reminder System
Today: {today}
Tomorrow: {tomorrow}
Next Week: {next_week}

Task: {task}
""",
    input_variables=["task"],
    partial_variables={
        "today": lambda: datetime.now().strftime("%Y-%m-%d"),
        "tomorrow": get_tomorrow,
        "next_week": get_next_week
    }
)

print(reminder_template.format(task="Complete project report"))
print()

print("✅ 场景4演示完成!")
```

---

## 最佳实践

### 1. 何时使用函数动态值 Partial

**适用场景:**
- ✅ 当前日期时间
- ✅ 动态系统状态
- ✅ 运行时计算的值
- ✅ 每次调用都需要更新的变量
- ✅ 随机值或唯一标识符

**不适用场景:**
- ❌ 固定不变的配置信息
- ❌ 计算开销大的值(考虑缓存)
- ❌ 需要在多次调用间保持一致的值

[来源: reference/search_partial_01.md]

### 2. 性能考虑

```python
# ✅ 推荐:轻量级函数
def get_date():
    return datetime.now().strftime("%Y-%m-%d")

# ⚠️ 注意:避免耗时操作
def get_expensive_data():
    # 避免在这里进行数据库查询或 API 调用
    # 每次 format() 都会执行
    return expensive_database_query()

# ✅ 推荐:如果需要耗时操作,考虑缓存
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_data():
    return expensive_database_query()
```

### 3. 函数签名要求

```python
# ✅ 正确:无参数函数
def get_timestamp():
    return datetime.now().isoformat()

# ❌ 错误:带参数的函数(会报错)
def get_formatted_date(format_string):
    return datetime.now().strftime(format_string)

# ✅ 解决方案:使用闭包
def create_date_formatter(format_string):
    def get_date():
        return datetime.now().strftime(format_string)
    return get_date

get_iso_date = create_date_formatter("%Y-%m-%d")
```

### 4. 调试技巧

```python
# 添加日志查看函数调用时机
def get_timestamp_with_log():
    timestamp = datetime.now().isoformat()
    print(f"[DEBUG] get_timestamp called: {timestamp}")
    return timestamp

template = PromptTemplate(
    template="{timestamp}: {message}",
    input_variables=["message"],
    partial_variables={"timestamp": get_timestamp_with_log}
)

# 观察何时调用函数
print("Creating template...")  # 函数不会被调用
print("Formatting template...")
template.format(message="test")  # 函数在这里被调用
```

---

## 常见问题

### Q1: 函数何时被调用?

**答:** 函数在每次调用 `format()` 或 `invoke()` 时被调用,而不是在模板创建时。

```python
def get_time():
    print("Function called!")
    return datetime.now().strftime("%H:%M:%S")

template = PromptTemplate(
    template="{time}: {msg}",
    input_variables=["msg"],
    partial_variables={"time": get_time}
)

print("Template created")  # 不会打印 "Function called!"
template.format(msg="test")  # 打印 "Function called!"
```

### Q2: 可以传递带参数的函数吗?

**答:** 不可以直接传递。需要使用闭包或 lambda 包装。

```python
# ❌ 错误
def format_date(format_string):
    return datetime.now().strftime(format_string)

template = PromptTemplate(
    template="{date}",
    input_variables=[],
    partial_variables={"date": format_date}  # 会报错
)

# ✅ 正确:使用闭包
def create_formatter(format_string):
    def get_date():
        return datetime.now().strftime(format_string)
    return get_date

template = PromptTemplate(
    template="{date}",
    input_variables=[],
    partial_variables={"date": create_formatter("%Y-%m-%d")}
)
```

### Q3: 函数返回值类型有限制吗?

**答:** 必须返回字符串或可转换为字符串的类型。

```python
# ✅ 正确:返回字符串
def get_date():
    return datetime.now().strftime("%Y-%m-%d")

# ✅ 正确:返回数字(会自动转换为字符串)
def get_count():
    return 42

# ❌ 错误:返回复杂对象
def get_object():
    return {"date": datetime.now()}  # 格式化时可能出错
```

### Q4: 如何在函数中访问外部状态?

**答:** 使用闭包或类方法。

```python
# 方式1:使用闭包
counter = {"value": 0}

def get_counter():
    counter["value"] += 1
    return str(counter["value"])

# 方式2:使用类
class Counter:
    def __init__(self):
        self.value = 0

    def get_count(self):
        self.value += 1
        return str(self.value)

counter = Counter()
template = PromptTemplate(
    template="Count: {count}",
    input_variables=[],
    partial_variables={"count": counter.get_count}
)
```

---

## 总结

**函数动态值 Partial Variables 的核心价值:**

1. **延迟计算:** 在需要时才计算值,而不是提前固定
2. **动态更新:** 每次调用都获取最新值
3. **灵活性:** 支持复杂的运行时逻辑
4. **追踪能力:** 为每次请求生成唯一标识
5. **实验支持:** 轻松实现 A/B 测试和实验追踪

**关键要点:**
- 传递函数对象,不是函数调用结果
- 函数在 `format()` 时调用,不是模板创建时
- 函数必须无参数(使用闭包解决参数需求)
- 返回值必须是字符串或可转换为字符串的类型
- 适用于需要动态计算的场景

**性能提示:**
- 避免在函数中执行耗时操作
- 考虑使用缓存优化重复计算
- 轻量级函数性能影响可忽略

---

**参考资料:**
- [LangChain PromptTemplate 官方文档](https://reference.langchain.com/v0.3/python/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
- [来源: reference/search_partial_01.md]
- [来源: reference/source_prompttemplate_01.md]

**下一步学习:**
- 07_实战代码_场景3_基础模板组合.md - 学习如何组合多个模板
- 07_实战代码_场景4_复杂模板组合.md - 学习多层次模板组合
