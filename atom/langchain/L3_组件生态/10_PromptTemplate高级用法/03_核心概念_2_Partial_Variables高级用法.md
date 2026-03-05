# 核心概念2：Partial Variables 高级用法

> 本文档深入讲解 LangChain PromptTemplate 中 Partial Variables 的高级用法，包括函数动态值、延迟计算、兼容性问题和最佳实践。

---

## 什么是 Partial Variables？

**Partial Variables（部分变量）** 是 PromptTemplate 中的一种高级特性，允许你预填充模板中的部分变量，而将其他变量留待后续填充。

**核心价值**：
- 减少重复传递相同的变量
- 支持动态计算的变量（如当前日期）
- 提升模板复用性和灵活性

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

---

## 1. 函数动态值实现

### 1.1 基础概念

Partial Variables 支持两种形式：
1. **字符串固定值**：预设常量
2. **函数动态值**：运行时计算

函数动态值是 Partial Variables 的高级用法，允许你在模板调用时动态计算变量值。

[来源: reference/search_partial_01.md | LangChain OpenTutorial]

### 1.2 实现原理

当你将函数作为 partial_variables 的值时，LangChain 会在模板格式化时自动调用该函数获取值。

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# 定义动态值函数
def get_current_date():
    """返回当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

# 创建带函数动态值的模板
prompt = PromptTemplate(
    template="今天是 {date}，请回答：{question}",
    input_variables=["question"],
    partial_variables={"date": get_current_date}  # 注意：传递函数对象，不是调用结果
)

# 每次调用时都会重新计算日期
result1 = prompt.format(question="天气如何？")
print(result1)  # 今天是 2026-02-26，请回答：天气如何？
```

**关键点**：
- 传递函数对象（`get_current_date`），而不是函数调用结果（`get_current_date()`）
- 函数会在每次 `format()` 或 `invoke()` 时被调用
- 函数应该是无参数的（或有默认参数）

[来源: reference/search_partial_01.md | LangChain 旧版文档]

### 1.3 常见应用场景

#### 场景1：当前时间戳

```python
from datetime import datetime

def get_timestamp():
    return datetime.now().isoformat()

prompt = PromptTemplate(
    template="[{timestamp}] 用户问题：{query}\n请提供答案。",
    input_variables=["query"],
    partial_variables={"timestamp": get_timestamp}
)

# 每次调用都会生成新的时间戳
print(prompt.format(query="什么是 RAG？"))
# [2026-02-26T10:30:45.123456] 用户问题：什么是 RAG？
# 请提供答案。
```

#### 场景2：系统信息

```python
import platform

def get_system_info():
    return f"{platform.system()} {platform.release()}"

prompt = PromptTemplate(
    template="系统环境：{system}\n任务：{task}",
    input_variables=["task"],
    partial_variables={"system": get_system_info}
)
```

#### 场景3：随机种子

```python
import random

def get_random_seed():
    return str(random.randint(1000, 9999))

prompt = PromptTemplate(
    template="随机种子：{seed}\n生成内容：{content}",
    input_variables=["content"],
    partial_variables={"seed": get_random_seed}
)
```

[来源: reference/search_partial_01.md | Medium 文章]

---

## 2. 延迟计算与懒加载

### 2.1 延迟计算原理

函数动态值的核心优势是**延迟计算（Lazy Evaluation）**：
- 函数不会在模板创建时执行
- 只在模板格式化时才执行
- 每次格式化都会重新执行

**类比**：
- **前端类比**：类似 React 的 `useMemo` 或 Vue 的 `computed`，但每次都重新计算
- **日常生活类比**：像是"现做现卖"的餐厅，每次点餐都重新烹饪

### 2.2 代码示例

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import time

# 定义一个有副作用的函数
call_count = 0

def get_date_with_counter():
    global call_count
    call_count += 1
    return f"{datetime.now().strftime('%Y-%m-%d')} (调用次数: {call_count})"

# 创建模板
prompt = PromptTemplate(
    template="日期：{date}\n问题：{question}",
    input_variables=["question"],
    partial_variables={"date": get_date_with_counter}
)

print("=== 模板创建完成，函数尚未执行 ===")
print(f"当前调用次数：{call_count}")  # 0

print("\n=== 第一次格式化 ===")
result1 = prompt.format(question="问题1")
print(result1)
print(f"当前调用次数：{call_count}")  # 1

time.sleep(1)

print("\n=== 第二次格式化 ===")
result2 = prompt.format(question="问题2")
print(result2)
print(f"当前调用次数：{call_count}")  # 2
```

**输出**：
```
=== 模板创建完成，函数尚未执行 ===
当前调用次数：0

=== 第一次格式化 ===
日期：2026-02-26 (调用次数: 1)
问题：问题1
当前调用次数：1

=== 第二次格式化 ===
日期：2026-02-26 (调用次数: 2)
问题：问题2
当前调用次数：2
```

### 2.3 性能考虑

**优点**：
- 始终获取最新值
- 避免缓存过期问题
- 适合动态变化的数据

**缺点**：
- 每次调用都会执行函数
- 如果函数耗时较长，会影响性能
- 不适合需要缓存的场景

**最佳实践**：
```python
# ❌ 不推荐：耗时操作
def get_expensive_data():
    # 假设这是一个耗时的数据库查询
    import time
    time.sleep(2)
    return "expensive_data"

# ✅ 推荐：使用缓存
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_data():
    import time
    time.sleep(2)
    return "cached_data"

prompt = PromptTemplate(
    template="数据：{data}\n问题：{question}",
    input_variables=["question"],
    partial_variables={"data": get_cached_data}
)
```

[来源: reference/search_partial_01.md | LangChain Tutorials]

---

## 3. 兼容性问题

### 3.1 ChatPromptTemplate 兼容性

**重要警告**：`partial_variables` 在 `ChatPromptTemplate` 中存在兼容性问题。

[来源: reference/search_partial_01.md | GitHub Issue #17560]

#### 问题描述

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

# ❌ 这可能不会按预期工作
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "今天是 {date}"),
    ("user", "{question}")
])

# partial_variables 在某些版本中不生效
chat_prompt_partial = chat_prompt.partial(date=get_date)
```

#### 历史版本行为差异

| LangChain 版本 | partial_variables 行为 | 说明 |
|----------------|------------------------|------|
| 0.1.x | 部分支持 | 仅支持字符串固定值 |
| 0.2.x | 不稳定 | 函数动态值可能失效 |
| 0.3.x (2025+) | 改进中 | 官方正在修复 |

#### 解决方案

**方案1：使用 PromptTemplate 替代**

```python
from langchain_core.prompts import PromptTemplate

# ✅ 使用 PromptTemplate（完全支持）
prompt = PromptTemplate(
    template="今天是 {date}\n用户问题：{question}",
    input_variables=["question"],
    partial_variables={"date": get_date}
)
```

**方案2：手动注入变量**

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "今天是 {date}"),
    ("user", "{question}")
])

# ✅ 在调用时手动注入
def invoke_with_date(question: str):
    return chat_prompt.invoke({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "question": question
    })
```

**方案3：使用 RunnablePassthrough**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "今天是 {date}"),
    ("user", "{question}")
])

# ✅ 使用 LCEL 管道注入
chain = (
    RunnablePassthrough.assign(date=lambda _: datetime.now().strftime("%Y-%m-%d"))
    | chat_prompt
)
```

[来源: reference/search_partial_01.md | GitHub Issue #17560]

### 3.2 版本兼容性检查

```python
import langchain_core

def check_partial_support():
    """检查当前版本是否完全支持 partial_variables"""
    version = langchain_core.__version__
    major, minor, patch = map(int, version.split('.')[:3])

    if major == 0 and minor < 2:
        return "部分支持（仅字符串固定值）"
    elif major == 0 and minor == 2:
        return "不稳定（建议升级）"
    else:
        return "完全支持"

print(f"LangChain 版本：{langchain_core.__version__}")
print(f"Partial Variables 支持：{check_partial_support()}")
```

---

## 4. 高级模式与最佳实践

### 4.1 模式1：多层次 Partial

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_time():
    return datetime.now().strftime("%H:%M:%S")

# 创建基础模板
base_prompt = PromptTemplate(
    template="日期：{date}\n时间：{time}\n系统：{system}\n问题：{question}",
    input_variables=["system", "question"],
    partial_variables={
        "date": get_date,
        "time": get_time
    }
)

# 进一步 partial
system_prompt = base_prompt.partial(system="RAG 助手")

# 最终使用
result = system_prompt.format(question="什么是向量检索？")
print(result)
```

**输出**：
```
日期：2026-02-26
时间：10:30:45
系统：RAG 助手
问题：什么是向量检索？
```

### 4.2 模式2：条件动态值

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_greeting():
    """根据时间返回不同的问候语"""
    hour = datetime.now().hour
    if hour < 12:
        return "早上好"
    elif hour < 18:
        return "下午好"
    else:
        return "晚上好"

prompt = PromptTemplate(
    template="{greeting}！{question}",
    input_variables=["question"],
    partial_variables={"greeting": get_greeting}
)

# 不同时间调用会得到不同的问候语
print(prompt.format(question="今天天气如何？"))
```

### 4.3 模式3：环境感知模板

```python
import os
from langchain_core.prompts import PromptTemplate

def get_environment():
    """获取当前运行环境"""
    return os.getenv("ENV", "development")

def get_log_level():
    """根据环境返回日志级别"""
    env = os.getenv("ENV", "development")
    return "DEBUG" if env == "development" else "INFO"

prompt = PromptTemplate(
    template="[{env}] [{log_level}] {message}",
    input_variables=["message"],
    partial_variables={
        "env": get_environment,
        "log_level": get_log_level
    }
)

# 根据环境变量自动调整
print(prompt.format(message="系统启动"))
# [development] [DEBUG] 系统启动
```

### 4.4 模式4：模板层次结构

```python
from langchain_core.prompts import PromptTemplate

# 基础模板（公司级别）
company_template = PromptTemplate(
    template="公司：{company}\n部门：{department}\n{content}",
    input_variables=["department", "content"],
    partial_variables={"company": lambda: "AI 科技公司"}
)

# 部门模板（部门级别）
dept_template = company_template.partial(department="研发部")

# 项目模板（项目级别）
project_template = PromptTemplate(
    template=dept_template.template + "\n项目：{project}\n任务：{task}",
    input_variables=["task"],
    partial_variables={
        **dept_template.partial_variables,
        "project": lambda: "RAG 系统"
    }
)
```

[来源: reference/search_partial_01.md | LateNode 完整指南]

---

## 5. 实战应用场景

### 5.1 RAG 系统日志模板

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import uuid

def get_request_id():
    """生成唯一请求 ID"""
    return str(uuid.uuid4())[:8]

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().isoformat()

# RAG 查询日志模板
rag_log_template = PromptTemplate(
    template="""
[请求ID: {request_id}]
[时间: {timestamp}]
[查询]: {query}
[检索结果数]: {num_results}
[响应]: {response}
""",
    input_variables=["query", "num_results", "response"],
    partial_variables={
        "request_id": get_request_id,
        "timestamp": get_timestamp
    }
)

# 使用
log_entry = rag_log_template.format(
    query="什么是向量检索？",
    num_results=5,
    response="向量检索是..."
)
print(log_entry)
```

### 5.2 多语言翻译模板

```python
from langchain_core.prompts import PromptTemplate
import locale

def get_system_language():
    """获取系统语言"""
    lang, _ = locale.getdefaultlocale()
    return lang[:2]  # 返回语言代码（如 'zh', 'en'）

translation_template = PromptTemplate(
    template="""
源语言：{source_lang}
目标语言：{target_lang}
文本：{text}

请翻译上述文本。
""",
    input_variables=["text", "target_lang"],
    partial_variables={"source_lang": get_system_language}
)
```

### 5.3 Agent 系统提示词

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y年%m月%d日")

def get_available_tools():
    """动态获取可用工具列表"""
    # 实际应用中可能从配置或数据库读取
    return "搜索、计算器、天气查询"

agent_template = PromptTemplate(
    template="""
你是一个智能助手。

当前日期：{date}
可用工具：{tools}

用户问题：{question}

请分析问题并选择合适的工具来回答。
""",
    input_variables=["question"],
    partial_variables={
        "date": get_current_date,
        "tools": get_available_tools
    }
)
```

[来源: reference/search_partial_01.md | Dynamic Prompts with LangChain]

---

## 6. 性能优化技巧

### 6.1 缓存策略

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=1)
def get_cached_date():
    """缓存日期（避免频繁调用）"""
    return datetime.now().strftime("%Y-%m-%d")

# 每天清除缓存
def clear_date_cache_daily():
    import schedule
    schedule.every().day.at("00:00").do(get_cached_date.cache_clear)
```

### 6.2 避免重复计算

```python
# ❌ 不推荐：每次都重新计算
def get_expensive_config():
    # 假设这是一个耗时的配置加载
    import json
    with open("config.json") as f:
        return json.load(f)

# ✅ 推荐：使用模块级缓存
_config_cache = None

def get_config():
    global _config_cache
    if _config_cache is None:
        import json
        with open("config.json") as f:
            _config_cache = json.load(f)
    return _config_cache
```

### 6.3 异步支持

```python
from langchain_core.prompts import PromptTemplate
import asyncio

async def get_async_data():
    """异步获取数据"""
    await asyncio.sleep(0.1)  # 模拟异步操作
    return "async_data"

# 注意：LangChain 的 partial_variables 不直接支持异步函数
# 需要在外部处理异步逻辑
async def format_with_async():
    data = await get_async_data()
    prompt = PromptTemplate(
        template="数据：{data}\n问题：{question}",
        input_variables=["question"],
        partial_variables={"data": lambda: data}
    )
    return prompt.format(question="测试")
```

---

## 7. 调试与故障排查

### 7.1 调试技巧

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_date_with_debug():
    """带调试信息的日期函数"""
    date = datetime.now().strftime("%Y-%m-%d")
    print(f"[DEBUG] get_date_with_debug 被调用，返回：{date}")
    return date

prompt = PromptTemplate(
    template="日期：{date}\n问题：{question}",
    input_variables=["question"],
    partial_variables={"date": get_date_with_debug}
)

# 调用时会看到调试信息
result = prompt.format(question="测试")
```

### 7.2 常见错误

**错误1：传递函数调用结果而非函数对象**

```python
# ❌ 错误
prompt = PromptTemplate(
    template="日期：{date}",
    input_variables=[],
    partial_variables={"date": get_date()}  # 错误：调用了函数
)

# ✅ 正确
prompt = PromptTemplate(
    template="日期：{date}",
    input_variables=[],
    partial_variables={"date": get_date}  # 正确：传递函数对象
)
```

**错误2：函数有必需参数**

```python
# ❌ 错误
def get_date_with_format(fmt):  # 有必需参数
    return datetime.now().strftime(fmt)

prompt = PromptTemplate(
    template="日期：{date}",
    input_variables=[],
    partial_variables={"date": get_date_with_format}  # 会报错
)

# ✅ 正确：使用 lambda 或默认参数
prompt = PromptTemplate(
    template="日期：{date}",
    input_variables=[],
    partial_variables={"date": lambda: get_date_with_format("%Y-%m-%d")}
)
```

---

## 8. 总结

### 核心要点

1. **函数动态值**：支持运行时计算，适合动态变化的数据
2. **延迟计算**：函数在模板格式化时才执行，每次都重新计算
3. **兼容性问题**：ChatPromptTemplate 存在兼容性问题，建议使用 PromptTemplate
4. **性能优化**：使用缓存避免重复计算，注意函数执行开销
5. **最佳实践**：合理使用多层次 partial、条件动态值、环境感知等模式

### 适用场景

- 需要动态时间戳的日志系统
- 环境感知的配置管理
- 多语言翻译系统
- Agent 系统的动态工具列表
- RAG 系统的请求追踪

### 注意事项

- 避免在函数中执行耗时操作
- 注意 ChatPromptTemplate 的兼容性问题
- 使用缓存优化性能
- 传递函数对象而非函数调用结果

---

**参考资料**：
- [LangChain 源码分析](reference/source_prompttemplate_01.md)
- [LangChain OpenTutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/02-prompt/01-prompttemplate)
- [GitHub Issue #17560](https://github.com/langchain-ai/langchain/issues/17560)
- [LateNode 完整指南](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples)

**版本**：v1.0
**最后更新**：2026-02-26
**适用于**：LangChain 0.3.x+
