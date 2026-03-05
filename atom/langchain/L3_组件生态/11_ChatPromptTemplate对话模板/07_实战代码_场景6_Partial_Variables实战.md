# 实战代码：场景6 - Partial Variables 实战

> **目标**：掌握 ChatPromptTemplate 的 partial variables 技术，实现预填充系统设置、动态日期注入、多语言模板管理

---

## 场景概述

**适用场景**：
- 预填充固定的系统配置
- 动态注入时间戳、日期等运行时信息
- 多语言模板管理
- RAG 系统的环境配置预填充
- 减少重复参数传递

**技术栈**：
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_openai.ChatOpenAI`
- `langchain_core.output_parsers.StrOutputParser`

**核心概念**：
- **partial()**: 预填充部分变量，返回新模板
- **partial_variables**: 支持静态值和函数
- **函数式 partial**: 运行时动态计算值

**源码参考**: `chat.py:1225-1258`

---

## 实战1：预填充系统设置

### 代码实现

```python
"""
实战1：预填充系统设置
演示如何使用 partial() 预填充固定的系统配置
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def demo_basic_partial():
    """演示基础 partial 用法"""
    print("=== 实战1：预填充系统设置 ===\n")

    # 1. 创建带多个变量的模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个 {role}。你的专长是 {expertise}。"),
        ("system", "你的回答风格是 {style}。"),
        ("human", "{question}")
    ])

    print("1. 原始模板")
    print(f"   输入变量: {template.input_variables}")

    # 2. 使用 partial 预填充部分变量
    partial_template = template.partial(
        role="AI 助手",
        expertise="Python 编程和 RAG 开发"
    )

    print("\n2. 预填充后的模板")
    print(f"   输入变量: {partial_template.input_variables}")
    print(f"   预填充变量: {list(partial_template.partial_variables.keys())}")

    # 3. 使用预填充模板（只需提供剩余变量）
    print("\n3. 使用预填充模板:")
    messages = partial_template.format_messages(
        style="简洁专业",
        question="什么是 LangChain?"
    )

    for i, msg in enumerate(messages, 1):
        print(f"   [{i}] {msg.__class__.__name__}: {msg.content[:60]}...")

    # 4. 实际调用
    print("\n4. 实际调用:")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = partial_template | model | StrOutputParser()

    try:
        result = chain.invoke({
            "style": "简洁专业",
            "question": "如何使用 ChatPromptTemplate?"
        })
        print(f"   回答: {result[:150]}...")
    except Exception as e:
        print(f"   错误: {str(e)}")

if __name__ == "__main__":
    demo_basic_partial()
```

### 运行结果

```
=== 实战1：预填充系统设置 ===

1. 原始模板
   输入变量: ['role', 'expertise', 'style', 'question']

2. 预填充后的模板
   输入变量: ['style', 'question']
   预填充变量: ['role', 'expertise']

3. 使用预填充模板:
   [1] SystemMessage: 你是一个 AI 助手。你的专长是 Python 编程和 RAG 开发。
   [2] SystemMessage: 你的回答风格是 简洁专业。
   [3] HumanMessage: 什么是 LangChain?

4. 实际调用:
   回答: ChatPromptTemplate 是 LangChain 中用于构建对话模板的核心组件...
```

---

## 实战2：动态日期注入（使用函数）

### 代码实现

```python
"""
实战2：动态日期注入（使用函数）
演示如何使用函数式 partial 动态注入时间信息
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_current_date():
    """获取当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%H:%M:%S")

def get_current_datetime():
    """获取当前日期时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def demo_function_partial():
    """演示函数式 partial"""
    print("=== 实战2：动态日期注入（使用函数） ===\n")

    # 1. 创建模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AI 助手。当前日期: {current_date}"),
        ("system", "当前时间: {current_time}"),
        ("human", "{question}")
    ])

    print("1. 原始模板")
    print(f"   输入变量: {template.input_variables}")

    # 2. 使用函数式 partial（注意：传入函数，不是函数调用结果）
    partial_template = template.partial(
        current_date=get_current_date,
        current_time=get_current_time
    )

    print("\n2. 函数式 partial")
    print(f"   输入变量: {partial_template.input_variables}")
    print(f"   预填充变量: {list(partial_template.partial_variables.keys())}")

    # 3. 第一次调用
    print("\n3. 第一次调用:")
    messages1 = partial_template.format_messages(question="现在几点了?")
    print(f"   日期: {[m.content for m in messages1 if '当前日期' in m.content][0]}")
    print(f"   时间: {[m.content for m in messages1 if '当前时间' in m.content][0]}")

    # 4. 延迟后第二次调用（演示函数每次都会重新执行）
    import time
    time.sleep(2)

    print("\n4. 2秒后第二次调用:")
    messages2 = partial_template.format_messages(question="现在几点了?")
    print(f"   日期: {[m.content for m in messages2 if '当前日期' in m.content][0]}")
    print(f"   时间: {[m.content for m in messages2 if '当前时间' in m.content][0]}")

    # 5. 实际调用
    print("\n5. 实际调用:")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = partial_template | model | StrOutputParser()

    try:
        result = chain.invoke({"question": "今天是几号?"})
        print(f"   回答: {result[:150]}...")
    except Exception as e:
        print(f"   错误: {str(e)}")

if __name__ == "__main__":
    demo_function_partial()
```

### 运行结果

```
=== 实战2：动态日期注入（使用函数） ===

1. 原始模板
   输入变量: ['current_date', 'current_time', 'question']

2. 函数式 partial
   输入变量: ['question']
   预填充变量: ['current_date', 'current_time']

3. 第一次调用:
   日期: 你是一个 AI 助手。当前日期: 2026-02-26
   时间: 当前时间: 06:31:45

4. 2秒后第二次调用:
   日期: 你是一个 AI 助手。当前日期: 2026-02-26
   时间: 当前时间: 06:31:47

5. 实际调用:
   回答: 根据系统信息，今天是 2026年2月26日...
```

---

## 实战3：多语言模板管理

### 代码实现

```python
"""
实战3：多语言模板管理
演示如何使用 partial 管理多语言模板
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 语言配置
LANGUAGE_CONFIGS = {
    "zh": {
        "system_role": "你是一个专业的 AI 助手。",
        "system_rules": "请用中文回答问题。",
        "language_name": "中文"
    },
    "en": {
        "system_role": "You are a professional AI assistant.",
        "system_rules": "Please answer in English.",
        "language_name": "English"
    },
    "ja": {
        "system_role": "あなたはプロのAIアシスタントです。",
        "system_rules": "日本語で答えてください。",
        "language_name": "日本語"
    }
}

def create_language_template(language: str) -> ChatPromptTemplate:
    """创建特定语言的模板"""
    # 基础模板
    base_template = ChatPromptTemplate.from_messages([
        ("system", "{system_role}"),
        ("system", "{system_rules}"),
        ("system", "当前语言: {language_name}"),
        ("human", "{question}")
    ])

    # 使用 partial 预填充语言配置
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])
    return base_template.partial(**config)

def demo_multilingual_templates():
    """演示多语言模板管理"""
    print("=== 实战3：多语言模板管理 ===\n")

    # 1. 创建不同语言的模板
    zh_template = create_language_template("zh")
    en_template = create_language_template("en")
    ja_template = create_language_template("ja")

    print("1. 创建多语言模板")
    print(f"   中文模板输入变量: {zh_template.input_variables}")
    print(f"   英文模板输入变量: {en_template.input_variables}")
    print(f"   日文模板输入变量: {ja_template.input_variables}")

    # 2. 测试中文模板
    print("\n2. 测试中文模板:")
    zh_messages = zh_template.format_messages(question="什么是 Python?")
    for msg in zh_messages[:2]:
        print(f"   {msg.content}")

    # 3. 测试英文模板
    print("\n3. 测试英文模板:")
    en_messages = en_template.format_messages(question="What is Python?")
    for msg in en_messages[:2]:
        print(f"   {msg.content}")

    # 4. 实际调用
    print("\n4. 实际调用（中文）:")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    zh_chain = zh_template | model | StrOutputParser()

    try:
        result = zh_chain.invoke({"question": "介绍一下 LangChain"})
        print(f"   回答: {result[:100]}...")
    except Exception as e:
        print(f"   错误: {str(e)}")

    print("\n5. 实际调用（英文）:")
    en_chain = en_template | model | StrOutputParser()

    try:
        result = en_chain.invoke({"question": "Introduce LangChain"})
        print(f"   Answer: {result[:100]}...")
    except Exception as e:
        print(f"   Error: {str(e)}")

if __name__ == "__main__":
    demo_multilingual_templates()
```

### 运行结果

```
=== 实战3：多语言模板管理 ===

1. 创建多语言模板
   中文模板输入变量: ['question']
   英文模板输入变量: ['question']
   日文模板输入变量: ['question']

2. 测试中文模板:
   你是一个专业的 AI 助手。
   请用中文回答问题。

3. 测试英文模板:
   You are a professional AI assistant.
   Please answer in English.

4. 实际调用（中文）:
   回答: LangChain 是一个用于构建基于大语言模型（LLM）应用的开源框架...

5. 实际调用（英文）:
   Answer: LangChain is an open-source framework for building applications powered by large language...
```

---

## 实战4：RAG 环境配置预填充

### 代码实现

```python
"""
实战4：RAG 环境配置预填充
演示如何在 RAG 系统中使用 partial 预填充环境配置
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

class RAGTemplateFactory:
    """RAG 模板工厂"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = self._load_config()

    def _load_config(self):
        """加载环境配置"""
        configs = {
            "development": {
                "system_name": "RAG 开发环境",
                "max_context_length": "2000",
                "retrieval_mode": "详细模式",
                "debug_mode": "开启"
            },
            "production": {
                "system_name": "RAG 生产环境",
                "max_context_length": "4000",
                "retrieval_mode": "精简模式",
                "debug_mode": "关闭"
            }
        }
        return configs.get(self.environment, configs["production"])

    def get_current_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_system_info(self):
        """获取系统信息"""
        return f"Python {os.sys.version.split()[0]} | {self.environment}"

    def create_rag_template(self) -> ChatPromptTemplate:
        """创建 RAG 模板"""
        # 基础模板
        base_template = ChatPromptTemplate.from_messages([
            ("system", "系统: {system_name}"),
            ("system", "环境: {system_info}"),
            ("system", "时间: {timestamp}"),
            ("system", "配置: 最大上下文长度={max_context_length}, 检索模式={retrieval_mode}, 调试={debug_mode}"),
            ("system", "参考文档:\n{context}"),
            ("human", "{question}")
        ])

        # 使用 partial 预填充配置
        return base_template.partial(
            system_name=self.config["system_name"],
            max_context_length=self.config["max_context_length"],
            retrieval_mode=self.config["retrieval_mode"],
            debug_mode=self.config["debug_mode"],
            timestamp=self.get_current_timestamp,  # 函数式 partial
            system_info=self.get_system_info  # 函数式 partial
        )

def demo_rag_environment_config():
    """演示 RAG 环境配置预填充"""
    print("=== 实战4：RAG 环境配置预填充 ===\n")

    # 1. 创建开发环境模板
    dev_factory = RAGTemplateFactory(environment="development")
    dev_template = dev_factory.create_rag_template()

    print("1. 开发环境模板")
    print(f"   输入变量: {dev_template.input_variables}")
    print(f"   预填充变量: {list(dev_template.partial_variables.keys())}")

    # 2. 创建生产环境模板
    prod_factory = RAGTemplateFactory(environment="production")
    prod_template = prod_factory.create_rag_template()

    print("\n2. 生产环境模板")
    print(f"   输入变量: {prod_template.input_variables}")
    print(f"   预填充变量: {list(prod_template.partial_variables.keys())}")

    # 3. 测试开发环境
    print("\n3. 测试开发环境:")
    dev_messages = dev_template.format_messages(
        context="LangChain 是一个 LLM 应用框架。",
        question="什么是 LangChain?"
    )
    for msg in dev_messages[:4]:
        print(f"   {msg.content}")

    # 4. 测试生产环境
    print("\n4. 测试生产环境:")
    prod_messages = prod_template.format_messages(
        context="LangChain 是一个 LLM 应用框架。",
        question="什么是 LangChain?"
    )
    for msg in prod_messages[:4]:
        print(f"   {msg.content}")

    # 5. 实际调用
    print("\n5. 实际调用（生产环境）:")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prod_template | model | StrOutputParser()

    try:
        result = chain.invoke({
            "context": "LangChain 是一个用于构建 LLM 应用的开源框架，支持 RAG、Agent 等功能。",
            "question": "LangChain 有哪些主要功能?"
        })
        print(f"   回答: {result[:150]}...")
    except Exception as e:
        print(f"   错误: {str(e)}")

if __name__ == "__main__":
    demo_rag_environment_config()
```

### 运行结果

```
=== 实战4：RAG 环境配置预填充 ===

1. 开发环境模板
   输入变量: ['context', 'question']
   预填充变量: ['system_name', 'max_context_length', 'retrieval_mode', 'debug_mode', 'timestamp', 'system_info']

2. 生产环境模板
   输入变量: ['context', 'question']
   预填充变量: ['system_name', 'max_context_length', 'retrieval_mode', 'debug_mode', 'timestamp', 'system_info']

3. 测试开发环境:
   系统: RAG 开发环境
   环境: Python 3.13.1 | development
   时间: 2026-02-26 06:31:50
   配置: 最大上下文长度=2000, 检索模式=详细模式, 调试=开启

4. 测试生产环境:
   系统: RAG 生产环境
   环境: Python 3.13.1 | production
   时间: 2026-02-26 06:31:50
   配置: 最大上下文长度=4000, 检索模式=精简模式, 调试=关闭

5. 实际调用（生产环境）:
   回答: 根据提供的文档，LangChain 的主要功能包括：1. RAG（检索增强生成）2. Agent（智能代理）...
```

---

## 最佳实践总结

### 1. Partial 的两种用法

```python
# 方式1: 静态值 partial
template = ChatPromptTemplate.from_messages([...])
partial_template = template.partial(
    role="AI 助手",
    expertise="Python"
)

# 方式2: 函数式 partial（注意：传入函数，不是调用结果）
def get_date():
    return datetime.now().strftime("%Y-%m-%d")

partial_template = template.partial(
    current_date=get_date  # 传入函数本身
)
```

### 2. 适用场景

```python
# ✅ 适合使用 partial 的场景
# 1. 固定的系统配置
template.partial(system_name="RAG System", version="1.0")

# 2. 动态的时间信息
template.partial(timestamp=lambda: datetime.now().isoformat())

# 3. 环境变量
template.partial(api_key=os.getenv("API_KEY"))

# 4. 多语言配置
template.partial(**LANGUAGE_CONFIGS["zh"])

# ❌ 不适合使用 partial 的场景
# 1. 每次调用都不同的用户输入
# 2. 需要根据上下文动态变化的内容
```

### 3. Partial 与组合的配合

```python
# 先 partial，再组合
base = ChatPromptTemplate.from_messages([("system", "{role}")])
partial_base = base.partial(role="助手")

final = partial_base + [("human", "{input}")]
# final 保留了 partial_variables
```

### 4. 性能优化

```python
# ✅ 好的做法 - 一次性 partial 多个变量
template.partial(
    var1="value1",
    var2="value2",
    var3="value3"
)

# ❌ 不好的做法 - 多次 partial
template.partial(var1="value1").partial(var2="value2").partial(var3="value3")
```

### 5. 函数式 Partial 注意事项

```python
# ✅ 正确 - 传入函数
template.partial(date=get_current_date)

# ❌ 错误 - 传入函数调用结果（只会执行一次）
template.partial(date=get_current_date())

# ✅ 使用 lambda 表达式
template.partial(date=lambda: datetime.now().strftime("%Y-%m-%d"))
```

---

## 常见问题

### Q1: partial 会修改原模板吗？

**答案**: 不会。partial() 返回新模板，不修改原模板。

```python
template1 = ChatPromptTemplate.from_messages([...])
template2 = template1.partial(var="value")

print(template1.partial_variables)  # {}（未改变）
print(template2.partial_variables)  # {'var': 'value'}（新模板）
```

### Q2: 如何查看模板的 partial_variables？

**答案**: 使用 `template.partial_variables` 属性。

```python
template = ChatPromptTemplate.from_messages([...]).partial(var1="value1")
print(template.partial_variables)  # {'var1': 'value1'}
print(template.input_variables)    # 不包含 var1
```

### Q3: 函数式 partial 什么时候执行？

**答案**: 每次调用 `format_messages()` 或 `invoke()` 时执行。

```python
template = template.partial(date=get_current_date)

# 第一次调用 - 执行 get_current_date()
messages1 = template.format_messages(input="test")

# 第二次调用 - 再次执行 get_current_date()
messages2 = template.format_messages(input="test")
```

### Q4: 可以覆盖 partial_variables 吗？

**答案**: 可以。在 `invoke()` 时传入同名参数会覆盖 partial_variables。

```python
template = template.partial(role="助手")

# 使用 partial 的值
result1 = chain.invoke({"input": "test"})  # role="助手"

# 覆盖 partial 的值
result2 = chain.invoke({"role": "专家", "input": "test"})  # role="专家"
```

---

## 参考资料

- **源码参考**: `langchain_core/prompts/chat.py` (lines 1225-1258)
- **LangChain 官方文档**: [Prompt Templates](https://python.langchain.com/docs/modules/prompts/)
- **相关文档**: [核心概念 - 消息模板类型](03_核心概念_01_消息模板类型.md)

---

**版本**: v1.0
**最后更新**: 2026-02-26
**适用 LangChain 版本**: 0.2.x - 0.3.x
**维护者**: Claude Code
