# 核心概念6：Agent选择决策树

> 实用选择指南和决策流程

---

## 什么是Agent选择决策树？

**Agent选择决策树是一套系统化的决策流程，帮助开发者根据模型能力、工具复杂度、成本预算等因素，快速选择最合适的Agent类型。**

在实际开发中，LangChain提供了多种Agent类型（OpenAI Functions、ReAct、Structured Chat等），每种类型都有其适用场景和限制。选择错误的Agent类型会导致：
- ❌ 工具调用失败或不稳定
- ❌ 成本过高（token消耗大）
- ❌ 性能不佳（响应慢、准确率低）
- ❌ 开发体验差（调试困难）

决策树通过结构化的问题引导，帮助你在5分钟内做出正确选择。

---

## 核心决策树

### 第一层：模型能力判断

```
你的模型是否支持函数调用（Function Calling）？
├─ 是 → 进入第二层（函数调用路径）
└─ 否 → 使用 ReAct Agent（唯一选择）
```

**如何判断模型是否支持函数调用？**

| 模型提供商 | 支持函数调用 | 推荐模型 |
|-----------|-------------|---------|
| OpenAI | ✅ | gpt-4, gpt-3.5-turbo |
| Anthropic | ✅ | claude-3-opus, claude-3-sonnet |
| Google | ✅ | gemini-pro |
| 开源模型 | ❌ (大部分) | llama-2, mistral, qwen |

**代码示例：检查模型能力**
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI 模型 - 支持函数调用
openai_model = ChatOpenAI(model="gpt-4")
print(f"支持函数调用: {hasattr(openai_model, 'bind_tools')}")  # True

# Anthropic 模型 - 支持函数调用
anthropic_model = ChatAnthropic(model="claude-3-sonnet-20240229")
print(f"支持函数调用: {hasattr(anthropic_model, 'bind_tools')}")  # True
```

---

### 第二层：工具复杂度判断（函数调用路径）

```
你的工具是否需要复杂参数？
├─ 是（多个输入/嵌套结构）→ Structured Chat Agent
└─ 否（简单参数）→ 进入第三层（性能优化路径）
```

**什么是"复杂参数"？**

**简单参数示例**（适合 OpenAI Functions）:
```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网页内容"""
    # 单一字符串参数
    return f"搜索结果: {query}"
```

**复杂参数示例**（需要 Structured Chat）:
```python
from langchain.tools import tool
from typing import List

@tool
def advanced_search(
    query: str,
    filters: List[str],
    date_range: dict,
    max_results: int = 10
) -> str:
    """高级搜索工具

    Args:
        query: 搜索关键词
        filters: 过滤条件列表
        date_range: 日期范围 {"start": "2024-01-01", "end": "2024-12-31"}
        max_results: 最大结果数
    """
    # 多个参数 + 嵌套结构
    return f"搜索 {query}，过滤 {filters}，日期 {date_range}"
```

**决策规则**:
- 参数 ≤ 2个 且 类型简单（str, int, bool）→ 简单参数
- 参数 > 2个 或 包含 List/Dict/嵌套结构 → 复杂参数

---

### 第三层：性能与成本权衡（性能优化路径）

```
你的优先级是什么？
├─ 最高可靠性 + 生产环境 → OpenAI Functions Agent
├─ 最新特性 + 统一API → Tool Calling Agent (create_agent)
└─ 调试与可解释性 → ReAct Agent
```

**三种选择的对比**:

| 维度 | OpenAI Functions | Tool Calling | ReAct |
|------|------------------|--------------|-------|
| **可靠性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Token消耗** | 中等 | 中等 | 较高 |
| **响应速度** | 快 | 快 | 较慢 |
| **调试难度** | 中等 | 中等 | 简单 |
| **2026状态** | 稳定 | 推荐 | 稳定 |

---

## 完整决策流程图

```
开始
  ↓
[1] 模型是否支持函数调用？
  ├─ 否 → ReAct Agent (唯一选择)
  └─ 是 ↓
[2] 工具参数是否复杂？
  ├─ 是 → Structured Chat Agent
  └─ 否 ↓
[3] 优先级是什么？
  ├─ 生产环境 → OpenAI Functions Agent
  ├─ 最新API → Tool Calling Agent (create_agent)
  └─ 调试需求 → ReAct Agent
```

---

## 10个真实场景的选择示例

### 场景1：客服机器人（简单工具）

**需求**:
- 工具: 查询订单、查询物流、退款申请
- 模型: OpenAI gpt-3.5-turbo
- 要求: 高可靠性、低成本

**决策过程**:
1. 模型支持函数调用 ✅
2. 工具参数简单（订单号、物流单号）✅
3. 优先级：生产环境 ✅

**推荐**: OpenAI Functions Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool

@tool
def query_order(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 状态: 已发货"

llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = create_openai_functions_agent(llm, [query_order], prompt)
executor = AgentExecutor(agent=agent, tools=[query_order])
```

---

### 场景2：数据分析助手（复杂工具）

**需求**:
- 工具: 数据库查询（多条件过滤、聚合、排序）
- 模型: OpenAI gpt-4
- 要求: 支持复杂SQL生成

**决策过程**:
1. 模型支持函数调用 ✅
2. 工具参数复杂（多个过滤条件、嵌套结构）✅

**推荐**: Structured Chat Agent

```python
from langchain.agents import create_structured_chat_agent
from langchain.tools import tool
from typing import List, Dict

@tool
def query_database(
    table: str,
    filters: List[Dict[str, str]],
    aggregations: List[str],
    order_by: str
) -> str:
    """复杂数据库查询

    Args:
        table: 表名
        filters: 过滤条件 [{"field": "age", "op": ">", "value": "18"}]
        aggregations: 聚合函数 ["COUNT(*)", "AVG(price)"]
        order_by: 排序字段
    """
    return f"查询 {table} 表，过滤 {filters}"

agent = create_structured_chat_agent(llm, [query_database], prompt)
```

---

### 场景3：开源模型应用（无函数调用）

**需求**:
- 工具: 搜索引擎、计算器
- 模型: Llama-2-70B（开源）
- 要求: 成本控制

**决策过程**:
1. 模型不支持函数调用 ❌

**推荐**: ReAct Agent（唯一选择）

```python
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent

llm = Ollama(model="llama2:70b")
agent = create_react_agent(llm, tools, prompt)
```

---

### 场景4：RAG知识库问答（2026最新）

**需求**:
- 工具: 向量检索、文档查询
- 模型: OpenAI gpt-4
- 要求: 使用最新API

**决策过程**:
1. 模型支持函数调用 ✅
2. 工具参数简单 ✅
3. 优先级：最新API ✅

**推荐**: Tool Calling Agent (create_agent)

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_agent(
    llm,
    tools=[vector_search_tool, doc_query_tool],
    system_prompt="你是一个知识库助手，帮助用户查询文档。"
)
```

---

### 场景5：多步骤工作流（调试需求）

**需求**:
- 工具: API调用、数据处理、结果汇总
- 模型: OpenAI gpt-4
- 要求: 需要查看推理过程

**决策过程**:
1. 模型支持函数调用 ✅
2. 工具参数简单 ✅
3. 优先级：调试与可解释性 ✅

**推荐**: ReAct Agent

```python
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 输出会显示完整的 Thought-Action-Observation 过程
result = executor.invoke({"input": "分析用户数据并生成报告"})
```

---

### 场景6：电商推荐系统（简单工具 + 高并发）

**需求**:
- 工具: 商品搜索、用户画像查询
- 模型: OpenAI gpt-3.5-turbo
- 要求: 高并发、低延迟

**推荐**: OpenAI Functions Agent

**原因**: 函数调用比ReAct更快，token消耗更少，适合高并发场景。

---

### 场景7：法律文档分析（复杂工具 + 高精度）

**需求**:
- 工具: 条款提取（多字段、嵌套结构）
- 模型: Anthropic Claude-3-Opus
- 要求: 高精度、支持复杂参数

**推荐**: Structured Chat Agent

**原因**: 法律条款提取需要多个字段（条款类型、生效日期、适用范围等），Structured Chat能更好地处理。

---

### 场景8：代码生成助手（简单工具 + 生产环境）

**需求**:
- 工具: 代码搜索、文档查询
- 模型: OpenAI gpt-4
- 要求: 生产环境、高可靠性

**推荐**: OpenAI Functions Agent

---

### 场景9：多模态应用（图片 + 文本）

**需求**:
- 工具: 图片识别、文本分析
- 模型: OpenAI gpt-4-vision
- 要求: 支持多模态输入

**推荐**: Tool Calling Agent (create_agent)

**原因**: 2026年的create_agent API对多模态支持更好。

---

### 场景10：实验性项目（快速原型）

**需求**:
- 工具: 任意工具
- 模型: 任意模型
- 要求: 快速开发、灵活调整

**推荐**: Tool Calling Agent (create_agent)

**原因**: 统一API，自动选择最佳Agent类型，减少决策成本。

---

## 快速选择速查表

### 按模型选择

| 模型 | 推荐Agent | 备选Agent |
|------|----------|----------|
| OpenAI gpt-4 | Tool Calling | OpenAI Functions |
| OpenAI gpt-3.5-turbo | OpenAI Functions | Tool Calling |
| Anthropic Claude-3 | Tool Calling | OpenAI Functions |
| Llama-2 / Mistral | ReAct | 无 |
| Gemini Pro | Tool Calling | OpenAI Functions |

### 按工具复杂度选择

| 工具类型 | 参数数量 | 推荐Agent |
|---------|---------|----------|
| 简单查询 | 1-2个 | OpenAI Functions |
| 中等复杂 | 3-4个 | OpenAI Functions |
| 高度复杂 | 5+个或嵌套 | Structured Chat |
| 任意复杂度 + 开源模型 | 任意 | ReAct |

### 按场景选择

| 场景 | 推荐Agent | 原因 |
|------|----------|------|
| 生产环境 | OpenAI Functions | 最可靠 |
| 快速原型 | Tool Calling | 最简单 |
| 调试开发 | ReAct | 最透明 |
| 复杂工具 | Structured Chat | 最灵活 |
| 开源模型 | ReAct | 唯一选择 |
| 成本敏感 | OpenAI Functions | Token最少 |

---

## 常见决策错误与纠正

### 错误1：盲目使用ReAct Agent ❌

**错误做法**:
```python
# 使用支持函数调用的模型，却选择ReAct
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools, prompt)  # ❌ 浪费模型能力
```

**正确做法**:
```python
# 使用函数调用能力
llm = ChatOpenAI(model="gpt-4")
agent = create_agent(llm, tools)  # ✅ 自动选择最佳类型
```

**为什么错**:
- ReAct需要更多token（Thought-Action-Observation循环）
- 函数调用更可靠（结构化输出）
- 成本更高、速度更慢

---

### 错误2：简单工具使用Structured Chat ❌

**错误做法**:
```python
@tool
def search(query: str) -> str:
    """简单搜索"""
    return f"结果: {query}"

# 使用Structured Chat处理简单工具
agent = create_structured_chat_agent(llm, [search], prompt)  # ❌ 过度设计
```

**正确做法**:
```python
# 使用OpenAI Functions
agent = create_openai_functions_agent(llm, [search], prompt)  # ✅ 简单高效
```

**为什么错**:
- Structured Chat增加了不必要的复杂度
- 简单工具用OpenAI Functions更快、更稳定

---

### 错误3：开源模型使用OpenAI Functions ❌

**错误做法**:
```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
agent = create_openai_functions_agent(llm, tools, prompt)  # ❌ 模型不支持
```

**正确做法**:
```python
# 使用ReAct
agent = create_react_agent(llm, tools, prompt)  # ✅ 兼容开源模型
```

**为什么错**:
- 开源模型大多不支持函数调用
- 会导致工具调用失败

---

## 决策树实战代码

### 自动化Agent选择器

```python
"""
Agent选择器 - 根据模型和工具自动选择最佳Agent类型
"""

from typing import List, Any
from langchain.tools import BaseTool
from langchain.agents import (
    create_agent,
    create_openai_functions_agent,
    create_react_agent,
    create_structured_chat_agent,
)

class AgentSelector:
    """Agent类型自动选择器"""

    @staticmethod
    def has_function_calling(llm: Any) -> bool:
        """检查模型是否支持函数调用"""
        return hasattr(llm, 'bind_tools')

    @staticmethod
    def is_complex_tool(tool: BaseTool) -> bool:
        """检查工具参数是否复杂"""
        # 获取工具参数
        args_schema = tool.args_schema
        if not args_schema:
            return False

        # 检查参数数量和类型
        fields = args_schema.__fields__

        # 规则1: 参数超过3个
        if len(fields) > 3:
            return True

        # 规则2: 包含List或Dict类型
        for field in fields.values():
            field_type = str(field.outer_type_)
            if 'List' in field_type or 'Dict' in field_type:
                return True

        return False

    @staticmethod
    def select_agent_type(
        llm: Any,
        tools: List[BaseTool],
        priority: str = "production"  # production, debug, latest
    ) -> str:
        """
        自动选择Agent类型

        Args:
            llm: 语言模型
            tools: 工具列表
            priority: 优先级 (production/debug/latest)

        Returns:
            推荐的Agent类型名称
        """
        # 第一层：检查函数调用支持
        if not AgentSelector.has_function_calling(llm):
            return "ReAct"

        # 第二层：检查工具复杂度
        has_complex_tools = any(
            AgentSelector.is_complex_tool(tool) for tool in tools
        )

        if has_complex_tools:
            return "Structured Chat"

        # 第三层：根据优先级选择
        if priority == "production":
            return "OpenAI Functions"
        elif priority == "latest":
            return "Tool Calling"
        elif priority == "debug":
            return "ReAct"
        else:
            return "Tool Calling"  # 默认

    @staticmethod
    def create_agent_auto(
        llm: Any,
        tools: List[BaseTool],
        prompt: Any,
        priority: str = "production"
    ):
        """
        自动创建最佳Agent

        Args:
            llm: 语言模型
            tools: 工具列表
            prompt: Prompt模板
            priority: 优先级

        Returns:
            创建的Agent实例
        """
        agent_type = AgentSelector.select_agent_type(llm, tools, priority)

        print(f"✅ 自动选择Agent类型: {agent_type}")

        if agent_type == "OpenAI Functions":
            return create_openai_functions_agent(llm, tools, prompt)
        elif agent_type == "ReAct":
            return create_react_agent(llm, tools, prompt)
        elif agent_type == "Structured Chat":
            return create_structured_chat_agent(llm, tools, prompt)
        elif agent_type == "Tool Calling":
            return create_agent(llm, tools)
        else:
            raise ValueError(f"未知Agent类型: {agent_type}")


# ===== 使用示例 =====

from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def simple_search(query: str) -> str:
    """简单搜索工具"""
    return f"搜索结果: {query}"

@tool
def complex_query(
    table: str,
    filters: List[dict],
    limit: int = 10
) -> str:
    """复杂查询工具"""
    return f"查询 {table}"

# 场景1: 简单工具 + 生产环境
llm = ChatOpenAI(model="gpt-4")
agent1 = AgentSelector.create_agent_auto(
    llm,
    [simple_search],
    prompt,
    priority="production"
)
# 输出: ✅ 自动选择Agent类型: OpenAI Functions

# 场景2: 复杂工具
agent2 = AgentSelector.create_agent_auto(
    llm,
    [complex_query],
    prompt,
    priority="production"
)
# 输出: ✅ 自动选择Agent类型: Structured Chat

# 场景3: 开源模型
from langchain_community.llms import Ollama
llm_open = Ollama(model="llama2")
agent3 = AgentSelector.create_agent_auto(
    llm_open,
    [simple_search],
    prompt,
    priority="production"
)
# 输出: ✅ 自动选择Agent类型: ReAct
```

---

## 性能对比测试

### 测试场景：简单工具调用

```python
"""
性能对比：OpenAI Functions vs ReAct vs Tool Calling
测试场景：简单搜索工具
"""

import time
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_functions_agent,
    create_react_agent,
    create_agent,
    AgentExecutor
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 测试查询
test_query = "搜索Python教程"

# 1. OpenAI Functions Agent
start = time.time()
agent1 = create_openai_functions_agent(llm, [search_tool], prompt)
executor1 = AgentExecutor(agent=agent1, tools=[search_tool])
result1 = executor1.invoke({"input": test_query})
time1 = time.time() - start

# 2. ReAct Agent
start = time.time()
agent2 = create_react_agent(llm, [search_tool], prompt)
executor2 = AgentExecutor(agent=agent2, tools=[search_tool])
result2 = executor2.invoke({"input": test_query})
time2 = time.time() - start

# 3. Tool Calling Agent
start = time.time()
agent3 = create_agent(llm, [search_tool])
result3 = agent3.invoke({"messages": [("user", test_query)]})
time3 = time.time() - start

print(f"""
性能对比结果:
- OpenAI Functions: {time1:.2f}秒
- ReAct: {time2:.2f}秒
- Tool Calling: {time3:.2f}秒

最快: {'OpenAI Functions' if time1 < min(time2, time3) else 'Tool Calling' if time3 < time2 else 'ReAct'}
""")