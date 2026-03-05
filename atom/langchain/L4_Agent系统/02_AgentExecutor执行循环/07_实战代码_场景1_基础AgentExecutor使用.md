# 实战代码 场景1：基础 AgentExecutor 使用

> 从零创建一个 AgentExecutor，跑通完整的 Think→Act→Observe 循环——先会用，再理解

---

## 场景概述

AgentExecutor 是 LangChain 中运行 Agent 的标准容器。本文从最简单的场景开始，演示：

- 定义工具 → 创建 Agent → 包装成 AgentExecutor → 运行
- 用 `verbose=True` 观察执行循环的每一步
- 用 `return_intermediate_steps=True` 拿到完整的推理轨迹
- 单工具调用 vs 多工具调用 vs 纯对话（不调用工具）

跑完这几个例子，你就能理解 AgentExecutor 的输入输出格式和基本行为模式。

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain langchain-openai python-dotenv

# 加载环境变量（需要 .env 文件中配置 OPENAI_API_KEY）
from dotenv import load_dotenv
load_dotenv()
```

---

## 示例1：最简 AgentExecutor——从创建到运行

**解决的问题：** 用最少的代码跑通一个完整的 Agent 执行循环，理解创建流程和输入输出格式。

```python
"""
最简 AgentExecutor 使用
演示：定义工具 → 创建 Agent → 包装 Executor → 运行
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


# ===== 1. 定义工具 =====
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_data = {
        "北京": "晴天 25°C，空气质量良好",
        "上海": "多云 22°C，有轻雾",
        "深圳": "阵雨 28°C，湿度较高",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def get_population(city: str) -> str:
    """获取城市人口信息"""
    pop_data = {
        "北京": "常住人口约 2189 万",
        "上海": "常住人口约 2487 万",
        "深圳": "常住人口约 1756 万",
    }
    return pop_data.get(city, f"未找到 {city} 的人口数据")


tools = [get_weather, get_population]


# ===== 2. 创建 Prompt =====
# agent_scratchpad 是 AgentExecutor 注入中间步骤的占位符，必须有
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的中文助手。简洁回答用户问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# ===== 3. 创建 Agent + Executor =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# create_tool_calling_agent：把 LLM + 工具 + Prompt 组合成一个 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor：包装 Agent，提供执行循环
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印每一步的执行过程
)


# ===== 4. 运行 =====
if __name__ == "__main__":
    result = executor.invoke({"input": "北京今天天气怎么样？"})

    # 输出格式：result 是一个 dict
    print(f"\n输入: {result['input']}")
    print(f"输出: {result['output']}")
```

### 预期输出

```
> Entering new AgentExecutor chain...

Invoking: `get_weather` with `{'city': '北京'}`

晴天 25°C，空气质量良好

北京今天晴天，气温 25°C，空气质量良好。

> Finished chain.

输入: 北京今天天气怎么样？
输出: 北京今天晴天，气温 25°C，空气质量良好。
```

> **关键观察：** `verbose=True` 让你看到了完整的执行循环——LLM 决定调用 `get_weather`，传入参数 `{'city': '北京'}`，拿到结果后生成最终回答。这就是一次完整的 Think→Act→Observe→Answer。

---

## 示例2：用 return_intermediate_steps 查看推理轨迹

**解决的问题：** `verbose` 只是打印日志，如果你想在代码中拿到每一步的详细信息（调了什么工具、传了什么参数、返回了什么），需要 `return_intermediate_steps`。

```python
"""
查看 Agent 的完整推理轨迹
演示：return_intermediate_steps 返回每一步的 AgentAction + observation
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def search_docs(query: str) -> str:
    """搜索知识库文档"""
    docs = {
        "RAG": "RAG（检索增强生成）通过检索外部知识增强 LLM 生成能力。核心流程：索引→检索→生成。",
        "Embedding": "Embedding 将文本映射到高维向量空间，语义相近的文本在向量空间中距离更近。",
        "Chunking": "文本分块是 RAG 的关键步骤。常见策略：固定大小分块、语义分块、递归字符分块。",
    }
    for key, value in docs.items():
        if key.lower() in query.lower():
            return value
    return f"未找到与 '{query}' 相关的文档"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


tools = [search_docs, calculate]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 RAG 学习助手。用中文简洁回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)

# 关键：return_intermediate_steps=True
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,  # 返回中间步骤
    verbose=False,  # 关掉 verbose，我们自己解析
)


if __name__ == "__main__":
    result = executor.invoke({"input": "什么是 RAG？它和 Embedding 有什么关系？"})

    # ===== 解析中间步骤 =====
    print("=" * 60)
    print("推理轨迹")
    print("=" * 60)

    for i, (action, observation) in enumerate(result["intermediate_steps"]):
        print(f"\n--- 步骤 {i + 1} ---")
        print(f"  工具: {action.tool}")
        print(f"  输入: {action.tool_input}")
        print(f"  结果: {observation[:80]}...")

    print(f"\n{'=' * 60}")
    print(f"最终回答: {result['output']}")
    print(f"总步骤数: {len(result['intermediate_steps'])}")
```

### 预期输出

```
============================================================
推理轨迹
============================================================

--- 步骤 1 ---
  工具: search_docs
  输入: {'query': 'RAG'}
  结果: RAG（检索增强生成）通过检索外部知识增强 LLM 生成能力。核心流程：索引→检索→生成。...

--- 步骤 2 ---
  工具: search_docs
  输入: {'query': 'Embedding'}
  结果: Embedding 将文本映射到高维向量空间，语义相近的文本在向量空间中距离更近。...

============================================================
最终回答: RAG 是检索增强生成技术...Embedding 是 RAG 的基础...
总步骤数: 2
```

> **关键洞察：** `intermediate_steps` 是一个列表，每个元素是 `(AgentAction, observation)` 元组。
> - `AgentAction.tool`：调用的工具名
> - `AgentAction.tool_input`：传给工具的参数
> - `observation`：工具返回的结果
>
> 这对应源码中的 `AgentStep(action, observation)`，是 AgentExecutor 执行循环的核心数据结构。

---

## 示例3：三种调用模式——单工具、多工具、纯对话

**解决的问题：** Agent 不是每次都调用工具。理解 LLM 什么时候调用工具、什么时候直接回答，是理解执行循环的关键。

```python
"""
三种调用模式对比
演示：Agent 根据问题自动决定是否调用工具、调用几个工具
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_data = {
        "北京": "晴天 25°C",
        "上海": "多云 22°C",
        "深圳": "阵雨 28°C",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def get_population(city: str) -> str:
    """获取城市人口信息"""
    pop_data = {
        "北京": "2189万",
        "上海": "2487万",
        "深圳": "1756万",
    }
    return pop_data.get(city, f"未找到 {city} 的人口数据")


tools = [get_weather, get_population]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的中文助手。简洁回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=False,
)


def run_and_report(question: str):
    """运行 Agent 并报告执行情况"""
    result = executor.invoke({"input": question})
    steps = result["intermediate_steps"]

    print(f"\n问题: {question}")
    print(f"工具调用次数: {len(steps)}")
    if steps:
        for action, obs in steps:
            print(f"  → {action.tool}({action.tool_input}) = {obs}")
    else:
        print("  → 未调用任何工具（LLM 直接回答）")
    print(f"回答: {result['output'][:100]}")
    print("-" * 50)


if __name__ == "__main__":
    # 模式 1：单工具调用
    # LLM 判断需要查天气 → 调用 get_weather → 基于结果回答
    print("=" * 50)
    print("模式 1：单工具调用")
    print("=" * 50)
    run_and_report("北京天气怎么样？")

    # 模式 2：多工具调用
    # LLM 判断需要天气 + 人口 → 调用两个工具 → 综合回答
    print("\n" + "=" * 50)
    print("模式 2：多工具调用")
    print("=" * 50)
    run_and_report("告诉我上海的天气和人口")

    # 模式 3：纯对话（不调用工具）
    # LLM 判断不需要工具 → 直接回答
    # 对应源码中 isinstance(output, AgentFinish) 在第一轮就为 True
    print("\n" + "=" * 50)
    print("模式 3：纯对话")
    print("=" * 50)
    run_and_report("你好，你是谁？")
```

### 预期输出

```
==================================================
模式 1：单工具调用
==================================================

问题: 北京天气怎么样？
工具调用次数: 1
  → get_weather({'city': '北京'}) = 晴天 25°C
回答: 北京今天晴天，气温 25°C。
--------------------------------------------------

==================================================
模式 2：多工具调用
==================================================

问题: 告诉我上海的天气和人口
工具调用次数: 2
  → get_weather({'city': '上海'}) = 多云 22°C
  → get_population({'city': '上海'}) = 2487万
回答: 上海目前多云，气温 22°C，常住人口约 2487 万。
--------------------------------------------------

==================================================
模式 3：纯对话
==================================================

问题: 你好，你是谁？
工具调用次数: 0
  → 未调用任何工具（LLM 直接回答）
回答: 你好！我是一个 AI 助手，可以帮你查询天气和人口信息。
--------------------------------------------------
```

> **关键洞察：** Agent 的"智能"体现在 LLM 自主决定是否调用工具。
> - 需要外部数据 → 调用工具（循环至少 2 轮：调用 + 回答）
> - 不需要外部数据 → 直接回答（循环只有 1 轮）
> - 需要多个数据 → 调用多个工具（可能在同一轮并行调用，也可能分多轮）
>
> 这对应源码中的判断：`if isinstance(next_step_output, AgentFinish)` —— 第一轮就 Finish 说明不需要工具。

---

## 与 AgentExecutor 源码的对照

三个示例覆盖了 AgentExecutor 主循环的所有分支：

```
AgentExecutor._call() 主循环：

while _should_continue():
    next_step = _take_next_step()

    ┌─ isinstance(next_step, AgentFinish)?
    │
    ├─ Yes → _return()                    ← 示例3：纯对话，第1轮就 Finish
    │
    └─ No → intermediate_steps.extend()
           iterations += 1
           继续循环
           │
           ├─ 下一轮 AgentFinish          ← 示例1：单工具，第2轮 Finish
           └─ 下一轮还有 AgentAction       ← 示例2：多工具，可能多轮
```

### 输入输出格式对照

| 方面 | 代码中的表现 | 源码中的对应 |
|------|-------------|-------------|
| 输入 | `{"input": "问题"}` | `_call(inputs)` 的 `inputs` 参数 |
| 输出 | `result["output"]` | `AgentFinish.return_values["output"]` |
| 中间步骤 | `result["intermediate_steps"]` | `List[Tuple[AgentAction, str]]` |
| 工具名 | `action.tool` | `AgentAction.tool` |
| 工具输入 | `action.tool_input` | `AgentAction.tool_input` |
| 工具结果 | `observation` | `AgentStep.observation` |

---

## 常见问题

### Q1：为什么 Prompt 必须有 `agent_scratchpad`？

`agent_scratchpad` 是 AgentExecutor 注入中间步骤的入口。每轮循环中，AgentExecutor 会把之前的 `intermediate_steps` 格式化后填入这个占位符，让 LLM 看到之前的推理历史。

没有它，LLM 每轮都是"失忆"状态，不知道之前调用过什么工具、得到了什么结果。

### Q2：`create_tool_calling_agent` 和 `AgentExecutor` 为什么要分开？

这是关注点分离：
- `create_tool_calling_agent`：负责"怎么让 LLM 做决策"（Prompt + LLM + 输出解析）
- `AgentExecutor`：负责"怎么执行循环"（调用工具、管理步骤、控制终止）

分开的好处是你可以换不同的 Agent 类型（ReAct、Tool Calling、OpenAI Functions）而不改执行循环逻辑。

### Q3：多工具调用是并行还是串行？

取决于 LLM 的输出。如果 LLM 在一次响应中返回多个 `tool_calls`（如同时请求天气和人口），AgentExecutor 会在同一轮中依次执行它们。异步模式下（`ainvoke`），多个工具可以通过 `asyncio.gather` 并发执行。

---

## 速查卡片

```
┌──────────────────────────────────────────────────────────┐
│           基础 AgentExecutor 使用速查                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  创建流程:                                                │
│    1. 定义工具: @tool 装饰器                               │
│    2. 创建 Prompt: 必须含 {agent_scratchpad}              │
│    3. 创建 Agent: create_tool_calling_agent(llm,tools,p) │
│    4. 包装 Executor: AgentExecutor(agent, tools)         │
│    5. 运行: executor.invoke({"input": "问题"})           │
│                                                          │
│  输出格式:                                                │
│    result["input"]                → 原始输入               │
│    result["output"]               → 最终回答               │
│    result["intermediate_steps"]   → 推理轨迹（需开启）     │
│                                                          │
│  常用参数:                                                │
│    verbose=True                   → 打印执行过程           │
│    return_intermediate_steps=True → 返回中间步骤           │
│                                                          │
│  三种调用模式:                                             │
│    纯对话   → 0 次工具调用，1 轮循环                        │
│    单工具   → 1 次工具调用，2 轮循环                        │
│    多工具   → N 次工具调用，2+ 轮循环                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

**下一步：** 阅读 [07_实战代码_场景2_循环控制与错误处理.md](./07_实战代码_场景2_循环控制与错误处理.md)，学习如何配置 max_iterations、handle_parsing_errors 等安全参数
