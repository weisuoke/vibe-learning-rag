# 核心概念 1：ReAct 循环模式

> Agent 的灵魂不是"一次性回答"，而是"想一步、做一步、看一步"的迭代循环

---

## 什么是 ReAct？

**ReAct = Reasoning + Acting。LLM 不再一次性给出答案，而是交替进行"推理"和"行动"，每次行动的结果反馈给 LLM 指导下一步推理，直到得出最终答案。**

**前端类比：** 就像 Redux 的 dispatch → reducer → state → re-render 循环。每次 dispatch 一个 action，reducer 处理后更新 state，UI 根据新 state 重新渲染，用户再触发下一个 action。

**日常生活类比：** 就像侦探破案 —— 先分析线索（Thought），然后去现场取证（Action），拿到证据后重新分析（Observation），再决定下一步调查方向，直到破案。

---

## Thought → Action → Observation 三步循环

ReAct 的核心是三个阶段不断交替：

```
┌─────────────────────────────────────────────────────┐
│                   ReAct 循环                         │
│                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │ Thought  │───→│  Action  │───→│ Observation  │  │
│   │ (推理)   │    │ (行动)   │    │ (观察结果)   │  │
│   └──────────┘    └──────────┘    └──────┬───────┘  │
│        ↑                                 │          │
│        │          ┌──────────┐           │          │
│        └──────────│ 够了吗？ │←──────────┘          │
│                   └────┬─────┘                      │
│                        │ 是                         │
│                   ┌────▼─────┐                      │
│                   │ 最终答案 │                      │
│                   └──────────┘                      │
└─────────────────────────────────────────────────────┘
```

### 三个阶段详解

| 阶段 | 执行者 | 做什么 | LangChain 对应 |
|------|--------|--------|----------------|
| Thought（推理） | LLM | 分析当前信息，决定下一步 | `agent.plan()` 调用 |
| Action（行动） | 工具 | 执行 LLM 选择的工具 | `tool.run()` 调用 |
| Observation（观察） | 系统 | 将工具结果返回给 LLM | `AgentStep.observation` |

### 一个完整的例子

用户问："北京今天天气怎么样？适合户外运动吗？"

```
第 1 轮：
  Thought:  用户想知道北京天气和是否适合户外运动，我需要先查天气
  Action:   调用 get_weather(city="北京")
  Observation: 北京今天晴，气温 22°C，PM2.5 指数 35

第 2 轮：
  Thought:  天气数据拿到了，22°C 晴天，空气质量优，适合户外运动
  Action:   无需再调用工具
  → 最终答案: 北京今天晴天，22°C，空气质量优良，非常适合户外运动！
```

---

## LangChain 的三个核心数据类型

LangChain 用三个数据类来表达 ReAct 循环中的每个环节。它们定义在 `langchain_core/agents.py` 中：

### 1. AgentAction —— "我要做什么"

```python
# 来源: langchain_core/agents.py
class AgentAction(Serializable):
    """Agent 请求执行的一个动作"""
    tool: str           # 工具名称，如 "get_weather"
    tool_input: str | dict  # 工具输入参数，如 {"city": "北京"}
    log: str            # LLM 的推理过程（Thought 部分）
```

**AgentAction 对应 ReAct 中的 Thought + Action。** `log` 字段记录了 LLM 的推理文本，`tool` 和 `tool_input` 是推理的结论 —— 决定调用哪个工具、传什么参数。

### 2. AgentFinish —— "我有答案了"

```python
class AgentFinish(Serializable):
    """Agent 达到停止条件，返回最终结果"""
    return_values: dict  # 最终返回值，如 {"output": "北京今天适合户外运动"}
    log: str             # LLM 的最终推理过程
```

**AgentFinish 是循环的出口。** 当 LLM 判断已经收集到足够信息，不需要再调用工具时，返回 AgentFinish 而不是 AgentAction。

### 3. AgentStep —— "做完了，结果是什么"

```python
class AgentStep(Serializable):
    """执行一个 AgentAction 后的结果"""
    action: AgentAction    # 执行了什么动作
    observation: str       # 工具返回的结果（Observation）
```

**AgentStep 对应 ReAct 中的 Observation。** 它把 action 和 observation 绑在一起，形成完整的一步记录。

### 三者的关系

```
LLM 推理
  ├── 需要工具 → AgentAction(tool, tool_input, log)
  │                  ↓ 执行工具
  │              AgentStep(action, observation)
  │                  ↓ 反馈给 LLM
  │              [下一轮推理...]
  │
  └── 不需要工具 → AgentFinish(return_values, log)
                       ↓
                   [循环结束，返回答案]
```

---

## 数据流完整示例

用代码展示一次完整的 ReAct 循环中数据是怎么流动的：

```python
"""
ReAct 循环数据流演示
展示 AgentAction → AgentStep → AgentFinish 的完整流转
"""

# ===== 第 1 轮：LLM 决定调用工具 =====

# agent.plan() 返回 AgentAction
action_1 = AgentAction(
    tool="get_weather",
    tool_input={"city": "北京"},
    log="用户想知道北京天气，我需要调用天气工具查询。\n"
        "Action: get_weather\n"
        "Action Input: {\"city\": \"北京\"}"
)

# 执行工具，得到 observation
observation_1 = "北京今天晴，气温 22°C，PM2.5 指数 35"

# 组装成 AgentStep
step_1 = AgentStep(
    action=action_1,
    observation=observation_1
)

# step_1 被追加到 intermediate_steps 列表
intermediate_steps = [step_1]

# ===== 第 2 轮：LLM 认为信息足够，返回最终答案 =====

# agent.plan(intermediate_steps) 返回 AgentFinish
finish = AgentFinish(
    return_values={
        "output": "北京今天晴天，22°C，空气质量优良，非常适合户外运动！"
    },
    log="我已经获取了天气信息，可以直接回答用户。\n"
        "Final Answer: 北京今天晴天，22°C，空气质量优良，非常适合户外运动！"
)

# 循环结束，返回 finish.return_values
```

**关键观察：**

1. `intermediate_steps` 是一个 `list[AgentStep]`，记录了所有历史步骤
2. 每轮调用 `agent.plan()` 时，都会把 `intermediate_steps` 传进去，让 LLM 看到之前做了什么
3. LLM 根据历史步骤决定：继续调用工具（AgentAction）还是给出答案（AgentFinish）

---

## 简单 Chain vs Agent 循环

理解 ReAct 最好的方式是和"没有循环"的简单 Chain 对比：

### 简单 Chain：一条直线

```
输入 → LLM → 输出
```

```python
# 简单 Chain：LLM 直接回答，没有工具，没有循环
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("北京今天天气怎么样？")
# LLM 只能根据训练数据猜测，无法获取实时天气
```

**问题：** LLM 的知识有截止日期，无法获取实时信息。

### Agent 循环：螺旋上升

```
输入 → LLM → 需要工具？ → 是 → 调用工具 → 结果反馈 → LLM → 需要工具？ → ...
                          → 否 → 输出
```

```python
# Agent 循环：LLM + 工具，迭代直到有答案
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    # 实际调用天气 API
    return f"{city}: 晴天, 22°C"

llm = ChatOpenAI(model="gpt-4o")

# 创建 Agent（省略 prompt 模板细节）
# agent = create_react_agent(llm, tools=[get_weather], prompt=react_prompt)
# executor = AgentExecutor(agent=agent, tools=[get_weather])
# result = executor.invoke({"input": "北京今天天气怎么样？"})
# LLM 会调用 get_weather 获取实时数据，再基于数据回答
```

### 核心区别

| 维度 | 简单 Chain | Agent（ReAct 循环） |
|------|-----------|-------------------|
| 执行模式 | 单次调用 | 多轮迭代 |
| 工具使用 | 无 | 按需调用 |
| 决策能力 | 无 | LLM 自主决策 |
| 信息来源 | 仅训练数据 | 训练数据 + 实时工具 |
| 适用场景 | 翻译、摘要 | 问答、数据分析、任务执行 |
| 前端类比 | 纯函数组件 | 带 useEffect 的有状态组件 |

---

## ReAct 循环的 ASCII 全景图

把所有概念串在一起：

```
用户输入: "北京天气怎么样？适合跑步吗？"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentExecutor._call()                     │
│                                                             │
│  intermediate_steps = []                                    │
│  iterations = 0                                             │
│                                                             │
│  ┌─── while _should_continue(iterations, time_elapsed): ───┐│
│  │                                                         ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │ agent.plan(intermediate_steps, **inputs)     │        ││
│  │  │                                             │        ││
│  │  │  LLM 推理: "我需要查天气"                    │        ││
│  │  │  → AgentAction(tool="get_weather",           │        ││
│  │  │                 tool_input={"city":"北京"})   │        ││
│  │  └──────────────────┬──────────────────────────┘        ││
│  │                     │                                   ││
│  │                     ▼                                   ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │ tool.run("get_weather", {"city":"北京"})     │        ││
│  │  │ → observation: "晴天, 22°C, PM2.5: 35"      │        ││
│  │  └──────────────────┬──────────────────────────┘        ││
│  │                     │                                   ││
│  │                     ▼                                   ││
│  │  intermediate_steps.append(                             ││
│  │      AgentStep(action, observation)                     ││
│  │  )                                                      ││
│  │  iterations += 1                                        ││
│  │                                                         ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │ agent.plan(intermediate_steps, **inputs)     │        ││
│  │  │                                             │        ││
│  │  │  LLM 推理: "天气数据够了，可以回答"           │        ││
│  │  │  → AgentFinish(return_values={               │        ││
│  │  │        "output": "晴天22°C，适合跑步！"      │        ││
│  │  │    })                                        │        ││
│  │  └──────────────────┬──────────────────────────┘        ││
│  │                     │                                   ││
│  └─────────────────────┼───────────────────────────────────┘│
│                        ▼                                    │
│  return {"output": "晴天22°C，适合跑步！"}                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 多轮工具调用示例

ReAct 的威力在于可以连续调用多个工具，逐步收集信息：

```python
"""
多轮 ReAct 循环示例：旅行规划
用户: "我下周去北京出差，帮我查下天气和推荐酒店"
"""

# === 第 1 轮 ===
# Thought: 用户需要天气和酒店两个信息，先查天气
# Action: get_weather("北京")
step_1 = AgentStep(
    action=AgentAction(
        tool="get_weather",
        tool_input={"city": "北京", "date": "下周"},
        log="用户需要北京下周的天气和酒店推荐。先查天气。"
    ),
    observation="北京下周：周一到周五，晴转多云，15-22°C，建议带薄外套"
)

# === 第 2 轮 ===
# Thought: 天气查到了，现在查酒店
# Action: search_hotels("北京")
step_2 = AgentStep(
    action=AgentAction(
        tool="search_hotels",
        tool_input={"city": "北京", "check_in": "下周一", "nights": 5},
        log="天气信息已获取。现在需要查询酒店推荐。"
    ),
    observation="推荐：1. 北京国贸大酒店 ¥800/晚 2. 亚朵酒店 ¥400/晚"
)

# === 第 3 轮 ===
# Thought: 天气和酒店都查到了，可以给出完整建议
# → AgentFinish
finish = AgentFinish(
    return_values={
        "output": (
            "北京下周天气不错，15-22°C，建议带薄外套。\n"
            "酒店推荐：国贸大酒店（¥800/晚）或亚朵（¥400/晚）。"
        )
    },
    log="天气和酒店信息都已收集完毕，可以给出完整建议。"
)

intermediate_steps = [step_1, step_2]
# 共 2 次工具调用，3 轮推理
```

**关键点：** LLM 自主决定调用工具的顺序和次数。它可能一轮调一个工具，也可能一轮调多个（MultiActionAgent）。

---

## 为什么 ReAct 比纯推理更强？

| 方法 | 做法 | 局限 |
|------|------|------|
| 纯推理（CoT） | LLM 只思考，不行动 | 无法获取外部信息 |
| 纯行动 | 按固定流程执行工具 | 不灵活，无法适应变化 |
| ReAct | 推理指导行动，行动反馈推理 | 兼具灵活性和信息获取能力 |

ReAct 的论文（Yao et al., 2022）证明：交替推理和行动比单独使用任何一种都更有效。LangChain 的 AgentExecutor 就是这个思想的工程实现。

---

## 与 RAG 的关系

ReAct 循环是 Agent 增强型 RAG 的基础：

```
传统 RAG:  查询 → 检索 → 生成（一次性，不迭代）

Agent RAG: 查询 → 推理"需要什么信息" → 检索 → 观察结果
                → 推理"信息够吗？" → 不够 → 换个关键词再检索
                → 推理"现在够了" → 生成最终答案
```

Agent RAG 通过 ReAct 循环实现了：
- **自适应检索**：根据第一次检索结果决定是否需要补充检索
- **多源整合**：可以从不同知识库、API 中收集信息
- **质量把关**：LLM 可以判断检索结果是否相关，不相关就重新检索

---

## 常见问题

**Q1: ReAct 循环会不会无限循环？**

不会。AgentExecutor 有两个安全阀：`max_iterations`（默认 15 次）和 `max_execution_time`。超出限制后执行 early stopping。详见 [03_核心概念_3_循环控制机制](./03_核心概念_3_循环控制机制.md)。

**Q2: LLM 怎么知道该返回 AgentAction 还是 AgentFinish？**

通过 Prompt 模板引导。ReAct 格式的 Prompt 会告诉 LLM：如果需要工具就输出 `Action: xxx`，如果有最终答案就输出 `Final Answer: xxx`。Agent 的 OutputParser 解析 LLM 输出，转换为对应的数据类型。

**Q3: 一轮可以调用多个工具吗？**

可以。`BaseMultiActionAgent` 的 `plan()` 方法可以返回 `list[AgentAction]`，一次请求多个工具并行执行。在异步模式下，多个工具通过 `asyncio.gather` 并发执行。

---

[来源: `langchain_core/agents.py`, `langchain_classic/agents/agent.py`, Context7 LangChain Agent 文档]
