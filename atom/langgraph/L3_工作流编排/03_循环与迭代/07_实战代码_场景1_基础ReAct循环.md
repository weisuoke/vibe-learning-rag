# 实战代码：基础 ReAct 循环

> Agent + 工具调用的经典循环模式

---

## 引用来源

本文档基于以下资料：
1. **参考资料**：
   - `reference/source_循环迭代_01.md` - LangGraph 循环迭代机制源码分析
   - `reference/context7_langgraph_01.md` - LangGraph 官方文档
   - `reference/search_循环迭代_01.md` - 社区资源搜索汇总
   - `reference/fetch_循环优化_01.md` - 循环优化最佳实践
2. **关键来源**：
   - [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/graph-api)
   - [Optimizing LangGraph Cycles](https://rajatpandit.com/optimizing-langgraph-cycles)

---

## 场景说明

### 什么是 ReAct 循环？

ReAct（Reasoning + Acting）是 AI Agent 最经典的循环模式：

```
思考(Reason) → 行动(Act) → 观察(Observe) → 思考 → 行动 → ... → 最终答案
```

Agent 不是一次性给出答案，而是**反复调用工具**来收集信息，直到有足够的信息生成最终回答。比如用户问"北京今天天气怎么样？适合户外运动吗？"，Agent 需要先查天气、再查空气质量、最后综合回答——每一轮都是循环。

### 流程图

```
START → agent_node → [需要工具?] ──Yes──→ tool_node ──→ agent_node → ...
                         │
                         No
                         ▼
                        END
```

---

## 完整代码

### 第 1 部分：状态定义

```python
"""
基础 ReAct 循环 - 完整可运行示例

场景：构建一个简单的 ReAct Agent，循环调用工具直到得到答案。
不依赖外部 API，使用模拟的 LLM 和工具。

运行方式：
  python 07_实战代码_场景1_基础ReAct循环.py
"""
import operator
from typing import Annotated, Literal, Any
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 状态定义
# ============================================================

class AgentState(TypedDict):
    """ReAct Agent 的状态

    - messages: 对话历史（使用 operator.add 实现追加）
    - steps: 循环计数器（安全守卫）
    """
    messages: Annotated[list, operator.add]
    steps: int
```

**设计要点：**
- `messages` 使用 `Annotated[list, operator.add]` 作为 reducer，每次更新都是追加而非覆盖
- `steps` 是一个简单的整数计数器，用于防止无限循环（TTL 模式）

### 第 2 部分：模拟 LLM 消息类型

```python
# ============================================================
# 2. 模拟消息类型（替代真实 LLM 的消息格式）
# ============================================================

@dataclass
class ToolCall:
    """模拟 LLM 返回的工具调用请求"""
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class AIMessage:
    """模拟 LLM 的回复消息"""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __str__(self):
        if self.tool_calls:
            calls = ", ".join(f"{tc.name}({tc.args})" for tc in self.tool_calls)
            return f"[AI 决策] 调用工具: {calls}"
        return f"[AI 回答] {self.content}"


@dataclass
class ToolMessage:
    """模拟工具执行的返回消息"""
    content: str
    tool_call_id: str = ""

    def __str__(self):
        return f"[工具结果] {self.content}"


@dataclass
class HumanMessage:
    """用户消息"""
    content: str

    def __str__(self):
        return f"[用户] {self.content}"
```

**为什么要模拟？** 代码可以直接运行，不需要 API Key。替换成真实 LLM 只需改 `agent_node`。

### 第 3 部分：模拟工具定义

```python
# ============================================================
# 3. 工具定义（模拟外部 API）
# ============================================================

def search_weather(city: str) -> str:
    """模拟天气查询工具"""
    weather_data = {
        "北京": "晴天，气温 22°C，湿度 45%",
        "上海": "多云，气温 26°C，湿度 70%",
        "深圳": "小雨，气温 28°C，湿度 85%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


def search_aqi(city: str) -> str:
    """模拟空气质量查询工具"""
    aqi_data = {
        "北京": "AQI 75，良好，适合户外活动",
        "上海": "AQI 120，轻度污染，敏感人群减少户外",
        "深圳": "AQI 50，优，非常适合户外活动",
    }
    return aqi_data.get(city, f"未找到 {city} 的空气质量数据")


def calculator(expression: str) -> str:
    """模拟计算器工具"""
    try:
        result = eval(expression)  # 仅用于演示，生产环境不要用 eval
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 工具注册表
TOOLS = {
    "search_weather": search_weather,
    "search_aqi": search_aqi,
    "calculator": calculator,
}
```

### 第 4 部分：模拟 LLM 决策逻辑

```python
# ============================================================
# 4. 模拟 LLM 决策（核心：根据对话历史决定下一步）
# ============================================================

def mock_llm_decide(messages: list) -> AIMessage:
    """模拟 LLM 的决策过程

    真实场景中这里会调用 OpenAI/Anthropic API。
    模拟逻辑：第1轮查天气 → 第2轮查AQI → 第3轮生成答案
    """
    # 统计已调用的工具
    called_tools = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if "气温" in msg.content:
                called_tools.add("search_weather")
            if "AQI" in msg.content:
                called_tools.add("search_aqi")

    if "search_weather" not in called_tools:
        return AIMessage(content="", tool_calls=[
            ToolCall(name="search_weather", args={"city": "北京"}, id="call_weather_1")
        ])
    elif "search_aqi" not in called_tools:
        return AIMessage(content="", tool_calls=[
            ToolCall(name="search_aqi", args={"city": "北京"}, id="call_aqi_1")
        ])
    else:
        # 收集工具结果，生成最终答案
        weather_info, aqi_info = "", ""
        for msg in messages:
            if isinstance(msg, ToolMessage):
                if "气温" in msg.content:
                    weather_info = msg.content
                if "AQI" in msg.content:
                    aqi_info = msg.content
        return AIMessage(
            content=f"根据查询结果：\n  天气：{weather_info}\n  空气：{aqi_info}\n"
                    f"结论：今天北京天气晴好，空气质量良好，非常适合户外运动！",
            tool_calls=[],
        )
```

### 第 5 部分：节点实现

```python
# ============================================================
# 5. 节点实现
# ============================================================

MAX_STEPS = 10  # 最大循环次数（安全守卫）


def agent_node(state: AgentState) -> dict:
    """Agent 节点：模拟 LLM 思考并决策

    职责：
    1. 读取当前对话历史
    2. 调用 LLM（这里是模拟）决定下一步
    3. 返回 AI 消息 + 递增步数计数器
    """
    current_step = state.get("steps", 0) + 1
    messages = state["messages"]

    # 调用模拟 LLM
    ai_message = mock_llm_decide(messages)

    print(f"  步骤 {current_step}: {ai_message}")

    return {
        "messages": [ai_message],
        "steps": current_step,
    }


def tool_node(state: AgentState) -> dict:
    """工具节点：执行 Agent 请求的工具调用

    职责：
    1. 从最后一条 AI 消息中提取 tool_calls
    2. 逐个执行工具
    3. 将工具结果作为 ToolMessage 写回 messages
    """
    messages = state["messages"]
    last_message = messages[-1]

    tool_results = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args

            if tool_name in TOOLS:
                # 执行工具
                result = TOOLS[tool_name](**tool_args)
                tool_msg = ToolMessage(
                    content=result,
                    tool_call_id=tool_call.id,
                )
                print(f"  {tool_msg}")
                tool_results.append(tool_msg)
            else:
                tool_results.append(
                    ToolMessage(
                        content=f"未知工具: {tool_name}",
                        tool_call_id=tool_call.id,
                    )
                )

    return {"messages": tool_results}
```

### 第 6 部分：路由函数（循环的核心）

```python
# ============================================================
# 6. 路由函数（决定循环是否继续）
# ============================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """条件路由：检查是否需要继续调用工具

    这是 ReAct 循环的核心控制点。

    判断逻辑：
    1. 安全守卫：步数超限 → 强制结束
    2. 业务逻辑：最后一条消息有 tool_calls → 继续调用工具
    3. 默认：没有 tool_calls → 结束（Agent 已给出最终答案）
    """
    # 安全守卫：步数限制（TTL 模式）
    if state.get("steps", 0) >= MAX_STEPS:
        print(f"  ⚠ 安全守卫触发：已执行 {state['steps']} 步，强制结束")
        return "end"

    # 业务逻辑：检查最后一条消息
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]

    # 如果 AI 消息包含 tool_calls → 需要执行工具 → 继续循环
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 没有 tool_calls → Agent 已给出最终答案 → 结束
    return "end"
```

**路由函数设计原则：** 安全守卫优先（先检查步数），确定性判断（纯 Python，不依赖 LLM），默认安全（意外情况走 "end"）。

### 第 7 部分：图构建与编译

```python
# ============================================================
# 7. 图构建
# ============================================================

def build_react_graph() -> StateGraph:
    """构建 ReAct 循环图

    图结构：
      START → agent → [should_continue] → tools → agent → ... → END
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # 定义边
    builder.add_edge(START, "agent")           # 入口 → agent

    builder.add_conditional_edges(             # agent → 条件路由
        "agent",
        should_continue,
        {
            "tools": "tools",                  # 需要工具 → tool_node
            "end": END,                        # 不需要 → 结束
        },
    )

    builder.add_edge("tools", "agent")         # 工具执行完 → 回到 agent（循环！）

    return builder.compile()
```

**关键：`builder.add_edge("tools", "agent")` 这一行创建了循环。** tool_node 执行完后，无条件回到 agent_node，让 Agent 根据工具结果继续思考。这就是 ReAct 循环的核心——工具结果反馈给 Agent，Agent 决定是否需要更多工具调用。

在源码层面，这条边通过 `attach_edge()` (`state.py` L1297) 向 agent 节点的 `branch:to:agent` channel 写入触发信号，激活 agent 节点的下一次执行。

### 第 8 部分：运行与输出

```python
# ============================================================
# 8. 运行
# ============================================================

def main():
    graph = build_react_graph()

    initial_state = {
        "messages": [HumanMessage(content="北京今天天气怎么样？适合户外运动吗？")],
        "steps": 0,
    }

    print("=" * 60)
    print(f"用户问题: {initial_state['messages'][0]}")
    print("-" * 60)

    result = graph.invoke(initial_state, config={"recursion_limit": 25})

    print("-" * 60)
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            print(f"最终答案: {msg.content}")
            break
    print(f"总循环次数: {result['steps']} | 消息总数: {len(result['messages'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 运行输出

```
============================================================
ReAct 循环演示
============================================================
用户问题: [用户] 北京今天天气怎么样？适合户外运动吗？
------------------------------------------------------------
  步骤 1: [AI 决策] 调用工具: search_weather({'city': '北京'})
  [工具结果] 晴天，气温 22°C，湿度 45%
  步骤 2: [AI 决策] 调用工具: search_aqi({'city': '北京'})
  [工具结果] AQI 75，良好，适合户外活动
  步骤 3: [AI 回答] 根据查询结果：
  天气：晴天，气温 22°C，湿度 45%  空气：AQI 75，良好
  结论：今天北京天气晴好，空气质量良好，非常适合户外运动！
------------------------------------------------------------
总循环次数: 3 | 消息总数: 6
============================================================
```

### 执行过程解析

| Superstep | 执行节点 | 状态变化 | 路由决策 |
|-----------|---------|---------|---------|
| 1 | agent_node | messages += [AI(tool_calls=[weather])], steps=1 | "tools" |
| 2 | tool_node | messages += [ToolMessage(天气结果)] | (静态边回 agent) |
| 3 | agent_node | messages += [AI(tool_calls=[aqi])], steps=2 | "tools" |
| 4 | tool_node | messages += [ToolMessage(AQI结果)] | (静态边回 agent) |
| 5 | agent_node | messages += [AI(最终答案)], steps=3 | "end" |

总共 5 个 superstep，远低于 `recursion_limit=25` 的限制。

---

## 源码关联

### 条件边的路由解析

当 `should_continue` 返回 `"tools"` 时，LangGraph 内部的处理流程：

1. `_branch.py` L146 的 `_route()` 调用 `should_continue(state)`，得到 `"tools"`
2. `_branch.py` L192 的 `_finish()` 通过 `path_map` 将 `"tools"` 映射为节点名 `"tools"`
3. `state.py` L1323 的 `attach_branch()` 中的 `get_writes()` 生成 `ChannelWriteEntry("branch:to:tools", None)`
4. 这个写入触发 tool_node 的执行

### 循环回边的实现

`builder.add_edge("tools", "agent")` 在编译时通过 `attach_edge()` (`state.py` L1297) 注册：

```python
# 编译后等价于：
# tool_node 执行完后，向 "branch:to:agent" channel 写入 None
# 这触发 agent_node 在下一个 superstep 中执行
self.nodes["tools"].writers.append(
    ChannelWrite(
        (ChannelWriteEntry("branch:to:agent", None),)
    )
)
```

这就是循环的底层机制：tool_node 的输出写入触发了 agent_node 的 trigger channel，形成闭环。

---

## 扩展建议

### 1. 接入真实 LLM

将 `mock_llm_decide` 替换为 `langchain-openai` 调用：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([search_weather, search_aqi])

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}
```

### 2. 集成 RemainingSteps 优雅降级

在状态中添加 `remaining_steps: RemainingSteps`，agent_node 中检查 `if remaining <= 2` 时跳过工具调用，直接返回当前最佳答案。详见 `03_核心概念_3_递归限制与安全机制.md`。

### 3. 语义缓存防止振荡

在状态中维护 `tool_cache: list[str]`，tool_node 执行前检查 `f"{tool_name}:{args}"` 是否已存在。重复调用时注入 `"系统提示：你已经调用过这个工具，请尝试不同方法"` 打破振荡。详见 [Optimizing LangGraph Cycles](https://rajatpandit.com/optimizing-langgraph-cycles)。

---

## 关键模式总结

ReAct 循环的核心就是三个组件的配合：

| 组件 | 职责 | 对应源码机制 |
|------|------|-------------|
| agent_node | LLM 思考，决定调用工具还是给出答案 | 节点执行，写入 messages channel |
| tool_node | 执行工具，返回结果 | 节点执行，通过 `branch:to:agent` 触发回边 |
| should_continue | 检查是否需要继续循环 | `_branch.py` 条件边路由，写入 `branch:to:{node}` channel |

这三者通过 LangGraph 的 channel 机制形成闭环，每一轮循环就是一个 superstep 序列：agent → (route) → tools → agent → ...，直到路由函数返回 `END`。
