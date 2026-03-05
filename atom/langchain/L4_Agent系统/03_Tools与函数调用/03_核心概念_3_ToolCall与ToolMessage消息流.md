# 核心概念 3：ToolCall 与 ToolMessage 消息流

> 模型通过 `AIMessage.tool_calls` 发出工具调用请求，工具执行后通过 `ToolMessage` 回传结果——这套请求→执行→响应的消息链路，是 Agent 能"做事"的底层通信机制。

---

## 概述

上一篇讲了函数调用协议的"怎么绑定、怎么触发"，这一篇深入消息层面：调用请求长什么样？结果怎么传回去？流式时怎么拼接？出错了怎么处理？

核心数据结构全部定义在 `langchain_core/messages/tool.py` 中：

| 类型 | 角色 | 一句话说明 |
|------|------|-----------|
| `ToolCall` | 请求 | 模型说"我要调用 get_weather，参数是 {city: 北京}" |
| `ToolMessage` | 响应 | 工具说"执行完了，结果是 晴，25°C" |
| `InvalidToolCall` | 异常 | 模型生成的 JSON 解析失败，记录原始错误 |
| `ToolCallChunk` | 流式 | 流式传输时 ToolCall 的碎片，逐步拼接 |

完整消息流：

```
HumanMessage("查天气")
    ↓ 模型推理
AIMessage(tool_calls=[ToolCall])     ← 请求
    ↓ 执行工具
ToolMessage(content="结果")          ← 响应
    ↓ 模型继续推理
AIMessage(content="最终回答")        ← 闭环
```

---

## 1. ToolCall — 模型的工具调用请求

### 一句话定义

**ToolCall 是一个 TypedDict，表示模型请求调用某个工具，包含工具名、参数和调用 ID。**

### 源码定义

[来源: `langchain_core/messages/tool.py`]

```python
class ToolCall(TypedDict):
    """表示模型对工具的调用请求"""
    name: str           # 工具名称，如 "get_weather"
    args: dict          # 参数字典，如 {"city": "北京"}（已解析的 JSON）
    id: Optional[str]   # 唯一调用 ID，如 "call_abc123"
    type: Literal["tool_call"]  # 固定值，标识类型
```

### 四个字段详解

| 字段 | 类型 | 作用 | 类比 |
|------|------|------|------|
| `name` | `str` | 指定调用哪个工具 | HTTP 请求的 URL 路径 |
| `args` | `dict` | 传递给工具的参数（已解析为 Python 字典） | HTTP 请求的 body |
| `id` | `str \| None` | 唯一标识这次调用，用于关联 ToolMessage | HTTP 请求的 request-id |
| `type` | `"tool_call"` | 固定值，区分 ToolCall 和 InvalidToolCall | Content-Type 头 |

### ToolCall 存在于 AIMessage.tool_calls 中

模型不会单独返回 ToolCall，它总是作为 `AIMessage.tool_calls` 列表的元素：

```python
from langchain_core.messages import AIMessage

response = AIMessage(
    content="",  # 调用工具时，content 通常为空
    tool_calls=[{
        "name": "get_weather",
        "args": {"city": "北京"},
        "id": "call_abc123",
        "type": "tool_call"
    }]
)

for tc in response.tool_calls:
    print(f"工具: {tc['name']}, 参数: {tc['args']}, ID: {tc['id']}")
```

### tool_call() 工厂函数

源码中提供了 `tool_call()` 工厂函数，用于创建 ToolCall 实例：

```python
from langchain_core.messages.tool import tool_call

# 工厂函数创建 ToolCall
tc = tool_call(
    name="get_weather",
    args={"city": "北京"},
    id="call_abc123"
)
# → {"name": "get_weather", "args": {"city": "北京"}, "id": "call_abc123", "type": "tool_call"}
```

工厂函数的好处是自动填充 `type: "tool_call"`，避免手动拼写出错。

---

## 2. ToolMessage — 工具执行结果

### 一句话定义

**ToolMessage 是工具执行完成后返回给模型的消息，通过 `tool_call_id` 关联到对应的 ToolCall 请求。**

### 源码定义

[来源: `langchain_core/messages/tool.py`]

```python
class ToolMessage(BaseMessage, ToolOutputMixin):
    """工具执行结果的消息"""
    tool_call_id: str           # 关联的 ToolCall ID（必填）
    artifact: Any = None        # 不发送给模型的完整数据（可选）
    status: str = "success"     # 执行状态："success" 或 "error"
    # 继承自 BaseMessage:
    # content: Union[str, list]  # 发送给模型的文本结果
    # type: Literal["tool"] = "tool"
```

### 关键字段详解

| 字段 | 类型 | 必填 | 作用 |
|------|------|------|------|
| `content` | `str \| list` | 是 | 发送给模型的执行结果文本 |
| `tool_call_id` | `str` | 是 | 关联到哪个 ToolCall（通过 ID 匹配） |
| `artifact` | `Any` | 否 | 完整数据，不发送给模型，供应用层使用 |
| `status` | `str` | 否 | `"success"` 或 `"error"`，默认 `"success"` |

### tool_call_id 的关联机制

`tool_call_id` 是 ToolCall 和 ToolMessage 之间的"纽带"：

```
AIMessage.tool_calls[0].id = "call_001"  ←──┐
AIMessage.tool_calls[1].id = "call_002"  ←──┤ 必须匹配
                                             │
ToolMessage.tool_call_id = "call_001"  ─────┘
ToolMessage.tool_call_id = "call_002"  ─────┘
```

```python
from langchain_core.messages import AIMessage, ToolMessage

ai_msg = AIMessage(content="", tool_calls=[
    {"name": "get_weather", "args": {"city": "北京"}, "id": "call_001", "type": "tool_call"},
    {"name": "get_weather", "args": {"city": "上海"}, "id": "call_002", "type": "tool_call"},
])

# 每个 ToolMessage 通过 tool_call_id 关联到对应的请求
tool_msg_1 = ToolMessage(content="北京：晴，25°C", tool_call_id="call_001")
tool_msg_2 = ToolMessage(content="上海：多云，28°C", tool_call_id="call_002")
```

### artifact — 不发送给模型的完整数据

`artifact` 字段解决一个实际问题：工具结果很大（DataFrame、图片等），全部发给模型浪费 Token。`artifact` 保留完整数据在应用层，只把摘要发给模型：

```python
tool_msg = ToolMessage(
    content="查询到 1000 条产品记录，前 2 条：产品A(¥99.9)、产品B(¥199.9)",  # 摘要给模型
    tool_call_id="call_db_query",
    artifact=[{"id": 1, "name": "产品A", "price": 99.9}, ...]  # 完整数据留在应用层
)
# 模型只看到 content，应用层通过 tool_msg.artifact 访问完整数据
```

**与 response_format 的配合**：当 Tool 的 `response_format="content_and_artifact"` 时，工具返回 `(content, artifact)` 元组，LangChain 自动拆分填充到 ToolMessage 的对应字段。

### status — 传递执行状态

```python
# 成功（默认）
ToolMessage(content="北京：晴，25°C", tool_call_id="call_001", status="success")

# 失败
ToolMessage(content="API 调用超时，请稍后重试", tool_call_id="call_001", status="error")
```

`status="error"` 告诉模型这次工具调用失败了，模型可以据此决定是重试、换个工具、还是直接告诉用户。

---

## 3. InvalidToolCall — 无效的工具调用

**当模型生成的工具调用 JSON 解析失败时，LangChain 不会直接报错，而是将其记录为 InvalidToolCall。**

```python
class InvalidToolCall(TypedDict):
    name: Optional[str]     # 工具名称（可能为 None）
    args: Optional[str]     # 原始 JSON 字符串（未解析，注意是 str 不是 dict）
    id: Optional[str]       # 调用 ID
    error: Optional[str]    # 错误信息
    type: Literal["invalid_tool_call"]
```

与 ToolCall 的关键区别：`args` 是原始字符串而非字典，多了 `error` 字段，存储在 `AIMessage.invalid_tool_calls`（而非 `tool_calls`）。

### default_tool_parser 的分离逻辑

源码中的 `default_tool_parser` 负责解析模型原始输出，将其分离为有效和无效两组：

```
模型原始输出 (raw_tool_calls)
    │
    ▼
default_tool_parser()
    ├─ JSON 解析成功 → ToolCall        → AIMessage.tool_calls
    └─ JSON 解析失败 → InvalidToolCall → AIMessage.invalid_tool_calls
```

这保证了即使模型偶尔生成格式错误的调用，系统也不会崩溃。

---

## 4. ToolCallChunk — 流式传输的碎片

**ToolCallChunk 是流式传输时 ToolCall 的碎片，多个 Chunk 逐步拼接成完整的 ToolCall。**

```python
class ToolCallChunk(TypedDict):
    name: Optional[str]     # 工具名称（可能分多次传输）
    args: Optional[str]     # JSON 字符串碎片（注意：是 str，不是 dict）
    id: Optional[str]       # 调用 ID
    index: Optional[int]    # 在 tool_calls 列表中的索引（区分并行调用）
    type: Literal["tool_call_chunk"]
```

与 ToolCall 的关键区别：`args` 是 JSON 字符串碎片（不是 dict），多了 `index` 字段用于区分多个并行调用的碎片。

### 流式拼接过程

```
Chunk 1: {"name": "get_we",  "args": "",              "index": 0}
Chunk 2: {"name": "ather",   "args": "{\"ci",         "index": 0}
Chunk 3: {"name": "",        "args": "ty\": \"北京\"}", "index": 0}
                    ↓ AIMessageChunk.__add__ 合并
完整结果: {"name": "get_weather", "args": {"city": "北京"}, "id": "call_123"}
```

`AIMessageChunk` 的 `__add__` 方法自动合并 `tool_call_chunks`——`name` 和 `args` 字符串拼接，`index` 用于区分并行调用。

### 流式传输代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}今天晴"

model_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])

# 流式传输，观察 ToolCallChunk
chunks = []
for chunk in model_with_tools.stream("北京天气怎么样？"):
    if chunk.tool_call_chunks:
        for tc_chunk in chunk.tool_call_chunks:
            print(f"Chunk: name={tc_chunk['name']!r}, args={tc_chunk['args']!r}")
    chunks.append(chunk)

# 合并所有 chunks 得到完整 ToolCall
full_message = chunks[0]
for c in chunks[1:]:
    full_message = full_message + c
print(f"完整 tool_calls: {full_message.tool_calls}")
```

---

## 5. 完整消息流 — 从请求到闭环

### 单工具调用流程

```
① HumanMessage("北京今天天气怎么样？")
      │
      ▼
② AIMessage(content="", tool_calls=[{name:"get_weather", args:{city:"北京"}, id:"call_001"}])
      │
      ▼  执行工具: get_weather(city="北京")
      │
③ ToolMessage(content="北京今天晴，25°C，湿度40%", tool_call_id="call_001")
      │
      ▼
④ AIMessage(content="北京今天天气晴朗，气温25°C，湿度40%，适合出行。")
```

### 完整可运行代码

```python
"""ToolCall 与 ToolMessage 消息流 - 完整流程演示"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息"""
    weather_data = {"北京": "晴，25°C，湿度 40%", "上海": "多云，28°C，湿度 65%"}
    return weather_data.get(city, f"{city}：暂无天气数据")

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools([get_weather])

# ① 用户提问
messages = [HumanMessage(content="北京今天天气怎么样？")]

# ② 模型生成 tool_calls
ai_response = model_with_tools.invoke(messages)
print(f"tool_calls: {ai_response.tool_calls}")
messages.append(ai_response)

# ③ 执行工具，构造 ToolMessage
for tc in ai_response.tool_calls:
    result = get_weather.invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# ④ 模型生成最终回答
final_response = model_with_tools.invoke(messages)
print(f"最终回答: {final_response.content}")
# → "北京今天天气晴朗，气温25°C，湿度40%，非常适合出行。"
```

---

## 6. 多工具并行调用

### 核心机制

一个 AIMessage 可以包含多个 tool_calls，表示模型希望同时调用多个工具。每个 ToolMessage 通过 `tool_call_id` 一一对应：

```
AIMessage.tool_calls = [
    {name: "get_weather", args: {city: "北京"}, id: "call_001"},
    {name: "get_weather", args: {city: "上海"}, id: "call_002"},
]
    │                                              │
    ▼                                              ▼
ToolMessage(content="北京：晴",     ToolMessage(content="上海：多云",
            tool_call_id="call_001")             tool_call_id="call_002")
```

### 完整可运行代码

```python
"""多工具并行调用演示"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息"""
    data = {"北京": "晴，25°C", "上海": "多云，28°C"}
    return data.get(city, f"{city}：暂无数据")

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools([get_weather])

messages = [HumanMessage(content="北京和上海的天气怎么样？")]
ai_response = model_with_tools.invoke(messages)
print(f"tool_calls 数量: {len(ai_response.tool_calls)}")  # 可能是 2
messages.append(ai_response)

# 用 tool_map 分发执行
tool_map = {"get_weather": get_weather}
for tc in ai_response.tool_calls:
    result = tool_map[tc["name"]].invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

final = model_with_tools.invoke(messages)
print(f"最终回答: {final.content}")
```

---

## 7. 错误处理

### 工具执行失败 + InvalidToolCall 处理

```python
ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)

# 处理有效调用（可能执行失败）
for tc in ai_response.tool_calls:
    try:
        result = get_weather.invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    except Exception as e:
        messages.append(ToolMessage(
            content=f"工具执行失败: {str(e)}",
            tool_call_id=tc["id"],
            status="error"  # 告知模型执行失败
        ))

# 处理无效调用（JSON 解析失败）
for itc in ai_response.invalid_tool_calls:
    messages.append(ToolMessage(
        content=f"工具调用格式错误: {itc['error']}，请重新生成",
        tool_call_id=itc.get("id", "unknown"),
        status="error"
    ))

final = model_with_tools.invoke(messages)
```

### 错误处理流程图

```
模型返回 AIMessage
    │
    ├─ tool_calls 不为空？
    │   ├─ 是 → 逐个执行工具
    │   │       ├─ 执行成功 → ToolMessage(status="success")
    │   │       └─ 执行异常 → ToolMessage(status="error")
    │   └─ 否 → 直接使用 content 作为最终回答
    │
    └─ invalid_tool_calls 不为空？
        └─ 是 → 记录错误，可选择要求模型重试
```

---

## 8. 在 RAG 中的应用

### 典型 RAG Agent 消息流

```
① HumanMessage("公司Q3营收是多少？")
② AIMessage(tool_calls=[{name: "search_kb", args: {query: "Q3营收"}, id: "call_001"}])
③ ToolMessage(content="根据2024年Q3财报，营收5.2亿元，同比增长15%", tool_call_id="call_001")
④ AIMessage("根据公司Q3财报数据，营收为5.2亿元，同比增长15%。")
```

多步检索场景——Agent 并行调用多个检索：

```
① HumanMessage("对比Q2和Q3的营收变化")
② AIMessage(tool_calls=[
       {name: "search_kb", args: {query: "Q2营收"}, id: "call_001"},
       {name: "search_kb", args: {query: "Q3营收"}, id: "call_002"},
   ])
③ ToolMessage(content="Q2营收4.5亿", tool_call_id="call_001")
④ ToolMessage(content="Q3营收5.2亿", tool_call_id="call_002")
⑤ AIMessage("Q2营收4.5亿，Q3营收5.2亿，环比增长15.6%。")
```

### artifact 在 RAG 中的价值

检索结果可能很长，用 artifact 保存完整文档，只把摘要发给模型：

```python
@tool(response_format="content_and_artifact")
def search_knowledge_base(query: str) -> tuple[str, list[dict]]:
    """在知识库中搜索相关文档"""
    docs = [
        {"content": "Q3营收5.2亿元...(500字)", "source": "财报.pdf", "page": 12},
        {"content": "同比增长15%...(300字)", "source": "分析报告.docx", "page": 5},
    ]
    summary = "找到2条相关文档：1) Q3营收5.2亿元 2) 同比增长15%"
    return summary, docs  # summary → content, docs → artifact
```

---

## 关键源码映射

| 概念 | 源码位置 |
|------|----------|
| `ToolCall` / `InvalidToolCall` / `ToolCallChunk` / `ToolMessage` | `langchain_core/messages/tool.py` |
| `tool_call()` 工厂函数 / `ToolOutputMixin` | `langchain_core/messages/tool.py` |
| `AIMessage.tool_calls` / `AIMessage.invalid_tool_calls` | `langchain_core/messages/ai.py` |
| `AIMessageChunk.__add__`（流式合并） | `langchain_core/messages/ai.py` |

---

## 总结

ToolCall 与 ToolMessage 消息流是 Agent 工具调用的底层通信协议：

1. **ToolCall** — 模型的调用请求，存在于 `AIMessage.tool_calls`
2. **ToolMessage** — 工具的执行结果，通过 `tool_call_id` 关联请求
3. **InvalidToolCall** — JSON 解析失败时的优雅降级
4. **ToolCallChunk** — 流式传输碎片，通过 `__add__` 拼接为完整 ToolCall
5. **artifact** — 完整数据留在应用层，摘要发给模型，节省 Token

掌握这套消息流，你就理解了 Agent 工具调用的"血液循环系统"。

---

**上一篇**: [03_核心概念_2_函数调用协议.md](./03_核心概念_2_函数调用协议.md) — bind_tools() 与格式转换
**下一篇**: [03_核心概念_4_Tool_Schema与参数验证.md](./03_核心概念_4_Tool_Schema与参数验证.md) — Pydantic BaseModel 与 JSON Schema
