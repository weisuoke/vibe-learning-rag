# 核心概念 4：PregelNode 状态绑定

> **文档类型**：核心概念深度解析
> **知识点**：L2_状态管理 / 02_状态传递与上下文
> **预计阅读时间**：15 分钟

---

## 文档来源

本文档基于以下资料综合整理：

**源码分析**：
- `libs/langgraph/langgraph/pregel/_read.py` - PregelNode 定义
- `libs/langgraph/langgraph/channels/base.py` - Channel 机制
- `libs/langgraph/langgraph/pregel/_write.py` - 写入机制

**官方文档**：
- Context7 LangGraph 文档 - 节点参数类型
- LangGraph 官方文档 - StateGraph API

**社区资源**：
- Medium: State Management in LangGraph
- LangChain Blog: Context Engineering

---

## 一、核心定义

### 什么是 PregelNode？

**PregelNode 是 LangGraph 中节点与状态系统的连接器，负责声明节点的输入、输出和触发条件。**

它不是节点本身，而是节点的"包装器"，管理着节点如何与 Channel（状态存储）交互。

### 源码定义

```python
class PregelNode:
    """A node in a Pregel graph."""

    channels: str | list[str]
    """The channels that will be passed as input to `bound`."""

    triggers: list[str]
    """If any of these channels is written to, this node will be triggered."""

    mapper: Callable[[Any], Any] | None
    """A function to transform the input before passing it to `bound`."""

    writers: list[Runnable]
    """Writers that will be executed after `bound`."""

    bound: Runnable[Any, Any]
    """The main logic of the node."""
```

**来源**：`libs/langgraph/langgraph/pregel/_read.py`

---

## 二、三大核心参数

### 2.1 channels - 输入声明

**作用**：声明节点需要从哪些 Channel 读取数据。

**类型**：
- `str` - 单个 channel
- `list[str]` - 多个 channels

**工作机制**：
```python
# 单个 channel
node = PregelNode(
    channels="messages",  # 只读取 messages channel
    bound=my_function
)

# 多个 channels
node = PregelNode(
    channels=["messages", "user_input", "context"],  # 读取多个
    bound=my_function
)
```

**底层实现**：
```python
# 来自 pregel/_read.py
class ChannelRead(RunnableCallable):
    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        if mapper:
            return mapper(read(select, fresh))
        else:
            return read(select, fresh)
```

**关键点**：
- `channels` 参数会被转换为 `ChannelRead` 对象
- 通过 `CONFIG_KEY_READ` 从 RunnableConfig 中注入读取函数
- 支持 `mapper` 进行数据转换

---

### 2.2 triggers - 触发条件

**作用**：声明哪些 Channel 被写入时会触发该节点执行。

**类型**：`list[str]`

**工作机制**：
```python
node = PregelNode(
    channels=["messages", "context"],
    triggers=["messages"],  # 只有 messages 被写入时才触发
    bound=my_function
)
```

**触发逻辑**：
```
1. 某个节点写入 "messages" channel
   ↓
2. Pregel 引擎检测到 "messages" 被更新
   ↓
3. 查找所有 triggers 包含 "messages" 的节点
   ↓
4. 将这些节点加入执行队列
```

**与 channels 的区别**：
- `channels`：节点**读取**哪些数据
- `triggers`：节点**何时执行**

**示例场景**：
```python
# 场景：验证节点只在生成完成后执行
generator_node = PregelNode(
    channels=["user_input"],
    triggers=["user_input"],
    bound=generate_response
)

validator_node = PregelNode(
    channels=["generated_response"],
    triggers=["generated_response"],  # 等待生成完成
    bound=validate_response
)
```

---

### 2.3 writers - 输出声明

**作用**：声明节点执行后如何写入状态。

**类型**：`list[Runnable]`（通常是 `ChannelWrite` 对象）

**工作机制**：
```python
# 来自 pregel/_write.py
class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = PASSTHROUGH
    skip_none: bool = False
    mapper: Callable | None = None

class ChannelWrite(RunnableCallable):
    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        write: TYPE_SEND = config[CONF][CONFIG_KEY_SEND]
        write(_assemble_writes(writes))
```

**写入模式**：

1. **PASSTHROUGH 模式**（直接传递）：
```python
ChannelWriteEntry(
    channel="output",
    value=PASSTHROUGH  # 直接传递节点返回值
)
```

2. **显式值模式**：
```python
ChannelWriteEntry(
    channel="status",
    value="completed"  # 写入固定值
)
```

3. **Mapper 转换模式**：
```python
ChannelWriteEntry(
    channel="processed_data",
    value=PASSTHROUGH,
    mapper=lambda x: {"result": x, "timestamp": time.time()}
)
```

4. **跳过 None 模式**：
```python
ChannelWriteEntry(
    channel="optional_field",
    value=PASSTHROUGH,
    skip_none=True  # 如果返回 None 则不写入
)
```

---

## 三、双重类比

### 3.1 前端开发类比

| PregelNode 概念 | 前端类比 | 说明 |
|----------------|----------|------|
| **channels** | React props | 声明组件需要哪些输入数据 |
| **triggers** | useEffect 依赖数组 | 声明何时重新执行 |
| **writers** | setState 调用 | 声明如何更新状态 |
| **bound** | 组件渲染函数 | 实际的业务逻辑 |
| **mapper** | props 转换函数 | 数据预处理 |

**React 示例对比**：
```javascript
// React 组件
function MessageProcessor({ messages, context }) {  // ← channels
  const [output, setOutput] = useState(null);

  useEffect(() => {  // ← triggers
    const result = processMessages(messages);  // ← bound
    setOutput(result);  // ← writers
  }, [messages]);  // ← triggers 依赖

  return <div>{output}</div>;
}

// LangGraph PregelNode
node = PregelNode(
    channels=["messages", "context"],  # 输入
    triggers=["messages"],             # 触发条件
    bound=process_messages,            # 逻辑
    writers=[ChannelWrite([           # 输出
        ChannelWriteEntry("output", PASSTHROUGH)
    ])]
)
```

---

### 3.2 日常生活类比

**PregelNode = 工厂生产线上的工作站**

| 概念 | 生活类比 |
|------|----------|
| **channels** | 工作站需要的原材料清单（"需要螺丝、电路板、外壳"） |
| **triggers** | 开工信号（"当传送带上有新零件到达时开始工作"） |
| **bound** | 工人的操作流程（"组装、测试、包装"） |
| **writers** | 产出物放置规则（"合格品放左边，次品放右边"） |
| **mapper** | 原材料预处理（"先清洗零件再使用"） |

**完整流程**：
```
1. 传送带（Channel）上有新零件到达
   ↓ (triggers 检测到)
2. 工作站启动
   ↓ (channels 读取)
3. 工人从指定位置拿取原材料
   ↓ (mapper 预处理)
4. 按照操作流程加工
   ↓ (bound 执行)
5. 将产品放到指定位置
   ↓ (writers 写入)
6. 触发下一个工作站
```

---

## 四、实战代码示例

### 4.1 基础示例：简单的消息处理节点

```python
"""
PregelNode 基础示例
演示：channels、triggers、writers 的基本使用
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# ===== 1. 定义状态 =====
class State(TypedDict):
    messages: Annotated[list, add_messages]
    processed_count: int

# ===== 2. 定义节点函数 =====
def process_message(state: State) -> dict:
    """处理消息的节点"""
    messages = state["messages"]
    last_message = messages[-1]

    # 简单处理：转换为大写
    processed = AIMessage(content=last_message.content.upper())

    return {
        "messages": [processed],
        "processed_count": state.get("processed_count", 0) + 1
    }

# ===== 3. 构建图 =====
graph = StateGraph(State)

# 添加节点（内部会创建 PregelNode）
graph.add_node("processor", process_message)

# 定义边
graph.add_edge(START, "processor")
graph.add_edge("processor", END)

# 编译
app = graph.compile()

# ===== 4. 运行 =====
result = app.invoke({
    "messages": [HumanMessage(content="hello world")],
    "processed_count": 0
})

print("处理后的消息:", result["messages"][-1].content)
print("处理次数:", result["processed_count"])
```

**输出**：
```
处理后的消息: HELLO WORLD
处理次数: 1
```

---

### 4.2 进阶示例：多 Channel 触发与条件写入

```python
"""
PregelNode 进阶示例
演示：多 channel 读取、条件触发、mapper 使用
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from operator import add

# ===== 1. 复杂状态定义 =====
class AgentState(TypedDict):
    user_input: str
    search_results: Annotated[list, add]
    generated_response: str
    validation_status: Literal["pending", "valid", "invalid"]
    retry_count: int

# ===== 2. 搜索节点 =====
def search_node(state: AgentState) -> dict:
    """
    channels: ["user_input"]
    triggers: ["user_input"]
    writers: ["search_results"]
    """
    query = state["user_input"]

    # 模拟搜索
    results = [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'"
    ]

    return {"search_results": results}

# ===== 3. 生成节点 =====
def generate_node(state: AgentState) -> dict:
    """
    channels: ["user_input", "search_results"]
    triggers: ["search_results"]
    writers: ["generated_response", "validation_status"]
    """
    query = state["user_input"]
    results = state["search_results"]

    # 模拟生成
    response = f"Based on {len(results)} results, here's the answer to '{query}'"

    return {
        "generated_response": response,
        "validation_status": "pending"
    }

# ===== 4. 验证节点 =====
def validate_node(state: AgentState) -> dict:
    """
    channels: ["generated_response"]
    triggers: ["generated_response"]
    writers: ["validation_status"]
    """
    response = state["generated_response"]

    # 简单验证：检查长度
    is_valid = len(response) > 20

    return {
        "validation_status": "valid" if is_valid else "invalid"
    }

# ===== 5. 构建图 =====
graph = StateGraph(AgentState)

graph.add_node("search", search_node)
graph.add_node("generate", generate_node)
graph.add_node("validate", validate_node)

graph.add_edge(START, "search")
graph.add_edge("search", "generate")
graph.add_edge("generate", "validate")
graph.add_edge("validate", END)

app = graph.compile()

# ===== 6. 运行 =====
result = app.invoke({
    "user_input": "What is LangGraph?",
    "search_results": [],
    "generated_response": "",
    "validation_status": "pending",
    "retry_count": 0
})

print("搜索结果数:", len(result["search_results"]))
print("生成的回答:", result["generated_response"])
print("验证状态:", result["validation_status"])
```

**输出**：
```
搜索结果数: 2
生成的回答: Based on 2 results, here's the answer to 'What is LangGraph?'
验证状态: valid
```

---

### 4.3 高级示例：手动创建 PregelNode（源码级别）

```python
"""
手动创建 PregelNode
演示：直接使用 PregelNode 类（通常不需要，仅用于理解底层机制）
"""

from langgraph.pregel import PregelNode
from langgraph.pregel._read import ChannelRead
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry, PASSTHROUGH
from langchain_core.runnables import RunnableLambda

# ===== 1. 定义节点逻辑 =====
def my_logic(state: dict) -> dict:
    """节点的实际逻辑"""
    return {"output": state["input"].upper()}

# ===== 2. 手动创建 PregelNode =====
node = PregelNode(
    # 输入声明
    channels=["input"],

    # 触发条件
    triggers=["input"],

    # 输入转换（可选）
    mapper=lambda x: {"input": x["input"].strip()},

    # 节点逻辑
    bound=RunnableLambda(my_logic),

    # 输出声明
    writers=[
        ChannelWrite([
            ChannelWriteEntry(
                channel="output",
                value=PASSTHROUGH,
                skip_none=False
            )
        ])
    ]
)

print("PregelNode 创建成功")
print(f"- 输入 channels: {node.channels}")
print(f"- 触发 triggers: {node.triggers}")
print(f"- 是否有 mapper: {node.mapper is not None}")
print(f"- 输出 writers 数量: {len(node.writers)}")
```

**输出**：
```
PregelNode 创建成功
- 输入 channels: ['input']
- 触发 triggers: ['input']
- 是否有 mapper: True
- 输出 writers 数量: 1
```

---

## 五、常见误区

### 误区 1：混淆 channels 和 triggers

❌ **错误理解**：
"channels 和 triggers 是一样的，都是声明节点需要的输入"

✅ **正确理解**：
- `channels`：节点**读取**哪些数据（输入声明）
- `triggers`：节点**何时执行**（触发条件）

**示例**：
```python
# 场景：验证节点需要读取多个数据，但只在生成完成时触发
validator_node = PregelNode(
    channels=["generated_response", "user_input", "context"],  # 读取 3 个
    triggers=["generated_response"],  # 只在这 1 个更新时触发
    bound=validate
)
```

---

### 误区 2：认为 writers 必须与节点返回值一致

❌ **错误理解**：
"节点返回什么，writers 就必须写入什么"

✅ **正确理解**：
- `writers` 可以写入**固定值**
- `writers` 可以通过 `mapper` **转换**节点返回值
- `writers` 可以**跳过** None 值

**示例**：
```python
# 节点返回 {"result": "data"}
# 但 writers 可以写入不同的内容
writers=[
    ChannelWrite([
        # 写入固定状态
        ChannelWriteEntry("status", "completed"),

        # 转换后写入
        ChannelWriteEntry(
            "processed_result",
            PASSTHROUGH,
            mapper=lambda x: {"data": x["result"], "timestamp": time.time()}
        )
    ])
]
```

---

### 误区 3：忽略 mapper 的作用

❌ **错误理解**：
"mapper 只是可选的，不重要"

✅ **正确理解**：
- `mapper` 可以**预处理**输入数据
- `mapper` 可以**适配**不同的状态 Schema
- `mapper` 可以**过滤**不需要的字段

**示例**：
```python
# 场景：节点只需要 messages 的最后一条
node = PregelNode(
    channels=["messages"],
    mapper=lambda state: {"last_message": state["messages"][-1]},  # 预处理
    bound=process_last_message
)
```

---

## 六、最佳实践

### 6.1 明确声明 triggers

**推荐**：
```python
# 明确声明触发条件
node = PregelNode(
    channels=["messages", "context"],
    triggers=["messages"],  # 只在 messages 更新时触发
    bound=my_function
)
```

**不推荐**：
```python
# 依赖默认行为（可能不明确）
node = PregelNode(
    channels=["messages", "context"],
    # 没有 triggers，行为不明确
    bound=my_function
)
```

---

### 6.2 使用 skip_none 避免覆盖

**推荐**：
```python
writers=[
    ChannelWrite([
        ChannelWriteEntry(
            "optional_field",
            PASSTHROUGH,
            skip_none=True  # 如果返回 None 则不写入
        )
    ])
]
```

**场景**：
- 节点可能不总是返回某个字段
- 避免用 None 覆盖已有的值

---

### 6.3 使用 mapper 适配状态 Schema

**推荐**：
```python
# 主图状态
class MainState(TypedDict):
    messages: list
    user_id: str
    context: dict

# 子图状态（更简单）
class SubState(TypedDict):
    query: str

# 使用 mapper 适配
subgraph_node = PregelNode(
    channels=["messages", "user_id"],
    mapper=lambda state: {
        "query": state["messages"][-1].content,
        "user": state["user_id"]
    },
    bound=subgraph
)
```

---

## 七、与其他概念的关系

### 7.1 PregelNode vs StateGraph

- **StateGraph**：图的顶层抽象，管理所有 Channel 和 Node
- **PregelNode**：单个节点的包装器，管理节点与 Channel 的连接

```
StateGraph
  ├── channels: dict[str, BaseChannel]  # 所有状态存储
  └── nodes: dict[str, PregelNode]      # 所有节点
        ├── channels: 输入声明
        ├── triggers: 触发条件
        └── writers: 输出声明
```

---

### 7.2 PregelNode vs Channel

- **Channel**：状态存储的容器（如 LastValue, BinaryOperator）
- **PregelNode**：声明如何读写 Channel

```
Channel (存储)
    ↑ read
PregelNode (声明)
    ↓ write
Channel (更新)
```

---

### 7.3 PregelNode vs RunnableConfig

- **RunnableConfig**：运行时配置的传递载体
- **PregelNode**：通过 RunnableConfig 注入读写函数

```python
# PregelNode 通过 RunnableConfig 获取读写函数
config[CONF][CONFIG_KEY_READ]  # 读取函数
config[CONF][CONFIG_KEY_SEND]  # 写入函数
```

---

## 八、参考资料

### 源码文件
1. `libs/langgraph/langgraph/pregel/_read.py` - PregelNode 定义
2. `libs/langgraph/langgraph/pregel/_write.py` - ChannelWrite 机制
3. `libs/langgraph/langgraph/channels/base.py` - BaseChannel 抽象

### 官方文档
1. [LangGraph StateGraph API](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph)
2. [LangGraph Nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes)

### 社区资源
1. [Medium: State Management in LangGraph](https://medium.com/algomart/state-management-in-langgraph-the-foundation-of-reliable-ai-workflows-db98dd1499ca)
2. [LangChain Blog: Context Engineering](https://blog.langchain.com/context-engineering-for-agents)

---

## 总结

**PregelNode 是 LangGraph 状态管理的核心组件，通过三大参数实现节点与状态的精确绑定：**

1. **channels**：声明输入（读取哪些数据）
2. **triggers**：声明触发（何时执行）
3. **writers**：声明输出（如何写入状态）

**关键要点**：
- PregelNode 不是节点本身，而是节点的"连接器"
- 通过 RunnableConfig 实现依赖注入
- 支持 mapper 进行数据转换
- 支持 skip_none 避免覆盖

**下一步学习**：
- 核心概念 5：状态流转路径
- 核心概念 6：RunnableConfig 上下文传递
