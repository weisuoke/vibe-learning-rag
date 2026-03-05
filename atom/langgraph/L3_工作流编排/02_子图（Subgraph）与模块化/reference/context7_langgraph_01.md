---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 02_子图（Subgraph）与模块化
context7_query: subgraph modular design state passing, Command.PARENT, streaming subgraphs, checkpointer
---

# Context7 文档：LangGraph 子图（Subgraph）

## 文档来源
- 库名称：langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/use-subgraphs

## 关键信息提取

### 1. 子图的两种集成方式

#### 方式一：共享状态键（直接添加编译后的子图）

当父图和子图有共享的 state key 时，可以直接将子图作为节点添加：

```python
class SubgraphState(TypedDict):
    foo: str  # 与父图共享
    bar: str  # 子图私有

subgraph = subgraph_builder.compile()

builder = StateGraph(ParentState)
builder.add_node("node_2", subgraph)  # 直接添加
```

共享 key 自动传递，子图私有 key 不影响父图。

#### 方式二：状态转换（包装函数调用子图）

当父子图状态完全不同时，用包装函数手动转换状态：

```python
def call_subgraph(state: State):
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    return {"foo": subgraph_output["bar"]}

builder.add_node("node_1", call_subgraph)
```

### 2. 多层嵌套子图

支持 parent → child → grandchild 多层嵌套，每层有独立的状态：

```python
# grandchild 只能访问 GrandChildState
# child 只能访问 ChildState
# parent 只能访问 ParentState
```

通过 `stream(input, subgraphs=True)` 可以监控所有层级的输出。

### 3. Command.PARENT 跨图导航

子图节点可以使用 `Command(graph=Command.PARENT)` 导航到父图节点：

```python
def node_a(state: State):
    return Command(
        update={"foo": value},
        goto="other_node",
        graph=Command.PARENT,
    )
```

**注意**：使用 Command.PARENT 更新共享 key 时，父图必须定义 reducer。

### 4. 子图流式输出

设置 `subgraphs=True` 可以流式获取子图输出：

```python
for chunk in graph.stream({"foo": "foo"}, stream_mode="updates", subgraphs=True):
    print(chunk)
# 输出格式: (namespace_tuple, {node_name: state_update})
```

### 5. 子图 Checkpointer 策略

- 父图编译时设置 checkpointer，自动传播到子图
- 子图可以 `compile(checkpointer=True)` 拥有独立记忆
- 独立 checkpointer 适用于多代理系统中各代理需要独立历史记录的场景

### 6. 多代理 Handoff

Command.PARENT 特别适合多代理系统中的 handoff：
- 每个代理是一个子图
- 通过 Command(goto="other_agent", graph=Command.PARENT) 切换代理
- 支持同时更新状态和控制流
