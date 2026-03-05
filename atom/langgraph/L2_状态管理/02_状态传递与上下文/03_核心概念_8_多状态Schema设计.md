# 核心概念 8：多状态 Schema 设计

> 本文档是"状态传递与上下文"知识点的第8个核心概念

---

## 什么是多状态 Schema 设计？

**多状态 Schema 设计**是 LangGraph 中一种强大的状态管理模式，允许不同节点使用不同的状态类型（InputState、OutputState、OverallState、PrivateState），实现状态隔离、共享和精确控制。

**核心价值：**
- **状态隔离**：不同节点只访问需要的状态字段
- **接口清晰**：明确定义输入输出边界
- **安全性**：私有状态不暴露给外部
- **灵活性**：支持复杂的多节点协作

---

## 为什么需要多状态 Schema？

### 问题场景

在复杂的工作流中，如果所有节点共享同一个大状态对象：

```python
# ❌ 问题：所有节点都能访问所有状态
class State(TypedDict):
    user_input: str
    search_results: list
    api_key: str  # 敏感信息
    internal_cache: dict  # 内部状态
    final_output: str
```

**问题：**
1. **安全风险**：敏感信息（如 api_key）可能被不该访问的节点读取
2. **耦合度高**：节点依赖整个状态结构，难以独立测试
3. **接口不清晰**：不知道节点实际需要哪些字段
4. **维护困难**：状态结构变化影响所有节点

### 解决方案：多状态 Schema

```python
# ✅ 解决方案：不同节点使用不同的状态 Schema
class InputState(TypedDict):
    user_input: str  # 只暴露输入

class OutputState(TypedDict):
    final_output: str  # 只暴露输出

class OverallState(TypedDict):
    user_input: str
    search_results: list
    final_output: str

class PrivateState(TypedDict):
    api_key: str  # 私有状态，不暴露
    internal_cache: dict
```

---

## 四种状态 Schema 类型

### 1. InputState - 输入状态

**定义：** 图的输入接口，定义外部调用时传入的数据结构

**使用场景：**
- 定义图的公开 API
- 限制外部输入的字段
- 验证输入数据类型

**示例：**
```python
from typing import TypedDict

class InputState(TypedDict):
    user_input: str
    language: str  # 可选配置
```

**特点：**
- 只包含必要的输入字段
- 外部调用时只能传入这些字段
- 类型安全，防止错误输入

---

### 2. OutputState - 输出状态

**定义：** 图的输出接口，定义返回给外部的数据结构

**使用场景：**
- 定义图的返回值
- 隐藏内部状态
- 只暴露必要的结果

**示例：**
```python
class OutputState(TypedDict):
    graph_output: str
    confidence_score: float  # 可选的元数据
```

**特点：**
- 只包含需要返回的字段
- 内部状态不会泄露
- 清晰的输出接口

---

### 3. OverallState - 全局状态

**定义：** 图内部的完整状态，包含所有节点可能需要的字段

**使用场景：**
- 节点间共享数据
- 存储中间结果
- 协调多个节点

**示例：**
```python
class OverallState(TypedDict):
    user_input: str  # 来自 InputState
    search_results: list  # 中间结果
    processed_data: dict  # 中间结果
    graph_output: str  # 最终输出
```

**特点：**
- 包含所有公共字段
- 节点可以读写任意字段
- 是 InputState 和 OutputState 的超集

---

### 4. PrivateState - 私有状态

**定义：** 节点内部的私有状态，不暴露给其他节点或外部

**使用场景：**
- 存储敏感信息（API key、密码）
- 缓存中间计算结果
- 节点内部的临时数据

**示例：**
```python
class PrivateState(TypedDict):
    api_key: str  # 敏感信息
    cache: dict  # 内部缓存
    debug_info: list  # 调试信息
```

**特点：**
- 只在特定节点内可见
- 不会传递给其他节点
- 不会出现在输出中

---

## 完整示例：多状态 Schema 实战

### 场景：智能搜索助手

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

# 1. 定义四种状态 Schema
class InputState(TypedDict):
    """用户输入接口"""
    user_input: str

class OutputState(TypedDict):
    """返回结果接口"""
    graph_output: str

class OverallState(TypedDict):
    """内部完整状态"""
    user_input: str
    search_query: str
    search_results: list
    graph_output: str

class PrivateState(TypedDict):
    """私有状态（API key 等）"""
    api_key: str
    cache: dict

# 2. 定义节点函数
def query_rewrite_node(state: InputState) -> OverallState:
    """
    节点1：查询改写
    - 输入：InputState（只读取 user_input）
    - 输出：OverallState（写入 search_query）
    """
    user_input = state["user_input"]
    # 改写查询逻辑
    search_query = f"优化后的查询: {user_input}"
    return {"search_query": search_query}

def search_node(state: OverallState, runtime: Runtime[PrivateState]) -> OverallState:
    """
    节点2：搜索
    - 输入：OverallState（读取 search_query）
    - Runtime：PrivateState（读取 api_key）
    - 输出：OverallState（写入 search_results）
    """
    search_query = state["search_query"]
    api_key = runtime.context["api_key"]  # 从私有状态读取

    # 模拟搜索
    search_results = [
        {"title": "结果1", "content": "内容1"},
        {"title": "结果2", "content": "内容2"}
    ]
    return {"search_results": search_results}

def generate_response_node(state: OverallState) -> OutputState:
    """
    节点3：生成回复
    - 输入：OverallState（读取 search_results）
    - 输出：OutputState（写入 graph_output）
    """
    search_results = state["search_results"]
    # 生成回复逻辑
    output = f"找到 {len(search_results)} 个结果"
    return {"graph_output": output}

# 3. 构建图
builder = StateGraph(
    OverallState,  # 内部使用 OverallState
    input_schema=InputState,  # 输入接口
    output_schema=OutputState,  # 输出接口
    context_schema=PrivateState  # 私有上下文
)

builder.add_node("query_rewrite", query_rewrite_node)
builder.add_node("search", search_node)
builder.add_node("generate", generate_response_node)

builder.add_edge(START, "query_rewrite")
builder.add_edge("query_rewrite", "search")
builder.add_edge("search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# 4. 调用图
result = graph.invoke(
    {"user_input": "LangGraph 教程"},  # 只传入 InputState 字段
    context={"api_key": "secret_key", "cache": {}}  # 私有上下文
)

print(result)  # 只返回 OutputState 字段
# 输出: {'graph_output': '找到 2 个结果'}
```

**运行结果：**
```
{'graph_output': '找到 2 个结果'}
```

**关键观察：**
1. 外部调用只需要传入 `user_input`（InputState）
2. 内部节点可以访问完整的 OverallState
3. 敏感信息（api_key）通过 context 传递，不出现在状态中
4. 最终只返回 `graph_output`（OutputState）

---

## 双重类比

### 类比 1：前端组件的 Props 和 State

**前端类比：**
```typescript
// InputState = Props（父组件传入）
interface Props {
  userInput: string;
}

// PrivateState = 组件内部 State（不暴露）
interface State {
  apiKey: string;
  cache: Map<string, any>;
}

// OutputState = 返回值（emit 给父组件）
interface Output {
  result: string;
}

// OverallState = Props + State（组件内部可访问）
```

**相似点：**
- InputState 像 Props，从外部传入
- PrivateState 像组件内部 State，不暴露
- OutputState 像 emit 的事件，返回给外部
- OverallState 像组件内部可访问的完整数据

---

### 类比 2：餐厅的点餐流程

**日常生活类比：**

| LangGraph 概念 | 餐厅类比 |
|----------------|----------|
| InputState | 顾客的点餐单（只包含菜品名称） |
| PrivateState | 厨房的配方和库存（顾客看不到） |
| OverallState | 厨房内部的完整订单信息（包含备注、优先级） |
| OutputState | 端上桌的菜品（顾客最终得到的） |

**流程：**
1. 顾客点餐（InputState）：只需要说菜名
2. 厨房内部（OverallState）：知道完整的订单信息、备注、优先级
3. 厨师使用配方（PrivateState）：配方和库存不告诉顾客
4. 端菜上桌（OutputState）：顾客只得到最终的菜品

---

## 状态流转路径

```
外部调用
  ↓
InputState（输入接口）
  ↓
OverallState（内部状态）
  ↓
节点1 → 节点2 → 节点3
  ↑         ↑
  PrivateState（私有上下文，通过 Runtime 访问）
  ↓
OutputState（输出接口）
  ↓
返回给外部
```

**关键点：**
1. InputState → OverallState：自动合并
2. OverallState 在节点间流转
3. PrivateState 通过 Runtime 访问，不参与状态流转
4. OverallState → OutputState：自动提取

---

## 状态 Schema 设计原则

### 1. 最小暴露原则

**原则：** 只暴露必要的字段，隐藏内部实现

```python
# ✅ 好的设计
class InputState(TypedDict):
    query: str  # 只暴露必要的输入

class OutputState(TypedDict):
    answer: str  # 只暴露最终结果

# ❌ 不好的设计
class InputState(TypedDict):
    query: str
    internal_config: dict  # 不应该暴露内部配置
    debug_mode: bool  # 不应该暴露调试选项
```

---

### 2. 类型安全原则

**原则：** 使用 TypedDict 定义所有 Schema，利用类型检查

```python
from typing import TypedDict, Literal

# ✅ 好的设计：类型明确
class InputState(TypedDict):
    query: str
    language: Literal["en", "zh"]  # 限制可选值
    max_results: int

# ❌ 不好的设计：类型不明确
class InputState(TypedDict):
    query: str
    options: dict  # 太宽泛，不知道包含什么
```

---

### 3. 单一职责原则

**原则：** 每个 Schema 只负责一个职责

```python
# ✅ 好的设计：职责清晰
class InputState(TypedDict):
    user_query: str  # 只负责输入

class SearchState(TypedDict):
    search_results: list  # 只负责搜索结果

class OutputState(TypedDict):
    final_answer: str  # 只负责输出

# ❌ 不好的设计：职责混乱
class State(TypedDict):
    user_query: str
    search_results: list
    final_answer: str
    api_key: str  # 混合了输入、中间状态、输出、私有信息
```

---

### 4. 向后兼容原则

**原则：** 新增字段时保持向后兼容

```python
from typing import TypedDict, NotRequired

# ✅ 好的设计：新增字段使用 NotRequired
class InputState(TypedDict):
    query: str
    language: NotRequired[str]  # 新增字段，可选

# ❌ 不好的设计：新增必填字段会破坏兼容性
class InputState(TypedDict):
    query: str
    language: str  # 新增必填字段，破坏兼容性
```

---

## 高级模式：嵌套状态 Schema

### 场景：多 Agent 系统

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

# 主图状态
class MainState(TypedDict):
    user_query: str
    agent1_result: str
    agent2_result: str
    final_output: str

# Agent1 的状态（子图）
class Agent1State(TypedDict):
    query: str  # 从主图传入
    result: str  # 返回给主图

# Agent2 的状态（子图）
class Agent2State(TypedDict):
    query: str
    result: str

# Agent1 子图
def build_agent1_graph():
    def agent1_node(state: Agent1State) -> Agent1State:
        return {"result": f"Agent1 处理: {state['query']}"}

    builder = StateGraph(Agent1State)
    builder.add_node("process", agent1_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile()

# Agent2 子图
def build_agent2_graph():
    def agent2_node(state: Agent2State) -> Agent2State:
        return {"result": f"Agent2 处理: {state['query']}"}

    builder = StateGraph(Agent2State)
    builder.add_node("process", agent2_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile()

# 主图
def build_main_graph():
    agent1_graph = build_agent1_graph()
    agent2_graph = build_agent2_graph()

    def call_agent1(state: MainState) -> MainState:
        result = agent1_graph.invoke({"query": state["user_query"]})
        return {"agent1_result": result["result"]}

    def call_agent2(state: MainState) -> MainState:
        result = agent2_graph.invoke({"query": state["user_query"]})
        return {"agent2_result": result["result"]}

    def merge_results(state: MainState) -> MainState:
        output = f"{state['agent1_result']} + {state['agent2_result']}"
        return {"final_output": output}

    builder = StateGraph(MainState)
    builder.add_node("agent1", call_agent1)
    builder.add_node("agent2", call_agent2)
    builder.add_node("merge", merge_results)

    builder.add_edge(START, "agent1")
    builder.add_edge(START, "agent2")
    builder.add_edge("agent1", "merge")
    builder.add_edge("agent2", "merge")
    builder.add_edge("merge", END)

    return builder.compile()

# 使用
main_graph = build_main_graph()
result = main_graph.invoke({"user_query": "测试查询"})
print(result["final_output"])
```

**关键点：**
1. 主图和子图使用不同的状态 Schema
2. 主图通过节点函数调用子图
3. 子图的输入输出通过状态映射传递

---

## 常见问题与解决方案

### 问题 1：状态字段冲突

**问题：** InputState 和 OutputState 有同名字段，但类型不同

```python
# ❌ 问题
class InputState(TypedDict):
    data: str  # 输入是字符串

class OutputState(TypedDict):
    data: list  # 输出是列表

class OverallState(TypedDict):
    data: ???  # 冲突！
```

**解决方案：** 使用不同的字段名

```python
# ✅ 解决方案
class InputState(TypedDict):
    input_data: str

class OutputState(TypedDict):
    output_data: list

class OverallState(TypedDict):
    input_data: str
    output_data: list
```

---

### 问题 2：节点需要访问私有状态

**问题：** 节点需要读取 API key 等私有信息

```python
# ❌ 问题：将私有信息放在 OverallState
class OverallState(TypedDict):
    api_key: str  # 不安全，会暴露

# ✅ 解决方案：使用 Runtime Context
class PrivateState(TypedDict):
    api_key: str

def node(state: OverallState, runtime: Runtime[PrivateState]):
    api_key = runtime.context["api_key"]  # 安全访问
```

---

### 问题 3：状态 Schema 过于复杂

**问题：** OverallState 包含太多字段，难以维护

```python
# ❌ 问题
class OverallState(TypedDict):
    field1: str
    field2: int
    field3: list
    # ... 50 个字段
```

**解决方案：** 拆分成多个子状态

```python
# ✅ 解决方案
class UserState(TypedDict):
    user_id: str
    user_name: str

class SearchState(TypedDict):
    query: str
    results: list

class OverallState(TypedDict):
    user: UserState  # 嵌套状态
    search: SearchState
```

---

## 引用来源

本文档基于以下资料编写：

1. **官方文档**：
   - `reference/context7_langgraph_01.md` - LangGraph 官方文档（多状态 Schema 定义）
   - 代码示例来自官方文档的 StateGraph API 说明

2. **技术博客**：
   - `reference/fetch_状态传递_01.md` - Medium: State Management in LangGraph
   - `reference/fetch_状态传递_03.md` - CloudThat: LangGraph State

3. **社区讨论**：
   - `reference/fetch_状态传递_11.md` - Reddit: 不同 State Schema 的使用

4. **源码分析**：
   - `reference/source_状态传递_01.md` - LangGraph 源码分析

---

## 总结

**多状态 Schema 设计**是 LangGraph 中实现状态隔离、接口清晰、安全可靠的关键模式：

1. **InputState**：定义输入接口，限制外部输入
2. **OutputState**：定义输出接口，隐藏内部状态
3. **OverallState**：内部完整状态，节点间共享
4. **PrivateState**：私有状态，通过 Runtime 访问

**设计原则：**
- 最小暴露：只暴露必要的字段
- 类型安全：使用 TypedDict 定义
- 单一职责：每个 Schema 职责清晰
- 向后兼容：新增字段使用 NotRequired

**实战价值：**
- 提高代码可维护性
- 增强系统安全性
- 支持复杂的多节点协作
- 便于单元测试和调试

---

**文档版本：** v1.0
**生成时间：** 2026-02-26
**文档长度：** 约 450 行
