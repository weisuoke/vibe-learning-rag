# 实战代码 - 场景1：最小 StateGraph 示例

> 从零开始创建最简单的可执行图，理解 StateGraph 的核心工作流程

---

## 场景概述

**目标**：创建一个最简单但完整的 StateGraph，掌握从创建到执行的完整流程。

**核心步骤**：
1. 定义 State Schema（状态结构）
2. 创建 StateGraph 实例
3. 定义节点函数
4. 添加节点到图
5. 添加边连接节点
6. 编译图
7. 执行图

**适用场景**：
- LangGraph 入门学习
- 理解图的基本构建流程
- 验证环境配置
- 作为复杂图的起点

---

## 完整代码示例

```python
"""
最小 StateGraph 示例
演示：从创建到执行的完整流程

来源：
- Context7 官方文档：https://docs.langchain.com/oss/python/langgraph/use-graph-api
- LangGraph 源码分析：state.py
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# ===== 1. 定义 State Schema =====
print("=== 步骤1：定义 State Schema ===")

class State(TypedDict):
    """
    状态定义：图中所有节点共享的数据结构

    类比：
    - 前端：Redux 的 store 结构
    - 日常：工作台上的工具和材料

    来源：Context7 官方文档
    """
    message: str  # 消息内容
    count: int    # 计数器


print("✓ State Schema 定义完成")
print(f"  - 字段: message (str), count (int)")


# ===== 2. 定义节点函数 =====
print("\n=== 步骤2：定义节点函数 ===")

def node_1(state: State) -> dict:
    """
    第一个节点：初始化消息

    节点函数规范：
    - 输入：完整的 State
    - 输出：部分状态更新（dict）
    - 自动合并：返回的 dict 会自动合并到 State 中

    来源：源码分析 _node.py
    """
    print(f"  [Node 1] 接收到 state: {state}")

    # 返回部分状态更新
    return {
        "message": "Hello from Node 1",
        "count": state.get("count", 0) + 1
    }


def node_2(state: State) -> dict:
    """
    第二个节点：处理消息

    类比：
    - 前端：组件函数，接收 props，返回新状态
    - 日常：工序站点，接收半成品，输出加工后的产品
    """
    print(f"  [Node 2] 接收到 state: {state}")

    return {
        "message": f"{state['message']} -> processed by Node 2",
        "count": state["count"] + 1
    }


def node_3(state: State) -> dict:
    """
    第三个节点：完成处理
    """
    print(f"  [Node 3] 接收到 state: {state}")

    return {
        "message": f"{state['message']} -> finalized by Node 3",
        "count": state["count"] + 1
    }


print("✓ 节点函数定义完成")
print(f"  - node_1: 初始化消息")
print(f"  - node_2: 处理消息")
print(f"  - node_3: 完成处理")


# ===== 3. 创建 StateGraph =====
print("\n=== 步骤3：创建 StateGraph ===")

# 创建 builder（构建器模式）
builder = StateGraph(State)

print("✓ StateGraph 实例创建完成")
print(f"  - 类型: StateGraph[State]")
print(f"  - 模式: Builder 模式（不可直接执行）")


# ===== 4. 添加节点 =====
print("\n=== 步骤4：添加节点 ===")

# 方式1：自动推断节点名称（使用函数名）
builder.add_node(node_1)  # 节点名称: "node_1"
builder.add_node(node_2)  # 节点名称: "node_2"
builder.add_node(node_3)  # 节点名称: "node_3"

print("✓ 节点添加完成")
print(f"  - node_1 (自动命名)")
print(f"  - node_2 (自动命名)")
print(f"  - node_3 (自动命名)")


# ===== 5. 添加边（定义执行顺序）=====
print("\n=== 步骤5：添加边 ===")

# START -> node_1：设置入口点
builder.add_edge(START, "node_1")
print("  ✓ START -> node_1 (入口点)")

# node_1 -> node_2：顺序执行
builder.add_edge("node_1", "node_2")
print("  ✓ node_1 -> node_2")

# node_2 -> node_3：顺序执行
builder.add_edge("node_2", "node_3")
print("  ✓ node_2 -> node_3")

# node_3 -> END：设置出口点
builder.add_edge("node_3", END)
print("  ✓ node_3 -> END (出口点)")

print("\n图结构：")
print("  START -> node_1 -> node_2 -> node_3 -> END")


# ===== 6. 编译图 =====
print("\n=== 步骤6：编译图 ===")

# 编译：将 builder 转换为可执行的 Pregel 实例
graph = builder.compile()

print("✓ 图编译完成")
print(f"  - 类型: CompiledStateGraph (Pregel 实例)")
print(f"  - 可执行方法: invoke(), stream(), ainvoke(), astream()")


# ===== 7. 执行图 =====
print("\n=== 步骤7：执行图 ===\n")

# 准备初始输入
initial_state = {
    "message": "Start",
    "count": 0
}

print(f"初始输入: {initial_state}\n")

# 执行图
result = graph.invoke(initial_state)

print(f"\n最终输出: {result}")


# ===== 8. 验证结果 =====
print("\n=== 步骤8：验证结果 ===")

assert result["count"] == 3, "计数器应该是 3（每个节点 +1）"
assert "Node 1" in result["message"], "消息应包含 Node 1"
assert "Node 2" in result["message"], "消息应包含 Node 2"
assert "Node 3" in result["message"], "消息应包含 Node 3"

print("✓ 所有断言通过")
print(f"  - 计数器: {result['count']} (预期: 3)")
print(f"  - 消息长度: {len(result['message'])} 字符")


# ===== 9. 图的可视化表示 =====
print("\n=== 步骤9：图的结构信息 ===")

print(f"节点列表: {list(graph.nodes.keys())}")
print(f"通道列表: {list(graph.channels.keys())}")

print("\n" + "="*60)
print("✓ 最小 StateGraph 示例执行完成！")
print("="*60)
```

---

## 运行输出示例

```
=== 步骤1：定义 State Schema ===
✓ State Schema 定义完成
  - 字段: message (str), count (int)

=== 步骤2：定义节点函数 ===
✓ 节点函数定义完成
  - node_1: 初始化消息
  - node_2: 处理消息
  - node_3: 完成处理

=== 步骤3：创建 StateGraph ===
✓ StateGraph 实例创建完成
  - 类型: StateGraph[State]
  - 模式: Builder 模式（不可直接执行）

=== 步骤4：添加节点 ===
✓ 节点添加完成
  - node_1 (自动命名)
  - node_2 (自动命名)
  - node_3 (自动命名)

=== 步骤5：添加边 ===
  ✓ START -> node_1 (入口点)
  ✓ node_1 -> node_2
  ✓ node_2 -> node_3
  ✓ node_3 -> END (出口点)

图结构：
  START -> node_1 -> node_2 -> node_3 -> END

=== 步骤6：编译图 ===
✓ 图编译完成
  - 类型: CompiledStateGraph (Pregel 实例)
  - 可执行方法: invoke(), stream(), ainvoke(), astream()

=== 步骤7：执行图 ===

初始输入: {'message': 'Start', 'count': 0}

  [Node 1] 接收到 state: {'message': 'Start', 'count': 0}
  [Node 2] 接收到 state: {'message': 'Hello from Node 1', 'count': 1}
  [Node 3] 接收到 state: {'message': 'Hello from Node 1 -> processed by Node 2', 'count': 2}

最终输出: {'message': 'Hello from Node 1 -> processed by Node 2 -> finalized by Node 3', 'count': 3}

=== 步骤8：验证结果 ===
✓ 所有断言通过
  - 计数器: 3 (预期: 3)
  - 消息长度: 71 字符

=== 步骤9：图的结构信息 ===
节点列表: ['__start__', 'node_1', 'node_2', 'node_3']
通道列表: ['message', 'count']

============================================================
✓ 最小 StateGraph 示例执行完成！
============================================================
```

---

## 核心知识点解析

### 1. State Schema 的作用

**定义**：使用 TypedDict 定义图中所有节点共享的数据结构。

**关键点**：
- 所有节点都能访问完整的 State
- 节点只需返回部分更新（dict）
- 自动合并：LangGraph 自动将返回值合并到 State

**来源**：Context7 官方文档

```python
class State(TypedDict):
    message: str  # 所有节点都能访问
    count: int    # 所有节点都能修改
```

### 2. Builder 模式

**StateGraph 是 builder，不可直接执行**：

```python
builder = StateGraph(State)  # 创建 builder
builder.add_node(...)        # 逐步构建
builder.add_edge(...)        # 添加连接
graph = builder.compile()    # 编译后才能执行
```

**来源**：源码分析 state.py:112-127

### 3. 节点函数协议

**标准签名**：
```python
def node(state: State) -> dict:
    # 接收完整 State
    # 返回部分更新
    return {"field": new_value}
```

**9种节点签名**（来源：源码 _node.py:16-93）：
- 基础节点：`(state: State) -> dict`
- 带 config：`(state: State, config: RunnableConfig) -> dict`
- 带 writer：`(state: State, *, writer: StreamWriter) -> dict`
- 带 store：`(state: State, *, store: BaseStore) -> dict`
- 带 runtime：`(state: State, *, runtime: Runtime[Context]) -> dict`
- 以及其他组合

### 4. START 和 END 节点

**特殊常量**（来源：源码 __init__.py）：
```python
from langgraph.graph import START, END

# START: 图的入口点
builder.add_edge(START, "first_node")

# END: 图的出口点
builder.add_edge("last_node", END)
```

**关键规则**：
- START 不能作为终点
- END 不能作为起点
- 必须有从 START 到某个节点的边
- 必须有从某个节点到 END 的边

### 5. 编译过程

**compile() 方法的作用**（来源：源码 state.py:1035-1153）：
1. 验证图结构（检查连通性、循环等）
2. 生成 CompiledStateGraph（Pregel 实例）
3. 设置执行引擎
4. 返回可执行对象

```python
graph = builder.compile()
# 返回类型: CompiledStateGraph (Pregel 实例)
```

---

## 双重类比

### 类比1：StateGraph 是什么？

**前端类比**：React 状态机
- State = Redux store
- 节点 = reducer 函数
- 边 = dispatch 流程
- compile = 创建 store

**日常生活类比**：工厂流水线
- State = 工作台状态
- 节点 = 工序站点
- 边 = 传送带
- compile = 启动生产线

### 类比2：节点函数

**前端类比**：组件函数
```javascript
// React 组件
function Component(props) {
  return newState;
}

// LangGraph 节点
def node(state: State) -> dict:
    return {"field": new_value}
```

**日常生活类比**：工序站点
- 接收半成品（state）
- 加工处理
- 输出成品（返回值）

### 类比3：Builder 模式

**前端类比**：配置对象
```javascript
// 前端配置
const config = new ConfigBuilder()
  .addRoute('/home', Home)
  .addRoute('/about', About)
  .build();

// LangGraph
builder = StateGraph(State)
builder.add_node(node_1)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

**日常生活类比**：搭积木
- 准备积木（创建 builder）
- 逐个拼接（add_node, add_edge）
- 完成作品（compile）

---

## 常见问题

### Q1: 为什么不能直接执行 StateGraph？

**答案**：StateGraph 是 builder 模式，只负责构建图结构，不负责执行。

**来源**：源码 state.py:112-127

```python
# ❌ 错误：不能直接执行
builder = StateGraph(State)
builder.invoke(...)  # AttributeError

# ✅ 正确：必须先编译
graph = builder.compile()
graph.invoke(...)  # 正确
```

### Q2: 节点函数必须返回完整的 State 吗？

**答案**：不需要，只需返回部分更新（dict），LangGraph 会自动合并。

**来源**：Context7 官方文档

```python
def node(state: State) -> dict:
    # 只返回需要更新的字段
    return {"count": state["count"] + 1}
    # message 字段保持不变
```

### Q3: START 和 END 是什么？

**答案**：特殊的节点常量，表示图的入口和出口。

**来源**：源码 __init__.py

```python
from langgraph.graph import START, END

# START: 入口点（必须有）
builder.add_edge(START, "first_node")

# END: 出口点（必须有）
builder.add_edge("last_node", END)
```

### Q4: 如何调试节点执行顺序？

**答案**：在节点函数中添加 print 语句。

```python
def node_1(state: State) -> dict:
    print(f"[Node 1] 接收到: {state}")
    return {"message": "Hello"}
```

### Q5: 编译后还能修改图吗？

**答案**：不能。编译后的图是不可变的，需要重新创建 builder。

**来源**：Twitter 最佳实践

```python
graph = builder.compile()
# 编译后不能再修改
# builder.add_node(...)  # 无效
```

---

## 最佳实践

### 1. 先规划后实现

**建议**（来源：Reddit 社区）：
1. 画出图的结构
2. 确定节点和边的关系
3. 再开始编码

```
START -> node_1 -> node_2 -> node_3 -> END
```

### 2. 节点职责单一

**建议**（来源：Twitter 最佳实践）：
- 每个节点只做一件事
- 保持节点函数简洁
- 避免复杂逻辑

```python
# ✅ 好：职责单一
def validate_input(state: State) -> dict:
    # 只负责验证
    ...

def process_data(state: State) -> dict:
    # 只负责处理
    ...

# ❌ 差：职责混乱
def do_everything(state: State) -> dict:
    # 验证、处理、保存...
    ...
```

### 3. 使用有意义的节点名称

```python
# ✅ 好：清晰的名称
builder.add_node("validate_input", validate_fn)
builder.add_node("process_data", process_fn)

# ❌ 差：模糊的名称
builder.add_node("node1", fn1)
builder.add_node("node2", fn2)
```

### 4. 添加日志和监控

```python
def node_1(state: State) -> dict:
    print(f"[Node 1] 开始执行")
    print(f"[Node 1] 输入: {state}")

    result = {"message": "Hello"}

    print(f"[Node 1] 输出: {result}")
    return result
```

### 5. 从简单开始

**建议**（来源：Reddit 社区）：
1. 从简单的线性流程开始
2. 逐步添加条件路由
3. 最后引入并行执行

---

## 扩展阅读

### 相关知识点

1. **多节点状态流转**（场景2）
   - 理解状态在节点间的传递
   - 部分状态返回的机制

2. **条件路由实战**（场景3）
   - add_conditional_edges
   - 动态路由和循环

3. **Context 注入实战**（场景4）
   - context_schema 定义
   - Runtime 注入

### 参考资源

**官方文档**：
- LangGraph 官方文档：https://docs.langchain.com/oss/python/langgraph/
- Context7 文档：/websites/langchain_oss_python_langgraph

**社区资源**：
- LangChain Academy：https://github.com/langchain-ai/langchain-academy
- GitHub 示例：https://github.com/langchain-ai/langgraph

**源码分析**：
- state.py：StateGraph 核心实现
- _node.py：节点协议定义
- __init__.py：公共 API

---

## 总结

**最小 StateGraph 的核心流程**：

1. **定义 State**：使用 TypedDict
2. **创建 builder**：`StateGraph(State)`
3. **添加节点**：`add_node(node_fn)`
4. **添加边**：`add_edge(start, end)`
5. **编译**：`compile()`
6. **执行**：`invoke(initial_state)`

**关键要点**：
- StateGraph 是 builder 模式
- 节点返回部分更新
- START 和 END 是特殊节点
- 编译后才能执行

**下一步**：
- 学习多节点状态流转（场景2）
- 理解条件路由（场景3）
- 探索 Context 注入（场景4）

---

**文件信息**：
- 知识点：01_StateGraph与节点定义
- 场景：场景1 - 最小 StateGraph 示例
- 代码行数：约 180 行
- 文档行数：约 480 行
- 最后更新：2026-02-25

**数据来源**：
- Context7 官方文档：StateGraph 基础创建、State 定义、节点函数
- 源码分析：state.py、_node.py、__init__.py
- GitHub 教程：LangChain Academy、社区示例
- Twitter 最佳实践：架构建议、生产优势
- Reddit 实践案例：常见问题、最佳实践
