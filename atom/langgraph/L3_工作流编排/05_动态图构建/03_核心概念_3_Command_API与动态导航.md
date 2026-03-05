# 核心概念3：Command API 与动态导航

## 一句话定义

**Command 让节点同时更新状态和控制路由，是动态图导航的统一接口——一个 Command 对象就能完成"把数据改成这样"和"接下来去那个节点"两件事，还支持跨子图导航。**

---

## 双重类比

### 前端类比：React Router 的 navigate + setState

在 React 中，你经常需要同时更新状态和跳转页面：

```javascript
// 前端：分两步操作
setState({ user: newUser });       // 1. 更新状态
navigate("/dashboard");            // 2. 跳转路由

// LangGraph Command：一步到位
return Command(
    update={"user": new_user},     // 1. 更新状态
    goto="dashboard"               // 2. 跳转路由
)
```

Command 就像把 `setState` 和 `navigate` 合并成一个原子操作。

### 日常生活类比：快递员的派送单

想象快递员拿到一张派送单：
- **update**（更新包裹状态）：在系统里标记"已取件"
- **goto**（下一站）：送到下一个中转站或直接送到客户手里
- **graph=Command.PARENT**（跨区域调度）：这个包裹不归我这个站点管，转交给上级调度中心

一张派送单同时包含了"做什么"和"去哪里"，这就是 Command 的设计哲学。

---

## Command 类详解

### 基本结构

```python
from langgraph.types import Command
from typing import Literal

@dataclass
class Command(Generic[N]):
    goto: Send | Sequence[Send | N] | N = ()     # 下一个节点
    update: Any | None = None                     # 状态更新
    graph: str | None = None                      # 目标图
    resume: dict[str, Any] | Any | None = None    # 中断恢复值
```

### 四个核心参数

#### 1. goto：指定下一个节点

```python
# 跳转到单个节点
return Command(goto="next_node")

# 跳转到多个节点（并行执行）
return Command(goto=["node_a", "node_b"])

# 使用 Send 对象（携带自定义状态）
return Command(goto=[Send("process", {"id": 1}), Send("process", {"id": 2})])
```

#### 2. update：状态更新

```python
# 更新状态字段
return Command(
    update={"status": "completed", "result": "success"},
    goto="next_node"
)
```

update 的值会像普通节点返回值一样，通过 Reducer 合并到图状态中。

#### 3. graph：目标图（跨图导航）

```python
# None（默认）：在当前图内导航
return Command(goto="node_a", graph=None)

# Command.PARENT：导航到父图
return Command(goto="other_subgraph", graph=Command.PARENT)
```

#### 4. resume：中断恢复值

```python
# 用于 human-in-the-loop 场景
# 当图被 interrupt() 暂停后，用 resume 恢复执行
return Command(resume={"approved": True})
```

---

## Command 与条件边的区别

### 条件边：路由逻辑在边上

```python
# 路由逻辑定义在边的路由函数中
def route_function(state: State) -> str:
    if state["score"] > 80:
        return "approve"
    return "reject"

# 节点本身不知道自己会被路由到哪里
def check_score(state: State) -> dict:
    return {"score": calculate_score(state)}

graph.add_conditional_edges("check_score", route_function)
```

节点和路由是分离的——节点只管计算，路由函数决定去向。

### Command：路由逻辑在节点内部

```python
# 路由逻辑和业务逻辑在同一个节点中
def check_score(state: State) -> Command[Literal["approve", "reject"]]:
    score = calculate_score(state)
    if score > 80:
        return Command(update={"score": score}, goto="approve")
    return Command(update={"score": score}, goto="reject")

# 不需要 add_conditional_edges，节点自己决定去向
graph.add_node("check_score", check_score)
```

路由决策和状态更新在同一个地方完成，代码更内聚。

### 对比表

| 特性 | 条件边 | Command |
|------|--------|---------|
| 路由逻辑位置 | 边上（路由函数） | 节点内部 |
| 状态更新 | 节点返回 dict | Command.update |
| 代码组织 | 分离（节点 + 路由函数） | 内聚（一个函数搞定） |
| 跨图导航 | 不支持 | 支持（graph=Command.PARENT） |
| 可视化 | 自动推断边 | 需要类型注解辅助 |
| 适用场景 | 简单分支、多节点共享路由逻辑 | 复杂决策、多代理切换 |

### Command 不会阻止静态边执行（重要！）

这是一个容易踩的坑：

```python
def my_node(state: State) -> Command[Literal["node_b"]]:
    return Command(goto="node_b")

# 如果同时存在静态边...
graph.add_node("my_node", my_node)
graph.add_edge("my_node", "node_c")  # 这条静态边也会执行！
```

当节点返回 Command 时，Command 的 `goto` 和已有的静态边会同时生效。如果你只想走 Command 指定的路径，不要给该节点添加静态边。

---

## 父图导航（Command.PARENT）

### 为什么需要父图导航

在多代理系统中，子图（子代理）需要把控制权交还给父图，让父图决定下一步调用哪个子代理：

```
父图
├── 子图A（客服代理）  ──→ Command(goto="子图B", graph=Command.PARENT)
├── 子图B（技术代理）  ──→ Command(goto="子图A", graph=Command.PARENT)
└── 子图C（销售代理）
```

### 基本用法

```python
def agent_node(state: AgentState) -> Command[Literal["other_agent"]]:
    """子图中的节点，需要切换到另一个子代理"""
    result = process(state)

    return Command(
        update={"messages": [result]},
        goto="other_agent",
        graph=Command.PARENT  # 关键：导航到父图
    )
```

### 共享状态键需要 Reducer

当子图通过 `Command.PARENT` 更新父图状态时，共享的状态键必须在父图中定义 Reducer：

```python
import operator
from typing import Annotated

# 父图状态
class ParentState(TypedDict):
    # 共享键必须有 Reducer，否则子图更新会报错
    messages: Annotated[list[str], operator.add]
    current_agent: str

# 子图状态
class ChildState(TypedDict):
    messages: Annotated[list[str], operator.add]  # 与父图共享
    internal_data: str  # 子图私有
```

### 多代理切换场景

```python
def customer_service_agent(state: State) -> Command[Literal["tech_support", "sales"]]:
    """客服代理：根据用户问题类型切换到对应代理"""
    intent = classify_intent(state["messages"][-1])

    if intent == "technical":
        return Command(
            update={"messages": ["正在转接技术支持..."]},
            goto="tech_support",
            graph=Command.PARENT
        )
    elif intent == "purchase":
        return Command(
            update={"messages": ["正在转接销售顾问..."]},
            goto="sales",
            graph=Command.PARENT
        )
    # 继续在当前子图处理
    return Command(update={"messages": [handle_query(state)]})
```

---

## 类型注解：Command[Literal[...]] 帮助可视化

LangGraph 使用节点函数的返回类型注解来推断图的边关系，这对可视化至关重要：

```python
from typing import Literal

# ✅ 有类型注解：LangGraph 知道这个节点可能跳转到 node_a 或 node_b
def my_node(state: State) -> Command[Literal["node_a", "node_b"]]:
    if state["condition"]:
        return Command(goto="node_a")
    return Command(goto="node_b")

# ❌ 没有类型注解：可视化时看不到边的连接关系
def my_node(state: State) -> Command:
    if state["condition"]:
        return Command(goto="node_a")
    return Command(goto="node_b")
```

类型注解不影响运行时行为，但会影响 `graph.get_graph().draw_mermaid()` 的输出。

---

## 完整代码示例：多代理切换系统

```python
"""
Command API 完整示例
场景：一个简单的多代理客服系统，根据用户意图在不同代理之间切换
"""

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command


# ===== 1. 定义状态 =====

class State(TypedDict):
    """共享状态"""
    user_input: str
    messages: Annotated[list[str], operator.add]
    intent: str
    resolved: bool


# ===== 2. 定义节点函数 =====

def router(state: State) -> Command[Literal["faq_agent", "tech_agent", "human_agent"]]:
    """
    路由节点：分析用户意图，决定交给哪个代理
    Command 让路由决策和状态更新在同一个地方完成
    """
    user_input = state["user_input"].lower()

    # 简单的意图分类（实际项目中用 LLM）
    if any(word in user_input for word in ["价格", "多少钱", "费用"]):
        intent = "faq"
        target = "faq_agent"
    elif any(word in user_input for word in ["报错", "bug", "无法", "失败"]):
        intent = "technical"
        target = "tech_agent"
    else:
        intent = "complex"
        target = "human_agent"

    print(f"[Router] 意图: {intent} → 转交: {target}")

    return Command(
        update={
            "intent": intent,
            "messages": [f"[系统] 识别意图为 {intent}，转交 {target}"],
        },
        goto=target,
    )


def faq_agent(state: State) -> Command[Literal["router", "__end__"]]:
    """
    FAQ 代理：处理常见问题
    如果无法回答，通过 Command 转回 router 重新分配
    """
    print(f"[FAQ] 处理问题: {state['user_input']}")

    # 模拟 FAQ 查询
    answer = f"关于「{state['user_input']}」的标准回答：请查看官网定价页面。"

    return Command(
        update={
            "messages": [f"[FAQ] {answer}"],
            "resolved": True,
        },
        goto=END,
    )


def tech_agent(state: State) -> Command[Literal["router", "__end__"]]:
    """
    技术代理：处理技术问题
    """
    print(f"[Tech] 处理技术问题: {state['user_input']}")

    answer = f"关于「{state['user_input']}」的技术解答：请尝试重启服务并检查日志。"

    return Command(
        update={
            "messages": [f"[Tech] {answer}"],
            "resolved": True,
        },
        goto=END,
    )


def human_agent(state: State) -> Command[Literal["__end__"]]:
    """
    人工代理：处理复杂问题
    """
    print(f"[Human] 转接人工处理: {state['user_input']}")

    return Command(
        update={
            "messages": ["[Human] 已转接人工客服，请稍候..."],
            "resolved": False,
        },
        goto=END,
    )


# ===== 3. 构建图 =====

builder = StateGraph(State)

# 添加节点
builder.add_node("router", router)
builder.add_node("faq_agent", faq_agent)
builder.add_node("tech_agent", tech_agent)
builder.add_node("human_agent", human_agent)

# 入口：START → router
builder.add_edge(START, "router")
# 注意：不需要 add_conditional_edges，因为 Command 自带路由

# 编译
graph = builder.compile()


# ===== 4. 运行示例 =====

if __name__ == "__main__":
    print("=== Command API 多代理切换示例 ===\n")

    # 测试用例 1：FAQ 问题
    print("--- 测试 1：FAQ 问题 ---")
    result = graph.invoke({
        "user_input": "你们的产品多少钱？",
        "messages": [],
        "intent": "",
        "resolved": False,
    })
    print(f"已解决: {result['resolved']}")
    for msg in result["messages"]:
        print(f"  {msg}")

    print("\n--- 测试 2：技术问题 ---")
    result = graph.invoke({
        "user_input": "服务报错了，无法连接数据库",
        "messages": [],
        "intent": "",
        "resolved": False,
    })
    print(f"已解决: {result['resolved']}")
    for msg in result["messages"]:
        print(f"  {msg}")

    print("\n--- 测试 3：复杂问题 ---")
    result = graph.invoke({
        "user_input": "我想定制一个企业方案",
        "messages": [],
        "intent": "",
        "resolved": False,
    })
    print(f"已解决: {result['resolved']}")
    for msg in result["messages"]:
        print(f"  {msg}")
```

**运行输出（测试1为例）：**
```
--- 测试 1：FAQ 问题 ---
[Router] 意图: faq → 转交: faq_agent
[FAQ] 处理问题: 你们的产品多少钱？
已解决: True
  [系统] 识别意图为 faq，转交 faq_agent
  [FAQ] 关于「你们的产品多少钱？」的标准回答：请查看官网定价页面。
```

三个测试分别命中 faq_agent、tech_agent、human_agent，验证了 Command 的动态路由能力。

---

## 源码层面：types.py 中 Command 类的实现

```python
# langgraph/types.py（简化版，行 367-418）
@dataclass
class Command(Generic[N], ToolOutputMixin):
    PARENT: ClassVar[str] = "__parent__"       # 父图标识常量
    graph: str | None = None                    # 目标图
    update: Any | None = None                   # 状态更新
    resume: dict[str, Any] | Any | None = None  # 中断恢复值
    goto: Send | Sequence[Send | N] | N = ()    # 下一个节点
```

关键设计点：
- `Generic[N]` 泛型参数用于类型注解，如 `Command[Literal["node_a", "node_b"]]`
- `ToolOutputMixin` 让 Command 可以作为工具调用的返回值
- `PARENT` 是类常量，值为 `"__parent__"`，用于跨图导航
- `goto` 支持多种类型：单个字符串、Send 对象、或混合列表

---

## 最佳实践与注意事项

### 1. 始终添加类型注解

```python
# ✅ 好：类型注解让图可视化正确显示边
def my_node(state: State) -> Command[Literal["node_a", "node_b"]]:
    ...

# ❌ 差：缺少类型注解，可视化看不到连接关系
def my_node(state: State) -> Command:
    ...
```

### 2. 不要混用 Command 和静态边

```python
# ❌ 危险：Command 的 goto 和静态边会同时执行
graph.add_node("my_node", my_node)  # 返回 Command(goto="node_a")
graph.add_edge("my_node", "node_b")  # 这条边也会执行！

# ✅ 安全：只用 Command 控制路由，不添加静态边
graph.add_node("my_node", my_node)  # 返回 Command(goto="node_a")
# 不添加任何从 my_node 出发的静态边
```

### 3. 父图导航时确保 Reducer 存在

```python
# 父图状态
class ParentState(TypedDict):
    # ✅ 有 Reducer：子图可以安全更新
    messages: Annotated[list[str], operator.add]

    # ❌ 没有 Reducer：子图通过 Command.PARENT 更新时会报错
    # messages: list[str]
```

### 4. Command 与 Send 的配合

Command 的 `goto` 参数也支持 Send 对象，可以实现"更新状态 + 动态并行"：

```python
def my_node(state: State) -> Command:
    return Command(
        update={"status": "processing"},
        goto=[
            Send("process", {"id": 1}),
            Send("process", {"id": 2}),
        ]
    )
```

### 5. 条件边 vs Command 的选择指南

| 场景 | 推荐方式 |
|------|---------|
| 简单的 if/else 分支 | 条件边（更直观） |
| 路由逻辑与业务逻辑紧耦合 | Command（更内聚） |
| 多个节点共享同一个路由逻辑 | 条件边（可复用路由函数） |
| 需要同时更新状态和路由 | Command（原子操作） |
| 子图到父图的通信 | Command（唯一选择） |
| 多代理切换 | Command + Command.PARENT |

---

## 引用来源

本文档基于以下资料编写：

1. **LangGraph 源码分析** - `langgraph/types.py` Command 类定义（行 367-418）、`langgraph/graph/state.py` StateGraph 构建器
   - 来源：`reference/source_dynamic_graph_01.md`

2. **Context7 官方文档** - Command API 基本用法、父图导航、类型注解、与条件边的区别
   - 来源：`reference/context7_langgraph_01.md`

3. **Medium 技术文章** - "A Beginner's Guide to Dynamic Routing in LangGraph with Command()" by Damilola Oyedunmade
   - 来源：`reference/fetch_command_routing_01.md`

4. **社区搜索结果** - Command API 社区趋势、可视化挑战、多代理架构实践
   - 来源：`reference/search_dynamic_graph_01.md`

---

## 总结

Command 是 LangGraph 动态导航的统一接口：通过 `goto` 控制路由、`update` 更新状态、`graph=Command.PARENT` 实现跨图导航，三者在一个对象中完成。与条件边相比，Command 让路由逻辑和业务逻辑更内聚，是多代理切换场景的首选方案。记住两个关键点：始终添加 `Command[Literal[...]]` 类型注解，以及不要混用 Command 和静态边。
