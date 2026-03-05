# 实战代码 - 场景4：Command 动态路由

> 使用 Command 类实现动态路由，节点根据运行时状态决定下一步去哪里、同时更新什么状态，无需 add_conditional_edges。

---

## 场景说明

**业务场景：智能助手动态决策**

用户输入一句话，系统自动判断意图（搜索、计算、闲聊），路由到对应的处理节点。关键在于：路由决策和状态更新在同一个节点内完成。

**为什么用 Command 而不是 add_conditional_edges？**

| 维度 | add_conditional_edges | Command |
|------|----------------------|---------|
| 路由逻辑位置 | 独立的路由函数 | 节点内部 |
| 状态更新 | 路由函数不能更新状态 | goto + update 一步到位 |
| 代码组织 | 路由逻辑和业务逻辑分离 | 路由逻辑和业务逻辑内聚 |
| 适用场景 | 简单的 if-else 分支 | 需要同时更新状态的复杂决策 |

**Command 的核心优势：决策和状态更新是原子操作，不会出现"状态更新了但路由错了"的不一致问题。**

---

## 完整可运行代码

### 基础版：智能助手意图路由

```python
"""
条件分支策略 - 实战场景4：Command 动态路由
演示：智能助手根据用户输入动态决策

核心要点：
- 节点返回 Command 对象，同时控制路由 + 更新状态
- 无需 add_conditional_edges，路由逻辑内聚在节点中
- Literal 类型注解提供类型安全
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ===== 1. 定义状态 =====

class AssistantState(TypedDict):
    """智能助手状态"""
    user_input: str           # 用户输入
    intent: str               # 识别的意图
    response: str             # 最终回复
    route_history: list[str]  # 路由轨迹（调试用）


# ===== 2. 路由节点：返回 Command 实现动态路由 =====

def router(state: AssistantState) -> Command[Literal["search_node", "calc_node", "chat_node"]]:
    """
    路由节点：分析用户输入，动态决定下一步

    关键：返回 Command 对象
    - goto: 指定下一个节点
    - update: 同时更新状态（意图、路由历史）
    - 类型注解 Command[Literal[...]] 确保只能路由到合法节点
    """
    text = state["user_input"].lower()
    history = state.get("route_history", [])

    print(f"\n🔍 路由分析: '{state['user_input']}'")

    # 意图识别（实际应用中可以用 LLM 分类）
    if any(w in text for w in ["搜索", "查找", "找", "search"]):
        print("  → 识别意图: 搜索")
        return Command(
            goto="search_node",
            update={
                "intent": "search",
                "route_history": [*history, "router→search"]
            }
        )
    elif any(w in text for w in ["计算", "算", "多少", "加", "减", "乘", "除"]):
        print("  → 识别意图: 计算")
        return Command(
            goto="calc_node",
            update={
                "intent": "calculation",
                "route_history": [*history, "router→calc"]
            }
        )
    else:
        print("  → 识别意图: 闲聊")
        return Command(
            goto="chat_node",
            update={
                "intent": "chat",
                "route_history": [*history, "router→chat"]
            }
        )


# ===== 3. 处理节点 =====

def search_node(state: AssistantState) -> dict:
    """搜索处理节点"""
    query = state["user_input"]
    print(f"  🔎 执行搜索: {query}")

    # 模拟搜索结果
    response = f"搜索结果: 关于「{query}」找到了 3 条相关信息"
    return {"response": response}


def calc_node(state: AssistantState) -> dict:
    """计算处理节点"""
    query = state["user_input"]
    print(f"  🧮 执行计算: {query}")

    # 模拟计算（简单提取数字）
    import re
    numbers = re.findall(r'\d+', query)
    if len(numbers) >= 2:
        result = int(numbers[0]) + int(numbers[1])
        response = f"计算结果: {numbers[0]} + {numbers[1]} = {result}"
    else:
        response = f"计算结果: 无法解析表达式「{query}」"

    return {"response": response}


def chat_node(state: AssistantState) -> dict:
    """闲聊处理节点"""
    query = state["user_input"]
    print(f"  💬 闲聊回复: {query}")

    response = f"你好！你说的是「{query}」，有什么我可以帮你的吗？"
    return {"response": response}


# ===== 4. 构建图 =====

builder = StateGraph(AssistantState)

# 添加节点
builder.add_node("router", router)
builder.add_node("search_node", search_node)
builder.add_node("calc_node", calc_node)
builder.add_node("chat_node", chat_node)

# 添加边
builder.add_edge(START, "router")
# 注意：不需要 add_conditional_edges！
# Command 会自动处理从 router 到各处理节点的路由
builder.add_edge("search_node", END)
builder.add_edge("calc_node", END)
builder.add_edge("chat_node", END)

# 编译
graph = builder.compile()


# ===== 5. 测试 =====

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Command 动态路由：智能助手")
    print("=" * 70)

    test_inputs = [
        "帮我搜索 Python 教程",
        "计算 15 加 27 等于多少",
        "今天天气真不错",
        "查找 LangGraph 文档",
        "算一下 100 减 30",
    ]

    for user_input in test_inputs:
        print(f"\n{'─' * 50}")
        result = graph.invoke({
            "user_input": user_input,
            "intent": "",
            "response": "",
            "route_history": []
        })
        print(f"  📋 意图: {result['intent']}")
        print(f"  📋 回复: {result['response']}")
        print(f"  📋 路由轨迹: {result['route_history']}")
```

---

## 对比：同样逻辑用 add_conditional_edges 实现

理解 Command 的价值，最好的方式是看同样的逻辑用传统方式怎么写。

### 传统方式：add_conditional_edges

```python
"""
对比：用 add_conditional_edges 实现同样的路由逻辑
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class AssistantState(TypedDict):
    user_input: str
    intent: str
    response: str
    route_history: list[str]


# ===== 传统方式：路由函数和业务逻辑分离 =====

def analyze_intent(state: AssistantState) -> dict:
    """
    分析节点：只负责识别意图，更新状态
    问题：路由逻辑在另一个函数里，意图识别做了两次
    """
    text = state["user_input"].lower()

    if any(w in text for w in ["搜索", "查找", "找", "search"]):
        intent = "search"
    elif any(w in text for w in ["计算", "算", "多少", "加", "减"]):
        intent = "calculation"
    else:
        intent = "chat"

    return {
        "intent": intent,
        "route_history": [*state.get("route_history", []), f"analyze→{intent}"]
    }


def route_by_intent(state: AssistantState) -> Literal["search_node", "calc_node", "chat_node"]:
    """
    路由函数：根据意图返回目标节点名
    问题1：这个函数不能更新状态
    问题2：意图识别逻辑和 analyze_intent 重复
    """
    intent = state["intent"]
    if intent == "search":
        return "search_node"
    elif intent == "calculation":
        return "calc_node"
    else:
        return "chat_node"


def search_node(state: AssistantState) -> dict:
    return {"response": f"搜索结果: 关于「{state['user_input']}」的信息"}

def calc_node(state: AssistantState) -> dict:
    return {"response": f"计算结果: 处理「{state['user_input']}」"}

def chat_node(state: AssistantState) -> dict:
    return {"response": f"闲聊回复: 你说的是「{state['user_input']}」"}


# 构建图
builder = StateGraph(AssistantState)

builder.add_node("analyze", analyze_intent)
builder.add_node("search_node", search_node)
builder.add_node("calc_node", calc_node)
builder.add_node("chat_node", chat_node)

builder.add_edge(START, "analyze")

# 传统方式：需要 add_conditional_edges + 独立的路由函数
builder.add_conditional_edges("analyze", route_by_intent)

builder.add_edge("search_node", END)
builder.add_edge("calc_node", END)
builder.add_edge("chat_node", END)

graph = builder.compile()
```

### 对比分析

```
┌─────────────────────────────────────────────────────────────┐
│                  传统方式 (add_conditional_edges)              │
│                                                             │
│  analyze_intent()     route_by_intent()     search_node()   │
│  ┌──────────────┐    ┌──────────────┐      ┌────────────┐  │
│  │ 识别意图      │ →  │ 再次读取意图  │  →   │ 处理搜索    │  │
│  │ 更新状态      │    │ 返回节点名    │      │            │  │
│  └──────────────┘    └──────────────┘      └────────────┘  │
│       节点函数           路由函数              节点函数       │
│    （可更新状态）      （不可更新状态）       （可更新状态）    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Command 方式                                │
│                                                             │
│  router()                               search_node()       │
│  ┌──────────────────────────────┐      ┌────────────┐      │
│  │ 识别意图                      │  →   │ 处理搜索    │      │
│  │ 更新状态 + 决定路由（原子操作） │      │            │      │
│  └──────────────────────────────┘      └────────────┘      │
│       节点函数（一步到位）                 节点函数           │
└─────────────────────────────────────────────────────────────┘
```

**Command 方式的优势：**
1. 少一个函数（不需要独立的路由函数）
2. 少一个节点（不需要 analyze 节点）
3. 状态更新和路由决策是原子操作，不会不一致
4. 代码更内聚，逻辑更清晰

---

## 进阶：Command 链式路由（多步决策）

### 场景：多轮质量检查

节点处理完后，根据结果决定是继续、重试还是结束。

```python
"""
进阶：Command 链式路由
场景：文档处理 → 质量检查 → 通过/重试/拒绝
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class DocProcessState(TypedDict):
    content: str
    processed: str
    quality_score: float
    attempts: int
    max_attempts: int
    status: str


def process_document(state: DocProcessState) -> Command[Literal["quality_check"]]:
    """处理文档，然后交给质量检查"""
    content = state["content"]
    attempts = state.get("attempts", 0) + 1

    # 模拟处理（每次重试结果会更好）
    quality_boost = attempts * 0.2
    processed = f"[v{attempts}] 处理后的内容: {content}"

    print(f"\n📄 处理文档 (第{attempts}次)")
    print(f"  内容: {content[:30]}...")

    return Command(
        goto="quality_check",
        update={
            "processed": processed,
            "attempts": attempts,
            "quality_score": min(0.3 + quality_boost, 1.0)  # 模拟质量提升
        }
    )


def quality_check(state: DocProcessState) -> Command[Literal["process_document", "publish", "reject"]]:
    """
    质量检查：根据评分决定下一步
    - 评分 > 0.8: 发布
    - 评分 > 0.4 且未超过重试次数: 重试
    - 否则: 拒绝
    """
    score = state["quality_score"]
    attempts = state["attempts"]
    max_attempts = state.get("max_attempts", 3)

    print(f"  ⭐ 质量评分: {score:.1f} (第{attempts}次, 最多{max_attempts}次)")

    if score > 0.8:
        print("  ✅ 质量达标，准备发布")
        return Command(
            goto="publish",
            update={"status": "approved"}
        )
    elif attempts < max_attempts:
        print(f"  🔄 质量不足，重试 ({attempts}/{max_attempts})")
        return Command(
            goto="process_document",
            update={"status": "retrying"}
        )
    else:
        print(f"  ❌ 超过最大重试次数，拒绝")
        return Command(
            goto="reject",
            update={"status": "rejected"}
        )


def publish(state: DocProcessState) -> dict:
    """发布文档"""
    print(f"  📢 文档已发布: {state['processed'][:40]}...")
    return {"status": "published"}


def reject(state: DocProcessState) -> dict:
    """拒绝文档"""
    print(f"  🚫 文档被拒绝，共尝试 {state['attempts']} 次")
    return {"status": "rejected"}


# 构建图
builder = StateGraph(DocProcessState)

builder.add_node("process_document", process_document)
builder.add_node("quality_check", quality_check)
builder.add_node("publish", publish)
builder.add_node("reject", reject)

builder.add_edge(START, "process_document")
# Command 自动处理所有路由，包括 quality_check → process_document 的循环
builder.add_edge("publish", END)
builder.add_edge("reject", END)

chain_graph = builder.compile()


# 测试
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Command 链式路由：多轮质量检查")
    print("=" * 70)

    result = chain_graph.invoke({
        "content": "这是一篇需要多次修改才能达标的文档",
        "processed": "",
        "quality_score": 0.0,
        "attempts": 0,
        "max_attempts": 3,
        "status": "pending"
    })

    print(f"\n📋 最终状态: {result['status']}")
    print(f"📋 尝试次数: {result['attempts']}")
    print(f"📋 最终评分: {result['quality_score']:.1f}")
```

### 链式路由执行流程

```
START → process_document → quality_check ─┬─→ publish → END
              ↑                           │
              └───── 重试（score <= 0.8）──┘
                                          │
                                          └─→ reject → END
                                          (超过 max_attempts)
```

---

## Command 调试技巧

### 技巧1：用 route_history 追踪路由轨迹

```python
class State(TypedDict):
    route_history: list[str]  # 记录每次路由决策

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    history = state.get("route_history", [])
    return Command(
        goto="node_b",
        update={"route_history": [*history, "a→b"]}  # 追加路由记录
    )

# 执行后检查 route_history 即可看到完整路由路径
```

### 技巧2：用 stream 模式观察每一步

```python
# stream 模式可以看到每个节点的执行和 Command 的路由决策
for event in graph.stream(initial_state):
    for node_name, output in event.items():
        print(f"节点: {node_name}")
        print(f"输出: {output}")
        print("---")
```

### 技巧3：Literal 类型注解防止拼写错误

```python
# ✅ 推荐：Literal 类型注解
def router(state) -> Command[Literal["search", "calc", "chat"]]:
    return Command(goto="search")  # IDE 会自动补全

# ❌ 不推荐：无类型注解
def router(state):
    return Command(goto="serach")  # 拼写错误，运行时才报错
```

### 技巧4：打印 Command 对象

```python
def router(state) -> Command[Literal["a", "b"]]:
    cmd = Command(goto="a", update={"key": "value"})
    print(f"Command: goto={cmd.goto}, update={cmd.update}")  # 调试输出
    return cmd
```

---

## 关键模式总结

### Command 三种用法

```python
# 用法1：只路由，不更新状态
return Command(goto="next_node")

# 用法2：路由 + 更新状态（最常用）
return Command(goto="next_node", update={"key": "value"})

# 用法3：路由到父图（子图专用）
return Command(goto=Command.PARENT, update={"result": "done"})
```

### Command vs add_conditional_edges 选择指南

```python
# 选 add_conditional_edges 的场景：
# - 路由逻辑简单（纯 if-else）
# - 不需要在路由时更新状态
# - 路由函数可以复用

# 选 Command 的场景：
# - 路由决策需要同时更新状态
# - 路由逻辑复杂，和业务逻辑紧密耦合
# - 需要循环（节点 A → B → A）
# - 子图需要返回父图
```

---

## 注意事项与常见错误

### 错误1：Command 节点没有对应的出边

```python
# ❌ 错误：router 用 Command 路由到 search_node，但没有 search_node → END 的边
builder.add_node("router", router)
builder.add_node("search_node", search_node)
builder.add_edge(START, "router")
# 缺少: builder.add_edge("search_node", END)
# 结果：search_node 执行完后图不知道去哪里
```

### 错误2：Command 的 goto 目标不存在

```python
# ❌ 错误：goto 指向一个未注册的节点
return Command(goto="nonexistent_node")
# 运行时报错：NodeNotFoundError

# ✅ 正确：使用 Literal 类型注解，编译时就能发现
def router(state) -> Command[Literal["search", "calc"]]:
    return Command(goto="search")  # 必须是 Literal 中列出的值
```

### 错误3：循环中没有终止条件

```python
# ❌ 危险：无限循环
def check(state) -> Command[Literal["process", "done"]]:
    if state["score"] < 0.8:
        return Command(goto="process")  # 如果 score 永远 < 0.8，就死循环了

# ✅ 安全：加上重试次数限制
def check(state) -> Command[Literal["process", "done", "fail"]]:
    if state["score"] < 0.8 and state["attempts"] < 5:
        return Command(goto="process", update={"attempts": state["attempts"] + 1})
    elif state["score"] >= 0.8:
        return Command(goto="done")
    else:
        return Command(goto="fail")  # 兜底退出
```

---

## RAG 应用场景

Command 动态路由在 RAG 系统中的典型应用：

| 场景 | Command 路由逻辑 |
|------|-----------------|
| Self-RAG | 检索质量评估 → 质量够高走生成，不够走重新检索 |
| Adaptive RAG | 问题分类 → 简单问题直接回答，复杂问题走 RAG 流程 |
| 多轮对话 | 判断是否需要检索 → 需要走检索链，不需要走对话链 |
| 幻觉检测 | 生成后检查 → 有幻觉走重新生成，无幻觉走输出 |

---

## 学习检查清单

- [ ] 理解 Command 的 goto + update 原子操作
- [ ] 能用 Command 替代 add_conditional_edges 实现路由
- [ ] 掌握 Command 链式路由（循环重试模式）
- [ ] 知道 Command 和 add_conditional_edges 的选择标准
- [ ] 能用 route_history 和 stream 调试路由问题

---

**文档版本**: v1.0
**最后更新**: 2026-03-01
**来源**: reference/fetch_command_routing_01.md, reference/source_conditional_branching_01.md
