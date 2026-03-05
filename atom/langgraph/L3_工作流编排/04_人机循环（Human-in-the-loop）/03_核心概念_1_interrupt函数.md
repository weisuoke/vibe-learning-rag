# 03_核心概念_1 - interrupt() 函数

> LangGraph 人机循环的核心机制：通过 interrupt() 在节点内部暂停图执行并与人类交互

---

## 概念定义

**interrupt() 是 LangGraph 的动态中断函数，在节点内部调用时暂停图执行，向客户端传递信息，等待人类决策后恢复执行并返回决策值。**

它是 LangGraph 官方推荐的人机循环（Human-in-the-loop）实现方式，替代了旧的 `interrupt_before` / `interrupt_after` 静态断点和已废弃的 `NodeInterrupt` 异常。

一句话理解：**interrupt() 就是分布式、可持久化的 `input()`**。

---

## 函数签名与参数

### 源码定义

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py:420]

def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.

    The interrupt function enables human-in-the-loop workflows by pausing
    graph execution at specific points to collect user input.

    Args:
        value: The value to send to the client. This value will be
               included in an Interrupt instance. Must be JSON serializable.

    Returns:
        The value provided when resuming the graph with Command(resume=value).
    """
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `Any` | 传递给客户端的中断信息。**必须是 JSON 可序列化的**。可以是字符串、字典、列表等 |

### 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| resume 值 | `Any` | 客户端通过 `Command(resume=value)` 传回的人类决策值 |

### 使用前提

1. **必须启用 Checkpointer**：`graph = builder.compile(checkpointer=MemorySaver())`
2. **必须在节点函数内部调用**：不能在条件边或其他位置调用
3. **value 必须 JSON 可序列化**：不能传递函数、类实例等不可序列化对象

---

## 工作原理（源码级别）

### 核心机制：双重身份

`interrupt()` 函数的行为取决于调用时机，它有两种完全不同的行为：

```
第一次执行（无 resume 值）：
    interrupt(value) → 抛出 GraphInterrupt 异常 → 暂停执行

恢复后重新执行（有 resume 值）：
    interrupt(value) → 直接返回 resume 值 → 继续执行
```

### 行为 1：第一次调用 —— 抛出 GraphInterrupt 异常

当节点第一次执行，遇到 `interrupt()` 调用时：

```python
def my_node(state):
    # 1. 执行到 interrupt()
    answer = interrupt("你确认要继续吗？")
    # ↑ 这里抛出 GraphInterrupt 异常
    # ↓ 下面的代码不会执行
    print(f"用户回答: {answer}")  # 不会执行！
```

**内部流程**：

```
interrupt("你确认要继续吗？")
    ↓
创建 Interrupt 对象：
    Interrupt(
        value="你确认要继续吗？",
        id=xxh3_128_hexdigest(...)  # 基于内容生成唯一 ID
    )
    ↓
抛出 GraphInterrupt 异常（继承自 GraphBubbleUp）
    ↓
异常沿调用栈向上冒泡
    ↓
PregelLoop 捕获异常
    ↓
Checkpointer 保存当前状态（包括中断信息）
    ↓
返回给客户端：
    result["__interrupt__"] = [Interrupt(value="你确认要继续吗？")]
```

### 行为 2：恢复后调用 —— 返回 resume 值

当客户端调用 `graph.invoke(Command(resume="是的"), config)` 后：

```python
def my_node(state):
    # 节点从头重新执行！
    # 2. interrupt() 检测到有匹配的 resume 值
    answer = interrupt("你确认要继续吗？")
    # ↑ 这次直接返回 "是的"（resume 值）
    # ↓ 继续执行后续代码
    print(f"用户回答: {answer}")  # 输出: 用户回答: 是的
```

**内部流程**：

```
graph.invoke(Command(resume="是的"), config)
    ↓
Checkpointer 恢复状态
    ↓
节点从头重新执行（不是从 interrupt 处继续！）
    ↓
interrupt("你确认要继续吗？")
    ↓
检查 resume 值列表：找到匹配的值 "是的"
    ↓
直接返回 "是的"（不抛异常）
    ↓
节点继续执行后续逻辑
```

### 行为 3：多次调用 —— 按顺序匹配

一个节点内可以有多个 `interrupt()` 调用，它们按顺序与 resume 值配对：

```python
def multi_step_node(state):
    # 第一个 interrupt
    name = interrupt("请输入你的名字")

    # 第二个 interrupt
    age = interrupt("请输入你的年龄")

    # 第三个 interrupt
    confirm = interrupt(f"确认：{name}, {age}岁？")

    return {"result": f"{name}, {age}岁, 确认={confirm}"}
```

**恢复过程**：

```
# 第一次执行：遇到第一个 interrupt，暂停
graph.invoke(input, config)
# → __interrupt__: [Interrupt(value="请输入你的名字")]

# 第一次恢复：第一个 interrupt 返回 "张三"，遇到第二个 interrupt，暂停
graph.invoke(Command(resume="张三"), config)
# → __interrupt__: [Interrupt(value="请输入你的年龄")]

# 第二次恢复：前两个 interrupt 返回缓存值，遇到第三个 interrupt，暂停
graph.invoke(Command(resume="25"), config)
# → __interrupt__: [Interrupt(value="确认：张三, 25岁？")]

# 第三次恢复：所有 interrupt 返回缓存值，节点完成执行
graph.invoke(Command(resume=True), config)
# → {"result": "张三, 25岁, 确认=True"}
```

**匹配规则**：
- resume 值按顺序与 interrupt 调用配对
- 已匹配的 interrupt 直接返回缓存的 resume 值
- 第一个未匹配的 interrupt 抛出新的 GraphInterrupt
- 作用域限定在执行该节点的特定任务内

---

## 与 Python input() 的类比

### 相似之处

```python
# Python input() —— 本地同步阻塞
def local_workflow():
    name = input("请输入名字: ")     # 阻塞等待用户输入
    print(f"你好, {name}")

# LangGraph interrupt() —— 分布式异步持久化
def distributed_workflow(state):
    name = interrupt("请输入名字")    # 暂停图执行，等待恢复
    return {"greeting": f"你好, {name}"}
```

### 关键区别

| 特性 | `input()` | `interrupt()` |
|------|-----------|---------------|
| 阻塞方式 | 同步阻塞当前线程 | 抛异常暂停图执行 |
| 持久化 | 无（进程退出就丢失） | 有（Checkpointer 保存状态） |
| 分布式 | 不支持（只能本地终端） | 支持（客户端可以在任何地方） |
| 并发 | 不支持 | 支持（不同 thread_id） |
| 恢复方式 | 用户直接在终端输入 | `Command(resume=value)` |
| 超时 | 无内置超时 | 可自定义超时逻辑 |
| 多次调用 | 每次独立 | 按顺序匹配 resume 值 |
| 节点重执行 | 不适用 | 恢复时节点从头执行 |

### 升级关系

```
input()                    →  interrupt()
─────────────────────────────────────────────
本地终端                    →  任意客户端（Web/App/API）
进程内存                    →  持久化存储（DB/Redis）
单线程阻塞                  →  异步非阻塞
单用户                      →  多用户并发
无状态                      →  有状态（Checkpoint）
```

---

## 完整代码示例

### 示例 1：最简单的 interrupt 用法

```python
"""
最简单的 interrupt 示例：单节点单中断
演示 interrupt() 的基本用法
"""

from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


# 1. 定义状态
class State(TypedDict):
    query: str
    answer: str


# 2. 定义节点 —— 包含 interrupt()
def ask_human(state: State):
    """在节点内部调用 interrupt() 暂停执行"""
    # interrupt() 第一次调用：抛出异常，暂停
    # interrupt() 恢复后调用：返回 resume 值
    human_input = interrupt(f"请回答问题: {state['query']}")
    return {"answer": human_input}


# 3. 构建图
builder = StateGraph(State)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "ask_human")
builder.add_edge("ask_human", END)

# 4. 编译 —— 必须提供 checkpointer！
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 5. 执行 —— 必须提供 thread_id！
config = {"configurable": {"thread_id": "demo-1"}}

# 第一次调用：遇到 interrupt，暂停
result = graph.invoke({"query": "你最喜欢的编程语言是什么？"}, config)
print("中断信息:", result)
# 输出包含 __interrupt__ 字段

# 6. 恢复 —— 使用 Command(resume=...)
result = graph.invoke(Command(resume="Python"), config)
print("最终结果:", result)
# 输出: {"query": "你最喜欢的编程语言是什么？", "answer": "Python"}
```

### 示例 2：传递结构化中断信息

```python
"""
传递结构化信息给客户端
interrupt 的 value 可以是任意 JSON 可序列化对象
"""

from typing import TypedDict, Optional, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class ApprovalState(TypedDict):
    action: str
    amount: float
    status: Optional[str]


def approval_node(state: ApprovalState) -> Command[Literal["execute", "reject"]]:
    """审批节点：传递结构化信息，根据决策路由"""

    # 传递结构化的中断信息
    decision = interrupt({
        "type": "approval_request",
        "question": "是否批准此操作？",
        "details": {
            "action": state["action"],
            "amount": state["amount"],
        },
        "options": ["approve", "reject"],
    })

    # 根据人类决策路由到不同节点
    if decision == "approve":
        return Command(goto="execute", update={"status": "approved"})
    else:
        return Command(goto="reject", update={"status": "rejected"})


def execute_node(state: ApprovalState):
    print(f"执行操作: {state['action']}, 金额: {state['amount']}")
    return {"status": "completed"}


def reject_node(state: ApprovalState):
    print(f"操作被拒绝: {state['action']}")
    return {"status": "rejected"}


# 构建图
builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("execute", execute_node)
builder.add_node("reject", reject_node)
builder.add_edge(START, "approval")
builder.add_edge("execute", END)
builder.add_edge("reject", END)

graph = builder.compile(checkpointer=MemorySaver())

# 执行
config = {"configurable": {"thread_id": "approval-001"}}

# 第一次调用：触发审批中断
result = graph.invoke(
    {"action": "转账", "amount": 5000.0, "status": "pending"},
    config
)
# result 包含中断信息：
# __interrupt__: [Interrupt(value={"type": "approval_request", ...})]

# 审批通过
result = graph.invoke(Command(resume="approve"), config)
print(result["status"])  # "completed"
```

### 示例 3：多个 interrupt 的顺序匹配

```python
"""
单节点内多个 interrupt 调用
演示按顺序匹配 resume 值的机制
"""

from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class FormState(TypedDict):
    result: str


def form_node(state: FormState):
    """模拟多步表单填写"""

    # 第一个 interrupt：收集姓名
    name = interrupt({
        "step": 1,
        "field": "name",
        "prompt": "请输入你的姓名"
    })

    # 第二个 interrupt：收集邮箱
    email = interrupt({
        "step": 2,
        "field": "email",
        "prompt": f"{name}，请输入你的邮箱"
    })

    # 第三个 interrupt：确认信息
    confirmed = interrupt({
        "step": 3,
        "field": "confirm",
        "prompt": f"确认信息：{name} <{email}>",
        "options": [True, False]
    })

    if confirmed:
        return {"result": f"注册成功：{name} <{email}>"}
    else:
        return {"result": "注册已取消"}


# 构建图
builder = StateGraph(FormState)
builder.add_node("form", form_node)
builder.add_edge(START, "form")
builder.add_edge("form", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "form-001"}}

# 第一步：触发第一个 interrupt
result = graph.invoke({"result": ""}, config)
# __interrupt__: [Interrupt(value={"step": 1, "field": "name", ...})]

# 第二步：回答姓名，触发第二个 interrupt
result = graph.invoke(Command(resume="张三"), config)
# __interrupt__: [Interrupt(value={"step": 2, "field": "email", ...})]
# 注意：此时第一个 interrupt 直接返回缓存的 "张三"

# 第三步：回答邮箱，触发第三个 interrupt
result = graph.invoke(Command(resume="zhangsan@example.com"), config)
# __interrupt__: [Interrupt(value={"step": 3, "field": "confirm", ...})]

# 第四步：确认，完成
result = graph.invoke(Command(resume=True), config)
print(result["result"])  # "注册成功：张三 <zhangsan@example.com>"
```

### 示例 4：条件中断（不是每次都中断）

```python
"""
条件中断：只在满足特定条件时才中断
展示 interrupt() 的灵活性
"""

from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


class TransferState(TypedDict):
    amount: float
    to_account: str
    approved: bool


def transfer_node(state: TransferState):
    """只有大额转账才需要人工审批"""

    if state["amount"] > 10000:
        # 大额转账：需要人工审批
        approved = interrupt({
            "question": f"大额转账 ¥{state['amount']} 到 {state['to_account']}，是否批准？",
            "risk_level": "high",
        })
    else:
        # 小额转账：自动批准
        approved = True

    return {"approved": approved}


# 构建图
builder = StateGraph(TransferState)
builder.add_node("transfer", transfer_node)
builder.add_edge(START, "transfer")
builder.add_edge("transfer", END)

graph = builder.compile(checkpointer=MemorySaver())

# 小额转账：不会中断，直接完成
config1 = {"configurable": {"thread_id": "transfer-small"}}
result = graph.invoke(
    {"amount": 500, "to_account": "ACC-001", "approved": False},
    config1
)
print(result["approved"])  # True（自动批准）

# 大额转账：触发中断
config2 = {"configurable": {"thread_id": "transfer-large"}}
result = graph.invoke(
    {"amount": 50000, "to_account": "ACC-002", "approved": False},
    config2
)
# 需要人工审批...
result = graph.invoke(Command(resume=True), config2)
print(result["approved"])  # True（人工批准）
```

---

## 重要注意事项

### 1. 必须启用 Checkpointer

```python
# ❌ 错误：没有 checkpointer，interrupt() 无法工作
graph = builder.compile()

# ✅ 正确：提供 checkpointer
graph = builder.compile(checkpointer=MemorySaver())
```

**原因**：interrupt() 需要 Checkpointer 保存中断时的状态，以便后续恢复。

### 2. value 必须 JSON 可序列化

```python
# ✅ 正确：字符串、数字、字典、列表
interrupt("请确认")
interrupt({"question": "批准？", "amount": 100})
interrupt(["选项A", "选项B", "选项C"])

# ❌ 错误：函数、类实例、自定义对象
interrupt(lambda x: x)           # 函数不可序列化
interrupt(MyCustomClass())       # 自定义类不可序列化
interrupt(datetime.now())        # datetime 不可序列化
```

### 3. 节点会重新执行

```python
def my_node(state):
    # ⚠️ 注意：恢复时这行代码会再次执行！
    print("节点开始执行")  # 第一次执行打印一次，恢复后又打印一次

    result = interrupt("请确认")

    # 只有恢复后才会执行到这里
    return {"result": result}
```

**最佳实践**：避免在 interrupt() 之前放置有副作用的操作（如发送邮件、写入数据库），或者使用幂等操作。

```python
def safe_node(state):
    # ✅ 幂等操作：重复执行不会有问题
    data = fetch_data(state["id"])  # 读取操作是幂等的

    result = interrupt({"data": data, "question": "确认？"})

    # 有副作用的操作放在 interrupt 之后
    if result:
        send_email(state["email"])  # 只在确认后执行
    return {"confirmed": result}
```

### 4. thread_id 必须一致

```python
config = {"configurable": {"thread_id": "my-thread-123"}}

# 第一次调用
graph.invoke(input, config)

# 恢复时必须使用相同的 config（相同的 thread_id）
graph.invoke(Command(resume=value), config)  # ✅ 相同的 thread_id

# ❌ 错误：使用不同的 thread_id
wrong_config = {"configurable": {"thread_id": "different-thread"}}
graph.invoke(Command(resume=value), wrong_config)  # 找不到中断状态！
```

### 5. 检查中断状态

```python
# 使用 get_state() 检查当前中断状态
state_snapshot = graph.get_state(config)

# 查看待处理的中断
print(state_snapshot.tasks)
# 每个 task 包含 interrupts 字段

# 查看下一步要执行的节点
print(state_snapshot.next)
# 如果有中断，next 会包含中断所在的节点名
```

---

## 与旧 API 的对比

### NodeInterrupt（已废弃）

```python
# ❌ 旧方式：NodeInterrupt（已废弃）
from langgraph.errors import NodeInterrupt

def old_node(state):
    if need_approval:
        raise NodeInterrupt("需要审批")  # 抛出异常中断
    return state

# ✅ 新方式：interrupt()（推荐）
from langgraph.types import interrupt

def new_node(state):
    if need_approval:
        result = interrupt("需要审批")  # 中断并获取返回值
    return {"approved": result}
```

**区别**：
- `NodeInterrupt` 只能中断，不能获取返回值
- `interrupt()` 既能中断，又能获取人类的决策值
- `NodeInterrupt` 已标记为 `@deprecated`

### interrupt_before / interrupt_after（静态断点）

```python
# 静态断点：在 compile() 时配置
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_node"],   # 节点执行前中断
    interrupt_after=["processing_node"],  # 节点执行后中断
)

# 动态中断：在节点内部调用
def approval_node(state):
    decision = interrupt("批准？")  # 更灵活
    return {"decision": decision}
```

**对比**：

| 特性 | interrupt() 函数 | interrupt_before/after |
|------|-----------------|----------------------|
| 类型 | 动态中断 | 静态断点 |
| 位置 | 节点内部任意位置 | 节点执行前/后 |
| 灵活性 | 高（支持条件逻辑） | 低（固定节点列表） |
| 数据传递 | 可传递任意 JSON 值 | 无数据传递 |
| 返回值 | 有（resume 值） | 无 |
| 推荐度 | 官方推荐 | 遗留方式，简单场景可用 |
| 配置方式 | 代码内调用 | compile() 参数 |

---

## 底层实现：关键类型

### Interrupt 数据类

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py:161]

@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any      # 中断关联的值（传递给客户端的信息）
    id: str         # 中断 ID，通过 xxh3_128_hexdigest 生成
```

### GraphInterrupt 异常

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/errors.py:84]

class GraphBubbleUp(Exception):
    """冒泡异常基类"""
    pass

class GraphInterrupt(GraphBubbleUp):
    """图中断异常，由 interrupt() 内部抛出
    子图中断时抛出，被根图抑制，不会直接暴露给用户"""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```

### Command 类（恢复相关部分）

```python
# [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py:368]

@dataclass
class Command(Generic[N]):
    resume: dict[str, Any] | Any | None = None  # 恢复值
    goto: Send | Sequence[Send | N] | N = ()     # 导航目标
    update: Any | None = None                    # 状态更新
    graph: str | None = None                     # 目标图
```

**resume 的两种形式**：
- **单个值**：`Command(resume="approved")` —— 恢复下一个中断
- **字典**：`Command(resume={interrupt_id: value})` —— 按 ID 恢复特定中断

---

## 常见问题

### Q1: interrupt() 之前的代码会重复执行吗？

**会的。** 恢复时节点从头重新执行，interrupt() 之前的所有代码都会再次运行。已匹配的 interrupt() 会直接返回缓存值而不暂停。

### Q2: 可以在循环中使用 interrupt() 吗？

**可以，但要小心。** 每次循环迭代中的 interrupt() 都是独立的，按顺序匹配 resume 值。

```python
def loop_node(state):
    results = []
    for item in state["items"]:
        # 每个 item 对应一个 interrupt
        decision = interrupt(f"处理 {item}？")
        if decision:
            results.append(item)
    return {"results": results}
```

### Q3: interrupt() 可以在子图中使用吗？

**可以。** 子图中的 interrupt() 会冒泡到根图，由根图统一处理。

### Q4: 如果不恢复中断会怎样？

状态会一直保存在 Checkpointer 中，直到被恢复或手动清理。不会有超时自动取消的机制（除非你自己实现）。

---

## 参考资料

### 源码
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py] - interrupt() 函数定义（第 420 行）
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py] - Interrupt 数据类（第 161 行）
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/types.py] - Command 类（第 368 行）
- [来源: sourcecode/langgraph/libs/langgraph/langgraph/errors.py] - GraphInterrupt 异常（第 84 行）

### 官方文档
- [Human-in-the-loop Concepts](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Interrupts Guide](https://docs.langchain.com/oss/python/langgraph/interrupts)

---

**版本**: v1.0
**最后更新**: 2026-02-28
**作者**: Claude Code
**知识点**: 人机循环（Human-in-the-loop） - interrupt() 函数
