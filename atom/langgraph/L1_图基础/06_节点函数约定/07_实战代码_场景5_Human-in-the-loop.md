# 实战代码 - 场景5：Human-in-the-loop

> 本文档展示如何在 LangGraph 节点函数中实现 Human-in-the-loop 模式，包括 interrupt 函数使用、用户输入处理、状态恢复和完整的交互式工作流。

---

## 场景概述

Human-in-the-loop (HITL) 是一种在自动化流程中引入人工干预的模式。在 LangGraph 中，通过 `interrupt()` 函数可以暂停图的执行，等待用户输入，然后恢复执行。

**核心特性**：
- 使用 `interrupt()` 函数暂停执行
- 需要 checkpointer 保存状态
- 使用 `Command(resume=...)` 恢复执行
- 支持多步骤交互式工作流

[来源: reference/context7_langgraph_01.md]

---

## 示例 1：基础 interrupt 使用

### 场景说明

最简单的 Human-in-the-loop 场景：在节点中暂停执行，等待用户输入一个值。

### 完整代码

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# 定义状态
class State(TypedDict):
    user_input: str
    result: str

# 节点函数：请求用户输入
def ask_user(state: State) -> dict:
    """暂停执行，等待用户输入"""
    user_input = interrupt("请输入您的名字：")
    return {"user_input": user_input}

# 节点函数：处理用户输入
def process_input(state: State) -> dict:
    """处理用户输入"""
    name = state["user_input"]
    result = f"你好，{name}！欢迎使用 LangGraph。"
    return {"result": result}

# 构建图
builder = StateGraph(State)
builder.add_node("ask", ask_user)
builder.add_node("process", process_input)
builder.add_edge(START, "ask")
builder.add_edge("ask", "process")
builder.add_edge("process", END)

# 编译图（必须使用 checkpointer）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

[来源: reference/context7_langgraph_01.md]

### 运行示例

```python
from langgraph.types import Command

# 初始化配置
config = {"configurable": {"thread_id": "1"}}

# 第一次调用：触发 interrupt
result = graph.invoke({"user_input": "", "result": ""}, config)
print("图已暂停，等待用户输入")
print(f"当前状态: {result}")

# 恢复执行：提供用户输入
result = graph.invoke(
    Command(resume="张三"),
    config
)
print(f"最终结果: {result['result']}")
# 输出: 你好，张三！欢迎使用 LangGraph。
```

### 关键点说明

1. **必须使用 checkpointer**：`interrupt()` 需要保存状态，因此必须在编译时提供 checkpointer
2. **使用 Command(resume=...)**：恢复执行时，使用 `Command(resume=value)` 提供用户输入
3. **thread_id 配置**：每个会话需要唯一的 thread_id 来保存状态

---

## 示例 2：表单收集场景

### 场景说明

多步骤表单收集：依次收集用户的姓名、邮箱，然后确认信息。

[来源: reference/context7_langgraph_01.md]

### 完整代码

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# 定义状态
class FormState(TypedDict):
    name: str
    email: str
    confirmed: bool

# 节点函数：收集用户信息
def collect_info(state: FormState) -> dict:
    """收集用户姓名和邮箱"""
    name = interrupt("请输入您的姓名：")
    email = interrupt("请输入您的邮箱：")
    return {"name": name, "email": email}

# 节点函数：确认信息
def confirm_info(state: FormState) -> dict:
    """确认收集的信息"""
    confirmation = interrupt(
        f"请确认您的信息 - 姓名: {state['name']}, 邮箱: {state['email']} (yes/no)"
    )
    return {"confirmed": confirmation == "yes"}

# 节点函数：处理结果
def process_form(state: FormState) -> dict:
    """处理表单提交"""
    if state["confirmed"]:
        print(f"表单已提交：{state['name']} ({state['email']})")
    else:
        print("表单已取消")
    return {}

# 构建图
builder = StateGraph(FormState)
builder.add_node("collect", collect_info)
builder.add_node("confirm", confirm_info)
builder.add_node("process", process_form)
builder.add_edge(START, "collect")
builder.add_edge("collect", "confirm")
builder.add_edge("confirm", "process")
builder.add_edge("process", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 运行示例

```python
from langgraph.types import Command

# 初始化配置
config = {"configurable": {"thread_id": "form_1"}}

# 第一次调用：触发第一个 interrupt（姓名）
result = graph.invoke(
    {"name": "", "email": "", "confirmed": False},
    config
)
print("等待输入姓名...")

# 提供姓名，触发第二个 interrupt（邮箱）
result = graph.invoke(Command(resume="张三"), config)
print("等待输入邮箱...")

# 提供邮箱，触发第三个 interrupt（确认）
result = graph.invoke(Command(resume="zhangsan@example.com"), config)
print("等待确认...")

# 提供确认，完成执行
result = graph.invoke(Command(resume="yes"), config)
print(f"最终状态: {result}")
# 输出: 表单已提交：张三 (zhangsan@example.com)
```

### 关键点说明

1. **多个 interrupt**：一个节点可以包含多个 `interrupt()` 调用，每次暂停都需要单独恢复
2. **状态传递**：每次恢复后，状态会更新并传递到下一个节点
3. **顺序执行**：interrupt 按照代码顺序依次触发

---

## 示例 3：条件性中断

### 场景说明

根据状态决定是否需要人工干预：只有在检测到异常情况时才暂停执行。

### 完整代码

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# 定义状态
class ValidationState(TypedDict):
    data: str
    is_valid: bool
    user_decision: str

# 节点函数：验证数据
def validate_data(state: ValidationState) -> dict:
    """验证数据，如果无效则请求人工干预"""
    data = state["data"]
    
    # 简单验证：检查数据长度
    if len(data) < 5:
        print("数据验证失败：长度不足")
        # 请求人工决策
        decision = interrupt(
            f"数据 '{data}' 验证失败。是否继续处理？(yes/no)"
        )
        return {"is_valid": False, "user_decision": decision}
    else:
        print("数据验证通过")
        return {"is_valid": True, "user_decision": "auto"}

# 节点函数：处理数据
def process_data(state: ValidationState) -> dict:
    """根据验证结果处理数据"""
    if state["is_valid"] or state["user_decision"] == "yes":
        print(f"处理数据: {state['data']}")
        return {}
    else:
        print("数据处理已取消")
        return {}

# 构建图
builder = StateGraph(ValidationState)
builder.add_node("validate", validate_data)
builder.add_node("process", process_data)
builder.add_edge(START, "validate")
builder.add_edge("validate", "process")
builder.add_edge("process", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 运行示例

```python
from langgraph.types import Command

# 场景 1：数据有效，自动通过
config1 = {"configurable": {"thread_id": "valid_1"}}
result = graph.invoke(
    {"data": "valid_data_12345", "is_valid": False, "user_decision": ""},
    config1
)
print(f"场景1结果: {result}")
# 输出: 数据验证通过
#       处理数据: valid_data_12345

# 场景 2：数据无效，需要人工决策
config2 = {"configurable": {"thread_id": "invalid_1"}}
result = graph.invoke(
    {"data": "bad", "is_valid": False, "user_decision": ""},
    config2
)
print("等待人工决策...")

# 用户决定继续处理
result = graph.invoke(Command(resume="yes"), config2)
print(f"场景2结果: {result}")
# 输出: 数据验证失败：长度不足
#       处理数据: bad

# 场景 3：数据无效，用户取消
config3 = {"configurable": {"thread_id": "invalid_2"}}
result = graph.invoke(
    {"data": "bad", "is_valid": False, "user_decision": ""},
    config3
)
result = graph.invoke(Command(resume="no"), config3)
print(f"场景3结果: {result}")
# 输出: 数据验证失败：长度不足
#       数据处理已取消
```

### 关键点说明

1. **条件性 interrupt**：只在特定条件下调用 `interrupt()`
2. **自动 vs 人工**：正常情况自动处理，异常情况人工介入
3. **决策记录**：将人工决策保存到状态中，供后续节点使用

---

## 示例 4：状态恢复机制

### 场景说明

展示如何在中断后恢复执行，包括查看当前状态、修改状态、继续执行。

### 完整代码

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# 定义状态
class WorkflowState(TypedDict):
    step: int
    data: list[str]
    user_input: str

# 节点函数：步骤 1
def step1(state: WorkflowState) -> dict:
    """第一步：初始化数据"""
    print("执行步骤 1")
    return {"step": 1, "data": ["初始化完成"]}

# 节点函数：步骤 2
def step2(state: WorkflowState) -> dict:
    """第二步：请求用户输入"""
    print("执行步骤 2")
    user_input = interrupt("请输入处理指令：")
    return {"step": 2, "user_input": user_input, "data": state["data"] + [f"用户输入: {user_input}"]}

# 节点函数：步骤 3
def step3(state: WorkflowState) -> dict:
    """第三步：处理数据"""
    print("执行步骤 3")
    return {"step": 3, "data": state["data"] + ["处理完成"]}

# 构建图
builder = StateGraph(WorkflowState)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_node("step3", step3)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 运行示例

```python
from langgraph.types import Command

# 初始化配置
config = {"configurable": {"thread_id": "workflow_1"}}

# 第一次调用：执行到 step2 的 interrupt
print("=== 第一次调用 ===")
result = graph.invoke(
    {"step": 0, "data": [], "user_input": ""},
    config
)
print(f"当前状态: step={result['step']}, data={result['data']}")
# 输出: 执行步骤 1
#       执行步骤 2
#       当前状态: step=2, data=['初始化完成']

# 查看保存的状态
print("\n=== 查看保存的状态 ===")
state_snapshot = checkpointer.get(config)
print(f"保存的状态: {state_snapshot}")

# 恢复执行：提供用户输入
print("\n=== 恢复执行 ===")
result = graph.invoke(Command(resume="处理数据"), config)
print(f"最终状态: step={result['step']}, data={result['data']}")
# 输出: 执行步骤 3
#       最终状态: step=3, data=['初始化完成', '用户输入: 处理数据', '处理完成']
```

### 关键点说明

1. **状态持久化**：checkpointer 自动保存每次中断时的状态
2. **状态查询**：可以使用 `checkpointer.get(config)` 查看保存的状态
3. **无缝恢复**：恢复执行时，从中断点继续，不会重新执行已完成的节点

---

## 示例 5：完整的交互式工作流

### 场景说明

一个完整的文档审批工作流：提交文档 → 自动检查 → 人工审批 → 处理结果。

### 完整代码

```python
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Literal

# 定义状态
class ApprovalState(TypedDict):
    document: str
    auto_check_passed: bool
    approval_status: Literal["pending", "approved", "rejected"]
    reviewer_comment: str
    final_result: str

# 节点函数：提交文档
def submit_document(state: ApprovalState) -> dict:
    """提交文档"""
    document = state["document"]
    print(f"文档已提交: {document}")
    return {"approval_status": "pending"}

# 节点函数：自动检查
def auto_check(state: ApprovalState) -> dict:
    """自动检查文档格式和内容"""
    document = state["document"]
    
    # 简单检查：文档长度
    if len(document) < 10:
        print("自动检查失败：文档内容过短")
        return {"auto_check_passed": False}
    else:
        print("自动检查通过")
        return {"auto_check_passed": True}

# 节点函数：人工审批
def manual_review(state: ApprovalState) -> dict:
    """人工审批文档"""
    document = state["document"]
    
    # 请求审批决策
    decision = interrupt(
        f"请审批文档: '{document}'\n选项: approve/reject"
    )
    
    # 如果拒绝，请求评论
    if decision == "reject":
        comment = interrupt("请输入拒绝原因：")
        return {
            "approval_status": "rejected",
            "reviewer_comment": comment
        }
    else:
        return {
            "approval_status": "approved",
            "reviewer_comment": "审批通过"
        }

# 节点函数：处理结果
def process_result(state: ApprovalState) -> dict:
    """处理审批结果"""
    status = state["approval_status"]
    
    if status == "approved":
        result = f"文档 '{state['document']}' 已批准。评论: {state['reviewer_comment']}"
    else:
        result = f"文档 '{state['document']}' 已拒绝。原因: {state['reviewer_comment']}"
    
    print(result)
    return {"final_result": result}

# 条件路由：根据自动检查结果决定下一步
def route_after_check(state: ApprovalState) -> Literal["manual_review", "reject"]:
    """路由：自动检查通过则进入人工审批，否则直接拒绝"""
    if state["auto_check_passed"]:
        return "manual_review"
    else:
        return "reject"

# 节点函数：自动拒绝
def auto_reject(state: ApprovalState) -> dict:
    """自动拒绝文档"""
    return {
        "approval_status": "rejected",
        "reviewer_comment": "自动检查未通过"
    }

# 构建图
builder = StateGraph(ApprovalState)
builder.add_node("submit", submit_document)
builder.add_node("check", auto_check)
builder.add_node("manual_review", manual_review)
builder.add_node("reject", auto_reject)
builder.add_node("process", process_result)

builder.add_edge(START, "submit")
builder.add_edge("submit", "check")
builder.add_conditional_edges(
    "check",
    route_after_check,
    {"manual_review": "manual_review", "reject": "reject"}
)
builder.add_edge("manual_review", "process")
builder.add_edge("reject", "process")
builder.add_edge("process", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 运行示例

```python
# 场景 1：文档通过自动检查，人工审批通过
print("=== 场景 1：审批通过 ===")
config1 = {"configurable": {"thread_id": "approval_1"}}

# 提交文档
result = graph.invoke(
    {
        "document": "这是一份完整的技术文档，包含详细的设计方案和实现细节。",
        "auto_check_passed": False,
        "approval_status": "pending",
        "reviewer_comment": "",
        "final_result": ""
    },
    config1
)
print("等待审批决策...")

# 审批通过
result = graph.invoke(Command(resume="approve"), config1)
print(f"最终结果: {result['final_result']}\n")

# 场景 2：文档通过自动检查，人工审批拒绝
print("=== 场景 2：审批拒绝 ===")
config2 = {"configurable": {"thread_id": "approval_2"}}

result = graph.invoke(
    {
        "document": "这是一份需要修改的文档。",
        "auto_check_passed": False,
        "approval_status": "pending",
        "reviewer_comment": "",
        "final_result": ""
    },
    config2
)
print("等待审批决策...")

# 审批拒绝
result = graph.invoke(Command(resume="reject"), config2)
print("等待拒绝原因...")

# 提供拒绝原因
result = graph.invoke(Command(resume="文档格式不符合规范"), config2)
print(f"最终结果: {result['final_result']}\n")

# 场景 3：文档未通过自动检查，直接拒绝
print("=== 场景 3：自动拒绝 ===")
config3 = {"configurable": {"thread_id": "approval_3"}}

result = graph.invoke(
    {
        "document": "短文档",
        "auto_check_passed": False,
        "approval_status": "pending",
        "reviewer_comment": "",
        "final_result": ""
    },
    config3
)
print(f"最终结果: {result['final_result']}")
```

### 关键点说明

1. **条件路由 + interrupt**：结合条件路由和人工干预，实现灵活的工作流
2. **多次 interrupt**：根据不同情况，可能需要多次人工输入
3. **状态管理**：清晰的状态定义，便于追踪工作流进度
4. **错误处理**：自动检查失败时，跳过人工审批，直接拒绝

---

## 最佳实践

### 1. Checkpointer 选择

```python
# 开发环境：使用内存 checkpointer
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# 生产环境：使用持久化 checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

### 2. Thread ID 管理

```python
import uuid

# 为每个会话生成唯一 ID
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 或使用用户 ID + 会话 ID
thread_id = f"user_{user_id}_session_{session_id}"
config = {"configurable": {"thread_id": thread_id}}
```

### 3. 错误处理

```python
def safe_interrupt(prompt: str, default: str = "") -> str:
    """安全的 interrupt 调用，带默认值"""
    try:
        return interrupt(prompt)
    except Exception as e:
        print(f"Interrupt 失败: {e}")
        return default
```

### 4. 超时处理

```python
import time

def timed_interrupt(prompt: str, timeout: int = 300) -> str:
    """带超时的 interrupt"""
    start_time = time.time()
    result = interrupt(prompt)
    
    if time.time() - start_time > timeout:
        raise TimeoutError("用户输入超时")
    
    return result
```

---

## 常见问题

### Q1: 为什么必须使用 checkpointer？

**A**: `interrupt()` 需要保存图的执行状态，以便在恢复时从中断点继续执行。没有 checkpointer，状态无法持久化，恢复执行会失败。

### Q2: 如何在 interrupt 后修改状态？

**A**: 使用 `Command(resume=value, update={...})` 可以在恢复时同时更新状态：

```python
result = graph.invoke(
    Command(resume="user_input", update={"extra_field": "value"}),
    config
)
```

### Q3: 可以在一个节点中使用多个 interrupt 吗？

**A**: 可以。每个 `interrupt()` 调用都会暂停执行，需要单独恢复。

### Q4: 如何取消一个已中断的执行？

**A**: 可以通过删除 checkpointer 中的状态来取消：

```python
checkpointer.delete(config)
```

---

## 总结

Human-in-the-loop 模式的核心要点：

1. **interrupt 函数**：暂停执行，等待用户输入
2. **Checkpointer**：必须使用 checkpointer 保存状态
3. **Command(resume=...)**：恢复执行时提供用户输入
4. **Thread ID**：每个会话需要唯一的 thread_id
5. **条件性中断**：根据状态决定是否需要人工干预
6. **多步骤交互**：支持复杂的交互式工作流

通过这些机制，可以构建灵活的人机协作系统，在自动化和人工控制之间找到平衡。

[来源: reference/context7_langgraph_01.md]
