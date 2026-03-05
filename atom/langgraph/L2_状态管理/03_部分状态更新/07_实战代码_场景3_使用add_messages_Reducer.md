# 实战代码：使用 add_messages Reducer

> 演示如何使用 `add_messages` 维护对话历史的部分状态更新

---

## 场景说明

本文档演示如何使用 `add_messages` Reducer 实现对话历史的增量更新。`add_messages` 是 LangGraph 专门为消息列表设计的 Reducer，它不仅能追加消息，还能智能处理消息的更新和去重。

**核心特性：**
- 使用 `Annotated[Sequence[BaseMessage], add_messages]` 定义消息字段
- 自动追加新消息到对话历史
- 支持消息更新（通过 ID）
- 适用于对话式应用和多轮交互

**[来源: reference/context7_langgraph_01.md, reference/source_部分状态更新_01.md]**

---

## 完整实战代码

```python
"""
使用 add_messages Reducer 维护对话历史
演示场景：多轮对话系统，每个节点追加新消息
"""

from typing import Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ===== 1. 定义状态 Schema =====
class ConversationState(dict):
    """
    对话系统的状态定义

    字段说明：
    - messages: 对话历史（追加模式，使用 add_messages）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ===== 2. 定义节点函数 =====

def user_input(state: ConversationState) -> dict:
    """
    节点1：接收用户输入

    返回部分状态：
    - messages: 追加用户消息
    """
    print("\n[用户输入节点]")

    # 模拟用户输入
    user_message = HumanMessage(content="你好，我想了解 LangGraph")

    print(f"用户: {user_message.content}")

    # 只返回新消息
    return {
        "messages": [user_message]
    }


def assistant_response(state: ConversationState) -> dict:
    """
    节点2：生成助手回复

    返回部分状态：
    - messages: 追加助手消息
    """
    print("\n[助手回复节点]")

    # 获取最后一条用户消息
    last_message = state["messages"][-1]

    # 模拟 LLM 生成回复
    if "LangGraph" in last_message.content:
        response = "LangGraph 是一个用于构建状态化工作流的框架，它基于图结构来组织节点和边。"
    else:
        response = "我理解了，请继续。"

    ai_message = AIMessage(content=response)

    print(f"助手: {ai_message.content}")

    return {
        "messages": [ai_message]
    }


def follow_up_question(state: ConversationState) -> dict:
    """
    节点3：用户追问

    返回部分状态：
    - messages: 追加用户消息
    """
    print("\n[用户追问节点]")

    # 模拟用户追问
    user_message = HumanMessage(content="它有什么核心特性？")

    print(f"用户: {user_message.content}")

    return {
        "messages": [user_message]
    }


def detailed_response(state: ConversationState) -> dict:
    """
    节点4：详细回复

    返回部分状态：
    - messages: 追加助手消息
    """
    print("\n[详细回复节点]")

    # 模拟详细回复
    response = """LangGraph 的核心特性包括：
1. 状态管理：通过 StateGraph 管理共享状态
2. 部分更新：节点可以只返回需要更新的字段
3. Reducer 函数：使用 add_messages 等 Reducer 实现增量更新
4. 持久化：支持 Checkpoint 机制实现断点续传"""

    ai_message = AIMessage(content=response)

    print(f"助手: {ai_message.content}")

    return {
        "messages": [ai_message]
    }


# ===== 3. 构建图 =====

def create_conversation_graph():
    """创建对话流程图"""

    # 创建 StateGraph
    builder = StateGraph(ConversationState)

    # 添加节点
    builder.add_node("user_input", user_input)
    builder.add_node("assistant_response", assistant_response)
    builder.add_node("follow_up", follow_up_question)
    builder.add_node("detailed_response", detailed_response)

    # 添加边（定义执行顺序）
    builder.add_edge(START, "user_input")
    builder.add_edge("user_input", "assistant_response")
    builder.add_edge("assistant_response", "follow_up")
    builder.add_edge("follow_up", "detailed_response")
    builder.add_edge("detailed_response", END)

    # 编译图（使用内存检查点）
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# ===== 4. 运行示例 =====

def run_example_1():
    """示例1：基础对话流程"""
    print("=" * 60)
    print("示例1：基础对话流程")
    print("=" * 60)

    graph = create_conversation_graph()

    # 初始状态
    initial_state = {
        "messages": []
    }

    # 执行图
    config = {"configurable": {"thread_id": "conversation-1"}}
    final_state = graph.invoke(initial_state, config)

    # 输出完整对话历史
    print("\n" + "=" * 60)
    print("完整对话历史")
    print("=" * 60)
    for i, msg in enumerate(final_state["messages"], 1):
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"\n[消息 {i}] {role}:")
        print(f"  {msg.content}")


def run_example_2():
    """示例2：演示 add_messages 的累积效果"""
    print("\n\n" + "=" * 60)
    print("示例2：演示 add_messages 的累积效果")
    print("=" * 60)

    # 手动演示 add_messages 的行为
    print("\n手动模拟 add_messages 的工作原理：")

    # 初始状态
    current_messages = []
    print(f"初始状态: messages = {current_messages}")

    # 节点1返回
    update_1 = [HumanMessage(content="你好")]
    current_messages = add_messages(current_messages, update_1)
    print(f"\n节点1返回: [HumanMessage('你好')]")
    print(f"合并后: {len(current_messages)} 条消息")
    print(f"  - {current_messages[-1].content}")

    # 节点2返回
    update_2 = [AIMessage(content="你好！有什么可以帮助你的？")]
    current_messages = add_messages(current_messages, update_2)
    print(f"\n节点2返回: [AIMessage('你好！有什么可以帮助你的？')]")
    print(f"合并后: {len(current_messages)} 条消息")
    for msg in current_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"  - {role}: {msg.content}")

    # 节点3返回
    update_3 = [HumanMessage(content="介绍一下 LangGraph")]
    current_messages = add_messages(current_messages, update_3)
    print(f"\n节点3返回: [HumanMessage('介绍一下 LangGraph')]")
    print(f"合并后: {len(current_messages)} 条消息")
    for msg in current_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"  - {role}: {msg.content}")

    print("\n关键点：")
    print("  1. 每个节点只返回新增的消息（列表）")
    print("  2. add_messages 自动将新消息追加到现有列表")
    print("  3. 保持完整的对话历史")


def run_example_3():
    """示例3：对比 operator.add vs add_messages"""
    print("\n\n" + "=" * 60)
    print("示例3：对比 operator.add vs add_messages")
    print("=" * 60)

    import operator

    print("\n使用 operator.add（简单列表追加）：")
    messages_1 = []
    messages_1 = operator.add(messages_1, [HumanMessage(content="你好")])
    messages_1 = operator.add(messages_1, [AIMessage(content="你好！")])
    print(f"结果: {len(messages_1)} 条消息")
    for msg in messages_1:
        print(f"  - {type(msg).__name__}: {msg.content}")

    print("\n使用 add_messages（智能消息管理）：")
    messages_2 = []
    messages_2 = add_messages(messages_2, [HumanMessage(content="你好")])
    messages_2 = add_messages(messages_2, [AIMessage(content="你好！")])
    print(f"结果: {len(messages_2)} 条消息")
    for msg in messages_2:
        print(f"  - {type(msg).__name__}: {msg.content}")

    print("\n区别：")
    print("  - operator.add: 简单的列表拼接")
    print("  - add_messages: 专门为消息设计，支持消息更新和去重")


# ===== 5. 实际应用场景 =====

def run_real_world_example():
    """实际应用：客服对话系统"""
    print("\n\n" + "=" * 60)
    print("实际应用：客服对话系统")
    print("=" * 60)

    class CustomerServiceState(dict):
        """客服系统的状态"""
        messages: Annotated[Sequence[BaseMessage], add_messages]
        customer_id: str
        issue_resolved: bool

    def greet_customer(state: CustomerServiceState) -> dict:
        """欢迎客户"""
        print("\n[欢迎节点]")
        return {
            "messages": [
                SystemMessage(content="客服系统已启动"),
                AIMessage(content=f"您好！我是客服助手，客户ID: {state['customer_id']}。请问有什么可以帮助您的？")
            ]
        }

    def receive_issue(state: CustomerServiceState) -> dict:
        """接收问题"""
        print("\n[接收问题节点]")
        return {
            "messages": [
                HumanMessage(content="我的订单还没有发货，订单号是 12345")
            ]
        }

    def check_order(state: CustomerServiceState) -> dict:
        """查询订单"""
        print("\n[查询订单节点]")
        return {
            "messages": [
                AIMessage(content="让我帮您查询一下订单 12345 的状态..."),
                ToolMessage(
                    content="订单状态: 已发货，预计明天送达",
                    tool_call_id="check_order_12345"
                )
            ]
        }

    def provide_solution(state: CustomerServiceState) -> dict:
        """提供解决方案"""
        print("\n[提供解决方案节点]")
        return {
            "messages": [
                AIMessage(content="您的订单 12345 已经发货，预计明天送达。您可以通过物流单号 SF1234567890 查询详细信息。")
            ],
            "issue_resolved": True
        }

    def confirm_resolution(state: CustomerServiceState) -> dict:
        """确认解决"""
        print("\n[确认解决节点]")
        return {
            "messages": [
                HumanMessage(content="好的，谢谢！"),
                AIMessage(content="不客气！如果还有其他问题，随时联系我们。祝您生活愉快！")
            ]
        }

    # 构建图
    builder = StateGraph(CustomerServiceState)
    builder.add_node("greet", greet_customer)
    builder.add_node("receive_issue", receive_issue)
    builder.add_node("check_order", check_order)
    builder.add_node("provide_solution", provide_solution)
    builder.add_node("confirm", confirm_resolution)

    builder.add_edge(START, "greet")
    builder.add_edge("greet", "receive_issue")
    builder.add_edge("receive_issue", "check_order")
    builder.add_edge("check_order", "provide_solution")
    builder.add_edge("provide_solution", "confirm")
    builder.add_edge("confirm", END)

    graph = builder.compile()

    # 执行
    initial_state = {
        "messages": [],
        "customer_id": "CUST-001",
        "issue_resolved": False
    }

    final_state = graph.invoke(initial_state)

    # 输出结果
    print("\n" + "=" * 60)
    print("对话记录")
    print("=" * 60)
    print(f"\n客户ID: {final_state['customer_id']}")
    print(f"问题已解决: {'是' if final_state['issue_resolved'] else '否'}")
    print(f"\n完整对话 ({len(final_state['messages'])} 条消息):")

    for i, msg in enumerate(final_state["messages"], 1):
        if isinstance(msg, SystemMessage):
            role = "系统"
        elif isinstance(msg, HumanMessage):
            role = "客户"
        elif isinstance(msg, AIMessage):
            role = "客服"
        elif isinstance(msg, ToolMessage):
            role = "工具"
        else:
            role = "未知"

        print(f"\n[{i}] {role}:")
        print(f"    {msg.content}")


# ===== 6. 高级特性：消息更新 =====

def run_advanced_example():
    """高级示例：消息更新和去重"""
    print("\n\n" + "=" * 60)
    print("高级示例：消息更新和去重")
    print("=" * 60)

    print("\nadd_messages 支持通过 ID 更新消息：")

    # 创建带 ID 的消息
    msg1 = HumanMessage(content="原始消息", id="msg-1")
    msg2 = AIMessage(content="回复消息", id="msg-2")

    messages = [msg1, msg2]
    print(f"\n初始消息:")
    for msg in messages:
        print(f"  - [{msg.id}] {msg.content}")

    # 更新消息（相同 ID）
    updated_msg1 = HumanMessage(content="更新后的消息", id="msg-1")
    messages = add_messages(messages, [updated_msg1])

    print(f"\n更新后:")
    for msg in messages:
        print(f"  - [{msg.id}] {msg.content}")

    print("\n关键点：")
    print("  - 如果新消息的 ID 已存在，会更新而不是追加")
    print("  - 这对于实现消息编辑功能很有用")


# ===== 7. 主函数 =====

if __name__ == "__main__":
    # 运行所有示例
    run_example_1()
    run_example_2()
    run_example_3()
    run_real_world_example()
    run_advanced_example()

    print("\n\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)
```

---

## 运行输出示例

```
============================================================
示例1：基础对话流程
============================================================

[用户输入节点]
用户: 你好，我想了解 LangGraph

[助手回复节点]
助手: LangGraph 是一个用于构建状态化工作流的框架，它基于图结构来组织节点和边。

[用户追问节点]
用户: 它有什么核心特性？

[详细回复节点]
助手: LangGraph 的核心特性包括：
1. 状态管理：通过 StateGraph 管理共享状态
2. 部分更新：节点可以只返回需要更新的字段
3. Reducer 函数：使用 add_messages 等 Reducer 实现增量更新
4. 持久化：支持 Checkpoint 机制实现断点续传

============================================================
完整对话历史
============================================================

[消息 1] 用户:
  你好，我想了解 LangGraph

[消息 2] 助手:
  LangGraph 是一个用于构建状态化工作流的框架，它基于图结构来组织节点和边。

[消息 3] 用户:
  它有什么核心特性？

[消息 4] 助手:
  LangGraph 的核心特性包括：
1. 状态管理：通过 StateGraph 管理共享状态
2. 部分更新：节点可以只返回需要更新的字段
3. Reducer 函数：使用 add_messages 等 Reducer 实现增量更新
4. 持久化：支持 Checkpoint 机制实现断点续传
```

---

## 核心要点总结

### 1. add_messages 的工作原理

**[来源: reference/context7_langgraph_01.md]**

```python
# add_messages 对消息列表的行为
current = [HumanMessage(content="你好")]
new = [AIMessage(content="你好！")]
result = add_messages(current, new)
# result = [HumanMessage("你好"), AIMessage("你好！")]
```

### 2. 状态定义模式

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(dict):
    # 使用 add_messages 维护对话历史
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

### 3. 节点返回值

```python
def my_node(state: State) -> dict:
    # 只返回新增的消息（列表形式）
    return {
        "messages": [
            HumanMessage(content="用户消息"),
            AIMessage(content="助手回复")
        ]
    }
```

### 4. 消息类型

**[来源: reference/context7_langgraph_01.md]**

- `HumanMessage`: 用户消息
- `AIMessage`: AI 助手消息
- `SystemMessage`: 系统消息
- `ToolMessage`: 工具调用结果消息

### 5. 适用场景

- 对话式应用
- 多轮交互
- 客服系统
- 聊天机器人
- 问答系统

---

## 常见问题

### Q1: add_messages 和 operator.add 有什么区别？

**A:** `add_messages` 是专门为消息设计的，支持消息更新和去重。

```python
# operator.add - 简单列表拼接
messages = operator.add(messages, [new_message])

# add_messages - 智能消息管理
messages = add_messages(messages, [new_message])
```

### Q2: 如何更新已有的消息？

**A:** 使用相同的消息 ID。

```python
# 原始消息
msg = HumanMessage(content="原始内容", id="msg-1")

# 更新消息（相同 ID）
updated_msg = HumanMessage(content="更新内容", id="msg-1")
messages = add_messages(messages, [updated_msg])
```

### Q3: 可以一次追加多条消息吗？

**A:** 可以，返回包含多条消息的列表即可。

```python
return {
    "messages": [
        HumanMessage(content="消息1"),
        AIMessage(content="消息2"),
        SystemMessage(content="消息3")
    ]
}
```

### Q4: 如何清空消息历史？

**A:** 使用 Overwrite 模式（将在场景6中详细讲解）。

---

## 与 operator.add 的对比

| 特性 | operator.add | add_messages |
|------|--------------|--------------|
| 用途 | 通用列表追加 | 专门用于消息 |
| 消息更新 | 不支持 | 支持（通过 ID） |
| 消息去重 | 不支持 | 支持 |
| 类型检查 | 无 | 有（BaseMessage） |
| 推荐场景 | 简单列表 | 对话历史 |

---

## 参考资料

- **[来源: reference/context7_langgraph_01.md]** - LangGraph 官方文档关于 add_messages 的说明
- **[来源: reference/source_部分状态更新_01.md]** - add_messages 源码分析
- **[来源: reference/search_部分状态更新_01.md]** - 社区最佳实践

---

**文档版本:** v1.0
**创建时间:** 2026-02-26
**适用于:** LangGraph 0.2.0+
**Python 版本:** 3.13+
