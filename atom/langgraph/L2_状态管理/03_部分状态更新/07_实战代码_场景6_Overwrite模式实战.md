# 07_实战代码_场景6_Overwrite模式实战

> 演示如何使用 Overwrite 模式显式覆盖 Reducer 定义的合并策略

---

## 场景概述

**学习目标：**
- 理解 Overwrite 模式的使用场景
- 掌握如何显式覆盖 Reducer 策略
- 学会在实际应用中使用 Overwrite 重置状态

**适用场景：**
- 需要重置累积的状态（如清空消息历史）
- 需要覆盖 Reducer 定义的合并行为
- 需要在特定条件下强制替换状态值

**技术要点：**
- `Overwrite` 类的使用
- `{OVERWRITE: value}` 字典语法
- 与 Reducer 函数的交互

---

## 完整实战代码

```python
"""
LangGraph 部分状态更新 - Overwrite 模式实战

演示场景：
1. 基础 Overwrite 使用
2. 覆盖 add_messages Reducer
3. 覆盖 operator.add Reducer
4. 条件性状态重置
5. 实际应用：对话历史管理

来源：
- [来源: reference/source_部分状态更新_01.md] - Overwrite 模式源码分析
- [来源: reference/context7_langgraph_01.md] - LangGraph 官方文档
- [来源: reference/search_部分状态更新_01.md] - 社区最佳实践
"""

import operator
from typing import Annotated, TypedDict, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.channels.binop import Overwrite
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ===== 场景1：基础 Overwrite 使用 =====
print("=" * 60)
print("场景1：基础 Overwrite 使用")
print("=" * 60)

class BasicState(TypedDict):
    """基础状态定义 - 使用 operator.add Reducer"""
    counter: Annotated[int, operator.add]  # 累加策略
    items: Annotated[list[str], operator.add]  # 列表追加策略

def increment_node(state: BasicState) -> dict:
    """正常累加节点"""
    print(f"  [increment_node] 当前 counter: {state['counter']}")
    return {
        "counter": 1,  # 使用 Reducer，累加 1
        "items": ["item"]  # 使用 Reducer，追加元素
    }

def reset_node(state: BasicState) -> dict:
    """使用 Overwrite 重置节点"""
    print(f"  [reset_node] 当前 counter: {state['counter']}")
    print(f"  [reset_node] 使用 Overwrite 重置为 0")
    return {
        "counter": Overwrite(0),  # 显式覆盖，重置为 0
        "items": Overwrite([])  # 显式覆盖，清空列表
    }

# 构建图
builder = StateGraph(BasicState)
builder.add_node("increment1", increment_node)
builder.add_node("increment2", increment_node)
builder.add_node("reset", reset_node)
builder.add_node("increment3", increment_node)

builder.add_edge(START, "increment1")
builder.add_edge("increment1", "increment2")
builder.add_edge("increment2", "reset")
builder.add_edge("reset", "increment3")
builder.add_edge("increment3", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({"counter": 0, "items": []})

print(f"\n最终结果：")
print(f"  counter: {result['counter']}")  # 应该是 1（重置后再累加）
print(f"  items: {result['items']}")  # 应该是 ['item']（重置后再追加）

# ===== 场景2：覆盖 add_messages Reducer =====
print("\n" + "=" * 60)
print("场景2：覆盖 add_messages Reducer")
print("=" * 60)

class ChatState(TypedDict):
    """对话状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def add_user_message(state: ChatState) -> dict:
    """添加用户消息"""
    print(f"  [add_user_message] 当前消息数: {len(state['messages'])}")
    return {
        "messages": [HumanMessage(content="Hello!")]
    }

def add_ai_response(state: ChatState) -> dict:
    """添加 AI 回复"""
    print(f"  [add_ai_response] 当前消息数: {len(state['messages'])}")
    return {
        "messages": [AIMessage(content="Hi there!")]
    }

def clear_history(state: ChatState) -> dict:
    """清空对话历史"""
    print(f"  [clear_history] 当前消息数: {len(state['messages'])}")
    print(f"  [clear_history] 使用 Overwrite 清空历史")
    return {
        "messages": Overwrite([])  # 显式覆盖，清空消息列表
    }

def add_new_conversation(state: ChatState) -> dict:
    """开始新对话"""
    print(f"  [add_new_conversation] 当前消息数: {len(state['messages'])}")
    return {
        "messages": [HumanMessage(content="New conversation")]
    }

# 构建图
builder = StateGraph(ChatState)
builder.add_node("user1", add_user_message)
builder.add_node("ai1", add_ai_response)
builder.add_node("clear", clear_history)
builder.add_node("user2", add_new_conversation)

builder.add_edge(START, "user1")
builder.add_edge("user1", "ai1")
builder.add_edge("ai1", "clear")
builder.add_edge("clear", "user2")
builder.add_edge("user2", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({"messages": []})

print(f"\n最终结果：")
print(f"  消息数: {len(result['messages'])}")
for i, msg in enumerate(result['messages']):
    print(f"  [{i}] {msg.__class__.__name__}: {msg.content}")

# ===== 场景3：使用字典语法的 Overwrite =====
print("\n" + "=" * 60)
print("场景3：使用字典语法的 Overwrite")
print("=" * 60)

from langgraph.constants import OVERWRITE

class DictState(TypedDict):
    """状态定义"""
    scores: Annotated[list[int], operator.add]

def add_scores(state: DictState) -> dict:
    """添加分数"""
    print(f"  [add_scores] 当前分数: {state['scores']}")
    return {"scores": [10, 20]}

def reset_with_dict(state: DictState) -> dict:
    """使用字典语法重置"""
    print(f"  [reset_with_dict] 当前分数: {state['scores']}")
    print(f"  [reset_with_dict] 使用 {{OVERWRITE: value}} 语法重置")
    return {
        "scores": {OVERWRITE: [100]}  # 字典语法覆盖
    }

# 构建图
builder = StateGraph(DictState)
builder.add_node("add1", add_scores)
builder.add_node("reset", reset_with_dict)
builder.add_node("add2", add_scores)

builder.add_edge(START, "add1")
builder.add_edge("add1", "reset")
builder.add_edge("reset", "add2")
builder.add_edge("add2", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({"scores": []})

print(f"\n最终结果：")
print(f"  scores: {result['scores']}")  # 应该是 [100, 10, 20]

# ===== 场景4：条件性状态重置 =====
print("\n" + "=" * 60)
print("场景4：条件性状态重置")
print("=" * 60)

class ConditionalState(TypedDict):
    """条件状态定义"""
    error_count: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]
    should_reset: bool

def process_with_error(state: ConditionalState) -> dict:
    """处理并可能产生错误"""
    print(f"  [process_with_error] 当前错误数: {state['error_count']}")
    return {
        "error_count": 1,
        "errors": ["Error occurred"],
        "should_reset": state["error_count"] + 1 >= 3  # 错误达到3次时标记重置
    }

def conditional_reset(state: ConditionalState) -> dict:
    """根据条件决定是否重置"""
    if state["should_reset"]:
        print(f"  [conditional_reset] 错误达到阈值，重置状态")
        return {
            "error_count": Overwrite(0),
            "errors": Overwrite([]),
            "should_reset": False
        }
    else:
        print(f"  [conditional_reset] 错误未达到阈值，保持状态")
        return {}

# 构建图
builder = StateGraph(ConditionalState)
builder.add_node("process1", process_with_error)
builder.add_node("process2", process_with_error)
builder.add_node("process3", process_with_error)
builder.add_node("reset", conditional_reset)

builder.add_edge(START, "process1")
builder.add_edge("process1", "process2")
builder.add_edge("process2", "process3")
builder.add_edge("process3", "reset")
builder.add_edge("reset", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({
    "error_count": 0,
    "errors": [],
    "should_reset": False
})

print(f"\n最终结果：")
print(f"  error_count: {result['error_count']}")
print(f"  errors: {result['errors']}")
print(f"  should_reset: {result['should_reset']}")

# ===== 场景5：实际应用 - 对话历史管理系统 =====
print("\n" + "=" * 60)
print("场景5：实际应用 - 对话历史管理系统")
print("=" * 60)

class ConversationState(TypedDict):
    """对话管理状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    conversation_id: str
    message_count: Annotated[int, operator.add]
    max_messages: int

def add_message(state: ConversationState) -> dict:
    """添加消息"""
    content = f"Message {state['message_count'] + 1}"
    print(f"  [add_message] 添加消息: {content}")
    return {
        "messages": [HumanMessage(content=content)],
        "message_count": 1
    }

def check_and_trim(state: ConversationState) -> dict:
    """检查并裁剪消息历史"""
    if state["message_count"] > state["max_messages"]:
        print(f"  [check_and_trim] 消息数 ({state['message_count']}) 超过限制 ({state['max_messages']})")
        print(f"  [check_and_trim] 保留最近 {state['max_messages']} 条消息")

        # 保留最近的消息
        recent_messages = list(state["messages"])[-state["max_messages"]:]

        return {
            "messages": Overwrite(recent_messages),
            "message_count": Overwrite(len(recent_messages))
        }
    else:
        print(f"  [check_and_trim] 消息数 ({state['message_count']}) 未超过限制")
        return {}

def start_new_conversation(state: ConversationState) -> dict:
    """开始新对话"""
    print(f"  [start_new_conversation] 清空历史，开始新对话")
    return {
        "messages": Overwrite([HumanMessage(content="New conversation started")]),
        "message_count": Overwrite(1),
        "conversation_id": "new_conversation_id"
    }

# 构建图
builder = StateGraph(ConversationState)
builder.add_node("msg1", add_message)
builder.add_node("msg2", add_message)
builder.add_node("msg3", add_message)
builder.add_node("msg4", add_message)
builder.add_node("trim", check_and_trim)
builder.add_node("new_conv", start_new_conversation)

builder.add_edge(START, "msg1")
builder.add_edge("msg1", "msg2")
builder.add_edge("msg2", "msg3")
builder.add_edge("msg3", "msg4")
builder.add_edge("msg4", "trim")
builder.add_edge("trim", "new_conv")
builder.add_edge("new_conv", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({
    "messages": [],
    "conversation_id": "initial_id",
    "message_count": 0,
    "max_messages": 3
})

print(f"\n最终结果：")
print(f"  conversation_id: {result['conversation_id']}")
print(f"  message_count: {result['message_count']}")
print(f"  消息列表:")
for i, msg in enumerate(result['messages']):
    print(f"    [{i}] {msg.content}")

# ===== 场景6：Overwrite vs 正常更新对比 =====
print("\n" + "=" * 60)
print("场景6：Overwrite vs 正常更新对比")
print("=" * 60)

class ComparisonState(TypedDict):
    """对比状态定义"""
    normal_list: Annotated[list[str], operator.add]
    overwrite_list: Annotated[list[str], operator.add]

def update_both_normal(state: ComparisonState) -> dict:
    """正常更新两个列表"""
    print(f"  [update_both_normal] 正常更新")
    return {
        "normal_list": ["A"],
        "overwrite_list": ["A"]
    }

def update_with_overwrite(state: ComparisonState) -> dict:
    """一个正常更新，一个使用 Overwrite"""
    print(f"  [update_with_overwrite] normal_list 正常追加，overwrite_list 使用 Overwrite")
    return {
        "normal_list": ["B"],  # 追加
        "overwrite_list": Overwrite(["B"])  # 覆盖
    }

def final_update(state: ComparisonState) -> dict:
    """最终更新"""
    print(f"  [final_update] 两个列表都正常追加")
    return {
        "normal_list": ["C"],
        "overwrite_list": ["C"]
    }

# 构建图
builder = StateGraph(ComparisonState)
builder.add_node("update1", update_both_normal)
builder.add_node("update2", update_with_overwrite)
builder.add_node("update3", final_update)

builder.add_edge(START, "update1")
builder.add_edge("update1", "update2")
builder.add_edge("update2", "update3")
builder.add_edge("update3", END)

graph = builder.compile()

# 执行图
print("\n执行流程：")
result = graph.invoke({
    "normal_list": [],
    "overwrite_list": []
})

print(f"\n最终结果对比：")
print(f"  normal_list (正常追加): {result['normal_list']}")  # ['A', 'B', 'C']
print(f"  overwrite_list (中间覆盖): {result['overwrite_list']}")  # ['B', 'C']

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
Overwrite 模式的关键要点：

1. **使用场景**：
   - 需要重置累积的状态
   - 需要覆盖 Reducer 定义的合并行为
   - 需要在特定条件下强制替换状态值

2. **两种语法**：
   - Overwrite(value) - 类语法
   - {OVERWRITE: value} - 字典语法

3. **与 Reducer 的关系**：
   - Overwrite 优先级高于 Reducer
   - 可以临时覆盖 Reducer 的合并策略
   - 不影响后续节点的 Reducer 行为

4. **实际应用**：
   - 对话历史管理（清空/裁剪）
   - 错误状态重置
   - 条件性状态覆盖
   - 会话切换

5. **最佳实践**：
   - 谨慎使用，避免破坏状态一致性
   - 明确注释 Overwrite 的使用原因
   - 考虑是否可以通过其他方式实现
   - 在条件分支中使用时要特别小心

[来源: reference/source_部分状态更新_01.md - Overwrite 模式源码分析]
[来源: reference/context7_langgraph_01.md - LangGraph 状态更新机制]
[来源: reference/search_部分状态更新_01.md - 社区最佳实践]
""")
```

---

## 运行输出示例

```
============================================================
场景1：基础 Overwrite 使用
============================================================

执行流程：
  [increment_node] 当前 counter: 0
  [increment_node] 当前 counter: 1
  [reset_node] 当前 counter: 2
  [reset_node] 使用 Overwrite 重置为 0
  [increment_node] 当前 counter: 0

最终结果：
  counter: 1
  items: ['item']

============================================================
场景2：覆盖 add_messages Reducer
============================================================

执行流程：
  [add_user_message] 当前消息数: 0
  [add_ai_response] 当前消息数: 1
  [clear_history] 当前消息数: 2
  [clear_history] 使用 Overwrite 清空历史
  [add_new_conversation] 当前消息数: 0

最终结果：
  消息数: 1
  [0] HumanMessage: New conversation

============================================================
场景3：使用字典语法的 Overwrite
============================================================

执行流程：
  [add_scores] 当前分数: []
  [reset_with_dict] 当前分数: [10, 20]
  [reset_with_dict] 使用 {OVERWRITE: value} 语法重置
  [add_scores] 当前分数: [100]

最终结果：
  scores: [100, 10, 20]

============================================================
场景4：条件性状态重置
============================================================

执行流程：
  [process_with_error] 当前错误数: 0
  [process_with_error] 当前错误数: 1
  [process_with_error] 当前错误数: 2
  [conditional_reset] 错误达到阈值，重置状态

最终结果：
  error_count: 0
  errors: []
  should_reset: False

============================================================
场景5：实际应用 - 对话历史管理系统
============================================================

执行流程：
  [add_message] 添加消息: Message 1
  [add_message] 添加消息: Message 2
  [add_message] 添加消息: Message 3
  [add_message] 添加消息: Message 4
  [check_and_trim] 消息数 (4) 超过限制 (3)
  [check_and_trim] 保留最近 3 条消息
  [start_new_conversation] 清空历史，开始新对话

最终结果：
  conversation_id: new_conversation_id
  message_count: 1
  消息列表:
    [0] New conversation started

============================================================
场景6：Overwrite vs 正常更新对比
============================================================

执行流程：
  [update_both_normal] 正常更新
  [update_with_overwrite] normal_list 正常追加，overwrite_list 使用 Overwrite
  [final_update] 两个列表都正常追加

最终结果对比：
  normal_list (正常追加): ['A', 'B', 'C']
  overwrite_list (中间覆盖): ['B', 'C']
```

---

## 技术要点总结

### 1. Overwrite 的两种语法

**类语法：**
```python
from langgraph.channels.binop import Overwrite

return {"field": Overwrite(new_value)}
```

**字典语法：**
```python
from langgraph.constants import OVERWRITE

return {"field": {OVERWRITE: new_value}}
```

### 2. Overwrite 的优先级

- Overwrite 优先级**高于** Reducer
- 可以临时覆盖 Reducer 的合并策略
- 不影响后续节点的 Reducer 行为

### 3. 使用场景

**适合使用 Overwrite：**
- 清空累积的状态（如消息历史）
- 重置错误计数器
- 切换会话/对话
- 条件性状态重置

**不适合使用 Overwrite：**
- 常规状态更新（应该使用 Reducer）
- 可以通过其他方式实现的场景
- 需要保持状态一致性的场景

### 4. 最佳实践

1. **明确注释**：使用 Overwrite 时要明确注释原因
2. **谨慎使用**：避免破坏状态一致性
3. **条件判断**：在条件分支中使用时要特别小心
4. **测试覆盖**：确保 Overwrite 逻辑有充分的测试

---

## 参考资料

- [来源: reference/source_部分状态更新_01.md] - LangGraph 源码分析，Overwrite 模式实现
- [来源: reference/context7_langgraph_01.md] - LangGraph 官方文档，状态更新机制
- [来源: reference/search_部分状态更新_01.md] - 社区最佳实践，Overwrite 使用场景

---

**文档版本：** v1.0
**创建时间：** 2026-02-26
**适用版本：** LangGraph 0.2.0+
**Python 版本：** 3.13+
