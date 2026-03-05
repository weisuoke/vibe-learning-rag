# 实战代码 场景1：基础TypedDict状态定义

> 完整可运行的TypedDict状态定义示例

## 场景描述

构建一个简单的聊天机器人，使用TypedDict定义状态，实现基础的对话管理。

## 完整代码

```python
"""
场景1：基础TypedDict状态定义
功能：简单聊天机器人
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# ============ 1. 状态定义 ============

class ChatState(TypedDict):
    """聊天机器人状态

    Attributes:
        messages: 对话历史（追加）
        user_id: 用户ID（覆盖）
        turn_count: 对话轮数（覆盖）
    """
    messages: Annotated[list[str], operator.add]
    user_id: str
    turn_count: int


# ============ 2. 节点函数 ============

def user_input_node(state: ChatState) -> dict:
    """处理用户输入"""
    print(f"\n[用户输入节点] 当前轮数: {state['turn_count']}")
    print(f"[用户输入节点] 历史消息数: {len(state['messages'])}")

    # 模拟用户输入
    user_message = f"用户消息 #{state['turn_count']}"

    return {
        "messages": [f"User: {user_message}"],
        "turn_count": state["turn_count"] + 1
    }


def bot_response_node(state: ChatState) -> dict:
    """生成机器人回复"""
    print(f"\n[机器人回复节点] 处理消息...")

    # 获取最后一条用户消息
    last_message = state["messages"][-1] if state["messages"] else ""

    # 简单的回复逻辑
    bot_message = f"收到: {last_message}"

    return {
        "messages": [f"Bot: {bot_message}"]
    }


# ============ 3. 构建图 ============

def create_chat_graph():
    """创建聊天图"""
    # 创建图
    graph = StateGraph(ChatState)

    # 添加节点
    graph.add_node("user_input", user_input_node)
    graph.add_node("bot_response", bot_response_node)

    # 添加边
    graph.add_edge("user_input", "bot_response")
    graph.add_edge("bot_response", END)

    # 设置入口
    graph.set_entry_point("user_input")

    # 编译
    return graph.compile()


# ============ 4. 运行示例 ============

def main():
    """主函数"""
    print("=" * 50)
    print("场景1：基础TypedDict状态定义")
    print("=" * 50)

    # 创建图
    app = create_chat_graph()

    # 初始状态
    initial_state = {
        "messages": [],
        "user_id": "user123",
        "turn_count": 1
    }

    print("\n初始状态:")
    print(f"  messages: {initial_state['messages']}")
    print(f"  user_id: {initial_state['user_id']}")
    print(f"  turn_count: {initial_state['turn_count']}")

    # 运行图
    result = app.invoke(initial_state)

    print("\n最终状态:")
    print(f"  messages: {result['messages']}")
    print(f"  user_id: {result['user_id']}")
    print(f"  turn_count: {result['turn_count']}")

    print("\n对话历史:")
    for msg in result["messages"]:
        print(f"  {msg}")


# ============ 5. 多轮对话示例 ============

def multi_turn_example():
    """多轮对话示例"""
    print("\n" + "=" * 50)
    print("多轮对话示例")
    print("=" * 50)

    app = create_chat_graph()

    # 第一轮
    print("\n--- 第1轮 ---")
    state1 = app.invoke({
        "messages": [],
        "user_id": "user123",
        "turn_count": 1
    })
    print(f"消息数: {len(state1['messages'])}")
    print(f"最后消息: {state1['messages'][-1]}")

    # 第二轮（继续之前的状态）
    print("\n--- 第2轮 ---")
    state2 = app.invoke({
        "messages": state1["messages"],  # 保留历史
        "user_id": state1["user_id"],
        "turn_count": state1["turn_count"]
    })
    print(f"消息数: {len(state2['messages'])}")
    print(f"最后消息: {state2['messages'][-1]}")

    # 第三轮
    print("\n--- 第3轮 ---")
    state3 = app.invoke({
        "messages": state2["messages"],
        "user_id": state2["user_id"],
        "turn_count": state2["turn_count"]
    })
    print(f"消息数: {len(state3['messages'])}")

    print("\n完整对话历史:")
    for i, msg in enumerate(state3["messages"], 1):
        print(f"  {i}. {msg}")


# ============ 6. 状态验证示例 ============

def validation_example():
    """状态验证示例"""
    print("\n" + "=" * 50)
    print("状态验证示例")
    print("=" * 50)

    from typing import get_type_hints

    # 获取类型提示
    hints = get_type_hints(ChatState, include_extras=True)

    print("\n状态字段类型:")
    for name, hint in hints.items():
        print(f"  {name}: {hint}")

    # 验证状态
    def validate_state(state: dict) -> bool:
        """验证状态是否符合ChatState定义"""
        required_fields = ["messages", "user_id", "turn_count"]

        for field in required_fields:
            if field not in state:
                print(f"  ❌ 缺少字段: {field}")
                return False

        if not isinstance(state["messages"], list):
            print(f"  ❌ messages 类型错误")
            return False

        if not isinstance(state["user_id"], str):
            print(f"  ❌ user_id 类型错误")
            return False

        if not isinstance(state["turn_count"], int):
            print(f"  ❌ turn_count 类型错误")
            return False

        print("  ✓ 状态验证通过")
        return True

    # 测试验证
    print("\n测试1: 正确的状态")
    validate_state({
        "messages": [],
        "user_id": "user123",
        "turn_count": 1
    })

    print("\n测试2: 缺少字段")
    validate_state({
        "messages": [],
        "user_id": "user123"
    })

    print("\n测试3: 类型错误")
    validate_state({
        "messages": "not a list",
        "user_id": "user123",
        "turn_count": 1
    })


# ============ 7. 运行所有示例 ============

if __name__ == "__main__":
    # 基础示例
    main()

    # 多轮对话
    multi_turn_example()

    # 状态验证
    validation_example()

    print("\n" + "=" * 50)
    print("所有示例运行完成")
    print("=" * 50)
```

## 运行结果

```
==================================================
场景1：基础TypedDict状态定义
==================================================

初始状态:
  messages: []
  user_id: user123
  turn_count: 1

[用户输入节点] 当前轮数: 1
[用户输入节点] 历史消息数: 0

[机器人回复节点] 处理消息...

最终状态:
  messages: ['User: 用户消息 #1', 'Bot: 收到: User: 用户消息 #1']
  user_id: user123
  turn_count: 2

对话历史:
  User: 用户消息 #1
  Bot: 收到: User: 用户消息 #1

==================================================
多轮对话示例
==================================================

--- 第1轮 ---
[用户输入节点] 当前轮数: 1
[用户输入节点] 历史消息数: 0
[机器人回复节点] 处理消息...
消息数: 2
最后消息: Bot: 收到: User: 用户消息 #1

--- 第2轮 ---
[用户输入节点] 当前轮数: 2
[用户输入节点] 历史消息数: 2
[机器人回复节点] 处理消息...
消息数: 4
最后消息: Bot: 收到: User: 用户消息 #2

--- 第3轮 ---
[用户输入节点] 当前轮数: 3
[用户输入节点] 历史消息数: 4
[机器人回复节点] 处理消息...
消息数: 6

完整对话历史:
  1. User: 用户消息 #1
  2. Bot: 收到: User: 用户消息 #1
  3. User: 用户消息 #2
  4. Bot: 收到: User: 用户消息 #2
  5. User: 用户消息 #3
  6. Bot: 收到: User: 用户消息 #3
```

## 关键知识点

### 1. TypedDict定义

```python
class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]  # 追加
    user_id: str                                   # 覆盖
    turn_count: int                                # 覆盖
```

**要点**：
- `Annotated[list, operator.add]`：列表追加
- 无Annotated的字段：覆盖更新
- 类型注解：提供静态类型检查

### 2. 节点返回部分更新

```python
def node(state: ChatState) -> dict:
    return {
        "messages": ["new message"],  # 只更新messages
        "turn_count": state["turn_count"] + 1  # 只更新turn_count
    }
```

**要点**：
- 节点只需返回要更新的字段
- 不需要返回完整状态
- StateGraph自动合并更新

### 3. 状态聚合

```python
# 初始状态
messages: []

# 节点1返回
{"messages": ["msg1"]}
# 聚合后: ["msg1"]

# 节点2返回
{"messages": ["msg2"]}
# 聚合后: ["msg1", "msg2"]
```

**要点**：
- `operator.add`自动追加
- 无需手动合并
- 保持消息顺序

## 扩展练习

### 练习1：添加时间戳

```python
import time

class ChatState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    user_id: str
    turn_count: int

def user_input_node(state: ChatState) -> dict:
    return {
        "messages": [{
            "role": "user",
            "content": "Hello",
            "timestamp": time.time()
        }]
    }
```

### 练习2：添加元数据

```python
class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]
    user_id: str
    turn_count: int
    metadata: dict  # 新增元数据字段

def node(state: ChatState) -> dict:
    return {
        "messages": ["msg"],
        "metadata": {
            "last_update": time.time(),
            "node": "user_input"
        }
    }
```

### 练习3：添加错误处理

```python
class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]
    user_id: str
    turn_count: int
    errors: Annotated[list[str], operator.add]

def node(state: ChatState) -> dict:
    try:
        result = process()
        return {"messages": [result]}
    except Exception as e:
        return {"errors": [str(e)]}
```

## 总结

**本场景展示了**：
1. 基础TypedDict状态定义
2. Annotated与operator.add的使用
3. 节点返回部分更新
4. 多轮对话状态管理
5. 状态验证方法

**关键要点**：
- TypedDict提供类型检查
- Annotated定义聚合规则
- 节点只返回要更新的字段
- StateGraph自动处理状态合并

## 参考资料

- TypedDict基础：`03_核心概念_01_TypedDict状态定义.md`
- Reducer详解：`03_核心概念_02_Annotated与Reducer.md`
- 最小可用：`04_最小可用.md`
