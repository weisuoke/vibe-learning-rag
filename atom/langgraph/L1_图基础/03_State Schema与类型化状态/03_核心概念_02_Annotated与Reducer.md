# 核心概念02：Annotated与Reducer

> LangGraph 状态聚合的核心机制

## 概念定义

**Annotated 是 Python 的类型注解工具，用于为类型附加元数据；Reducer 是状态聚合函数，定义如何合并多个状态更新。**

## 为什么需要 Reducer

### 问题场景

```python
# 场景：两个节点都返回 messages
def node1(state):
    return {"messages": ["hello"]}

def node2(state):
    return {"messages": ["world"]}

# 问题：如何合并这两个更新？
# 选项1：覆盖 - 只保留 ["world"]
# 选项2：追加 - 合并为 ["hello", "world"]
# 选项3：自定义 - 去重、排序等
```

**Reducer 解决了状态聚合的问题。**

## Annotated 基础

### 1. 什么是 Annotated

```python
from typing import Annotated

# Annotated[类型, 元数据1, 元数据2, ...]
x: Annotated[int, "这是一个整数", "范围: 0-100"]
```

**本质**：Annotated 是类型注解的容器，可以附加任意元数据。

### 2. 在 LangGraph 中的使用

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    # 类型 = list
    # 元数据 = operator.add (reducer 函数)
    messages: Annotated[list, operator.add]
```

**解释**：
- `list` 是字段类型
- `operator.add` 是 reducer 函数
- StateGraph 读取 Annotated 的元数据，自动应用 reducer

### 3. 提取元数据

```python
from typing import get_type_hints, get_args

class State(TypedDict):
    messages: Annotated[list, operator.add]

# 获取类型提示
hints = get_type_hints(State, include_extras=True)
print(hints["messages"])
# Annotated[list, <built-in function add>]

# 提取元数据
args = get_args(hints["messages"])
print(args)
# (list, <built-in function add>)

# 类型
print(args[0])  # list
# Reducer
print(args[1])  # <built-in function add>
```

## Reducer 函数

### 1. Reducer 的签名

```python
def reducer(existing_value, new_value):
    """
    合并两个值

    Args:
        existing_value: 当前状态中的值
        new_value: 节点返回的新值

    Returns:
        合并后的值
    """
    return merged_value
```

**关键点**：
- 接收两个参数
- 返回合并后的值
- 必须是纯函数（无副作用）

### 2. 内置 Reducer

#### operator.add - 追加/相加

```python
import operator

class State(TypedDict):
    # 列表追加
    messages: Annotated[list, operator.add]
    # 字符串拼接
    text: Annotated[str, operator.add]
    # 数值相加
    count: Annotated[int, operator.add]

# 示例
# messages: [] + ["a"] = ["a"]
# messages: ["a"] + ["b"] = ["a", "b"]
# text: "hello" + " world" = "hello world"
# count: 1 + 2 = 3
```

#### operator.or_ - 字典合并

```python
class State(TypedDict):
    metadata: Annotated[dict, operator.or_]

# 示例
# metadata: {"a": 1} | {"b": 2} = {"a": 1, "b": 2}
# metadata: {"a": 1} | {"a": 2} = {"a": 2}  # 后者覆盖
```

#### operator.and_ - 集合交集

```python
class State(TypedDict):
    tags: Annotated[set, operator.and_]

# 示例
# tags: {1, 2, 3} & {2, 3, 4} = {2, 3}
```

#### operator.sub - 集合差集

```python
class State(TypedDict):
    excluded: Annotated[set, operator.sub]

# 示例
# excluded: {1, 2, 3} - {2} = {1, 3}
```

### 3. 自定义 Reducer

#### 去重追加

```python
def unique_append(existing: list, new: list) -> list:
    """追加新元素，去重"""
    seen = set(existing)
    return existing + [x for x in new if x not in seen]

class State(TypedDict):
    items: Annotated[list, unique_append]

# 示例
# items: [1, 2] + [2, 3] = [1, 2, 3]
```

#### 保留最新 N 个

```python
def keep_last_n(n: int):
    """工厂函数：创建保留最新 N 个元素的 reducer"""
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        return combined[-n:]
    return reducer

class State(TypedDict):
    recent_messages: Annotated[list, keep_last_n(10)]

# 示例
# recent_messages: [1,2,3,...,10] + [11] = [2,3,...,10,11]
```

#### 按时间戳合并

```python
def merge_by_timestamp(existing: list, new: list) -> list:
    """按时间戳排序合并"""
    combined = existing + new
    return sorted(combined, key=lambda x: x["timestamp"])

class State(TypedDict):
    events: Annotated[list, merge_by_timestamp]

# 示例
# events: [{"ts": 2, "msg": "b"}] + [{"ts": 1, "msg": "a"}]
# = [{"ts": 1, "msg": "a"}, {"ts": 2, "msg": "b"}]
```

#### 字典深度合并

```python
def deep_merge(existing: dict, new: dict) -> dict:
    """递归合并字典"""
    result = existing.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class State(TypedDict):
    config: Annotated[dict, deep_merge]

# 示例
# config: {"a": {"b": 1}} + {"a": {"c": 2}}
# = {"a": {"b": 1, "c": 2}}
```

## Reducer 执行时机

### 1. 节点返回后

```python
def node(state: State) -> dict:
    return {"messages": ["new"]}

# StateGraph 自动调用 reducer
# state["messages"] = reducer(state["messages"], ["new"])
```

### 2. 并行执行后

```python
# 并行执行两个节点
def node1(state: State) -> dict:
    return {"messages": ["a"]}

def node2(state: State) -> dict:
    return {"messages": ["b"]}

# StateGraph 聚合
# 1. 收集所有更新: [["a"], ["b"]]
# 2. 依次应用 reducer:
#    state["messages"] = reducer([], ["a"])  # ["a"]
#    state["messages"] = reducer(["a"], ["b"])  # ["a", "b"]
```

### 3. 手动更新时

```python
# 使用 update_state 手动更新
graph.update_state(config, {"messages": ["manual"]})

# StateGraph 调用 reducer
# state["messages"] = reducer(state["messages"], ["manual"])
```

## 无 Reducer 的字段

### 默认行为：覆盖

```python
class State(TypedDict):
    # 无 Annotated，默认覆盖
    user_id: str
    count: int

def node1(state: State) -> dict:
    return {"user_id": "user1", "count": 1}

def node2(state: State) -> dict:
    return {"user_id": "user2", "count": 2}

# 并行执行后
# user_id = "user2"  # 后者覆盖（不确定顺序）
# count = 2
```

**注意**：无 reducer 的字段在并行执行时，最终值是不确定的（取决于执行顺序）。

### 显式使用 LastValue

```python
from langgraph.channels import LastValue

class State(TypedDict):
    # 显式指定使用最后一个值
    user_id: Annotated[str, LastValue()]
```

## 在 LangGraph 中的应用

### 1. 聊天机器人

```python
from typing import TypedDict, Annotated
import operator

class ChatState(TypedDict):
    # 对话历史 - 追加
    messages: Annotated[list, operator.add]

    # 用户信息 - 覆盖
    user_id: str
    user_name: str

    # 上下文 - 覆盖
    context: str

def user_input_node(state: ChatState) -> dict:
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user123"
    }

def llm_node(state: ChatState) -> dict:
    # 读取历史
    messages = state["messages"]
    # 生成回复
    response = llm.invoke(messages)
    return {
        "messages": [{"role": "assistant", "content": response}]
    }

# 执行后
# messages = [
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hi!"}
# ]
```

### 2. 工作流状态

```python
class WorkflowState(TypedDict):
    # 任务列表 - 追加
    tasks: Annotated[list, operator.add]

    # 错误列表 - 追加
    errors: Annotated[list, operator.add]

    # 当前状态 - 覆盖
    status: str

    # 重试次数 - 累加
    retry_count: Annotated[int, operator.add]

def task_node(state: WorkflowState) -> dict:
    try:
        result = execute_task()
        return {
            "tasks": [{"name": "task1", "result": result}],
            "status": "running"
        }
    except Exception as e:
        return {
            "errors": [{"task": "task1", "error": str(e)}],
            "status": "failed",
            "retry_count": 1
        }
```

### 3. 数据处理管道

```python
class PipelineState(TypedDict):
    # 处理步骤 - 追加
    steps: Annotated[list, operator.add]

    # 统计信息 - 字典合并
    stats: Annotated[dict, operator.or_]

    # 处理后的数据 - 追加
    processed_data: Annotated[list, operator.add]

def transform_node(state: PipelineState) -> dict:
    data = transform(state["processed_data"])
    return {
        "steps": ["transform"],
        "stats": {"transformed": len(data)},
        "processed_data": data
    }

def filter_node(state: PipelineState) -> dict:
    data = filter_data(state["processed_data"])
    return {
        "steps": ["filter"],
        "stats": {"filtered": len(data)},
        "processed_data": data
    }
```

## 高级技巧

### 1. 条件 Reducer

```python
def conditional_append(condition_key: str):
    """根据条件决定是否追加"""
    def reducer(existing: list, new: list) -> list:
        # 假设 new 是 [{"condition": bool, "data": ...}]
        return existing + [item["data"] for item in new if item.get(condition_key)]
    return reducer

class State(TypedDict):
    valid_items: Annotated[list, conditional_append("is_valid")]
```

### 2. 带验证的 Reducer

```python
def validated_append(validator):
    """追加前验证"""
    def reducer(existing: list, new: list) -> list:
        validated = [x for x in new if validator(x)]
        return existing + validated
    return reducer

def is_positive(x):
    return x > 0

class State(TypedDict):
    positive_numbers: Annotated[list, validated_append(is_positive)]
```

### 3. 带限制的 Reducer

```python
def limited_append(max_size: int):
    """限制列表大小"""
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        if len(combined) > max_size:
            return combined[-max_size:]
        return combined
    return reducer

class State(TypedDict):
    recent_logs: Annotated[list, limited_append(100)]
```

## 常见错误

### 1. Reducer 签名错误

```python
# ❌ 错误：只有一个参数
def bad_reducer(value):
    return value

# ✅ 正确：两个参数
def good_reducer(existing, new):
    return existing + new
```

### 2. Reducer 有副作用

```python
# ❌ 错误：修改了输入
def bad_reducer(existing: list, new: list) -> list:
    existing.extend(new)  # 修改了 existing
    return existing

# ✅ 正确：不修改输入
def good_reducer(existing: list, new: list) -> list:
    return existing + new  # 创建新列表
```

### 3. 类型不匹配

```python
class State(TypedDict):
    count: Annotated[int, operator.add]

def node(state: State) -> dict:
    # ❌ 返回字符串
    return {"count": "123"}

    # ✅ 返回整数
    return {"count": 123}
```

### 4. 忘记导入 Annotated

```python
# ❌ 错误
from typing import TypedDict

class State(TypedDict):
    messages: Annotated[list, operator.add]  # NameError

# ✅ 正确
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
```

## 调试技巧

### 1. 打印 Reducer 调用

```python
def debug_reducer(name: str):
    """包装 reducer，打印调用信息"""
    def wrapper(reducer):
        def inner(existing, new):
            print(f"[{name}] existing: {existing}, new: {new}")
            result = reducer(existing, new)
            print(f"[{name}] result: {result}")
            return result
        return inner
    return wrapper

class State(TypedDict):
    messages: Annotated[list, debug_reducer("messages")(operator.add)]
```

### 2. 验证 Reducer 行为

```python
def test_reducer():
    """测试 reducer"""
    reducer = operator.add

    # 测试列表
    assert reducer([1, 2], [3]) == [1, 2, 3]

    # 测试字符串
    assert reducer("hello", " world") == "hello world"

    # 测试数值
    assert reducer(1, 2) == 3

    print("All tests passed!")

test_reducer()
```

### 3. 检查 Annotated 元数据

```python
from typing import get_type_hints, get_args

def inspect_state(state_class):
    """检查状态定义"""
    hints = get_type_hints(state_class, include_extras=True)
    for name, hint in hints.items():
        if hasattr(hint, "__metadata__"):
            args = get_args(hint)
            print(f"{name}: type={args[0]}, reducer={args[1]}")
        else:
            print(f"{name}: type={hint}, reducer=None")

inspect_state(State)
```

## 最佳实践

### 1. 明确 Reducer 语义

```python
# ✅ 清晰的命名
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 追加消息
    user_id: str  # 覆盖用户ID
    total_count: Annotated[int, operator.add]  # 累加计数
```

### 2. 文档化自定义 Reducer

```python
def merge_unique(existing: list, new: list) -> list:
    """
    合并列表，去除重复元素

    Args:
        existing: 现有列表
        new: 新列表

    Returns:
        合并后的去重列表

    Example:
        >>> merge_unique([1, 2], [2, 3])
        [1, 2, 3]
    """
    seen = set(existing)
    return existing + [x for x in new if x not in seen]
```

### 3. 测试 Reducer

```python
import unittest

class TestReducers(unittest.TestCase):
    def test_merge_unique(self):
        result = merge_unique([1, 2], [2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_merge_unique_empty(self):
        result = merge_unique([], [1, 2])
        self.assertEqual(result, [1, 2])
```

### 4. 避免复杂 Reducer

```python
# ❌ 过于复杂
def complex_reducer(existing, new):
    # 100 行代码...
    pass

# ✅ 简单清晰
def simple_reducer(existing, new):
    return existing + new
```

## 总结

**Annotated 与 Reducer 是 LangGraph 状态管理的核心**：
- Annotated 附加元数据到类型
- Reducer 定义状态聚合规则
- 自动处理并行执行的状态合并
- 支持自定义聚合逻辑

**关键点**：
- Reducer 必须是纯函数
- 接收两个参数，返回合并值
- 内置 reducer：add, or_, and_, sub
- 自定义 reducer 满足特殊需求

## 参考资料

- 详细实战：`07_实战代码_场景2_Reducer实战应用.md`
- TypedDict 基础：`03_核心概念_01_TypedDict状态定义.md`
- Channel 系统：`03_核心概念_03_Channel类型系统.md`
