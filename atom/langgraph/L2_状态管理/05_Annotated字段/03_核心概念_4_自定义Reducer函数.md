# 核心概念 4：自定义 Reducer 函数

## 概述

虽然 LangGraph 提供了 `operator.add`、`operator.or_` 和 `add_messages` 等内置 Reducer，但在实际应用中，我们经常需要编写自定义 Reducer 来实现特定的合并逻辑。本文将深入讲解如何编写各种类型的自定义 Reducer 函数，包括 Lambda 函数、普通函数、工厂函数以及常见的实用模式。

[来源: reference/search_annotated_github_01.md | 技术文章]

## 第一性原理

### 为什么需要自定义 Reducer？

内置 Reducer 虽然覆盖了常见场景，但无法满足所有需求。例如：
- **去重合并**：追加新项时自动去重
- **限制大小**：保持列表在固定大小内
- **条件覆盖**：只在满足特定条件时更新
- **复杂合并**：基于业务逻辑的自定义合并

自定义 Reducer 让我们能够：
1. **精确控制**：完全控制状态合并逻辑
2. **业务适配**：适配特定的业务需求
3. **性能优化**：针对特定场景优化性能
4. **代码复用**：将合并逻辑封装为可复用的函数

[来源: reference/search_annotated_github_01.md | 技术文章]

## Lambda 函数作为 Reducer

### 基本用法

Lambda 函数是最简洁的自定义 Reducer 形式，适合简单的合并逻辑：

```python
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    # 优先使用新值（非空时）
    value: Annotated[str, lambda old, new: new or old]

    # 条件覆盖（新值非 None 时）
    data: Annotated[dict | None, lambda old, new: new if new is not None else old]

    # 总是覆盖
    result: Annotated[str, lambda old, new: new]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 常见 Lambda 模式

#### 模式 1：优先使用新值

```python
class State(TypedDict):
    # 新值非空则使用新值，否则保留旧值
    primary_issue: Annotated[str, lambda x, y: y or x]
```

**使用场景**：
- 可选字段的更新
- 默认值保留

**示例**：
```python
old = "default"
new = ""
result = (lambda x, y: y or x)(old, new)  # "default"

old = "default"
new = "updated"
result = (lambda x, y: y or x)(old, new)  # "updated"
```

[来源: reference/source_annotated_01.md | 源码分析]

#### 模式 2：条件覆盖

```python
class State(TypedDict):
    # 新值非 None 则覆盖
    user_info: Annotated[dict | None, lambda x, y: y if y is not None else x]

    # 新值为真则覆盖
    flag: Annotated[bool, lambda x, y: y if y else x]
```

**使用场景**：
- 可选参数更新
- 避免 None 值覆盖

[来源: reference/search_annotated_reddit_01.md | Reddit 社区讨论]

#### 模式 3：总是覆盖

```python
class State(TypedDict):
    # 总是使用新值（忽略旧值）
    autoresponse: Annotated[dict | None, lambda _, y: y]
```

**使用场景**：
- 强制更新字段
- 忽略历史值

[来源: reference/source_annotated_01.md | 源码分析]

### Lambda 函数的限制

```python
# ❌ 不好：逻辑复杂，难以理解
class State(TypedDict):
    data: Annotated[list, lambda x, y: x + [i for i in y if i not in x]]

# ✅ 好：使用普通函数
def merge_unique(old: list, new: list) -> list:
    """去重合并"""
    seen = set(old)
    return old + [i for i in new if i not in seen]

class State(TypedDict):
    data: Annotated[list, merge_unique]
```

**建议**：
- Lambda 适合单行简单逻辑
- 复杂逻辑使用普通函数
- 需要文档注释时使用普通函数

[来源: reference/search_annotated_github_01.md | 技术文章]

## 普通函数作为 Reducer

### 基本结构

```python
def custom_reducer(old_value: T, new_value: T) -> T:
    """
    自定义 Reducer 函数

    Args:
        old_value: 当前状态中的值
        new_value: 节点返回的新值

    Returns:
        合并后的值
    """
    # 实现合并逻辑
    return merged_value
```

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

### 实用模式 1：去重合并

```python
def merge_with_dedup(old: list, new: list) -> list:
    """
    去重合并列表

    Args:
        old: 旧列表
        new: 新列表

    Returns:
        合并后的列表（去重）
    """
    seen = set(old)
    return old + [item for item in new if item not in seen]

class State(TypedDict):
    items: Annotated[list[str], merge_with_dedup]
```

**完整示例**：
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

def merge_with_dedup(old: list, new: list) -> list:
    """去重合并"""
    seen = set(old)
    return old + [item for item in new if item not in seen]

class State(TypedDict):
    items: Annotated[list[str], merge_with_dedup]

def node1(state: State) -> dict:
    return {"items": ["a", "b", "c"]}

def node2(state: State) -> dict:
    return {"items": ["b", "c", "d"]}  # b, c 重复

# 构建图
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()

# 执行
result = graph.invoke({"items": []})
print(result["items"])  # ["a", "b", "c", "d"] - 自动去重
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 实用模式 2：限制大小

```python
def merge_with_limit(old: list, new: list, max_size: int = 100) -> list:
    """
    限制大小的合并

    Args:
        old: 旧列表
        new: 新列表
        max_size: 最大大小

    Returns:
        合并后的列表（保留最近的 max_size 项）
    """
    combined = old + new
    return combined[-max_size:]

# 使用闭包固定 max_size
def create_limited_merger(max_size: int):
    def merger(old: list, new: list) -> list:
        combined = old + new
        return combined[-max_size:]
    return merger

class State(TypedDict):
    # 保留最近 10 条消息
    recent_messages: Annotated[list, create_limited_merger(10)]
```

**完整示例**：
```python
def create_limited_merger(max_size: int):
    """创建限制大小的 Reducer"""
    def merger(old: list, new: list) -> list:
        combined = old + new
        return combined[-max_size:]
    return merger

class State(TypedDict):
    logs: Annotated[list[str], create_limited_merger(5)]

def node(state: State) -> dict:
    return {"logs": [f"Log {len(state['logs']) + 1}"]}

# 构建图
builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_edge("node", "node")  # 循环
builder.add_edge("node", END)

graph = builder.compile()

# 执行多次
result = graph.invoke({"logs": []})
for _ in range(10):
    result = graph.invoke(result)

print(len(result["logs"]))  # 5 - 始终保持最大 5 条
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 实用模式 3：按键合并字典

```python
def merge_dict_by_key(old: dict, new: dict) -> dict:
    """
    按键合并字典（深度合并）

    Args:
        old: 旧字典
        new: 新字典

    Returns:
        合并后的字典
    """
    result = old.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_dict_by_key(result[key], value)
        else:
            result[key] = value
    return result

class State(TypedDict):
    config: Annotated[dict, merge_dict_by_key]
```

**示例**：
```python
old = {"a": {"x": 1}, "b": 2}
new = {"a": {"y": 2}, "c": 3}
result = merge_dict_by_key(old, new)
# {"a": {"x": 1, "y": 2}, "b": 2, "c": 3}
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 实用模式 4：条件合并

```python
def merge_if_valid(old: list, new: list) -> list:
    """
    只合并有效的新项

    Args:
        old: 旧列表
        new: 新列表

    Returns:
        合并后的列表
    """
    # 过滤掉无效项
    valid_new = [item for item in new if item and len(item) > 0]
    return old + valid_new

class State(TypedDict):
    items: Annotated[list[str], merge_if_valid]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## Reducer 工厂函数

### 什么是 Reducer 工厂？

Reducer 工厂函数是返回 Reducer 函数的高阶函数，用于创建可配置的 Reducer：

```python
def create_reducer(config) -> Callable:
    """Reducer 工厂函数"""
    def reducer(old, new):
        # 使用 config 参数
        return merge(old, new, config)
    return reducer
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 工厂模式 1：可配置的限制大小

```python
def create_size_limited_reducer(max_size: int, keep: str = "recent"):
    """
    创建限制大小的 Reducer

    Args:
        max_size: 最大大小
        keep: "recent" 保留最近的，"oldest" 保留最早的

    Returns:
        Reducer 函数
    """
    def reducer(old: list, new: list) -> list:
        combined = old + new
        if len(combined) <= max_size:
            return combined

        if keep == "recent":
            return combined[-max_size:]
        elif keep == "oldest":
            return combined[:max_size]
        else:
            raise ValueError(f"Invalid keep value: {keep}")

    return reducer

class State(TypedDict):
    recent_logs: Annotated[list, create_size_limited_reducer(10, "recent")]
    oldest_logs: Annotated[list, create_size_limited_reducer(10, "oldest")]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 工厂模式 2：可配置的去重策略

```python
def create_dedup_reducer(key_func=None):
    """
    创建去重 Reducer

    Args:
        key_func: 用于提取唯一键的函数（默认使用项本身）

    Returns:
        Reducer 函数
    """
    def reducer(old: list, new: list) -> list:
        if key_func is None:
            # 简单去重
            seen = set(old)
            return old + [item for item in new if item not in seen]
        else:
            # 基于键去重
            seen_keys = {key_func(item) for item in old}
            result = old.copy()
            for item in new:
                key = key_func(item)
                if key not in seen_keys:
                    result.append(item)
                    seen_keys.add(key)
            return result

    return reducer

# 使用示例
class State(TypedDict):
    # 简单去重
    items: Annotated[list[str], create_dedup_reducer()]

    # 基于 ID 去重
    users: Annotated[list[dict], create_dedup_reducer(lambda x: x["id"])]
```

**完整示例**：
```python
def create_dedup_reducer(key_func=None):
    """创建去重 Reducer"""
    def reducer(old: list, new: list) -> list:
        if key_func is None:
            seen = set(old)
            return old + [item for item in new if item not in seen]
        else:
            seen_keys = {key_func(item) for item in old}
            result = old.copy()
            for item in new:
                key = key_func(item)
                if key not in seen_keys:
                    result.append(item)
                    seen_keys.add(key)
            return result
    return reducer

class State(TypedDict):
    users: Annotated[list[dict], create_dedup_reducer(lambda x: x["id"])]

def node1(state: State) -> dict:
    return {"users": [{"id": 1, "name": "Alice"}]}

def node2(state: State) -> dict:
    return {"users": [{"id": 1, "name": "Alice Updated"}, {"id": 2, "name": "Bob"}]}

# 构建图
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()

# 执行
result = graph.invoke({"users": []})
print(result["users"])
# [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
# 注意：ID 1 的用户没有被更新，因为已存在
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 工厂模式 3：可配置的合并策略

```python
def create_merge_reducer(strategy: str = "append"):
    """
    创建可配置的合并 Reducer

    Args:
        strategy: "append", "prepend", "replace", "merge"

    Returns:
        Reducer 函数
    """
    def reducer(old: list, new: list) -> list:
        if strategy == "append":
            return old + new
        elif strategy == "prepend":
            return new + old
        elif strategy == "replace":
            return new
        elif strategy == "merge":
            # 去重合并
            seen = set(old)
            return old + [item for item in new if item not in seen]
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    return reducer

class State(TypedDict):
    append_list: Annotated[list, create_merge_reducer("append")]
    prepend_list: Annotated[list, create_merge_reducer("prepend")]
    replace_list: Annotated[list, create_merge_reducer("replace")]
    merge_list: Annotated[list, create_merge_reducer("merge")]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 复杂 Reducer 示例

### 示例 1：带优先级的合并

```python
def merge_with_priority(old: list[dict], new: list[dict]) -> list[dict]:
    """
    按优先级合并项

    Args:
        old: 旧列表（每项包含 priority 字段）
        new: 新列表

    Returns:
        按优先级排序的合并列表
    """
    # 合并
    combined = old + new

    # 按 ID 去重（保留优先级高的）
    by_id = {}
    for item in combined:
        item_id = item.get("id")
        if item_id not in by_id or item.get("priority", 0) > by_id[item_id].get("priority", 0):
            by_id[item_id] = item

    # 按优先级排序
    result = sorted(by_id.values(), key=lambda x: x.get("priority", 0), reverse=True)
    return result

class State(TypedDict):
    tasks: Annotated[list[dict], merge_with_priority]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 示例 2：时间窗口合并

```python
from datetime import datetime, timedelta

def create_time_window_reducer(window_hours: int = 24):
    """
    创建时间窗口 Reducer（只保留最近 N 小时的数据）

    Args:
        window_hours: 时间窗口（小时）

    Returns:
        Reducer 函数
    """
    def reducer(old: list[dict], new: list[dict]) -> list[dict]:
        now = datetime.now()
        cutoff = now - timedelta(hours=window_hours)

        # 合并
        combined = old + new

        # 过滤旧数据
        result = [
            item for item in combined
            if datetime.fromisoformat(item["timestamp"]) > cutoff
        ]

        return result

    return reducer

class State(TypedDict):
    events: Annotated[list[dict], create_time_window_reducer(24)]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 示例 3：加权平均合并

```python
def merge_with_weighted_average(old: dict, new: dict, weight: float = 0.5) -> dict:
    """
    加权平均合并数值字典

    Args:
        old: 旧字典
        new: 新字典
        weight: 新值的权重（0-1）

    Returns:
        加权平均后的字典
    """
    result = {}
    all_keys = set(old.keys()) | set(new.keys())

    for key in all_keys:
        old_val = old.get(key, 0)
        new_val = new.get(key, 0)
        result[key] = old_val * (1 - weight) + new_val * weight

    return result

def create_weighted_average_reducer(weight: float = 0.5):
    """创建加权平均 Reducer"""
    def reducer(old: dict, new: dict) -> dict:
        return merge_with_weighted_average(old, new, weight)
    return reducer

class State(TypedDict):
    metrics: Annotated[dict, create_weighted_average_reducer(0.3)]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 最佳实践

### 1. 选择合适的形式

```python
# ✅ 好：简单逻辑用 Lambda
class State(TypedDict):
    value: Annotated[str, lambda x, y: y or x]

# ✅ 好：复杂逻辑用普通函数
def merge_complex(old: list, new: list) -> list:
    """复杂的合并逻辑"""
    # 多行逻辑
    return result

class State(TypedDict):
    data: Annotated[list, merge_complex]

# ✅ 好：可配置逻辑用工厂函数
class State(TypedDict):
    items: Annotated[list, create_dedup_reducer(lambda x: x["id"])]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 2. 添加类型注解

```python
# ✅ 好：明确类型
def merge_items(old: list[str], new: list[str]) -> list[str]:
    """合并字符串列表"""
    return old + new

# ❌ 不好：类型不明确
def merge_items(old, new):
    return old + new
```

### 3. 编写文档注释

```python
def merge_with_dedup(old: list, new: list) -> list:
    """
    去重合并列表

    Args:
        old: 当前状态中的列表
        new: 节点返回的新列表

    Returns:
        合并后的列表（自动去重）

    Example:
        >>> merge_with_dedup([1, 2], [2, 3])
        [1, 2, 3]
    """
    seen = set(old)
    return old + [item for item in new if item not in seen]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

### 4. 避免副作用

```python
# ❌ 不好：修改输入参数
def bad_reducer(old: list, new: list) -> list:
    old.extend(new)  # 修改了 old
    return old

# ✅ 好：创建新对象
def good_reducer(old: list, new: list) -> list:
    return old + new  # 创建新列表
```

### 5. 处理边界情况

```python
def safe_merge(old: list | None, new: list | None) -> list:
    """安全的合并（处理 None）"""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 常见陷阱

### 陷阱 1：修改可变对象

```python
# ❌ 危险：修改了原始列表
def bad_reducer(old: list, new: list) -> list:
    old.extend(new)
    return old

# ✅ 安全：创建新列表
def good_reducer(old: list, new: list) -> list:
    return old + new
```

### 陷阱 2：忘记处理 None

```python
# ❌ 错误：可能抛出 TypeError
def bad_reducer(old: list, new: list) -> list:
    return old + new  # 如果 old 或 new 是 None 会报错

# ✅ 正确：处理 None
def good_reducer(old: list | None, new: list | None) -> list:
    return (old or []) + (new or [])
```

### 陷阱 3：性能问题

```python
# ❌ 慢：每次都遍历整个列表
def slow_dedup(old: list, new: list) -> list:
    result = old.copy()
    for item in new:
        if item not in result:  # O(n) 查找
            result.append(item)
    return result

# ✅ 快：使用集合
def fast_dedup(old: list, new: list) -> list:
    seen = set(old)  # O(1) 查找
    return old + [item for item in new if item not in seen]
```

[来源: reference/search_annotated_github_01.md | 技术文章]

## 总结

### 核心要点

1. **Lambda 函数**：适合单行简单逻辑
2. **普通函数**：适合复杂逻辑和需要文档的场景
3. **工厂函数**：适合可配置的 Reducer
4. **类型注解**：提高代码可读性和类型安全
5. **避免副作用**：不要修改输入参数

### 选择指南

- 简单条件判断？使用 Lambda
- 复杂合并逻辑？使用普通函数
- 需要配置参数？使用工厂函数
- 需要复用？封装为独立函数

### 实用模式

- 去重合并：`merge_with_dedup`
- 限制大小：`create_size_limited_reducer`
- 条件覆盖：`lambda x, y: y if condition else x`
- 深度合并：`merge_dict_by_key`

[来源: reference/search_annotated_github_01.md | 技术文章]

---

**参考资料**：
- [LangGraph 源码分析](reference/source_annotated_01.md)
- [Annotated 字段解析机制](reference/source_annotated_02.md)
- [LangGraph 官方文档](reference/context7_langgraph_01.md)
- [Reddit 社区讨论](reference/search_annotated_reddit_01.md)
- [技术文章与教程](reference/search_annotated_github_01.md)
