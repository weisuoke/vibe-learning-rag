# 实战代码 - 场景3：自定义 Reducer 实战

## 场景概述

本场景演示如何编写自定义 Reducer 函数，实现更复杂的状态管理逻辑。

**核心目标**：
- 掌握自定义 Reducer 的编写方法
- 实现去重合并、限制大小等常用模式
- 理解 Reducer 工厂函数
- 处理复杂的状态更新逻辑

**适用场景**：
- 需要去重的列表管理
- 有大小限制的历史记录
- 条件覆盖的状态更新
- 复杂的数据合并逻辑

[来源: reference/search_annotated_github_01.md]

---

## 原理讲解

### 1. Reducer 函数签名

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
    return merge_logic(old_value, new_value)
```

**关键要求**：
1. 必须接受两个参数
2. 返回值类型与参数类型一致
3. 不修改原始对象（返回新对象）
4. 幂等性（多次调用结果一致）

[来源: reference/source_annotated_02.md]

### 2. 常用 Reducer 模式

| 模式 | 用途 | 示例 |
|------|------|------|
| **去重合并** | 避免重复项 | `merge_with_dedup` |
| **限制大小** | 控制列表长度 | `merge_with_limit` |
| **条件覆盖** | 按条件更新 | `conditional_merge` |
| **优先级合并** | 按优先级选择 | `priority_merge` |
| **时间窗口** | 保留时间范围内的数据 | `time_window_merge` |

[来源: reference/search_annotated_github_01.md]

### 3. Reducer 工厂函数

```python
def create_reducer(config):
    """
    Reducer 工厂函数
    返回配置化的 Reducer
    """
    def reducer(old, new):
        # 使用 config 参数
        return merge(old, new, config)
    return reducer
```

**优势**：
- 参数化配置
- 代码复用
- 灵活性高

[来源: reference/search_annotated_github_01.md]

---

## 完整代码示例

### 示例 1：去重合并 Reducer

```python
"""
去重合并 Reducer
避免列表中出现重复项
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 1. 定义去重 Reducer
def merge_with_dedup(old: list, new: list) -> list:
    """
    去重合并列表

    Args:
        old: 旧列表
        new: 新列表

    Returns:
        合并后的去重列表
    """
    # 使用 set 去重，保持顺序
    seen = set(old)
    result = old.copy()

    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)

    return result


# 2. 定义状态
class State(TypedDict):
    """状态定义"""
    items: Annotated[list[str], merge_with_dedup]


# 3. 定义节点
def add_items_1(state: State) -> dict:
    """添加第一批项目"""
    return {"items": ["A", "B", "C"]}


def add_items_2(state: State) -> dict:
    """添加第二批项目（包含重复）"""
    return {"items": ["B", "C", "D"]}


def add_items_3(state: State) -> dict:
    """添加第三批项目（包含重复）"""
    return {"items": ["C", "D", "E"]}


# 4. 构建图
def create_graph():
    """创建图"""
    builder = StateGraph(State)

    builder.add_node("add_1", add_items_1)
    builder.add_node("add_2", add_items_2)
    builder.add_node("add_3", add_items_3)

    builder.add_edge(START, "add_1")
    builder.add_edge("add_1", "add_2")
    builder.add_edge("add_2", "add_3")
    builder.add_edge("add_3", END)

    return builder.compile()


# 5. 运行
def main():
    """主函数"""
    graph = create_graph()

    result = graph.invoke({"items": []})

    print("=" * 60)
    print("去重合并 Reducer 演示")
    print("=" * 60)
    print(f"\n最终项目列表: {result['items']}")
    print(f"项目数量: {len(result['items'])}")
    print("\n说明: B、C、D 只出现一次，没有重复")


if __name__ == "__main__":
    main()
```

**运行输出**：

```
============================================================
去重合并 Reducer 演示
============================================================

最终项目列表: ['A', 'B', 'C', 'D', 'E']
项目数量: 5

说明: B、C、D 只出现一次，没有重复
```

**关键观察**：
- 第一批：添加 A, B, C
- 第二批：尝试添加 B, C, D，但 B 和 C 已存在，只添加 D
- 第三批：尝试添加 C, D, E，但 C 和 D 已存在，只添加 E
- 最终：A, B, C, D, E（无重复）

[来源: reference/search_annotated_github_01.md]

---

### 示例 2：限制大小的 Reducer

```python
"""
限制大小的 Reducer
保持列表在指定大小内
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 1. Reducer 工厂函数
def create_limited_reducer(max_size: int):
    """
    创建限制大小的 Reducer

    Args:
        max_size: 最大列表大小

    Returns:
        Reducer 函数
    """
    def merge_with_limit(old: list, new: list) -> list:
        """
        合并并限制大小
        保留最新的 max_size 个项目
        """
        combined = old + new

        # 保留最新的 max_size 个项目
        if len(combined) > max_size:
            return combined[-max_size:]

        return combined

    return merge_with_limit


# 2. 定义状态
class State(TypedDict):
    """状态定义"""
    recent_items: Annotated[list[str], create_limited_reducer(5)]


# 3. 定义节点
def add_batch_1(state: State) -> dict:
    """添加第一批"""
    return {"recent_items": ["Item 1", "Item 2", "Item 3"]}


def add_batch_2(state: State) -> dict:
    """添加第二批"""
    return {"recent_items": ["Item 4", "Item 5", "Item 6"]}


def add_batch_3(state: State) -> dict:
    """添加第三批"""
    return {"recent_items": ["Item 7", "Item 8"]}


# 4. 构建图
def create_graph():
    """创建图"""
    builder = StateGraph(State)

    builder.add_node("batch_1", add_batch_1)
    builder.add_node("batch_2", add_batch_2)
    builder.add_node("batch_3", add_batch_3)

    builder.add_edge(START, "batch_1")
    builder.add_edge("batch_1", "batch_2")
    builder.add_edge("batch_2", "batch_3")
    builder.add_edge("batch_3", END)

    return builder.compile()


# 5. 运行
def main():
    """主函数"""
    graph = create_graph()

    result = graph.invoke({"recent_items": []})

    print("=" * 60)
    print("限制大小 Reducer 演示")
    print("=" * 60)
    print(f"\n最终项目列表: {result['recent_items']}")
    print(f"项目数量: {len(result['recent_items'])} (限制为 5)")
    print("\n说明: 只保留最新的 5 个项目")


if __name__ == "__main__":
    main()
```

**运行输出**：

```
============================================================
限制大小 Reducer 演示
============================================================

最终项目列表: ['Item 4', 'Item 5', 'Item 6', 'Item 7', 'Item 8']
项目数量: 5 (限制为 5)

说明: 只保留最新的 5 个项目
```

**关键观察**：
- 总共添加了 8 个项目
- 只保留最新的 5 个：Item 4-8
- Item 1-3 被自动移除

[来源: reference/search_annotated_github_01.md]

---

### 示例 3：条件覆盖 Reducer

```python
"""
条件覆盖 Reducer
根据条件决定是否更新
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 1. 定义条件 Reducer
def merge_if_not_none(old: str, new: str) -> str:
    """
    只有新值非 None 时才更新
    """
    if new is not None and new != "":
        return new
    return old


def merge_if_higher_priority(old: dict, new: dict) -> dict:
    """
    只有新值优先级更高时才更新
    """
    if not old:
        return new

    if not new:
        return old

    old_priority = old.get("priority", 0)
    new_priority = new.get("priority", 0)

    if new_priority > old_priority:
        return new

    return old


# 2. 定义状态
class State(TypedDict):
    """状态定义"""
    message: Annotated[str, merge_if_not_none]
    task: Annotated[dict, merge_if_higher_priority]


# 3. 定义节点
def set_initial(state: State) -> dict:
    """设置初始值"""
    return {
        "message": "Initial message",
        "task": {"name": "Task 1", "priority": 1}
    }


def try_update_low_priority(state: State) -> dict:
    """尝试用低优先级任务更新"""
    return {
        "message": "",  # 空字符串，不应更新
        "task": {"name": "Task 2", "priority": 0}  # 低优先级，不应更新
    }


def update_high_priority(state: State) -> dict:
    """用高优先级任务更新"""
    return {
        "message": "Updated message",  # 非空，应更新
        "task": {"name": "Task 3", "priority": 5}  # 高优先级，应更新
    }


# 4. 构建图
def create_graph():
    """创建图"""
    builder = StateGraph(State)

    builder.add_node("initial", set_initial)
    builder.add_node("try_low", try_update_low_priority)
    builder.add_node("update_high", update_high_priority)

    builder.add_edge(START, "initial")
    builder.add_edge("initial", "try_low")
    builder.add_edge("try_low", "update_high")
    builder.add_edge("update_high", END)

    return builder.compile()


# 5. 运行
def main():
    """主函数"""
    graph = create_graph()

    result = graph.invoke({
        "message": "",
        "task": {}
    })

    print("=" * 60)
    print("条件覆盖 Reducer 演示")
    print("=" * 60)
    print(f"\n最终消息: {result['message']}")
    print(f"最终任务: {result['task']}")
    print("\n说明:")
    print("- 消息: 空字符串不更新，非空字符串更新")
    print("- 任务: 只有更高优先级的任务才会更新")


if __name__ == "__main__":
    main()
```

**运行输出**：

```
============================================================
条件覆盖 Reducer 演示
============================================================

最终消息: Updated message
最终任务: {'name': 'Task 3', 'priority': 5}

说明:
- 消息: 空字符串不更新，非空字符串更新
- 任务: 只有更高优先级的任务才会更新
```

[来源: reference/search_annotated_github_01.md]

---

### 示例 4：复杂对象合并 Reducer

```python
"""
复杂对象合并 Reducer
处理嵌套字典的智能合并
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from datetime import datetime


# 1. 定义复杂合并 Reducer
def deep_merge_dict(old: dict, new: dict) -> dict:
    """
    深度合并字典
    - 递归合并嵌套字典
    - 列表追加而非覆盖
    - 保留时间戳
    """
    if not old:
        return new

    if not new:
        return old

    result = old.copy()

    for key, value in new.items():
        if key in result:
            # 如果都是字典，递归合并
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dict(result[key], value)
            # 如果都是列表，追加
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            # 否则覆盖
            else:
                result[key] = value
        else:
            result[key] = value

    return result


# 2. 定义状态
class State(TypedDict):
    """状态定义"""
    config: Annotated[dict, deep_merge_dict]


# 3. 定义节点
def set_base_config(state: State) -> dict:
    """设置基础配置"""
    return {
        "config": {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "features": ["feature_a"],
            "version": "1.0"
        }
    }


def add_cache_config(state: State) -> dict:
    """添加缓存配置"""
    return {
        "config": {
            "cache": {
                "enabled": True,
                "ttl": 3600
            },
            "features": ["feature_b"]
        }
    }


def add_logging_config(state: State) -> dict:
    """添加日志配置"""
    return {
        "config": {
            "logging": {
                "level": "INFO",
                "file": "/var/log/app.log"
            },
            "features": ["feature_c"],
            "version": "1.1"
        }
    }


# 4. 构建图
def create_graph():
    """创建图"""
    builder = StateGraph(State)

    builder.add_node("base", set_base_config)
    builder.add_node("cache", add_cache_config)
    builder.add_node("logging", add_logging_config)

    builder.add_edge(START, "base")
    builder.add_edge("base", "cache")
    builder.add_edge("cache", "logging")
    builder.add_edge("logging", END)

    return builder.compile()


# 5. 运行
def main():
    """主函数"""
    graph = create_graph()

    result = graph.invoke({"config": {}})

    print("=" * 60)
    print("复杂对象合并 Reducer 演示")
    print("=" * 60)
    print("\n最终配置:")

    import json
    print(json.dumps(result["config"], indent=2))

    print("\n说明:")
    print("- 嵌套字典被递归合并")
    print("- features 列表被追加（包含 a, b, c）")
    print("- version 被覆盖为最新值")


if __name__ == "__main__":
    main()
```

**运行输出**：

```
============================================================
复杂对象合并 Reducer 演示
============================================================

最终配置:
{
  "database": {
    "host": "localhost",
    "port": 5432
  },
  "features": [
    "feature_a",
    "feature_b",
    "feature_c"
  ],
  "version": "1.1",
  "cache": {
    "enabled": true,
    "ttl": 3600
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/app.log"
  }
}

说明:
- 嵌套字典被递归合并
- features 列表被追加（包含 a, b, c）
- version 被覆盖为最新值
```

[来源: reference/search_annotated_github_01.md]

---

## 实际应用场景

### 场景 1：智能日志收集

```python
"""
智能日志收集系统
- 去重日志
- 限制日志数量
- 按级别过滤
"""

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from datetime import datetime


# 日志级别
LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


def create_smart_log_reducer(max_logs: int = 100, min_level: str = "INFO"):
    """
    创建智能日志 Reducer

    Args:
        max_logs: 最大日志数量
        min_level: 最小日志级别
    """
    def reducer(old: list[dict], new: list[dict]) -> list[dict]:
        # 过滤低级别日志
        min_level_value = LOG_LEVELS.get(min_level, 0)
        filtered = [
            log for log in new
            if LOG_LEVELS.get(log.get("level", "INFO"), 0) >= min_level_value
        ]

        # 合并并去重（按消息内容）
        seen_messages = {log["message"] for log in old}
        unique_new = [
            log for log in filtered
            if log["message"] not in seen_messages
        ]

        # 合并并限制大小
        combined = old + unique_new
        if len(combined) > max_logs:
            return combined[-max_logs:]

        return combined

    return reducer


class State(TypedDict):
    """状态定义"""
    logs: Annotated[list[dict], create_smart_log_reducer(max_logs=5, min_level="INFO")]


def node_a(state: State) -> dict:
    """节点 A"""
    return {
        "logs": [
            {"level": "DEBUG", "message": "Debug message"},
            {"level": "INFO", "message": "Node A started"},
        ]
    }


def node_b(state: State) -> dict:
    """节点 B"""
    return {
        "logs": [
            {"level": "INFO", "message": "Node A started"},  # 重复
            {"level": "WARNING", "message": "Node B warning"},
        ]
    }


def node_c(state: State) -> dict:
    """节点 C"""
    return {
        "logs": [
            {"level": "ERROR", "message": "Node C error"},
        ]
    }


def create_graph():
    """创建图"""
    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    return builder.compile()


def main():
    """主函数"""
    graph = create_graph()
    result = graph.invoke({"logs": []})

    print("=" * 60)
    print("智能日志收集")
    print("=" * 60)
    print(f"\n日志数量: {len(result['logs'])}")
    print("\n日志列表:")
    for log in result["logs"]:
        print(f"  [{log['level']}] {log['message']}")

    print("\n说明:")
    print("- DEBUG 日志被过滤")
    print("- 重复日志被去重")
    print("- 只保留最近 5 条")


if __name__ == "__main__":
    main()
```

[来源: reference/search_annotated_github_01.md]

---

## 常见陷阱与解决方案

### 陷阱 1：修改原始对象

```python
# ❌ 错误：修改原始列表
def bad_reducer(old: list, new: list) -> list:
    old.extend(new)  # 修改了原始对象
    return old

# ✅ 正确：创建新列表
def good_reducer(old: list, new: list) -> list:
    return old + new  # 返回新对象
```

[来源: reference/search_annotated_github_01.md]

---

### 陷阱 2：忘记处理空值

```python
# ❌ 错误：未处理 None
def bad_reducer(old: list, new: list) -> list:
    return old + new  # 如果 old 或 new 是 None 会报错

# ✅ 正确：处理空值
def good_reducer(old: list, new: list) -> list:
    if not old:
        return new or []
    if not new:
        return old
    return old + new
```

---

### 陷阱 3：性能问题

```python
# ❌ 不好：每次都遍历整个列表
def slow_dedup(old: list, new: list) -> list:
    result = old.copy()
    for item in new:
        if item not in result:  # O(n) 查找
            result.append(item)
    return result

# ✅ 好：使用 set 加速
def fast_dedup(old: list, new: list) -> list:
    seen = set(old)  # O(1) 查找
    result = old.copy()
    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
```

[来源: reference/search_annotated_github_01.md]

---

## 总结

### 核心要点

1. **Reducer 签名**：`(old, new) -> merged`
2. **不修改原对象**：返回新对象
3. **处理边界情况**：空值、None、空列表
4. **性能优化**：使用 set、避免深拷贝
5. **工厂函数**：参数化配置

### 常用模式

- 去重合并
- 限制大小
- 条件覆盖
- 深度合并
- 智能过滤

### 下一步

- 学习复杂状态管理（场景4）

---

**参考资料**：
- [来源: reference/search_annotated_github_01.md]
- [来源: reference/search_annotated_reddit_01.md]
- [来源: reference/source_annotated_02.md]
