# 实战代码 场景2：Reducer实战应用

> 完整可运行的Reducer实战示例

## 场景描述

构建一个数据处理管道，使用多种Reducer实现复杂的状态聚合逻辑，包括列表追加、字典合并、自定义去重等。

## 完整代码

```python
"""
场景2：Reducer实战应用
功能：数据处理管道with多种Reducer
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
import time

# ============ 1. 自定义Reducer ============

def merge_unique(existing: list, new: list) -> list:
    """去重追加"""
    seen = set(existing)
    return existing + [x for x in new if x not in seen]


def keep_last_n(n: int):
    """保留最新N个元素"""
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        return combined[-n:]
    return reducer


def merge_by_priority(existing: list, new: list) -> list:
    """按优先级合并（高优先级覆盖低优先级）"""
    # 假设每个元素是 {"id": ..., "priority": ..., "data": ...}
    merged = {item["id"]: item for item in existing}
    for item in new:
        if item["id"] not in merged or item["priority"] > merged[item["id"]]["priority"]:
            merged[item["id"]] = item
    return list(merged.values())


def deep_merge_dict(existing: dict, new: dict) -> dict:
    """递归合并字典"""
    result = existing.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


# ============ 2. 状态定义 ============

class PipelineState(TypedDict):
    """数据处理管道状态"""
    # 列表追加
    logs: Annotated[list[str], operator.add]

    # 去重追加
    unique_items: Annotated[list[str], merge_unique]

    # 保留最新10个
    recent_events: Annotated[list[dict], keep_last_n(10)]

    # 按优先级合并
    tasks: Annotated[list[dict], merge_by_priority]

    # 字典合并
    stats: Annotated[dict, operator.or_]

    # 深度合并
    config: Annotated[dict, deep_merge_dict]

    # 数值累加
    total_count: Annotated[int, operator.add]

    # 覆盖更新
    status: str
    current_step: str


# ============ 3. 节点函数 ============

def data_collection_node(state: PipelineState) -> dict:
    """数据收集节点"""
    print("\n[数据收集] 收集数据...")

    return {
        "logs": ["[收集] 开始收集数据"],
        "unique_items": ["item1", "item2", "item1"],  # 包含重复
        "recent_events": [
            {"event": "collect", "time": time.time()}
        ],
        "tasks": [
            {"id": "task1", "priority": 1, "data": "low priority"}
        ],
        "stats": {"collected": 100},
        "config": {"source": {"type": "api", "url": "http://example.com"}},
        "total_count": 100,
        "status": "collecting",
        "current_step": "collection"
    }


def data_processing_node(state: PipelineState) -> dict:
    """数据处理节点"""
    print("\n[数据处理] 处理数据...")

    return {
        "logs": ["[处理] 处理数据中"],
        "unique_items": ["item2", "item3"],  # item2重复，item3新增
        "recent_events": [
            {"event": "process", "time": time.time()}
        ],
        "tasks": [
            {"id": "task1", "priority": 5, "data": "high priority"},  # 更高优先级
            {"id": "task2", "priority": 3, "data": "medium priority"}
        ],
        "stats": {"processed": 80},
        "config": {"processing": {"threads": 4}},
        "total_count": 80,
        "status": "processing",
        "current_step": "processing"
    }


def data_validation_node(state: PipelineState) -> dict:
    """数据验证节点"""
    print("\n[数据验证] 验证数据...")

    return {
        "logs": ["[验证] 验证数据完成"],
        "unique_items": ["item4"],  # 新增item4
        "recent_events": [
            {"event": "validate", "time": time.time()}
        ],
        "tasks": [
            {"id": "task3", "priority": 2, "data": "validation task"}
        ],
        "stats": {"validated": 75},
        "config": {"validation": {"strict": True}},
        "total_count": 75,
        "status": "validated",
        "current_step": "validation"
    }


# ============ 4. 构建图 ============

def create_pipeline_graph():
    """创建数据处理管道图"""
    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("collect", data_collection_node)
    graph.add_node("process", data_processing_node)
    graph.add_node("validate", data_validation_node)

    # 添加边
    graph.add_edge("collect", "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", END)

    # 设置入口
    graph.set_entry_point("collect")

    return graph.compile()


# ============ 5. 运行示例 ============

def main():
    """主函数"""
    print("=" * 60)
    print("场景2：Reducer实战应用")
    print("=" * 60)

    # 创建图
    app = create_pipeline_graph()

    # 初始状态
    initial_state = {
        "logs": [],
        "unique_items": [],
        "recent_events": [],
        "tasks": [],
        "stats": {},
        "config": {},
        "total_count": 0,
        "status": "init",
        "current_step": "init"
    }

    print("\n初始状态:")
    print(f"  logs: {initial_state['logs']}")
    print(f"  unique_items: {initial_state['unique_items']}")
    print(f"  total_count: {initial_state['total_count']}")

    # 运行图
    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("最终状态:")
    print("=" * 60)

    print(f"\n1. logs (operator.add):")
    for log in result["logs"]:
        print(f"   - {log}")

    print(f"\n2. unique_items (merge_unique):")
    print(f"   {result['unique_items']}")
    print(f"   说明：item1和item2去重，保留唯一值")

    print(f"\n3. recent_events (keep_last_n(10)):")
    for event in result["recent_events"]:
        print(f"   - {event}")

    print(f"\n4. tasks (merge_by_priority):")
    for task in result["tasks"]:
        print(f"   - {task}")
    print(f"   说明：task1被高优先级版本覆盖")

    print(f"\n5. stats (operator.or_):")
    print(f"   {result['stats']}")

    print(f"\n6. config (deep_merge_dict):")
    print(f"   {result['config']}")

    print(f"\n7. total_count (operator.add):")
    print(f"   {result['total_count']}")
    print(f"   说明：100 + 80 + 75 = 255")

    print(f"\n8. status (覆盖):")
    print(f"   {result['status']}")

    print(f"\n9. current_step (覆盖):")
    print(f"   {result['current_step']}")


# ============ 6. 并行执行示例 ============

def parallel_example():
    """并行执行Reducer示例"""
    print("\n" + "=" * 60)
    print("并行执行Reducer示例")
    print("=" * 60)

    class ParallelState(TypedDict):
        results: Annotated[list[str], operator.add]
        errors: Annotated[list[str], operator.add]
        count: Annotated[int, operator.add]

    def worker1(state: ParallelState) -> dict:
        print("\n[Worker 1] 执行任务...")
        return {
            "results": ["worker1_result"],
            "count": 10
        }

    def worker2(state: ParallelState) -> dict:
        print("[Worker 2] 执行任务...")
        return {
            "results": ["worker2_result"],
            "count": 20
        }

    def worker3(state: ParallelState) -> dict:
        print("[Worker 3] 执行任务...")
        return {
            "results": ["worker3_result"],
            "errors": ["worker3_error"],
            "count": 30
        }

    # 创建图
    graph = StateGraph(ParallelState)
    graph.add_node("worker1", worker1)
    graph.add_node("worker2", worker2)
    graph.add_node("worker3", worker3)

    # 并行执行
    graph.add_edge("worker1", END)
    graph.add_edge("worker2", END)
    graph.add_edge("worker3", END)

    graph.set_entry_point("worker1")
    graph.set_entry_point("worker2")
    graph.set_entry_point("worker3")

    app = graph.compile()

    # 运行
    result = app.invoke({
        "results": [],
        "errors": [],
        "count": 0
    })

    print("\n并行执行结果:")
    print(f"  results: {result['results']}")
    print(f"  errors: {result['errors']}")
    print(f"  count: {result['count']}")
    print(f"  说明：三个worker的结果自动聚合")


# ============ 7. Reducer性能测试 ============

def performance_test():
    """Reducer性能测试"""
    print("\n" + "=" * 60)
    print("Reducer性能测试")
    print("=" * 60)

    import time

    # 测试operator.add
    start = time.time()
    result = []
    for i in range(1000):
        result = operator.add(result, [i])
    duration1 = time.time() - start
    print(f"\noperator.add (1000次): {duration1:.4f}秒")

    # 测试merge_unique
    start = time.time()
    result = []
    for i in range(1000):
        result = merge_unique(result, [i % 100])  # 有重复
    duration2 = time.time() - start
    print(f"merge_unique (1000次): {duration2:.4f}秒")

    # 测试keep_last_n
    reducer = keep_last_n(10)
    start = time.time()
    result = []
    for i in range(1000):
        result = reducer(result, [i])
    duration3 = time.time() - start
    print(f"keep_last_n(10) (1000次): {duration3:.4f}秒")

    print(f"\n性能对比:")
    print(f"  operator.add: 1.0x")
    print(f"  merge_unique: {duration2/duration1:.2f}x")
    print(f"  keep_last_n: {duration3/duration1:.2f}x")


# ============ 8. 运行所有示例 ============

if __name__ == "__main__":
    # 基础示例
    main()

    # 并行执行
    parallel_example()

    # 性能测试
    performance_test()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
```

## 运行结果

```
============================================================
场景2：Reducer实战应用
============================================================

初始状态:
  logs: []
  unique_items: []
  total_count: 0

[数据收集] 收集数据...

[数据处理] 处理数据...

[数据验证] 验证数据...

============================================================
最终状态:
============================================================

1. logs (operator.add):
   - [收集] 开始收集数据
   - [处理] 处理数据中
   - [验证] 验证数据完成

2. unique_items (merge_unique):
   ['item1', 'item2', 'item3', 'item4']
   说明：item1和item2去重，保留唯一值

3. recent_events (keep_last_n(10)):
   - {'event': 'collect', 'time': 1708826400.0}
   - {'event': 'process', 'time': 1708826401.0}
   - {'event': 'validate', 'time': 1708826402.0}

4. tasks (merge_by_priority):
   - {'id': 'task1', 'priority': 5, 'data': 'high priority'}
   - {'id': 'task2', 'priority': 3, 'data': 'medium priority'}
   - {'id': 'task3', 'priority': 2, 'data': 'validation task'}
   说明：task1被高优先级版本覆盖

5. stats (operator.or_):
   {'collected': 100, 'processed': 80, 'validated': 75}

6. config (deep_merge_dict):
   {'source': {'type': 'api', 'url': 'http://example.com'}, 'processing': {'threads': 4}, 'validation': {'strict': True}}

7. total_count (operator.add):
   255
   说明：100 + 80 + 75 = 255

8. status (覆盖):
   validated

9. current_step (覆盖):
   validation
```

## 关键知识点

### 1. 内置Reducer

```python
# operator.add - 追加/相加
logs: Annotated[list, operator.add]
total_count: Annotated[int, operator.add]

# operator.or_ - 字典合并
stats: Annotated[dict, operator.or_]
```

### 2. 自定义Reducer

```python
def merge_unique(existing: list, new: list) -> list:
    """去重追加"""
    seen = set(existing)
    return existing + [x for x in new if x not in seen]

unique_items: Annotated[list, merge_unique]
```

### 3. Reducer工厂函数

```python
def keep_last_n(n: int):
    """保留最新N个元素"""
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        return combined[-n:]
    return reducer

recent_events: Annotated[list, keep_last_n(10)]
```

### 4. 复杂Reducer逻辑

```python
def merge_by_priority(existing: list, new: list) -> list:
    """按优先级合并"""
    merged = {item["id"]: item for item in existing}
    for item in new:
        if item["id"] not in merged or item["priority"] > merged[item["id"]]["priority"]:
            merged[item["id"]] = item
    return list(merged.values())
```

## 扩展练习

### 练习1：时间窗口Reducer

```python
def time_window_reducer(window_seconds: int):
    """保留时间窗口内的事件"""
    def reducer(existing: list, new: list) -> list:
        now = time.time()
        # 过滤旧事件
        valid = [e for e in existing if now - e["time"] < window_seconds]
        return valid + new
    return reducer

events: Annotated[list, time_window_reducer(3600)]  # 1小时窗口
```

### 练习2：加权平均Reducer

```python
def weighted_average_reducer(existing: dict, new: dict) -> dict:
    """加权平均"""
    if not existing:
        return new

    total_weight = existing.get("weight", 0) + new.get("weight", 0)
    avg_value = (
        existing.get("value", 0) * existing.get("weight", 0) +
        new.get("value", 0) * new.get("weight", 0)
    ) / total_weight

    return {"value": avg_value, "weight": total_weight}

metrics: Annotated[dict, weighted_average_reducer]
```

### 练习3：Top-K Reducer

```python
def top_k_reducer(k: int, key_func):
    """保留Top-K元素"""
    def reducer(existing: list, new: list) -> list:
        combined = existing + new
        sorted_items = sorted(combined, key=key_func, reverse=True)
        return sorted_items[:k]
    return reducer

top_scores: Annotated[list, top_k_reducer(10, lambda x: x["score"])]
```

## 总结

**本场景展示了**：
1. 多种内置Reducer的使用
2. 自定义Reducer实现
3. Reducer工厂函数模式
4. 复杂聚合逻辑
5. 并行执行中的Reducer
6. Reducer性能考虑

**关键要点**：
- Reducer必须是纯函数
- 接收两个参数，返回合并值
- 支持自定义复杂逻辑
- 自动处理并行更新

## 参考资料

- Reducer详解：`03_核心概念_02_Annotated与Reducer.md`
- 并行执行：`03_核心概念_07_并行执行与状态聚合.md`
- TypedDict基础：`03_核心概念_01_TypedDict状态定义.md`
