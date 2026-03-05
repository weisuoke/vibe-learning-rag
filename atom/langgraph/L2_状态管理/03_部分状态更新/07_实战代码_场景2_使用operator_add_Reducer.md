# 实战代码：使用 operator.add Reducer

> 演示如何使用 `operator.add` 实现列表追加的部分状态更新

---

## 场景说明

本文档演示如何使用 `operator.add` Reducer 实现列表的增量更新。在 LangGraph 中，当你需要累积数据（如日志、步骤记录、结果列表）而不是覆盖时，`operator.add` 是最常用的 Reducer。

**核心特性：**
- 使用 `Annotated[list, operator.add]` 定义累积字段
- 节点函数只返回新增的元素
- 框架自动将新元素追加到现有列表

**[来源: reference/context7_langgraph_01.md, reference/source_部分状态更新_01.md]**

---

## 完整实战代码

```python
"""
使用 operator.add Reducer 实现列表追加
演示场景：多步骤数据处理流程，每个节点追加处理结果
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ===== 1. 定义状态 Schema =====
class ProcessState(TypedDict):
    """
    数据处理流程的状态定义

    字段说明：
    - input_data: 输入数据（覆盖模式）
    - steps: 处理步骤记录（追加模式，使用 operator.add）
    - results: 处理结果列表（追加模式，使用 operator.add）
    - counter: 步骤计数器（覆盖模式）
    """
    input_data: str
    # 使用 operator.add 作为 Reducer，新元素会追加到列表末尾
    steps: Annotated[list[str], operator.add]
    results: Annotated[list[dict], operator.add]
    counter: int


# ===== 2. 定义节点函数 =====

def load_data(state: ProcessState) -> dict:
    """
    步骤1：加载数据

    返回部分状态：
    - steps: 追加一条步骤记录
    - results: 追加一个结果字典
    - counter: 更新计数器
    """
    print(f"\n[步骤 {state['counter'] + 1}] 加载数据...")

    # 模拟数据加载
    loaded_data = state["input_data"].upper()

    # 只返回需要更新的字段
    return {
        "steps": [f"步骤1: 加载数据 '{state['input_data']}'"],
        "results": [{
            "step": 1,
            "action": "load",
            "data": loaded_data,
            "status": "success"
        }],
        "counter": state["counter"] + 1
    }


def validate_data(state: ProcessState) -> dict:
    """
    步骤2：验证数据

    返回部分状态：
    - steps: 追加一条步骤记录
    - results: 追加一个结果字典
    - counter: 更新计数器
    """
    print(f"\n[步骤 {state['counter'] + 1}] 验证数据...")

    # 获取上一步的结果
    last_result = state["results"][-1]
    data = last_result["data"]

    # 模拟数据验证
    is_valid = len(data) > 0

    return {
        "steps": [f"步骤2: 验证数据，结果={'通过' if is_valid else '失败'}"],
        "results": [{
            "step": 2,
            "action": "validate",
            "is_valid": is_valid,
            "status": "success"
        }],
        "counter": state["counter"] + 1
    }


def transform_data(state: ProcessState) -> dict:
    """
    步骤3：转换数据

    返回部分状态：
    - steps: 追加一条步骤记录
    - results: 追加一个结果字典
    - counter: 更新计数器
    """
    print(f"\n[步骤 {state['counter'] + 1}] 转换数据...")

    # 获取加载的数据
    loaded_data = state["results"][0]["data"]

    # 模拟数据转换
    transformed = f"[TRANSFORMED] {loaded_data}"

    return {
        "steps": [f"步骤3: 转换数据为 '{transformed}'"],
        "results": [{
            "step": 3,
            "action": "transform",
            "data": transformed,
            "status": "success"
        }],
        "counter": state["counter"] + 1
    }


def save_data(state: ProcessState) -> dict:
    """
    步骤4：保存数据

    返回部分状态：
    - steps: 追加一条步骤记录
    - results: 追加一个结果字典
    - counter: 更新计数器
    """
    print(f"\n[步骤 {state['counter'] + 1}] 保存数据...")

    # 获取转换后的数据
    transformed_data = state["results"][-1]["data"]

    # 模拟数据保存
    save_path = f"/data/{state['input_data']}.txt"

    return {
        "steps": [f"步骤4: 保存数据到 '{save_path}'"],
        "results": [{
            "step": 4,
            "action": "save",
            "path": save_path,
            "status": "success"
        }],
        "counter": state["counter"] + 1
    }


# ===== 3. 构建图 =====

def create_processing_graph():
    """创建数据处理流程图"""

    # 创建 StateGraph
    builder = StateGraph(ProcessState)

    # 添加节点
    builder.add_node("load", load_data)
    builder.add_node("validate", validate_data)
    builder.add_node("transform", transform_data)
    builder.add_node("save", save_data)

    # 添加边（定义执行顺序）
    builder.add_edge(START, "load")
    builder.add_edge("load", "validate")
    builder.add_edge("validate", "transform")
    builder.add_edge("transform", "save")
    builder.add_edge("save", END)

    # 编译图（使用内存检查点）
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# ===== 4. 运行示例 =====

def run_example_1():
    """示例1：基础数据处理流程"""
    print("=" * 60)
    print("示例1：基础数据处理流程")
    print("=" * 60)

    graph = create_processing_graph()

    # 初始状态
    initial_state = {
        "input_data": "hello world",
        "steps": [],
        "results": [],
        "counter": 0
    }

    # 执行图
    config = {"configurable": {"thread_id": "example-1"}}
    final_state = graph.invoke(initial_state, config)

    # 输出结果
    print("\n" + "=" * 60)
    print("执行完成！")
    print("=" * 60)
    print(f"\n输入数据: {final_state['input_data']}")
    print(f"总步骤数: {final_state['counter']}")
    print(f"\n处理步骤:")
    for step in final_state["steps"]:
        print(f"  - {step}")
    print(f"\n处理结果:")
    for result in final_state["results"]:
        print(f"  - 步骤{result['step']}: {result['action']} -> {result['status']}")


def run_example_2():
    """示例2：演示 operator.add 的累积效果"""
    print("\n\n" + "=" * 60)
    print("示例2：演示 operator.add 的累积效果")
    print("=" * 60)

    # 手动演示 operator.add 的行为
    print("\n手动模拟 operator.add 的工作原理：")

    # 初始状态
    current_steps = []
    print(f"初始状态: steps = {current_steps}")

    # 节点1返回
    update_1 = ["步骤1: 加载数据"]
    current_steps = operator.add(current_steps, update_1)
    print(f"节点1返回: {update_1}")
    print(f"合并后: steps = {current_steps}")

    # 节点2返回
    update_2 = ["步骤2: 验证数据"]
    current_steps = operator.add(current_steps, update_2)
    print(f"节点2返回: {update_2}")
    print(f"合并后: steps = {current_steps}")

    # 节点3返回
    update_3 = ["步骤3: 转换数据"]
    current_steps = operator.add(current_steps, update_3)
    print(f"节点3返回: {update_3}")
    print(f"合并后: steps = {current_steps}")

    print(f"\n最终结果: {current_steps}")
    print("\n关键点：")
    print("  1. 每个节点只返回新增的元素（列表）")
    print("  2. operator.add 自动将新元素追加到现有列表")
    print("  3. 节点不需要知道之前的完整列表")


def run_example_3():
    """示例3：对比覆盖模式 vs 追加模式"""
    print("\n\n" + "=" * 60)
    print("示例3：对比覆盖模式 vs 追加模式")
    print("=" * 60)

    # 定义两种状态
    class OverwriteState(TypedDict):
        logs: list[str]  # 覆盖模式（没有 Reducer）

    class AppendState(TypedDict):
        logs: Annotated[list[str], operator.add]  # 追加模式

    # 模拟节点更新
    print("\n覆盖模式（没有 Reducer）：")
    state_1 = {"logs": ["初始日志"]}
    print(f"初始状态: {state_1}")

    update_1 = {"logs": ["节点1的日志"]}
    state_1.update(update_1)
    print(f"节点1更新后: {state_1}")
    print("  ❌ 初始日志被覆盖了！")

    print("\n追加模式（使用 operator.add）：")
    state_2 = {"logs": ["初始日志"]}
    print(f"初始状态: {state_2}")

    # 模拟 operator.add 的行为
    update_2 = ["节点1的日志"]
    state_2["logs"] = operator.add(state_2["logs"], update_2)
    print(f"节点1更新后: {state_2}")
    print("  ✅ 初始日志被保留，新日志被追加！")


# ===== 5. 实际应用场景 =====

def run_real_world_example():
    """实际应用：日志收集系统"""
    print("\n\n" + "=" * 60)
    print("实际应用：日志收集系统")
    print("=" * 60)

    class LogState(TypedDict):
        """日志收集系统的状态"""
        task_id: str
        logs: Annotated[list[dict], operator.add]
        errors: Annotated[list[str], operator.add]
        warnings: Annotated[list[str], operator.add]

    def task_1(state: LogState) -> dict:
        """任务1：数据库查询"""
        print("\n执行任务1：数据库查询")
        return {
            "logs": [{
                "timestamp": "2026-02-26 10:00:00",
                "level": "INFO",
                "message": "开始查询数据库"
            }],
            "warnings": ["数据库连接较慢"]
        }

    def task_2(state: LogState) -> dict:
        """任务2：数据处理"""
        print("执行任务2：数据处理")
        return {
            "logs": [{
                "timestamp": "2026-02-26 10:00:05",
                "level": "INFO",
                "message": "处理了 1000 条记录"
            }]
        }

    def task_3(state: LogState) -> dict:
        """任务3：结果保存"""
        print("执行任务3：结果保存")
        return {
            "logs": [{
                "timestamp": "2026-02-26 10:00:10",
                "level": "INFO",
                "message": "结果已保存"
            }],
            "errors": ["磁盘空间不足，使用临时存储"]
        }

    # 构建图
    builder = StateGraph(LogState)
    builder.add_node("task1", task_1)
    builder.add_node("task2", task_2)
    builder.add_node("task3", task_3)
    builder.add_edge(START, "task1")
    builder.add_edge("task1", "task2")
    builder.add_edge("task2", "task3")
    builder.add_edge("task3", END)

    graph = builder.compile()

    # 执行
    initial_state = {
        "task_id": "task-12345",
        "logs": [],
        "errors": [],
        "warnings": []
    }

    final_state = graph.invoke(initial_state)

    # 输出结果
    print("\n" + "=" * 60)
    print("任务执行完成")
    print("=" * 60)
    print(f"\n任务ID: {final_state['task_id']}")
    print(f"\n日志记录 ({len(final_state['logs'])} 条):")
    for log in final_state["logs"]:
        print(f"  [{log['timestamp']}] {log['level']}: {log['message']}")

    if final_state["warnings"]:
        print(f"\n警告 ({len(final_state['warnings'])} 条):")
        for warning in final_state["warnings"]:
            print(f"  ⚠️  {warning}")

    if final_state["errors"]:
        print(f"\n错误 ({len(final_state['errors'])} 条):")
        for error in final_state["errors"]:
            print(f"  ❌ {error}")


# ===== 6. 主函数 =====

if __name__ == "__main__":
    # 运行所有示例
    run_example_1()
    run_example_2()
    run_example_3()
    run_real_world_example()

    print("\n\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)
```

---

## 运行输出示例

```
============================================================
示例1：基础数据处理流程
============================================================

[步骤 1] 加载数据...

[步骤 2] 验证数据...

[步骤 3] 转换数据...

[步骤 4] 保存数据...

============================================================
执行完成！
============================================================

输入数据: hello world
总步骤数: 4

处理步骤:
  - 步骤1: 加载数据 'hello world'
  - 步骤2: 验证数据，结果=通过
  - 步骤3: 转换数据为 '[TRANSFORMED] HELLO WORLD'
  - 步骤4: 保存数据到 '/data/hello world.txt'

处理结果:
  - 步骤1: load -> success
  - 步骤2: validate -> success
  - 步骤3: transform -> success
  - 步骤4: save -> success


============================================================
示例2：演示 operator.add 的累积效果
============================================================

手动模拟 operator.add 的工作原理：
初始状态: steps = []
节点1返回: ['步骤1: 加载数据']
合并后: steps = ['步骤1: 加载数据']
节点2返回: ['步骤2: 验证数据']
合并后: steps = ['步骤1: 加载数据', '步骤2: 验证数据']
节点3返回: ['步骤3: 转换数据']
合并后: steps = ['步骤1: 加载数据', '步骤2: 验证数据', '步骤3: 转换数据']

最终结果: ['步骤1: 加载数据', '步骤2: 验证数据', '步骤3: 转换数据']

关键点：
  1. 每个节点只返回新增的元素（列表）
  2. operator.add 自动将新元素追加到现有列表
  3. 节点不需要知道之前的完整列表
```

---

## 核心要点总结

### 1. operator.add 的工作原理

**[来源: reference/source_部分状态更新_01.md]**

```python
# operator.add 对列表的行为
current = [1, 2, 3]
new = [4, 5]
result = operator.add(current, new)
# result = [1, 2, 3, 4, 5]
```

### 2. 状态定义模式

```python
class State(TypedDict):
    # 覆盖模式（默认）
    counter: int

    # 追加模式（使用 operator.add）
    logs: Annotated[list[str], operator.add]
```

### 3. 节点返回值

```python
def my_node(state: State) -> dict:
    # 只返回新增的元素（列表形式）
    return {
        "logs": ["新日志1", "新日志2"]  # 会被追加到现有列表
    }
```

### 4. 适用场景

- ✅ 日志收集
- ✅ 步骤记录
- ✅ 结果累积
- ✅ 历史追踪
- ✅ 事件流

---

## 常见问题

### Q1: 为什么要返回列表而不是单个元素？

**A:** `operator.add` 对列表执行拼接操作，所以必须返回列表。

```python
# ✅ 正确
return {"logs": ["新日志"]}

# ❌ 错误
return {"logs": "新日志"}  # 会导致类型错误
```

### Q2: 可以一次追加多个元素吗？

**A:** 可以，返回包含多个元素的列表即可。

```python
return {
    "logs": ["日志1", "日志2", "日志3"]
}
```

### Q3: 如何清空列表？

**A:** 使用 Overwrite 模式（将在场景6中详细讲解）。

---

## 参考资料

- **[来源: reference/context7_langgraph_01.md]** - LangGraph 官方文档关于 Annotated 字段的说明
- **[来源: reference/source_部分状态更新_01.md]** - BinaryOperatorAggregate 源码分析
- **[来源: reference/search_部分状态更新_01.md]** - 社区最佳实践

---

**文档版本:** v1.0
**创建时间:** 2026-02-26
**适用于:** LangGraph 0.2.0+
**Python 版本:** 3.13+
