---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-25
knowledge_point: 04_图的编译与执行
context7_query: Pregel algorithm execution loop graph runtime
---

# Context7 文档：LangGraph Pregel 算法与执行流程

## 文档来源
- 库名称：LangGraph
- 版本：latest
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/

## 关键信息提取

### 1. LangGraph Pregel Runtime Overview

**来源**：https://docs.langchain.com/oss/python/langgraph/pregel

提供 LangGraph 的 Pregel 运行时的高层概述，解释其核心组件：actors 和 channels。详细说明每个步骤内执行的三个阶段：Plan、Execution 和 Update，遵循 Pregel Algorithm/Bulk Synchronous Parallel 模型。

**核心组件**：
- **Actors（执行者）**：图中的节点，执行计算任务
- **Channels（通道）**：用于在 actors 之间传递数据

**执行阶段**：
1. **Plan（计划）**：确定要执行哪些 actors
2. **Execution（执行）**：并行执行选定的 actors
3. **Update（更新）**：使用 actors 写入的值更新 channels

### 2. LangGraph Pregel: Cycle Example

**来源**：https://docs.langchain.com/oss/python/langgraph/pregel

展示如何在 Pregel 图中创建循环，其中节点写入它订阅的通道。执行继续直到产生 None 值，停止循环。

```python
from langgraph.channels import EphemeralValue, ChannelWriteEntry
from langgraph.pregel import Pregel, NodeBuilder

example_node = (
    NodeBuilder().subscribe_only("value")
    .do(lambda x: x + x if len(x) < 10 else None)
    .write_to(ChannelWriteEntry("value", skip_none=True))
)

app = Pregel(
    nodes={"example_node": example_node},
    channels={
        "value": EphemeralValue(str),
    },
    input_channels=["value"],
    output_channels=["value"],
)

app.invoke({"value": "a"})
```

**关键点**：
- 节点可以订阅它写入的通道，形成循环
- 使用 `skip_none=True` 在返回 None 时停止循环
- `EphemeralValue` 用于临时存储值

### 3. LangGraph runtime > Overview

**来源**：https://docs.langchain.com/oss/python/langgraph/pregel

在 LangGraph 中，Pregel 将 actors 和 channels 组合成单个应用程序。Actors 从 channels 读取数据并向 channels 写入数据。Pregel 将应用程序的执行组织成多个步骤，遵循 Pregel Algorithm/Bulk Synchronous Parallel 模型。

**每个步骤包含三个阶段**：
1. **Plan（计划）**：确定要执行哪些 actors
2. **Execution（执行）**：并行执行选定的 actors
3. **Update（更新）**：使用 actors 写入的值更新 channels

**执行流程**：
- 这个过程重复，直到没有 actors 被选中或达到最大步骤数

### 4. LangGraph runtime

**来源**：https://docs.langchain.com/oss/python/langgraph/pregel

Pregel 实现 LangGraph 的运行时，管理 LangGraph 应用程序的执行。编译 StateGraph 或创建 @entrypoint 会产生一个可以用输入调用的 Pregel 实例。本指南在高层次上解释运行时，并提供直接使用 Pregel 实现应用程序的说明。

**Pregel 运行时命名**：
- 以 Google 的 Pregel 算法命名
- Pregel 算法描述了一种使用图进行大规模并行计算的高效方法

**关键特性**：
- 管理图的执行
- 支持并行计算
- 基于 BSP（Bulk Synchronous Parallel）模型

### 5. Create and control loops

**来源**：https://docs.langchain.com/oss/python/langgraph/use-graph-api

当创建带有循环的图时，需要一种方法来停止执行。最常见的方法是使用条件边，一旦满足特定的终止条件，就将流程引导到 `END` 节点。此外，可以通过在调用或流式传输期间设置递归限制来控制图可以运行的最大步骤数。此限制通过在图超过允许的超步数时引发错误来防止无限循环。

**循环控制方法**：
1. **条件边**：使用条件边引导到 END 节点
2. **递归限制**：设置最大步骤数防止无限循环

**示例**：
```python
# 设置递归限制
config = {"recursion_limit": 10}
graph.invoke(input, config=config)
```

## 总结

### Pregel 算法核心特性

1. **BSP 模型**：
   - Bulk Synchronous Parallel（批量同步并行）
   - 将计算组织成超步（supersteps）
   - 每个超步包含三个阶段：Plan, Execution, Update

2. **Actors 和 Channels**：
   - **Actors**：图中的节点，执行计算
   - **Channels**：数据传递通道
   - Actors 从 channels 读取，向 channels 写入

3. **执行流程**：
   ```
   开始
     ↓
   Plan（选择要执行的 actors）
     ↓
   Execution（并行执行 actors）
     ↓
   Update（更新 channels）
     ↓
   检查终止条件
     ↓
   重复或结束
   ```

4. **循环控制**：
   - 条件边引导到 END
   - 递归限制防止无限循环
   - 节点返回 None 停止循环

5. **并行执行**：
   - 在 Execution 阶段并行执行多个 actors
   - 提高执行效率
   - 适合大规模图计算

### 实际应用场景

1. **迭代式工作流**：使用循环实现迭代式任务
2. **并行处理**：利用 Pregel 的并行执行能力
3. **大规模图计算**：适合处理复杂的图结构
4. **状态机实现**：通过 channels 管理状态流转
5. **可恢复执行**：结合 checkpoint 实现断点续传

### 与 Google Pregel 的关系

LangGraph 的 Pregel 运行时借鉴了 Google Pregel 算法的核心思想：
- **BSP 模型**：批量同步并行计算
- **消息传递**：通过 channels 传递数据
- **超步迭代**：将执行组织成多个步骤
- **并行执行**：在每个超步中并行执行节点

但 LangGraph 的 Pregel 针对 AI 工作流进行了优化：
- 支持 checkpoint 持久化
- 支持人机交互（interrupt）
- 支持流式输出
- 集成 LangChain 生态系统
