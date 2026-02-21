# 核心概念 02：Agent 进度流式

## 概述

深入理解 `stream_mode="updates"` 模式，掌握如何追踪和监控 Agent 执行的每个步骤。

---

## Agent 执行模型

### 状态机视角

**Agent 本质上是一个状态机，通过节点（Node）转换状态（State）。**

```
初始状态
↓
[节点 A] 执行 → 状态更新 → 触发 updates 流式
↓
[节点 B] 执行 → 状态更新 → 触发 updates 流式
↓
[节点 C] 执行 → 状态更新 → 触发 updates 流式
↓
最终状态
```

---

### 节点类型

#### 1. Model 节点
```python
# LLM 推理节点
{
    "model": {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {...}}]
            )
        ]
    }
}
```

#### 2. Tools 节点
```python
# 工具执行节点
{
    "tools": {
        "messages": [
            ToolMessage(
                content="北京的天气是晴天",
                tool_call_id="call_123"
            )
        ]
    }
}
```

#### 3. 自定义节点
```python
# 用户定义的节点
{
    "custom_node": {
        "messages": [...],
        "custom_data": {...}
    }
}
```

---

## 基础使用

### 简单示例

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    return f"{city}的天气是晴天"

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather]
)

# 追踪执行步骤
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气？"}]},
    stream_mode="updates"
):
    for node_name, data in chunk.items():
        print(f"\n{'='*50}")
        print(f"节点: {node_name}")
        print(f"消息数: {len(data.get('messages', []))}")

        if data.get('messages'):
            last_msg = data['messages'][-1]
            print(f"消息类型: {type(last_msg).__name__}")
```

**输出**：
```
==================================================
节点: model
消息数: 1
消息类型: AIMessage

==================================================
节点: tools
消息数: 1
消息类型: ToolMessage

==================================================
节点: model
消息数: 1
消息类型: AIMessage
```

---

### 详细监控

```python
import json

for chunk in agent.stream(input, stream_mode="updates"):
    for node_name, data in chunk.items():
        print(f"\n[{node_name}]")

        # 显示消息详情
        for msg in data.get('messages', []):
            if hasattr(msg, 'content'):
                print(f"  内容: {msg.content[:100]}")

            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  工具调用:")
                for tc in msg.tool_calls:
                    print(f"    - {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

            if hasattr(msg, 'tool_call_id'):
                print(f"  工具ID: {msg.tool_call_id}")
```

---

## 高级特性

### 1. Subgraph 流式

**追踪嵌套 Agent 的执行**：

```python
from langchain.agents import create_agent

# 子 Agent
sub_agent = create_agent(
    model="gpt-4o-mini",
    tools=[...],
    name="weather_agent"
)

def call_weather_agent(query: str) -> str:
    result = sub_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# 主 Agent
main_agent = create_agent(
    model="gpt-4o-mini",
    tools=[call_weather_agent],
    name="supervisor"
)

# 追踪所有层级
for namespace, mode, data in main_agent.stream(
    input,
    stream_mode="updates",
    subgraphs=True  # 启用子图流式
):
    level = len(namespace)
    indent = "  " * level

    for node_name, state in data.items():
        print(f"{indent}[L{level}:{node_name}]")
```

**输出**：
```
[L0:model]
  [L1:model]
  [L1:tools]
  [L1:model]
[L0:tools]
[L0:model]
```

---

### 2. 命名空间解析

```python
def parse_namespace(namespace: tuple) -> dict:
    """解析命名空间"""
    if not namespace:
        return {"level": 0, "path": "root"}

    path = []
    for ns in namespace:
        # 格式: "node_name:task_id"
        parts = ns.split(":")
        path.append({
            "node": parts[0],
            "task_id": parts[1] if len(parts) > 1 else None
        })

    return {
        "level": len(namespace),
        "path": path
    }

# 使用
for namespace, mode, data in agent.stream(
    input,
    stream_mode="updates",
    subgraphs=True
):
    info = parse_namespace(namespace)
    print(f"Level {info['level']}: {info['path']}")
```

---

### 3. 执行路径追踪

```python
class ExecutionTracker:
    def __init__(self):
        self.path = []
        self.node_counts = {}

    def track(self, chunk: dict):
        """追踪执行路径"""
        for node_name in chunk.keys():
            self.path.append(node_name)
            self.node_counts[node_name] = self.node_counts.get(node_name, 0) + 1

    def get_summary(self) -> dict:
        """获取执行摘要"""
        return {
            "total_steps": len(self.path),
            "execution_path": " → ".join(self.path),
            "node_counts": self.node_counts
        }

# 使用
tracker = ExecutionTracker()

for chunk in agent.stream(input, stream_mode="updates"):
    tracker.track(chunk)

summary = tracker.get_summary()
print(f"总步骤: {summary['total_steps']}")
print(f"执行路径: {summary['execution_path']}")
print(f"节点统计: {summary['node_counts']}")
```

**输出**：
```
总步骤: 3
执行路径: model → tools → model
节点统计: {'model': 2, 'tools': 1}
```

---

## 实战场景

### 场景 1：进度条显示

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def stream_with_progress(agent, input):
    """带进度条的流式执行"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("执行中...", total=None)

        for chunk in agent.stream(input, stream_mode="updates"):
            for node_name, data in chunk.items():
                progress.update(task, description=f"[{node_name}] 执行中...")

        progress.update(task, description="✅ 完成")

# 使用
stream_with_progress(agent, input)
```

---

### 场景 2：实时日志

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stream_with_logging(agent, input):
    """带日志的流式执行"""
    start_time = datetime.now()

    for i, chunk in enumerate(agent.stream(input, stream_mode="updates")):
        elapsed = (datetime.now() - start_time).total_seconds()

        for node_name, data in chunk.items():
            logger.info(
                f"Step {i+1} [{elapsed:.2f}s]: {node_name} "
                f"(messages: {len(data.get('messages', []))})"
            )

# 使用
stream_with_logging(agent, input)
```

**输出**：
```
INFO:__main__:Step 1 [0.52s]: model (messages: 1)
INFO:__main__:Step 2 [1.23s]: tools (messages: 1)
INFO:__main__:Step 3 [2.15s]: model (messages: 1)
```

---

### 场景 3：条件中断

```python
def stream_with_interrupt(agent, input, max_steps=10):
    """带中断条件的流式执行"""
    step_count = 0

    for chunk in agent.stream(input, stream_mode="updates"):
        step_count += 1

        # 检查中断条件
        if step_count > max_steps:
            print(f"\n⚠️ 超过最大步骤数 ({max_steps})，中断执行")
            break

        for node_name, data in chunk.items():
            print(f"步骤 {step_count}: {node_name}")

            # 检查错误条件
            if "error" in data:
                print(f"\n❌ 检测到错误，中断执行")
                break

# 使用
stream_with_interrupt(agent, input, max_steps=5)
```

---

### 场景 4：性能分析

```python
import time

class PerformanceAnalyzer:
    def __init__(self):
        self.node_times = {}
        self.current_node = None
        self.current_start = None

    def start_node(self, node_name: str):
        """开始计时"""
        self.current_node = node_name
        self.current_start = time.time()

    def end_node(self):
        """结束计时"""
        if self.current_node and self.current_start:
            elapsed = time.time() - self.current_start
            if self.current_node not in self.node_times:
                self.node_times[self.current_node] = []
            self.node_times[self.current_node].append(elapsed)

    def get_report(self) -> str:
        """生成性能报告"""
        report = ["性能分析报告", "="*50]

        for node, times in self.node_times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            report.append(
                f"{node}: 平均 {avg_time:.3f}s, "
                f"总计 {total_time:.3f}s, "
                f"调用 {len(times)} 次"
            )

        return "\n".join(report)

# 使用
analyzer = PerformanceAnalyzer()

for chunk in agent.stream(input, stream_mode="updates"):
    for node_name in chunk.keys():
        analyzer.start_node(node_name)
        # 处理数据
        analyzer.end_node()

print(analyzer.get_report())
```

**输出**：
```
性能分析报告
==================================================
model: 平均 0.523s, 总计 1.046s, 调用 2 次
tools: 平均 0.234s, 总计 0.234s, 调用 1 次
```

---

## 调试技巧

### 技巧 1：状态快照

```python
def capture_state_snapshots(agent, input):
    """捕获每个步骤的状态快照"""
    snapshots = []

    for chunk in agent.stream(input, stream_mode="updates"):
        snapshot = {
            "timestamp": time.time(),
            "nodes": list(chunk.keys()),
            "state": chunk
        }
        snapshots.append(snapshot)

    return snapshots

# 使用
snapshots = capture_state_snapshots(agent, input)

# 分析快照
for i, snap in enumerate(snapshots):
    print(f"\n快照 {i+1}:")
    print(f"  节点: {snap['nodes']}")
    print(f"  时间: {snap['timestamp']}")
```

---

### 技巧 2：差异对比

```python
def compare_states(prev_state: dict, curr_state: dict) -> dict:
    """对比两个状态的差异"""
    diff = {
        "added_nodes": set(curr_state.keys()) - set(prev_state.keys()),
        "removed_nodes": set(prev_state.keys()) - set(curr_state.keys()),
        "modified_nodes": set()
    }

    for node in set(prev_state.keys()) & set(curr_state.keys()):
        if prev_state[node] != curr_state[node]:
            diff["modified_nodes"].add(node)

    return diff

# 使用
prev_chunk = None

for chunk in agent.stream(input, stream_mode="updates"):
    if prev_chunk:
        diff = compare_states(prev_chunk, chunk)
        print(f"差异: {diff}")
    prev_chunk = chunk
```

---

### 技巧 3：可视化执行图

```python
def visualize_execution(agent, input):
    """可视化执行流程"""
    nodes = []
    edges = []
    prev_node = None

    for chunk in agent.stream(input, stream_mode="updates"):
        for node_name in chunk.keys():
            if node_name not in nodes:
                nodes.append(node_name)

            if prev_node:
                edge = (prev_node, node_name)
                if edge not in edges:
                    edges.append(edge)

            prev_node = node_name

    # 生成 Mermaid 图
    mermaid = ["graph TD"]
    for i, node in enumerate(nodes):
        mermaid.append(f"    {chr(65+i)}[{node}]")

    for i, (src, dst) in enumerate(edges):
        src_idx = nodes.index(src)
        dst_idx = nodes.index(dst)
        mermaid.append(f"    {chr(65+src_idx)} --> {chr(65+dst_idx)}")

    return "\n".join(mermaid)

# 使用
mermaid_graph = visualize_execution(agent, input)
print(mermaid_graph)
```

**输出**：
```
graph TD
    A[model]
    B[tools]
    A --> B
    B --> A
```

---

## 最佳实践

### 1. 选择性监控

```python
# ❌ 不推荐：监控所有节点
for chunk in agent.stream(input, stream_mode="updates"):
    for node, data in chunk.items():
        print(f"{node}: {data}")  # 输出过多

# ✅ 推荐：只监控关键节点
MONITORED_NODES = {"model", "tools"}

for chunk in agent.stream(input, stream_mode="updates"):
    for node, data in chunk.items():
        if node in MONITORED_NODES:
            print(f"{node}: 执行完成")
```

---

### 2. 异步处理

```python
import asyncio

async def async_stream_with_updates(agent, input):
    """异步流式处理"""
    async for chunk in agent.astream(input, stream_mode="updates"):
        # 异步处理每个更新
        await process_update_async(chunk)

# 使用
asyncio.run(async_stream_with_updates(agent, input))
```

---

### 3. 错误恢复

```python
def stream_with_recovery(agent, input, max_retries=3):
    """带错误恢复的流式执行"""
    retry_count = 0

    while retry_count < max_retries:
        try:
            for chunk in agent.stream(input, stream_mode="updates"):
                for node, data in chunk.items():
                    print(f"{node}: 执行完成")
            break  # 成功完成

        except Exception as e:
            retry_count += 1
            print(f"错误: {e}, 重试 {retry_count}/{max_retries}")

            if retry_count >= max_retries:
                print("达到最大重试次数，执行失败")
                raise

# 使用
stream_with_recovery(agent, input)
```

---

## 性能考虑

### 开销分析

```python
# updates 模式的开销
基础开销: ~2%
- 状态序列化: ~1%
- 回调触发: ~0.5%
- 队列管理: ~0.5%

# Subgraph 流式的额外开销
subgraphs=True: +1%
- 命名空间管理: ~0.5%
- 嵌套追踪: ~0.5%
```

### 优化建议

```python
# 1. 禁用不需要的 subgraph 流式
agent.stream(input, stream_mode="updates", subgraphs=False)  # 默认

# 2. 减少状态大小
# 只在状态中保留必要的数据

# 3. 批量处理更新
updates_buffer = []
for chunk in agent.stream(input, stream_mode="updates"):
    updates_buffer.append(chunk)
    if len(updates_buffer) >= 10:
        process_batch(updates_buffer)
        updates_buffer.clear()
```

---

## 总结

### 核心要点

1. **updates 模式追踪 Agent 执行图的每个节点**
2. **返回 `{node_name: state_update}` 格式的数据**
3. **支持 Subgraph 流式追踪嵌套 Agent**
4. **适用于监控、调试、进度追踪场景**
5. **性能开销 ~2%，Subgraph 额外 +1%**

### 使用场景

- ✅ 多步推理监控
- ✅ 工作流调试
- ✅ 进度追踪
- ✅ 性能分析
- ✅ 执行路径可视化

### 避免误区

- ❌ 不要监控所有节点（选择性监控）
- ❌ 不要总是启用 subgraphs（按需启用）
- ❌ 不要忽略性能开销（合理使用）

---

## 参考资源

- **官方文档**: https://docs.langchain.com/oss/python/langgraph/streaming
- **相关知识点**:
  - 03_核心概念_01 - Stream 模式详解
  - 03_核心概念_03 - LLM 令牌流式
  - 07_实战代码 - 完整代码示例

---

**版本**: LangChain 0.3.x (2025-2026)
**最后更新**: 2026-02-21
