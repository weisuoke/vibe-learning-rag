---
type: context7_documentation
library: LangGraph (Python)
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 11_状态调试技巧
context7_query: state debugging stream_mode debug get_state observability LangSmith
---

# Context7 文档：LangGraph 状态调试

## 文档来源
- 库名称：LangGraph (Python)
- Context7 ID: /websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph

## 关键信息提取

### 1. Debug Stream Mode

使用 `stream_mode="debug"` 可以在图执行过程中捕获最详细的信息，包括节点名称和每步后的完整状态。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
):
    print(chunk)
```

### 2. 获取状态快照 (get_state)

```python
# 获取最新状态快照
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# 获取特定检查点的状态
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

返回 `StateSnapshot` 对象，包含：
- 消息历史
- 配置信息
- 元数据
- 时间戳

### 3. 状态历史 (get_state_history)

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

返回 `StateSnapshot` 对象列表，展示线程状态的演变过程。

### 4. 流式模式对比

| 模式 | 输出内容 | 适用场景 |
|------|----------|----------|
| `values` | 每步后完整状态 | 查看状态全貌 |
| `updates` | 仅节点名和更新 | 追踪增量变化 |
| `checkpoints` | 检查点事件 | 状态持久化调试 |
| `tasks` | 任务开始/完成事件 | 执行流程调试 |
| `debug` | checkpoints + tasks | 全面调试 |
| `messages` | LLM 消息逐 token | LLM 输出调试 |
| `custom` | 自定义数据 | 特定场景调试 |

### 5. LangSmith 可观测性

启用 LangSmith 追踪：
```shell
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

添加元数据和标签：
```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a welcome email"}]},
    config={
        "tags": ["production", "email-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)
```

### 6. 自定义流式数据

```python
from langgraph.config import get_stream_writer

def node(state: State):
    writer = get_stream_writer()
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}
```
