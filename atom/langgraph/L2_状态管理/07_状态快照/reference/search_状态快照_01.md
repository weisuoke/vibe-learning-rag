---
type: search_result
search_query: LangGraph state snapshot get_state time travel debugging 2025 2026
search_engine: web_search
searched_at: 2026-02-27
knowledge_point: 07_状态快照
---

# 搜索结果：LangGraph 状态快照与时间旅行

## 搜索摘要

搜索到丰富的官方教程和社区资源，涵盖状态快照的使用、时间旅行调试、子图状态管理等。

## 相关链接

- [Use time travel (官方 How-to)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/) - 完整的时间旅行操作指南
- [Time Travel 概念文档](https://langchain-ai.github.io/langgraphjs/concepts/time-travel/) - Replay 和 Fork 的概念解释
- [Persistence 概念文档](https://langchain-ai.github.io/langgraph/concepts/persistence/) - Checkpointer 持久化层概述
- [Time Travel 入门教程](https://langchain-ai.github.io/langgraph/tutorials/get-started/6-time-travel/) - 聊天机器人中的时间旅行
- [Checkpoint-Based State Replay 实践](https://scour.ing/@abnv/p/https:/dev.to/sreeni5018/debugging-non-deterministic-llm-agents-implementing-checkpoint-based-state-replay-with-langgraph-5171) - 社区实践案例
- [子图状态管理](https://langchain-ai.github.io/langgraph/how-tos/subgraphs-manage-state/) - 子图中的状态查看与更新
- [Server API 时间旅行](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_time_travel/) - 通过 Server API 实现时间旅行

## 关键信息提取

### 时间旅行两大操作
1. **Replaying 🔁** - 从过去的 checkpoint 重放执行
2. **Forking 🔀** - 从过去的 checkpoint 分叉，修改状态后探索替代路径

### 核心 API
- `graph.get_state(config)` - 获取当前/指定 checkpoint 的状态快照
- `graph.get_state_history(config)` - 获取完整执行历史
- `graph.update_state(config, values)` - 修改状态创建新 checkpoint
- `graph.invoke(None, config)` - 从指定 checkpoint 恢复执行

### 社区实践要点
- LLM 天然非确定性，时间旅行是调试 Agent 的关键工具
- Checkpoint 在每个 super-step 自动保存
- 子图状态可以通过 `subgraphs=True` 递归获取
