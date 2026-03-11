---
type: fetched_content
source: https://python.langchain.com/docs/how_to/debugging/
title: LangChain Debugging Guide (2026)
fetched_at: 2026-03-06
status: redirected_to_overview
author: LangChain
knowledge_point: 09_Agent调试与优化
fetch_tool: grok-mcp
---

# LangChain 调试指南（2026 最新版）

## 重要变更说明

截至 2026 年，LangChain 官方文档的调试页面已从传统的 `set_debug`/`set_verbose` 方式转向以 **LangSmith** 为核心的调试方案。原始 URL (`/docs/how_to/debugging/`) 现重定向到 LangChain 概览页面。

## 当前官方推荐的调试方式

### 1. LangSmith 追踪（首选）

> "Debug with LangSmith: Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics."

- 可视化追踪执行路径
- 捕获状态转换
- 提供详细运行时指标

### 2. create_agent API (2026 新)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

### 3. Deep Agents

LangChain 推荐使用 Deep Agents 作为"自带电池"的解决方案，包含：
- 自动压缩长对话
- 虚拟文件系统
- 子代理生成以管理和隔离上下文

### 4. Python Logging 配置

```python
import logging

# 配置基本日志
logging.basicConfig(level=logging.WARNING)

# 增加 LangSmith 详细程度
import langsmith
langsmith_logger = logging.getLogger("langsmith")
langsmith_logger.setLevel(level=logging.DEBUG)
```

## 传统调试方法（仍可用但不推荐）

### set_debug() 和 set_verbose()
```python
from langchain_core.globals import set_debug, set_verbose

set_debug(True)      # 启用 debug 模式
set_verbose(True)    # 启用 verbose 日志
```

### StdOutCallbackHandler
```python
from langchain_core.callbacks import StdOutCallbackHandler

chain.invoke(input, {"callbacks": [StdOutCallbackHandler()]})
```
