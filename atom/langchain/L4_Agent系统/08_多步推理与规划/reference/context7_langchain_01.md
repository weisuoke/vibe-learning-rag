---
type: context7_documentation
library: LangChain
version: latest (2026)
fetched_at: 2026-03-06
knowledge_point: 08_多步推理与规划
context7_query: deep agents planning decomposition write_todos middleware subagent
---

# Context7 文档：LangChain 规划与任务分解

## 文档来源
- 库名称：LangChain
- Library ID: /websites/langchain
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. TodoListMiddleware (任务规划中间件)

LangChain 2026 新增的 `TodoListMiddleware`，为 Agent 提供自动任务规划和跟踪能力：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()]
)
```

**功能：**
- 自动提供 `write_todos` 工具
- 系统提示词引导 Agent 进行任务规划
- 适用于复杂、多步骤或长时间运行的操作
- 提高进度可见性和任务协调能力

### 2. Deep Agents SDK (深度代理)

Deep Agents SDK 构建在 LangGraph 之上，专为复杂多步任务设计：

**核心能力：**
- **规划能力**：内置任务分解和规划工具
- **文件系统**：上下文管理和长期记忆存储
- **子代理派生**：能够动态创建子代理处理子任务
- **模块化中间件**：TodoListMiddleware、FilesystemMiddleware、SubAgentMiddleware

**核心特性：**
- `write_todos` 工具：将复杂任务分解为离散步骤
- 进度跟踪：追踪每个步骤的完成状态
- 动态适应：根据新信息调整计划
- 子代理协作：复杂任务可分发给专门的子代理

### 3. RAG Agent 多步推理

RAG Agent 展示了多步推理过程：
1. 生成查询搜索标准方法
2. 收到答案后生成第二个查询
3. 收到所有必要上下文后回答原始问题

这是 Plan-and-Execute 模式在 RAG 场景的典型应用。

### 4. 中间件架构

Deep Agents 采用模块化中间件架构：
- **TodoListMiddleware**: 任务规划和跟踪
- **FilesystemMiddleware**: 文件系统和长期记忆
- **SubAgentMiddleware**: 子代理管理
- 可按需组合使用

```python
# 完整的 Deep Agent 自动包含所有中间件
from langchain.agents import create_deep_agent

agent = create_deep_agent(
    model="gpt-4.1",
    tools=[...],
    # 自动附加 TodoListMiddleware, FilesystemMiddleware, SubAgentMiddleware
)
```
