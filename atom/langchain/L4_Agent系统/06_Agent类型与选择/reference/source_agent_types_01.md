---
type: source_code_analysis
source: sourcecode/langchain/libs/langchain/langchain_classic/agents/agent_types.py
analyzed_files: [agent_types.py]
analyzed_at: 2026-03-02
knowledge_point: Agent类型与选择
---

# 源码分析：Agent 类型枚举定义

## 分析的文件
- `sourcecode/langchain/libs/langchain/langchain_classic/agents/agent_types.py` - Agent 类型枚举定义

## 关键发现

### 1. Deprecation 警告

**重要**: `AgentType` 枚举已在 0.1.0 版本标记为 deprecated，将在 1.0 版本移除。

```python
@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class AgentType(str, Enum):
    """An enum for agent types."""
```

**影响**:
- 不应在新代码中使用 `AgentType` 枚举
- 应迁移到 `create_*_agent()` 函数或 `create_agent()` 统一 API
- 现有代码需要在 1.0 版本前完成迁移

### 2. 支持的 Agent 类型

#### ReAct 系列

**ZERO_SHOT_REACT_DESCRIPTION**
```python
ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
```
- 描述: "A zero shot agent that does a reasoning step before acting."
- 特点: 在执行动作前进行推理步骤
- 适用: 通用场景，任何 LLM

**REACT_DOCSTORE**
```python
REACT_DOCSTORE = "react-docstore"
```
- 描述: "A zero shot agent that does a reasoning step before acting. This agent has access to a document store that allows it to look up relevant information to answering the question."
- 特点: 带文档存储的 ReAct Agent
- 适用: 需要查询文档的场景

**CHAT_ZERO_SHOT_REACT_DESCRIPTION**
```python
CHAT_ZERO_SHOT_REACT_DESCRIPTION = "chat-zero-shot-react-description"
```
- 描述: "A zero shot agent that does a reasoning step before acting. This agent is designed to be used in conjunction"
- 特点: 针对聊天模型优化的 ReAct
- 适用: 对话场景

#### Conversational 系列

**CONVERSATIONAL_REACT_DESCRIPTION**
```python
CONVERSATIONAL_REACT_DESCRIPTION = "conversational-react-description"
```
- 特点: 带对话记忆的 ReAct Agent
- 适用: 多轮对话场景

**CHAT_CONVERSATIONAL_REACT_DESCRIPTION**
```python
CHAT_CONVERSATIONAL_REACT_DESCRIPTION = "chat-conversational-react-description"
```
- 特点: 聊天模型 + 对话记忆
- 适用: 聊天机器人场景

#### Structured Chat

**STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION**
```python
STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION = (
    "structured-chat-zero-shot-react-description"
)
```
- 描述: "An zero-shot react agent optimized for chat models. This agent is capable of invoking tools that have multiple inputs."
- 特点: 支持多输入工具
- 适用: 工具参数复杂的场景

#### OpenAI Functions 系列

**OPENAI_FUNCTIONS**
```python
OPENAI_FUNCTIONS = "openai-functions"
```
- 描述: "An agent optimized for using open AI functions."
- 特点: 使用 OpenAI 函数调用
- 适用: OpenAI 模型 + 简单工具

**OPENAI_MULTI_FUNCTIONS**
```python
OPENAI_MULTI_FUNCTIONS = "openai-multi-functions"
```
- 特点: 支持多函数调用
- 适用: 需要同时调用多个工具

#### 其他类型

**SELF_ASK_WITH_SEARCH**
```python
SELF_ASK_WITH_SEARCH = "self-ask-with-search"
```
- 描述: "An agent that breaks down a complex question into a series of simpler questions. This agent uses a search tool to look up answers to the simpler questions in order to answer the original complex question."
- 特点: 问题分解 + 搜索
- 适用: 复杂问题需要分步解决

## 架构洞察

### 1. 命名模式

Agent 类型命名遵循以下模式:
- `{CONTEXT}_{SHOT}_{REASONING}_{DESCRIPTION}`
- 例如: `CHAT_ZERO_SHOT_REACT_DESCRIPTION`
  - CHAT: 针对聊天模型
  - ZERO_SHOT: 零样本学习
  - REACT: 推理-行动模式
  - DESCRIPTION: 基于描述

### 2. 演化趋势

从枚举定义可以看出 Agent 类型的演化:
1. **早期**: 简单 ReAct (ZERO_SHOT_REACT_DESCRIPTION)
2. **中期**: 针对特定场景优化 (CONVERSATIONAL, CHAT)
3. **现代**: 函数调用优化 (OPENAI_FUNCTIONS)
4. **未来**: 统一 API (create_agent) 取代枚举

### 3. 设计权衡

**为什么弃用枚举?**
- ❌ 枚举限制了扩展性（新增类型需要修改核心代码）
- ❌ 字符串枚举不够类型安全
- ❌ 与现代 Python 类型提示不兼容
- ✅ 函数式 API 更灵活（create_*_agent）
- ✅ 更好的文档和类型提示

## 迁移建议

### 从枚举迁移到函数

**旧方式 (Deprecated)**:
```python
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
```

**新方式 (推荐)**:
```python
from langchain.agents import create_openai_functions_agent, AgentExecutor

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**最新方式 (2026)**:
```python
from langchain.agents import create_agent

agent = create_agent(llm, tools, system_prompt="...")
```

## 总结

1. **AgentType 枚举已弃用**，不应在新代码中使用
2. **三大主流类型**: OpenAI Functions, ReAct, Structured Chat
3. **迁移路径**: 枚举 → create_*_agent() → create_agent()
4. **设计理念**: 从静态枚举到动态函数，提升灵活性和可扩展性
