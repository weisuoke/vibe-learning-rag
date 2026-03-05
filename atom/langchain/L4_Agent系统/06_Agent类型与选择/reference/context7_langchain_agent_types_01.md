---
type: context7_documentation
library: langchain
version: latest (2026-02-26)
fetched_at: 2026-03-02
knowledge_point: Agent类型与选择
context7_query: agent types OpenAI Functions ReAct Structured Chat create_agent
---

# Context7 文档：LangChain Agent 类型

## 文档来源
- 库名称: LangChain
- 版本: latest (更新至 2026-02-26)
- 官方文档链接: https://docs.langchain.com
- Context7 ID: /websites/langchain

## 关键信息提取

### 1. create_agent() - 2026 推荐 API

**统一的 Agent 创建接口**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)
```

**特点**:
- 自动根据模型能力选择最佳 Agent 类型
- 统一的接口，简化 Agent 创建
- 支持所有主流模型（OpenAI, Anthropic, 开源模型）

**来源**: https://docs.langchain.com/oss/python/langchain/sql-agent

### 2. OpenAI Functions Agent

**创建方式**:
```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
```

**使用场景**:
- 需要结构化函数调用
- 工具参数简单（单一输入或少量参数）
- 使用支持函数调用的模型（OpenAI, Anthropic）

**配合 AgentExecutor**:
```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=False,
)
```

**来源**: https://docs.langchain.com/oss/python/integrations/tools/semanticscholar

### 3. ReAct Agent

**创建方式**:
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0)
agent = create_agent(model, tools=[tool])
```

**推理模式**:
- Thought (思考): Agent 分析当前状态
- Action (行动): 选择并执行工具
- Observation (观察): 获取工具执行结果
- 循环直到得出最终答案

**使用场景**:
- 开源模型（不支持函数调用）
- 需要显式推理过程
- 调试和可解释性要求高

**来源**: https://docs.langchain.com/oss/python/integrations/tools/google_drive

### 4. 多 Agent 架构

**专业化 Agent 模式**:
```typescript
// GitHub 专家 Agent
const githubAgent = createAgent({
  model: llm,
  tools: [searchCode, searchIssues, searchPrs],
  systemPrompt: `You are a GitHub expert...`,
});

// Notion 专家 Agent
const notionAgent = createAgent({
  model: llm,
  tools: [searchNotion, getPage],
  systemPrompt: `You are a Notion expert...`,
});

// Slack 专家 Agent
const slackAgent = createAgent({
  model: llm,
  tools: [searchSlack, getThread],
  systemPrompt: `You are a Slack expert...`,
});
```

**设计原则**:
- 每个 Agent 专注于特定领域
- 通过 system prompt 定义专业知识
- 工具集与领域匹配

**来源**: https://docs.langchain.com/oss/javascript/langchain/multi-agent/router-knowledge-base

### 5. Pandas DataFrame Agent - 类型选择示例

**使用 OpenAI Functions**:
```python
# 方法 1: 使用 OPENAI_FUNCTIONS agent type
agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type=AgentType.OPENAI_FUNCTIONS,  # 推荐
    verbose=True
)
```

**对比 ZERO_SHOT_REACT_DESCRIPTION**:
```python
# 方法 2: 使用 ReAct agent type
agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

**选择建议**:
- OpenAI Functions: 更可靠、更高效、推荐用于生产环境
- ReAct: 更灵活、适合调试和开源模型

**来源**: https://docs.langchain.com/oss/python/integrations/tools/pandas

### 6. Agent 架构与工具链

**核心概念**:
> "LangChain enables the creation of agents that can chain together multiple tools and language models to accomplish complex tasks. The OpenAI Functions Agent is a specialized agent implementation that leverages OpenAI's function calling capabilities to dynamically select and execute tools based on user input."

**关键能力**:
- 动态工具选择
- 多步骤工作流
- 外部数据源访问
- 专业化操作执行

**来源**: https://docs.langchain.com/oss/python/integrations/tools/asknews

## 实践建议

### 1. 2026 年推荐方式

**优先级**:
1. `create_agent()` - 最简单，自动选择
2. `create_openai_functions_agent()` - 明确使用函数调用
3. `create_react_agent()` - 开源模型或需要显式推理

### 2. 模型兼容性

| Agent 类型 | OpenAI | Anthropic | 开源模型 |
|-----------|--------|-----------|---------|
| OpenAI Functions | ✅ | ✅ | ❌ |
| ReAct | ✅ | ✅ | ✅ |
| Structured Chat | ✅ | ✅ | ⚠️ |

### 3. 工具复杂度匹配

- **简单工具** (1-2 参数): OpenAI Functions
- **复杂工具** (多输入/嵌套): Structured Chat
- **任何工具** + 开源模型: ReAct

## 总结

1. **create_agent()** 是 2026 年推荐的统一 API
2. **OpenAI Functions** 适合生产环境，可靠性高
3. **ReAct** 适合开源模型和调试场景
4. **Structured Chat** 适合复杂工具参数
5. **多 Agent 架构** 通过专业化提升性能
