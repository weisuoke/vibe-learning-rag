---
type: context7_documentation
library: langchain, langgraph
fetched_at: 2026-03-06
knowledge_point: 10_Agent最佳实践
context7_query: agent best practices, multi-agent architecture, LangGraph vs AgentExecutor
sources:
  - /websites/langchain (Benchmark: 88.8, Snippets: 20161)
  - /websites/langchain_oss_python_langgraph (Benchmark: 86.9, Snippets: 900)
---

# Context7 文档：LangChain Agent Best Practices

## 关键信息提取

### 1. Agent 设计模式

#### 1.1 多代理模式概览 (Multi-agent Patterns)

> Source: https://docs.langchain.com/oss/python/langchain/multi-agent/index

LangChain 官方定义了五种核心多代理模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Subagents（子代理）** | 主代理将子代理包装为工具，协调所有路由决策 | 需要精细控制子代理输入输出时 |
| **Handoffs（切换）** | 基于状态动态切换行为，工具调用更新状态变量触发路由或配置更改 | 对话式系统中需要动态切换代理角色 |
| **Skills（技能）** | 按需加载专门的提示词和知识，单代理保持控制权 | 单代理需要在不同专业领域间切换 |
| **Router（路由器）** | 对输入进行分类并导向专门代理，综合结果为统一响应 | 输入类型多样，需要不同专家处理 |
| **Custom Workflow（自定义工作流）** | 使用 LangGraph 构建定制执行流程，混合确定性逻辑和代理行为 | 复杂业务流程，需要精确控制 |

#### 1.2 子代理工具模式 (Tool per Agent Pattern)

> Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

两种暴露子代理为工具的方式：

- **Tool per Agent（每个代理一个工具）**: 对每个子代理的输入输出有精细控制，自定义程度高，但设置更多
- **Single Dispatch Tool（单一分发工具）**: 适合大量代理和分布式团队，遵循约定优于配置的原则，组合更简单

**最佳实践 - 最小化工具集：**

```python
# Good: Focused tool set - 每个子代理只给必需的工具
email_agent = {
    "name": "email-sender",
    "tools": [send_email, validate_email],  # Only email-related
}

# Bad: Too many tools - 工具太多导致不聚焦
email_agent = {
    "name": "email-sender",
    "tools": [send_email, web_search, database_query, file_upload],  # Unfocused
}
```

#### 1.3 Orchestrator-Worker 工作流模式

> Source: https://docs.langchain.com/oss/python/langgraph/workflows-agents

使用 LangGraph 的 Functional API 实现编排者-工作者模式：

```python
from typing import List

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

@task
def orchestrator(topic: str):
    """Orchestrator that generates a plan for the report"""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {topic}"),
    ])
    return report_sections.sections

@task
def llm_call(section: Section):
    """Worker writes a section of the report"""
    result = llm.invoke([
        SystemMessage(content="Write a report section."),
        HumanMessage(content=f"Section: {section.name}, Description: {section.description}"),
    ])
    return result.content

@task
def synthesizer(completed_sections: list[str]):
    """Synthesize full report from sections"""
    return "\n\n---\n\n".join(completed_sections)

@entrypoint()
def orchestrator_worker(topic: str):
    sections = orchestrator(topic).result()
    section_futures = [llm_call(section) for section in sections]
    final_report = synthesizer(
        [section_fut.result() for section_fut in section_futures]
    ).result()
    return final_report
```

---

### 2. LangGraph vs AgentExecutor

> Source: https://docs.langchain.com/oss/python/reference/langchain-python
> Source: https://docs.langchain.com/oss/python/migrate/langgraph-v1

#### 2.1 官方推荐

- **LangChain agents** 适合快速构建代理和自主应用
- **LangGraph** 适合更高级需求：确定性与代理工作流的混合、高度自定义、精确延迟控制
- **LangChain agents 底层基于 LangGraph**，提供持久执行、流式传输、人机协作、持久化等功能

#### 2.2 迁移路径

从 `create_react_agent` (LangGraph v0) 迁移到 `create_agent` (LangChain v1)：

```python
# 新方式 (LangChain v1) - 推荐
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant.",
)

# 旧方式 (LangGraph v0) - 已弃用
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model,
    tools,
    prompt="You are a helpful assistant.",
)
```

关键变化：
- `create_react_agent` -> `create_agent`
- `prompt` 参数 -> `system_prompt` 参数
- 新 API 引入了灵活的中间件系统 (middleware system)

---

### 3. Agent 工作流构建 (LangGraph)

#### 3.1 Graph API 构建代理工作流

> Source: https://docs.langchain.com/oss/python/langgraph/workflows-agents

核心模式：LLM 调用节点 + 工具执行节点 + 条件边路由

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import Literal

# 节点定义
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant.")]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 条件边：决定是否继续调用工具
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# 构建工作流
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
```

#### 3.2 Functional API 构建代理

> Source: https://docs.langchain.com/oss/python/langgraph/quickstart

使用 `@entrypoint()` 和 `@task` 装饰器的函数式 API：

```python
@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break
        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages
```

#### 3.3 Agentic RAG 工作流

> Source: https://docs.langchain.com/oss/python/langgraph/agentic-rag

将检索、问题重写、文档评分等组合为代理式 RAG：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# 定义节点
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# 定义边
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
```

#### 3.4 SQL Agent 工作流

> Source: https://docs.langchain.com/oss/python/langgraph/sql-agent

复杂多步骤代理工作流示例：

```python
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()
```

---

### 4. Agent 测试策略

> Source: https://docs.langchain.com/langsmith/test-react-agent-pytest

#### 4.1 关键测试场景

1. **工具调用功能验证**: 验证代理的工具调用能力是否正确
2. **领域边界测试**: 确认代理是否会忽略领域外的不相关问题
3. **多步骤复杂查询**: 验证代理能否处理需要组合使用多个工具的复杂查询

#### 4.2 测试建议

- 在部署到生产环境前，运行这些测试以确保代理行为符合预期
- 使用 LangSmith 进行跟踪和评估
- 使用 pytest 组织测试用例

---

### 5. 工具设计最佳实践

> Source: https://docs.langchain.com/oss/python/deepagents/subagents

#### 5.1 核心原则

- **最小权限原则**: 每个子代理/工具只获得完成特定任务所需的最少工具
- **聚焦职责**: 工具集应与代理的具体职责紧密对齐
- **安全考虑**: 限制不必要的工具访问，防止意外复杂性

#### 5.2 工具组织模式

- **Tool per Agent**: 每个子代理包装为独立工具，精细控制输入输出
- **Single Dispatch Tool**: 统一分发入口，约定优于配置
- 选择取决于代理数量、团队结构和定制需求

---

## LangGraph 核心特性总结

| 特性 | 说明 |
|------|------|
| **Durable Execution（持久执行）** | 长时间运行的代理可以持久化状态 |
| **Streaming（流式传输）** | 支持实时流式输出 |
| **Human-in-the-loop（人机协作）** | 支持人工审批和干预 |
| **Persistence（持久化）** | 状态持久化和恢复 |
| **Graph API** | 使用 StateGraph 声明式构建工作流 |
| **Functional API** | 使用 @entrypoint/@task 装饰器函数式构建 |
| **Conditional Edges** | 条件路由支持动态工作流 |

---

## 关键结论

1. **LangChain agents 已建立在 LangGraph 之上** - 新项目推荐直接使用 LangChain v1 的 `create_agent`
2. **AgentExecutor 已弃用** - 迁移到 LangGraph/LangChain v1
3. **多代理设计要遵循最小工具集原则** - 每个子代理只给必需的工具
4. **五种核心多代理模式** - Subagents, Handoffs, Skills, Router, Custom Workflow
5. **测试覆盖三个维度** - 工具调用、领域边界、多步骤复杂查询
6. **Graph API vs Functional API** - Graph API 适合复杂工作流，Functional API 适合简单代理逻辑
