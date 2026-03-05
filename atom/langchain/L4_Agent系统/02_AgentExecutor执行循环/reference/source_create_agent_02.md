---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain_v1/langchain/agents/factory.py
  - libs/langchain_v1/langchain/agents/middleware/_retry.py
  - libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py
analyzed_at: 2026-02-28
knowledge_point: 02_AgentExecutor执行循环
---

# 源码分析：v1 create_agent 与中间件系统（AgentExecutor 的现代替代）

## 分析的文件

- `libs/langchain_v1/langchain/agents/factory.py` (71K) - create_agent 工厂函数
- `libs/langchain_v1/langchain/agents/middleware/_retry.py` (3.7K) - 重试逻辑
- `libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py` (18K) - 工具调用限制

## 关键发现

### 1. create_agent 函数签名（factory.py:658）

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
```

### 2. create_agent 执行循环描述

官方文档注释：
> The agent node calls the language model with the messages list (after applying
> the system prompt). If the resulting AIMessage contains tool_calls, the graph
> will then call the tools. The tools node executes the tools and adds the
> responses to the messages list as ToolMessage objects. The agent node then calls
> the language model again. The process repeats until no more tool_calls are present
> in the response.

### 3. 与 AgentExecutor 的对比

| 方面 | AgentExecutor | create_agent |
|------|--------------|-------------|
| 底层 | Chain (while 循环) | LangGraph StateGraph |
| 循环控制 | max_iterations, max_execution_time | ToolCallLimitMiddleware |
| 错误处理 | handle_parsing_errors | ModelRetryMiddleware, ToolRetryMiddleware |
| 工具验证 | validate_tools | 中间件 wrap_tool_call |
| 状态 | intermediate_steps 列表 | AgentState TypedDict |
| 持久化 | 无内置 | Checkpointer |
| 扩展 | 子类化 | 中间件组合 |

### 4. 重试中间件（_retry.py）

- RetryOn: 支持异常类型元组或自定义判断函数
- OnFailure: "error"（抛出）、"continue"（注入错误消息继续）、自定义函数
- calculate_delay: 指数退避 + 可选 jitter（±25%）
- validate_retry_params: 参数验证

### 5. 工具调用限制中间件（tool_call_limit.py）

- ToolCallLimitState: 跟踪 thread 和 run 级别的工具调用计数
- ExitBehavior: "continue"（阻止超限工具）、"error"（抛出异常）、"end"（立即停止）
- 支持按工具名称和全局限制
- 替代了 AgentExecutor 的 max_iterations

### 6. 中间件系统架构

中间件可以拦截和修改 agent 行为的各个阶段：
- wrap_model_call: 拦截 LLM 调用
- wrap_tool_call: 拦截工具执行
- 支持动态工具注入
- 支持结构化输出
