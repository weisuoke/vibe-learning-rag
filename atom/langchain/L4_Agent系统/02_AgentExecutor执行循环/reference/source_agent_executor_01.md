---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/agents/agent.py
  - libs/langchain/langchain_classic/agents/agent_iterator.py
  - libs/core/langchain_core/agents.py
analyzed_at: 2026-02-28
knowledge_point: 02_AgentExecutor执行循环
---

# 源码分析：AgentExecutor 执行循环核心实现

## 分析的文件

- `libs/langchain/langchain_classic/agents/agent.py` (61K) - AgentExecutor 主类
- `libs/langchain/langchain_classic/agents/agent_iterator.py` (17K) - 迭代器模式
- `libs/core/langchain_core/agents.py` (8.2K) - 核心数据类型

## 关键发现

### 1. 核心数据类型（langchain_core/agents.py）

三个核心数据类：
- **AgentAction**: 表示 Agent 请求执行的动作（tool + tool_input + log）
- **AgentFinish**: 表示 Agent 达到停止条件的最终返回值（return_values + log）
- **AgentStep**: AgentAction 执行后的结果（action + observation）

基本工作流程（文档注释）：
1. 给定 prompt，Agent 使用 LLM 请求一个动作
2. Agent 执行动作，接收 observation
3. Agent 将 observation 返回给 LLM，生成下一个动作
4. 当 Agent 达到停止条件时，返回最终值

### 2. AgentExecutor 类（agent.py:1012）

关键属性：
- `agent`: BaseSingleActionAgent | BaseMultiActionAgent | Runnable
- `tools`: Sequence[BaseTool]
- `max_iterations`: int = 15（默认最多15次迭代）
- `max_execution_time`: float | None = None
- `early_stopping_method`: str = "force"（force 或 generate）
- `handle_parsing_errors`: bool | str | Callable = False
- `trim_intermediate_steps`: int | Callable = -1
- `return_intermediate_steps`: bool = False

### 3. 主执行循环（agent.py:1570 _call方法）

```python
def _call(self, inputs, run_manager=None):
    name_to_tool_map = {tool.name: tool for tool in self.tools}
    color_mapping = get_color_mapping(...)
    intermediate_steps = []
    iterations = 0
    time_elapsed = 0.0
    start_time = time.time()

    while self._should_continue(iterations, time_elapsed):
        next_step_output = self._take_next_step(
            name_to_tool_map, color_mapping,
            inputs, intermediate_steps, run_manager
        )
        if isinstance(next_step_output, AgentFinish):
            return self._return(next_step_output, intermediate_steps, run_manager)

        intermediate_steps.extend(next_step_output)
        if len(next_step_output) == 1:
            tool_return = self._get_tool_return(next_step_output[0])
            if tool_return is not None:
                return self._return(tool_return, intermediate_steps, run_manager)

        iterations += 1
        time_elapsed = time.time() - start_time

    # 超出限制，执行 early stopping
    output = self._action_agent.return_stopped_response(
        self.early_stopping_method, intermediate_steps, **inputs
    )
    return self._return(output, intermediate_steps, run_manager)
```

### 4. 单步执行（agent.py:1301 _iter_next_step）

每一步的逻辑：
1. 准备 intermediate_steps（可能裁剪）
2. 调用 agent.plan() 获取 LLM 决策
3. 如果解析失败（OutputParserException）：
   - handle_parsing_errors=True: 将错误作为 observation 反馈给 LLM
   - handle_parsing_errors=False: 抛出异常
   - handle_parsing_errors=str: 使用自定义字符串
   - handle_parsing_errors=callable: 调用函数处理
4. 如果返回 AgentFinish: 结束循环
5. 如果返回 AgentAction: 执行工具，获取 observation

### 5. 工具执行（agent.py:1380 _perform_agent_action）

- 在 name_to_tool_map 中查找工具
- 找到: 执行 tool.run()，获取 observation
- 未找到: 使用 InvalidTool 返回错误信息
- 支持 return_direct: 工具可以直接返回最终结果

### 6. 循环控制（agent.py:1235 _should_continue）

```python
def _should_continue(self, iterations, time_elapsed):
    if self.max_iterations is not None and iterations >= self.max_iterations:
        return False
    return self.max_execution_time is None or time_elapsed < self.max_execution_time
```

### 7. Early Stopping 策略

- `"force"`: 直接返回固定字符串 "Agent stopped due to iteration limit or time limit."
- `"generate"`: 调用 LLM 基于已有步骤生成最终答案

### 8. 迭代器模式（agent_iterator.py）

AgentExecutorIterator 提供逐步迭代能力：
- 支持 yield_actions: 逐步产出 AgentAction 和 AgentStep
- 支持同步和异步迭代
- 内部复用 AgentExecutor 的 _iter_next_step 和 _should_continue

### 9. 异步执行（agent.py:1624 _acall）

- 使用 asyncio_timeout 包装整个循环
- 工具执行使用 asyncio.gather 并发执行多个工具
- TimeoutError 时执行 early stopping

## 代码片段

### 解析错误处理
```python
except OutputParserException as e:
    if isinstance(self.handle_parsing_errors, bool):
        raise_error = not self.handle_parsing_errors
    # ... 多种处理方式
    output = AgentAction("_Exception", observation, text)
    observation = ExceptionTool().run(output.tool_input, ...)
    yield AgentStep(action=output, observation=observation)
```

### 工具查找与执行
```python
if agent_action.tool in name_to_tool_map:
    tool = name_to_tool_map[agent_action.tool]
    observation = tool.run(agent_action.tool_input, ...)
else:
    observation = InvalidTool().run({
        "requested_tool_name": agent_action.tool,
        "available_tool_names": list(name_to_tool_map.keys()),
    }, ...)
```
