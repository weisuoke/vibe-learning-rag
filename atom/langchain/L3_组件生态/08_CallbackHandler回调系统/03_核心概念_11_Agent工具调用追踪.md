# Agent工具调用追踪

## 1. 【30字核心】

**Agent工具调用追踪是通过重写on_tool_start/end方法监控Agent推理过程和工具执行的技术，实现完整的Agent行为可观测性。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Agent工具调用追踪的第一性原理

#### 1. 最基础的定义

**Agent工具调用追踪 = 捕获Agent决策过程中的工具选择、参数传递和执行结果**

仅此而已！没有更基础的了。

#### 2. 为什么需要Agent工具调用追踪？

**核心问题：Agent是黑盒系统，不追踪就无法理解其推理过程和调试失败原因**

Agent的执行流程：
```
用户输入 → Agent推理 → 选择工具 → 传递参数 → 执行工具 → 获取结果 → 继续推理 → 最终输出
```

在这个流程中，如果没有追踪：
- 不知道Agent为什么选择某个工具
- 不知道传递了什么参数
- 不知道工具返回了什么结果
- 无法调试Agent失败的原因
- 无法优化Agent的性能

#### 3. Agent工具调用追踪的三层价值

##### 价值1：可调试性

**问题**：Agent失败时，不知道是哪个环节出错

**解决**：
```python
class DebugCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        print(f"[Tool Start] {tool_name}")
        print(f"[Input] {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"[Tool End] Output: {output}")

    def on_tool_error(self, error, **kwargs):
        print(f"[Tool Error] {error}")

# 使用
agent_executor.run("query", callbacks=[DebugCallbackHandler()])
```

通过追踪，可以清楚看到：
- Agent选择了哪个工具
- 传递了什么参数
- 工具返回了什么
- 哪里出错了

##### 价值2：性能优化

**问题**：不知道哪个工具调用最慢，无法优化

**解决**：
```python
import time

class PerformanceCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tool_times = {}
        self.current_tool = None
        self.start_time = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_tool = serialized.get("name", "unknown")
        self.start_time = time.time()

    def on_tool_end(self, output, **kwargs):
        if self.current_tool and self.start_time:
            elapsed = time.time() - self.start_time
            if self.current_tool not in self.tool_times:
                self.tool_times[self.current_tool] = []
            self.tool_times[self.current_tool].append(elapsed)
            print(f"[{self.current_tool}] took {elapsed:.2f}s")

    def get_stats(self):
        return {
            tool: {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times)
            }
            for tool, times in self.tool_times.items()
        }
```

通过追踪，可以：
- 识别慢工具
- 统计调用频率
- 优化工具实现
- 减少不必要的调用

##### 价值3：成本控制

**问题**：不知道Agent调用了多少次LLM和工具，成本不可控

**解决**：
```python
class CostTrackingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0
        self.total_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1

    def on_llm_end(self, response, **kwargs):
        # 统计token使用
        if hasattr(response, 'llm_output'):
            tokens = response.llm_output.get('token_usage', {})
            self.total_tokens += tokens.get('total_tokens', 0)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_calls += 1

    def get_cost_summary(self):
        # 假设 gpt-4o-mini: $0.15/1M input, $0.60/1M output
        estimated_cost = (self.total_tokens / 1_000_000) * 0.375  # 平均价格
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": estimated_cost
        }
```

通过追踪，可以：
- 统计LLM调用次数
- 统计工具调用次数
- 计算token使用量
- 估算成本

#### 4. 从第一性原理推导Agent追踪最佳实践

**推理链：**
```
1. Agent是多步推理系统，每步都可能失败
   ↓
2. 需要追踪每个工具调用的输入输出
   ↓
3. 重写on_tool_start/end方法捕获工具调用
   ↓
4. 记录工具名称、参数、结果、耗时
   ↓
5. 将追踪数据发送到可观测性平台
   ↓
6. 在dashboard中分析Agent行为
   ↓
最终：实现Agent的完整可观测性和可调试性
```

#### 5. 一句话总结第一性原理

**Agent工具调用追踪的本质是捕获Agent推理过程中的每个决策点和执行结果，使黑盒系统变得透明可调试。**

---

## 3. 【核心概念】

### 核心概念1：on_tool_start方法重写

**on_tool_start在工具开始执行前触发，可以捕获工具名称和输入参数**

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

class ToolTrackingCallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """工具开始执行时调用"""
        # 获取工具名称
        tool_name = serialized.get("name", "unknown_tool")

        # 获取run_id（用于关联start和end）
        run_id = kwargs.get("run_id")

        # 获取父run_id（用于追踪调用链）
        parent_run_id = kwargs.get("parent_run_id")

        # 获取tags和metadata
        tags = kwargs.get("tags", [])
        metadata = kwargs.get("metadata", {})

        print(f"[Tool Start] {tool_name}")
        print(f"  Run ID: {run_id}")
        print(f"  Input: {input_str}")
        print(f"  Tags: {tags}")
        print(f"  Metadata: {metadata}")
```

**详细解释**：

`on_tool_start`方法的参数：
1. **serialized**: 工具的序列化信息
   - `name`: 工具名称（如 "search", "calculator"）
   - `description`: 工具描述
   - 其他工具元数据

2. **input_str**: 传递给工具的输入（字符串形式）
   - 对于简单工具，直接是参数值
   - 对于复杂工具，可能是JSON字符串

3. **kwargs**: 额外信息
   - `run_id`: 当前运行的唯一标识
   - `parent_run_id`: 父运行的ID（Agent的run_id）
   - `tags`: 标签列表
   - `metadata`: 元数据字典

**常见使用场景**：
- 记录工具调用日志
- 统计工具使用频率
- 验证工具参数
- 发送追踪数据到监控系统

**在RAG开发中的应用**：
在RAG Agent中，工具可能包括：检索器、重排序器、文档加载器等。追踪这些工具的调用可以优化检索策略。

---

### 核心概念2：on_tool_end方法重写

**on_tool_end在工具执行完成后触发，可以捕获工具输出和执行状态**

```python
class ToolTrackingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tool_executions = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = kwargs.get("run_id")
        tool_name = serialized.get("name", "unknown")

        self.tool_executions[run_id] = {
            "tool_name": tool_name,
            "input": input_str,
            "start_time": time.time()
        }

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """工具执行完成时调用"""
        run_id = kwargs.get("run_id")

        if run_id in self.tool_executions:
            execution = self.tool_executions[run_id]
            execution["output"] = output
            execution["end_time"] = time.time()
            execution["duration"] = execution["end_time"] - execution["start_time"]

            print(f"[Tool End] {execution['tool_name']}")
            print(f"  Duration: {execution['duration']:.2f}s")
            print(f"  Output: {output[:100]}...")  # 只显示前100字符
```

**on_tool_end方法的参数**：
1. **output**: 工具的输出结果（字符串形式）
2. **kwargs**: 额外信息
   - `run_id`: 与on_tool_start相同的run_id
   - `parent_run_id`: 父运行ID
   - `tags`: 标签
   - `metadata`: 元数据

**关键点**：
- 使用`run_id`关联start和end事件
- 计算工具执行时间
- 记录工具输出
- 分析工具成功率

**在RAG开发中的应用**：
追踪检索器返回的文档数量、相关性分数、检索耗时等，用于优化检索策略。

---

### 核心概念3：on_tool_error方法重写

**on_tool_error在工具执行失败时触发，用于错误处理和告警**

```python
class ToolTrackingCallbackHandler(BaseCallbackHandler):
    def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any
    ) -> None:
        """工具执行出错时调用"""
        run_id = kwargs.get("run_id")

        if run_id in self.tool_executions:
            execution = self.tool_executions[run_id]
            execution["error"] = str(error)
            execution["error_type"] = type(error).__name__
            execution["end_time"] = time.time()
            execution["duration"] = execution["end_time"] - execution["start_time"]
            execution["status"] = "failed"

            print(f"[Tool Error] {execution['tool_name']}")
            print(f"  Error: {error}")
            print(f"  Duration: {execution['duration']:.2f}s")

            # 发送告警
            self.send_alert(execution)

    def send_alert(self, execution):
        """发送告警到监控系统"""
        # 实现告警逻辑
        pass
```

**错误处理策略**：
1. **记录错误详情**：错误类型、错误消息、堆栈跟踪
2. **发送告警**：关键工具失败时发送告警
3. **重试逻辑**：某些错误可以触发重试
4. **降级策略**：工具失败时使用备用方案

**常见错误类型**：
- `TimeoutError`: 工具执行超时
- `ValueError`: 参数验证失败
- `ConnectionError`: 网络连接失败
- `PermissionError`: 权限不足

**在RAG开发中的应用**：
检索器失败时，可以降级到关键词搜索；向量数据库连接失败时，可以使用缓存。

---

### 核心概念4：Agent推理过程追踪

**Agent的推理过程包括多个LLM调用和工具调用，需要完整追踪整个链路**

```python
class AgentTrackingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.agent_runs = {}
        self.current_agent_run = None

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Agent开始执行时调用"""
        run_id = kwargs.get("run_id")
        chain_type = serialized.get("name", "unknown")

        # 检测是否是Agent
        if "agent" in chain_type.lower():
            self.current_agent_run = run_id
            self.agent_runs[run_id] = {
                "input": inputs,
                "start_time": time.time(),
                "llm_calls": [],
                "tool_calls": [],
                "steps": []
            }

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM调用开始"""
        if self.current_agent_run:
            parent_run_id = kwargs.get("parent_run_id")
            if parent_run_id == self.current_agent_run:
                self.agent_runs[self.current_agent_run]["llm_calls"].append({
                    "run_id": kwargs.get("run_id"),
                    "prompts": prompts,
                    "start_time": time.time()
                })

    def on_llm_end(self, response, **kwargs):
        """LLM调用结束"""
        if self.current_agent_run:
            run_id = kwargs.get("run_id")
            for llm_call in self.agent_runs[self.current_agent_run]["llm_calls"]:
                if llm_call["run_id"] == run_id:
                    llm_call["response"] = response.generations[0][0].text
                    llm_call["end_time"] = time.time()
                    llm_call["duration"] = llm_call["end_time"] - llm_call["start_time"]

    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具调用开始"""
        if self.current_agent_run:
            parent_run_id = kwargs.get("parent_run_id")
            if parent_run_id == self.current_agent_run:
                self.agent_runs[self.current_agent_run]["tool_calls"].append({
                    "run_id": kwargs.get("run_id"),
                    "tool_name": serialized.get("name"),
                    "input": input_str,
                    "start_time": time.time()
                })

    def on_tool_end(self, output, **kwargs):
        """工具调用结束"""
        if self.current_agent_run:
            run_id = kwargs.get("run_id")
            for tool_call in self.agent_runs[self.current_agent_run]["tool_calls"]:
                if tool_call["run_id"] == run_id:
                    tool_call["output"] = output
                    tool_call["end_time"] = time.time()
                    tool_call["duration"] = tool_call["end_time"] - tool_call["start_time"]

    def on_agent_action(self, action, **kwargs):
        """Agent决策时调用"""
        if self.current_agent_run:
            self.agent_runs[self.current_agent_run]["steps"].append({
                "type": "action",
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log
            })

    def on_agent_finish(self, finish, **kwargs):
        """Agent完成时调用"""
        if self.current_agent_run:
            self.agent_runs[self.current_agent_run]["steps"].append({
                "type": "finish",
                "output": finish.return_values,
                "log": finish.log
            })
            self.agent_runs[self.current_agent_run]["end_time"] = time.time()
            self.agent_runs[self.current_agent_run]["total_duration"] = (
                self.agent_runs[self.current_agent_run]["end_time"] -
                self.agent_runs[self.current_agent_run]["start_time"]
            )

    def on_chain_end(self, outputs, **kwargs):
        """Agent执行结束"""
        run_id = kwargs.get("run_id")
        if run_id == self.current_agent_run:
            self.agent_runs[run_id]["output"] = outputs
            self.current_agent_run = None

    def get_agent_summary(self, run_id):
        """获取Agent执行摘要"""
        if run_id not in self.agent_runs:
            return None

        run = self.agent_runs[run_id]
        return {
            "total_duration": run.get("total_duration", 0),
            "llm_calls_count": len(run["llm_calls"]),
            "tool_calls_count": len(run["tool_calls"]),
            "steps_count": len(run["steps"]),
            "llm_total_time": sum(call.get("duration", 0) for call in run["llm_calls"]),
            "tool_total_time": sum(call.get("duration", 0) for call in run["tool_calls"]),
            "tools_used": list(set(call["tool_name"] for call in run["tool_calls"]))
        }
```

**Agent推理链路**：
```
1. on_chain_start (Agent开始)
   ↓
2. on_llm_start (LLM推理：选择工具)
   ↓
3. on_llm_end (LLM返回：工具名称和参数)
   ↓
4. on_agent_action (Agent决策：执行工具)
   ↓
5. on_tool_start (工具开始执行)
   ↓
6. on_tool_end (工具执行完成)
   ↓
7. on_llm_start (LLM推理：分析工具结果)
   ↓
8. on_llm_end (LLM返回：继续或结束)
   ↓
9. on_agent_finish (Agent完成)
   ↓
10. on_chain_end (Agent结束)
```

**在RAG开发中的应用**：
追踪RAG Agent的完整推理过程：查询改写 → 检索 → 重排序 → 生成，优化每个环节。

---

## 4. 【最小可用】

掌握以下内容，就能追踪Agent工具调用：

### 4.1 基础工具追踪：重写on_tool_start/end

```python
from langchain.callbacks.base import BaseCallbackHandler

class SimpleToolTracker(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        print(f"[Tool] {tool_name} started")
        print(f"  Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"  Output: {output[:100]}...")

# 使用
from langchain.agents import AgentExecutor
agent_executor.run("query", callbacks=[SimpleToolTracker()])
```

### 4.2 错误追踪：重写on_tool_error

```python
class ToolErrorTracker(BaseCallbackHandler):
    def on_tool_error(self, error, **kwargs):
        print(f"[Tool Error] {error}")
        # 可以发送告警或记录日志
```

### 4.3 性能追踪：记录工具执行时间

```python
import time

class ToolPerformanceTracker(BaseCallbackHandler):
    def __init__(self):
        self.tool_times = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = kwargs.get("run_id")
        self.tool_times[run_id] = {
            "tool": serialized.get("name"),
            "start": time.time()
        }

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id in self.tool_times:
            duration = time.time() - self.tool_times[run_id]["start"]
            tool = self.tool_times[run_id]["tool"]
            print(f"[{tool}] took {duration:.2f}s")
```

### 4.4 Agent推理追踪：追踪完整链路

```python
class AgentReasoningTracker(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        print(f"[Agent Action] Tool: {action.tool}")
        print(f"  Input: {action.tool_input}")

    def on_agent_finish(self, finish, **kwargs):
        print(f"[Agent Finish] Output: {finish.return_values}")
```

**这些知识足以：**
- 追踪Agent的工具调用
- 调试Agent失败原因
- 优化工具性能
- 为后续学习复杂追踪打基础

---

## 5. 【双重类比】

### 类比1：on_tool_start/end

**前端类比：** API请求拦截器

在前端中，axios拦截器可以在请求前后执行逻辑：
```javascript
// 请求拦截器
axios.interceptors.request.use(config => {
  console.log('Request:', config.url);
  config.startTime = Date.now();
  return config;
});

// 响应拦截器
axios.interceptors.response.use(response => {
  const duration = Date.now() - response.config.startTime;
  console.log('Response:', response.data, `(${duration}ms)`);
  return response;
});
```

**日常生活类比：** 快递追踪

快递系统在每个节点记录状态：
- 揽件（on_tool_start）
- 运输中（执行中）
- 签收（on_tool_end）

```python
# 就像快递追踪
class ToolTracker(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[揽件] 工具: {serialized.get('name')}")

    def on_tool_end(self, output, **kwargs):
        print(f"[签收] 结果: {output}")
```

---

### 类比2：Agent推理链路追踪

**前端类比：** React组件生命周期

React组件有多个生命周期方法，追踪组件的完整生命周期：
```javascript
class Component extends React.Component {
  componentDidMount() {
    console.log('Component mounted');
  }

  componentDidUpdate() {
    console.log('Component updated');
  }

  componentWillUnmount() {
    console.log('Component unmounted');
  }
}
```

**日常生活类比：** 医生诊断流程

医生诊断病人的完整流程：
1. 问诊（on_chain_start）
2. 思考（on_llm_start）
3. 决定检查项目（on_agent_action）
4. 执行检查（on_tool_start）
5. 获取检查结果（on_tool_end）
6. 分析结果（on_llm_start）
7. 给出诊断（on_agent_finish）

```python
# 就像医生诊断
class DiagnosisTracker(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("[问诊] 患者症状:", inputs)

    def on_agent_action(self, action, **kwargs):
        print(f"[决定] 需要做: {action.tool}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[检查] 执行: {serialized.get('name')}")

    def on_agent_finish(self, finish, **kwargs):
        print("[诊断] 结论:", finish.return_values)
```

---

### 类比3：工具参数传递

**前端类比：** 函数调用参数

在JavaScript中，函数调用时传递参数：
```javascript
function search(query, filters) {
  console.log('Searching:', query, 'with filters:', filters);
  return results;
}

// 调用
search('RAG', { category: 'AI', year: 2024 });
```

**日常生活类比：** 餐厅点餐

在餐厅点餐时，需要告诉服务员：
- 菜名（工具名称）
- 要求（工具参数）：不要辣、多加醋

```python
# 就像点餐
class OrderTracker(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        dish = serialized.get("name")  # 菜名
        requirements = input_str  # 要求
        print(f"[点餐] {dish}, 要求: {requirements}")
```

---

### 类比总结表

| Agent概念 | 前端类比 | 日常生活类比 |
|----------|---------|-------------|
| on_tool_start/end | API拦截器 | 快递追踪 |
| Agent推理链路 | React生命周期 | 医生诊断流程 |
| 工具参数传递 | 函数参数 | 餐厅点餐 |
| on_tool_error | try-catch | 快递丢失 |
| 性能追踪 | Performance API | 计时器 |

---

