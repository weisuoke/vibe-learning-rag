# 实战代码2：Agent调用链追踪

> 实现 AI Agent 调用链追踪系统，支持 LangGraph、LangSmith 集成

---

## 学习目标

- 实现完整的 Agent 调用链追踪器
- 集成 LangGraph 状态机追踪
- 使用 LangSmith 进行调试
- 可视化调用链和性能分析

---

## 1. 基础调用链追踪器

### 1.1 简单追踪器

```python
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class CallRecord:
    """调用记录"""
    function_name: str
    args: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    depth: int = 0
    duration: float = 0.0

class SimpleCallTracer:
    """简单调用链追踪器"""

    def __init__(self):
        self.call_stack: List[CallRecord] = []
        self.execution_log: List[CallRecord] = []

    def enter(self, func_name: str, **kwargs):
        """进入函数"""
        record = CallRecord(
            function_name=func_name,
            args=kwargs,
            start_time=datetime.now(),
            depth=len(self.call_stack)
        )
        self.call_stack.append(record)

        indent = "  " * record.depth
        args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"{indent}→ {func_name}({args_str})")

    def exit(self, result: Any = None, error: str = None):
        """退出函数"""
        if not self.call_stack:
            raise RuntimeError("调用栈为空")

        record = self.call_stack.pop()
        record.end_time = datetime.now()
        record.duration = (record.end_time - record.start_time).total_seconds()
        record.result = result
        record.error = error

        self.execution_log.append(record)

        indent = "  " * len(self.call_stack)
        status = "✗" if error else "✓"
        print(f"{indent}← {status} {record.function_name} ({record.duration:.3f}s)")

    def get_call_chain(self) -> List[str]:
        """获取当前调用链"""
        return [r.function_name for r in self.call_stack]

    def print_summary(self):
        """打印执行摘要"""
        print("\n" + "="*60)
        print("执行摘要:")
        print("="*60)

        total_time = sum(r.duration for r in self.execution_log)
        print(f"总耗时: {total_time:.3f}s")
        print(f"总调用: {len(self.execution_log)} 次")

        print("\n调用详情:")
        for record in self.execution_log:
            indent = "  " * record.depth
            status = "✗" if record.error else "✓"
            print(f"{indent}{status} {record.function_name}: {record.duration:.3f}s")

# 使用示例
tracer = SimpleCallTracer()

def research_agent(query: str):
    tracer.enter('research_agent', query=query)
    try:
        results = search_web(query)
        analysis = analyze_results(results)
        tracer.exit(result=analysis)
        return analysis
    except Exception as e:
        tracer.exit(error=str(e))
        raise

def search_web(query: str):
    tracer.enter('search_web', query=query)
    import time
    time.sleep(0.1)
    results = [f"结果 for {query}"]
    tracer.exit(result=results)
    return results

def analyze_results(results: List[str]):
    tracer.enter('analyze_results', count=len(results))
    import time
    time.sleep(0.15)
    analysis = f"分析了 {len(results)} 个结果"
    tracer.exit(result=analysis)
    return analysis

# 执行
print("=== 简单调用链追踪 ===\n")
result = research_agent("AI Agent 2026")
tracer.print_summary()
```

---

## 2. 装饰器版本追踪器

### 2.1 使用装饰器自动追踪

```python
from functools import wraps
import time

class DecoratorTracer:
    """装饰器版本的追踪器"""

    def __init__(self):
        self.call_stack = []
        self.execution_log = []

    def trace(self, func):
        """追踪装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 进入函数
            func_name = func.__name__
            depth = len(self.call_stack)
            start_time = time.time()

            self.call_stack.append(func_name)

            indent = "  " * depth
            print(f"{indent}→ {func_name}()")

            try:
                # 执行函数
                result = func(*args, **kwargs)

                # 退出函数
                duration = time.time() - start_time
                self.call_stack.pop()

                self.execution_log.append({
                    'function': func_name,
                    'depth': depth,
                    'duration': duration,
                    'success': True
                })

                print(f"{indent}← ✓ {func_name} ({duration:.3f}s)")
                return result

            except Exception as e:
                # 异常处理
                duration = time.time() - start_time
                self.call_stack.pop()

                self.execution_log.append({
                    'function': func_name,
                    'depth': depth,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })

                print(f"{indent}← ✗ {func_name} ({duration:.3f}s) - {e}")
                raise

        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_calls = len(self.execution_log)
        successful_calls = sum(1 for log in self.execution_log if log['success'])
        total_time = sum(log['duration'] for log in self.execution_log)

        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': total_calls - successful_calls,
            'total_time': total_time,
            'avg_time': total_time / total_calls if total_calls > 0 else 0
        }

# 使用示例
tracer = DecoratorTracer()

@tracer.trace
def main_agent(task: str):
    step1()
    step2()
    return "完成"

@tracer.trace
def step1():
    time.sleep(0.1)
    substep1()

@tracer.trace
def substep1():
    time.sleep(0.05)

@tracer.trace
def step2():
    time.sleep(0.12)

# 执行
print("=== 装饰器追踪 ===\n")
result = main_agent("测试任务")

print("\n统计信息:")
stats = tracer.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
```

---

## 3. LangGraph 集成追踪

### 3.1 LangGraph 状态追踪器

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    """Agent 状态"""
    query: str
    search_results: list
    analysis: str
    messages: Annotated[list, operator.add]

class LangGraphTracer:
    """LangGraph 状态追踪器"""

    def __init__(self):
        self.state_history = []
        self.node_executions = []

    def trace_node(self, node_name: str):
        """节点追踪装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(state: AgentState):
                # 记录进入节点
                start_time = time.time()
                print(f"→ 节点: {node_name}")
                print(f"  输入状态: {list(state.keys())}")

                # 执行节点
                result = func(state)

                # 记录退出节点
                duration = time.time() - start_time
                print(f"← 节点: {node_name} ({duration:.3f}s)")
                print(f"  输出状态: {list(result.keys())}")

                # 保存执行记录
                self.node_executions.append({
                    'node': node_name,
                    'duration': duration,
                    'input_keys': list(state.keys()),
                    'output_keys': list(result.keys())
                })

                # 保存状态快照
                self.state_history.append({
                    'node': node_name,
                    'state': result.copy()
                })

                return result

            return wrapper
        return decorator

    def visualize_flow(self):
        """可视化执行流程"""
        print("\n" + "="*60)
        print("LangGraph 执行流程:")
        print("="*60)

        for i, execution in enumerate(self.node_executions, 1):
            print(f"\n{i}. {execution['node']}")
            print(f"   耗时: {execution['duration']:.3f}s")
            print(f"   输入: {execution['input_keys']}")
            print(f"   输出: {execution['output_keys']}")

# 创建追踪器
tracer = LangGraphTracer()

# 定义节点
@tracer.trace_node("search_node")
def search_node(state: AgentState):
    """搜索节点"""
    time.sleep(0.1)
    return {
        **state,
        "search_results": [f"结果 for {state['query']}"],
        "messages": ["搜索完成"]
    }

@tracer.trace_node("analyze_node")
def analyze_node(state: AgentState):
    """分析节点"""
    time.sleep(0.15)
    return {
        **state,
        "analysis": f"分析了 {len(state['search_results'])} 个结果",
        "messages": ["分析完成"]
    }

@tracer.trace_node("summarize_node")
def summarize_node(state: AgentState):
    """总结节点"""
    time.sleep(0.08)
    return {
        **state,
        "messages": ["总结完成"]
    }

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

# 执行
print("=== LangGraph 追踪 ===\n")
initial_state = {
    "query": "AI Agent 2026",
    "search_results": [],
    "analysis": "",
    "messages": []
}

result = app.invoke(initial_state)
tracer.visualize_flow()
```

---

## 4. LangSmith 集成

### 4.1 LangSmith 追踪配置

```python
import os
from langsmith import Client
from langsmith.run_helpers import traceable

# 配置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "agent-call-tracing"

class LangSmithTracer:
    """LangSmith 集成追踪器"""

    def __init__(self, project_name: str = "agent-tracing"):
        self.client = Client()
        self.project_name = project_name

    @traceable(name="research_agent")
    def research_agent(self, query: str) -> str:
        """研究 Agent"""
        search_results = self.search_tool(query)
        analysis = self.analyze_tool(search_results)
        return analysis

    @traceable(name="search_tool")
    def search_tool(self, query: str) -> list:
        """搜索工具"""
        time.sleep(0.1)
        return [f"结果1 for {query}", f"结果2 for {query}"]

    @traceable(name="analyze_tool")
    def analyze_tool(self, results: list) -> str:
        """分析工具"""
        time.sleep(0.15)
        return f"分析了 {len(results)} 个结果"

    def get_recent_runs(self, limit: int = 10):
        """获取最近的运行记录"""
        runs = self.client.list_runs(
            project_name=self.project_name,
            limit=limit
        )
        return list(runs)

    def print_run_summary(self, run_id: str):
        """打印运行摘要"""
        run = self.client.read_run(run_id)

        print(f"\n运行 ID: {run.id}")
        print(f"名称: {run.name}")
        print(f"状态: {run.status}")
        print(f"开始时间: {run.start_time}")
        print(f"结束时间: {run.end_time}")
        print(f"耗时: {run.execution_time}ms")

        if run.inputs:
            print(f"输入: {run.inputs}")
        if run.outputs:
            print(f"输出: {run.outputs}")

# 使用示例
tracer = LangSmithTracer()

print("=== LangSmith 追踪 ===\n")
result = tracer.research_agent("AI Agent 2026")
print(f"\n结果: {result}")

# 获取最近的运行记录
recent_runs = tracer.get_recent_runs(limit=1)
if recent_runs:
    tracer.print_run_summary(recent_runs[0].id)
```

---

## 5. 性能分析追踪器

### 5.1 带性能分析的追踪器

```python
import cProfile
import pstats
from io import StringIO

class PerformanceTracer:
    """性能分析追踪器"""

    def __init__(self):
        self.call_stack = []
        self.execution_log = []
        self.profiler = cProfile.Profile()

    def trace(self, func):
        """追踪装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            depth = len(self.call_stack)

            self.call_stack.append(func_name)

            # 开始性能分析
            self.profiler.enable()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                self.profiler.disable()
                self.call_stack.pop()

                # 记录执行信息
                self.execution_log.append({
                    'function': func_name,
                    'depth': depth,
                    'duration': duration,
                    'success': True
                })

                return result

            except Exception as e:
                duration = time.time() - start_time
                self.profiler.disable()
                self.call_stack.pop()

                self.execution_log.append({
                    'function': func_name,
                    'depth': depth,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })

                raise

        return wrapper

    def print_profile(self, top_n: int = 10):
        """打印性能分析结果"""
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(top_n)

        print("\n" + "="*60)
        print("性能分析:")
        print("="*60)
        print(s.getvalue())

    def print_call_tree(self):
        """打印调用树"""
        print("\n" + "="*60)
        print("调用树:")
        print("="*60)

        for log in self.execution_log:
            indent = "  " * log['depth']
            status = "✓" if log['success'] else "✗"
            print(f"{indent}{status} {log['function']}: {log['duration']:.3f}s")

# 使用示例
tracer = PerformanceTracer()

@tracer.trace
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@tracer.trace
def calculate_sum(n):
    total = 0
    for i in range(n):
        total += fibonacci(i)
    return total

# 执行
print("=== 性能分析追踪 ===\n")
result = calculate_sum(10)
print(f"结果: {result}")

tracer.print_call_tree()
tracer.print_profile(top_n=5)
```

---

## 6. 可视化调用链

### 6.1 ASCII 艺术可视化

```python
class CallChainVisualizer:
    """调用链可视化器"""

    def __init__(self):
        self.execution_log = []

    def record(self, func_name: str, depth: int, duration: float, success: bool = True):
        """记录执行"""
        self.execution_log.append({
            'function': func_name,
            'depth': depth,
            'duration': duration,
            'success': success
        })

    def visualize_tree(self):
        """树形可视化"""
        print("\n" + "="*60)
        print("调用链树形图:")
        print("="*60)

        for i, log in enumerate(self.execution_log):
            is_last = i == len(self.execution_log) - 1
            next_depth = self.execution_log[i + 1]['depth'] if not is_last else 0

            # 绘制缩进
            for d in range(log['depth']):
                if d < log['depth'] - 1:
                    print("│   ", end="")
                else:
                    if next_depth > log['depth']:
                        print("├── ", end="")
                    else:
                        print("└── ", end="")

            # 绘制节点
            status = "✓" if log['success'] else "✗"
            print(f"{status} {log['function']} ({log['duration']:.3f}s)")

    def visualize_timeline(self):
        """时间线可视化"""
        print("\n" + "="*60)
        print("执行时间线:")
        print("="*60)

        max_duration = max(log['duration'] for log in self.execution_log)
        scale = 50 / max_duration if max_duration > 0 else 1

        for log in self.execution_log:
            indent = "  " * log['depth']
            bar_length = int(log['duration'] * scale)
            bar = "█" * bar_length

            status = "✓" if log['success'] else "✗"
            print(f"{indent}{status} {log['function']:<20} {bar} {log['duration']:.3f}s")

# 使用示例
visualizer = CallChainVisualizer()

# 模拟执行记录
visualizer.record("main_agent", 0, 0.350, True)
visualizer.record("search_tool", 1, 0.100, True)
visualizer.record("api_call", 2, 0.080, True)
visualizer.record("analyze_tool", 1, 0.150, True)
visualizer.record("summarize_tool", 1, 0.100, True)

visualizer.visualize_tree()
visualizer.visualize_timeline()
```

---

## 7. 完整示例：多层 Agent 追踪

### 7.1 复杂 Agent 系统

```python
from typing import List, Dict
import json

class CompleteAgentTracer:
    """完整的 Agent 追踪系统"""

    def __init__(self, enable_profiling: bool = False):
        self.call_stack = []
        self.execution_log = []
        self.enable_profiling = enable_profiling

    def trace(self, func):
        """追踪装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            depth = len(self.call_stack)
            start_time = time.time()

            # 记录调用
            call_info = {
                'function': func_name,
                'args': str(args)[:100],
                'kwargs': str(kwargs)[:100],
                'depth': depth,
                'start_time': start_time
            }
            self.call_stack.append(call_info)

            indent = "  " * depth
            print(f"{indent}→ {func_name}()")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                self.call_stack.pop()

                # 记录执行日志
                self.execution_log.append({
                    **call_info,
                    'duration': duration,
                    'success': True,
                    'result': str(result)[:100]
                })

                print(f"{indent}← ✓ {func_name} ({duration:.3f}s)")
                return result

            except Exception as e:
                duration = time.time() - start_time
                self.call_stack.pop()

                self.execution_log.append({
                    **call_info,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })

                print(f"{indent}← ✗ {func_name} ({duration:.3f}s) - {e}")
                raise

        return wrapper

    def export_trace(self, filename: str):
        """导出追踪数据"""
        with open(filename, 'w') as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        print(f"\n追踪数据已导出到: {filename}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_calls = len(self.execution_log)
        successful = sum(1 for log in self.execution_log if log['success'])
        total_time = sum(log['duration'] for log in self.execution_log)

        # 按函数统计
        func_stats = {}
        for log in self.execution_log:
            func = log['function']
            if func not in func_stats:
                func_stats[func] = {'count': 0, 'total_time': 0, 'failures': 0}

            func_stats[func]['count'] += 1
            func_stats[func]['total_time'] += log['duration']
            if not log['success']:
                func_stats[func]['failures'] += 1

        return {
            'total_calls': total_calls,
            'successful_calls': successful,
            'failed_calls': total_calls - successful,
            'total_time': total_time,
            'function_stats': func_stats
        }

    def print_report(self):
        """打印完整报告"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("执行报告:")
        print("="*60)

        print(f"\n总体统计:")
        print(f"  总调用次数: {stats['total_calls']}")
        print(f"  成功: {stats['successful_calls']}")
        print(f"  失败: {stats['failed_calls']}")
        print(f"  总耗时: {stats['total_time']:.3f}s")

        print(f"\n函数统计:")
        for func, data in stats['function_stats'].items():
            avg_time = data['total_time'] / data['count']
            print(f"  {func}:")
            print(f"    调用次数: {data['count']}")
            print(f"    总耗时: {data['total_time']:.3f}s")
            print(f"    平均耗时: {avg_time:.3f}s")
            print(f"    失败次数: {data['failures']}")

# 使用示例
tracer = CompleteAgentTracer()

@tracer.trace
def research_agent(query: str):
    """研究 Agent"""
    search_results = search_web(query)
    analysis = analyze_results(search_results)
    summary = summarize(analysis)
    return summary

@tracer.trace
def search_web(query: str):
    """搜索网络"""
    time.sleep(0.1)
    return [f"结果1 for {query}", f"结果2 for {query}"]

@tracer.trace
def analyze_results(results: List[str]):
    """分析结果"""
    time.sleep(0.15)
    detailed_analysis = []
    for result in results:
        detailed_analysis.append(analyze_single(result))
    return detailed_analysis

@tracer.trace
def analyze_single(result: str):
    """分析单个结果"""
    time.sleep(0.05)
    return f"分析: {result}"

@tracer.trace
def summarize(analysis: List[str]):
    """总结"""
    time.sleep(0.08)
    return f"总结了 {len(analysis)} 个分析"

# 执行
print("=== 完整 Agent 追踪系统 ===\n")
result = research_agent("AI Agent 2026")
print(f"\n最终结果: {result}")

tracer.print_report()
tracer.export_trace("agent_trace.json")
```

---

## 总结

**调用链追踪的核心功能：**
- 记录函数调用顺序
- 测量执行时间
- 捕获异常信息
- 可视化调用链

**实现方式：**
- 手动调用（enter/exit）
- 装饰器自动追踪
- 上下文管理器
- LangGraph/LangSmith 集成

**2026 年最佳实践：**
- LangGraph: 状态机追踪
- LangSmith: 云端调试
- 性能分析: cProfile 集成
- 可视化: ASCII 艺术 + JSON 导出

**记住：** 调用链追踪是调试 AI Agent 的关键工具，选择合适的追踪方式能大幅提升开发效率！
