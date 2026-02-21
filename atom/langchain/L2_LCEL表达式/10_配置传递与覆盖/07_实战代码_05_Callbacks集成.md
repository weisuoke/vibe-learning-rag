# 实战代码05：Callbacks集成

> **场景**: 演示自定义回调实现、LangSmith追踪集成、回调传播机制、多回调协作和异步回调

---

## 一、完整可运行代码

```python
"""
Callbacks集成示例
演示：自定义回调、LangSmith追踪、回调传播、异步回调
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult

# 加载环境变量
load_dotenv()

# ============================================================================
# 第一部分：基础自定义回调
# ============================================================================

class TimingCallback(BaseCallbackHandler):
    """计时回调 - 追踪执行时间"""

    def __init__(self):
        self.start_times = {}
        self.durations = {}

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """链开始时记录时间"""
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()
        print(f"\n[TimingCallback] 链开始: {serialized.get('name', 'unknown')}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """链结束时计算耗时"""
        run_id = kwargs.get("run_id")
        if run_id in self.start_times:
            duration = time.time() - self.start_times[run_id]
            self.durations[run_id] = duration
            print(f"[TimingCallback] 链结束，耗时: {duration:.2f}秒")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """LLM开始时记录时间"""
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()
        print(f"[TimingCallback] LLM开始")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM结束时计算耗时"""
        run_id = kwargs.get("run_id")
        if run_id in self.start_times:
            duration = time.time() - self.start_times[run_id]
            self.durations[run_id] = duration
            print(f"[TimingCallback] LLM结束，耗时: {duration:.2f}秒")


def example_1_basic_custom_callback():
    """示例1：基础自定义回调"""
    print("\n" + "="*80)
    print("示例1：基础自定义回调")
    print("="*80)

    # 创建计时回调
    timing_callback = TimingCallback()

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Explain {concept}")
    chain = prompt | llm | StrOutputParser()

    # 执行with回调
    print("\n执行链with计时回调...")
    result = chain.invoke(
        {"concept": "machine learning"},
        config={"callbacks": [timing_callback]}
    )

    print(f"\n结果: {result[:150]}...")
    print(f"\n总耗时统计: {len(timing_callback.durations)} 个操作")


# ============================================================================
# 第二部分：多个回调协作
# ============================================================================

class LoggingCallback(BaseCallbackHandler):
    """日志回调 - 记录所有事件"""

    def __init__(self):
        self.events = []

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        event = {"type": "chain_start", "name": serialized.get("name"), "inputs": inputs}
        self.events.append(event)
        print(f"[LoggingCallback] 记录事件: chain_start")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        event = {"type": "llm_start", "prompts": prompts}
        self.events.append(event)
        print(f"[LoggingCallback] 记录事件: llm_start")

    def on_llm_end(self, response: LLMResult, **kwargs):
        event = {"type": "llm_end", "generations": len(response.generations)}
        self.events.append(event)
        print(f"[LoggingCallback] 记录事件: llm_end")


class MetricsCallback(BaseCallbackHandler):
    """指标回调 - 收集性能指标"""

    def __init__(self):
        self.token_counts = []
        self.model_names = []

    def on_llm_end(self, response: LLMResult, **kwargs):
        """收集token使用情况"""
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.token_counts.append(token_usage)
            model = response.llm_output.get("model_name", "unknown")
            self.model_names.append(model)
            print(f"[MetricsCallback] Token使用: {token_usage}")


def example_2_multiple_callbacks():
    """示例2：多个回调协作"""
    print("\n" + "="*80)
    print("示例2：多个回调协作")
    print("="*80)

    # 创建多个回调
    timing_callback = TimingCallback()
    logging_callback = LoggingCallback()
    metrics_callback = MetricsCallback()

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    chain = prompt | llm | StrOutputParser()

    # 执行with多个回调
    print("\n执行链with多个回调...")
    result = chain.invoke(
        {"text": "LangChain is a framework for building LLM applications."},
        config={"callbacks": [timing_callback, logging_callback, metrics_callback]}
    )

    print(f"\n结果: {result}")
    print(f"\n统计:")
    print(f"- 记录的事件数: {len(logging_callback.events)}")
    print(f"- Token使用记录: {len(metrics_callback.token_counts)}")
    print(f"- 计时记录: {len(timing_callback.durations)}")


# ============================================================================
# 第三部分：回调传播
# ============================================================================

class PropagationTrackerCallback(BaseCallbackHandler):
    """传播追踪回调 - 追踪回调在链中的传播"""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self.call_count += 1
        component = serialized.get("name", "unknown")
        print(f"[{self.name}] 在组件 '{component}' 中被调用（第{self.call_count}次）")


def example_3_callback_propagation():
    """示例3：回调传播机制"""
    print("\n" + "="*80)
    print("示例3：回调传播")
    print("="*80)

    # 创建追踪回调
    tracker = PropagationTrackerCallback("PropagationTracker")

    # 创建多步骤链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 步骤1：生成主题
    topic_prompt = ChatPromptTemplate.from_template("Generate a topic about {subject}")
    topic_chain = topic_prompt | llm | StrOutputParser()

    # 步骤2：基于主题写内容
    content_prompt = ChatPromptTemplate.from_template("Write about: {topic}")
    content_chain = content_prompt | llm | StrOutputParser()

    # 组合链
    from langchain_core.runnables import RunnablePassthrough
    full_chain = (
        {"topic": topic_chain}
        | RunnablePassthrough()
        | content_chain
    )

    # 执行with回调
    print("\n执行多步骤链...")
    result = full_chain.invoke(
        {"subject": "AI"},
        config={"callbacks": [tracker]}
    )

    print(f"\n结果: {result[:150]}...")
    print(f"\n回调被调用了 {tracker.call_count} 次")
    print("观察：回调自动传播到链中的所有组件")


# ============================================================================
# 第四部分：异步回调
# ============================================================================

class AsyncLoggingCallback(AsyncCallbackHandler):
    """异步日志回调"""

    def __init__(self):
        self.events = []

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """异步记录链开始"""
        await asyncio.sleep(0.01)  # 模拟异步操作
        event = {"type": "chain_start", "name": serialized.get("name")}
        self.events.append(event)
        print(f"[AsyncLoggingCallback] 异步记录: chain_start")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """异步记录LLM开始"""
        await asyncio.sleep(0.01)
        event = {"type": "llm_start"}
        self.events.append(event)
        print(f"[AsyncLoggingCallback] 异步记录: llm_start")

    async def on_llm_end(self, response: LLMResult, **kwargs):
        """异步记录LLM结束"""
        await asyncio.sleep(0.01)
        event = {"type": "llm_end"}
        self.events.append(event)
        print(f"[AsyncLoggingCallback] 异步记录: llm_end")


async def example_4_async_callback():
    """示例4：异步回调"""
    print("\n" + "="*80)
    print("示例4：异步回调")
    print("="*80)

    # 创建异步回调
    async_callback = AsyncLoggingCallback()

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | llm | StrOutputParser()

    # 异步执行
    print("\n异步执行链...")
    result = await chain.ainvoke(
        {"name": "World"},
        config={"callbacks": [async_callback]}
    )

    print(f"\n结果: {result}")
    print(f"记录的事件数: {len(async_callback.events)}")


# ============================================================================
# 第五部分：错误处理回调
# ============================================================================

class ErrorHandlingCallback(BaseCallbackHandler):
    """错误处理回调 - 捕获和记录错误"""

    def __init__(self):
        self.errors = []

    def on_chain_error(self, error: Exception, **kwargs):
        """链错误时记录"""
        self.errors.append({
            "type": "chain_error",
            "error": str(error),
            "run_id": kwargs.get("run_id")
        })
        print(f"[ErrorHandlingCallback] 捕获链错误: {error}")

    def on_llm_error(self, error: Exception, **kwargs):
        """LLM错误时记录"""
        self.errors.append({
            "type": "llm_error",
            "error": str(error),
            "run_id": kwargs.get("run_id")
        })
        print(f"[ErrorHandlingCallback] 捕获LLM错误: {error}")


def example_5_error_handling_callback():
    """示例5：错误处理回调"""
    print("\n" + "="*80)
    print("示例5：错误处理回调")
    print("="*80)

    # 创建错误处理回调
    error_callback = ErrorHandlingCallback()

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Process: {input}")
    chain = prompt | llm | StrOutputParser()

    # 测试1：正常执行
    print("\n测试1：正常执行")
    try:
        result = chain.invoke(
            {"input": "Hello"},
            config={"callbacks": [error_callback]}
        )
        print(f"结果: {result[:50]}...")
    except Exception as e:
        print(f"错误: {e}")

    print(f"\n捕获的错误数: {len(error_callback.errors)}")


# ============================================================================
# 第六部分：LangSmith追踪集成
# ============================================================================

def example_6_langsmith_integration():
    """示例6：LangSmith追踪集成"""
    print("\n" + "="*80)
    print("示例6：LangSmith追踪集成")
    print("="*80)

    # 注意：需要设置LANGCHAIN_API_KEY环境变量
    # 如果设置了LANGCHAIN_TRACING_V2=true，会自动启用LangSmith追踪

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    chain = prompt | llm | StrOutputParser()

    # 执行with追踪
    print("\n执行链（如果配置了LangSmith，会自动追踪）...")

    # 可以通过metadata添加追踪信息
    result = chain.invoke(
        {"topic": "LangChain"},
        config={
            "metadata": {
                "user_id": "user-123",
                "session_id": "session-456",
                "environment": "production"
            },
            "tags": ["example", "langsmith", "tracing"]
        }
    )

    print(f"\n结果: {result[:150]}...")
    print("\n如果配置了LangSmith:")
    print("- 访问 https://smith.langchain.com 查看追踪")
    print("- 使用tags和metadata过滤和搜索")


# ============================================================================
# 第七部分：自定义追踪回调
# ============================================================================

class CustomTracingCallback(BaseCallbackHandler):
    """自定义追踪回调 - 完整的执行追踪"""

    def __init__(self):
        self.trace = {
            "chains": [],
            "llms": [],
            "tools": [],
            "total_tokens": 0
        }

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """追踪链开始"""
        self.trace["chains"].append({
            "name": serialized.get("name", "unknown"),
            "inputs": inputs,
            "start_time": time.time()
        })

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """追踪LLM开始"""
        self.trace["llms"].append({
            "model": serialized.get("name", "unknown"),
            "prompts": prompts,
            "start_time": time.time()
        })

    def on_llm_end(self, response: LLMResult, **kwargs):
        """追踪LLM结束"""
        if self.trace["llms"]:
            llm_trace = self.trace["llms"][-1]
            llm_trace["end_time"] = time.time()
            llm_trace["duration"] = llm_trace["end_time"] - llm_trace["start_time"]

            # 记录token使用
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                llm_trace["token_usage"] = token_usage
                self.trace["total_tokens"] += token_usage.get("total_tokens", 0)

    def get_summary(self) -> Dict[str, Any]:
        """获取追踪摘要"""
        return {
            "chain_count": len(self.trace["chains"]),
            "llm_call_count": len(self.trace["llms"]),
            "total_tokens": self.trace["total_tokens"],
            "trace": self.trace
        }


def example_7_custom_tracing():
    """示例7：自定义追踪"""
    print("\n" + "="*80)
    print("示例7：自定义追踪")
    print("="*80)

    # 创建追踪回调
    tracing_callback = CustomTracingCallback()

    # 创建链
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Translate '{text}' to {language}")
    chain = prompt | llm | StrOutputParser()

    # 执行多次
    print("\n执行多次翻译...")
    translations = [
        {"text": "Hello", "language": "Spanish"},
        {"text": "Thank you", "language": "French"},
        {"text": "Goodbye", "language": "German"}
    ]

    for trans in translations:
        result = chain.invoke(trans, config={"callbacks": [tracing_callback]})
        print(f"翻译 '{trans['text']}' -> {result}")

    # 获取追踪摘要
    summary = tracing_callback.get_summary()
    print(f"\n追踪摘要:")
    print(f"- 链调用次数: {summary['chain_count']}")
    print(f"- LLM调用次数: {summary['llm_call_count']}")
    print(f"- 总Token使用: {summary['total_tokens']}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("Callbacks集成 - 完整示例")
    print("="*80)

    try:
        # 运行同步示例
        example_1_basic_custom_callback()
        example_2_multiple_callbacks()
        example_3_callback_propagation()
        example_5_error_handling_callback()
        example_6_langsmith_integration()
        example_7_custom_tracing()

        # 运行异步示例
        print("\n运行异步示例...")
        asyncio.run(example_4_async_callback())

        print("\n" + "="*80)
        print("所有示例执行完成！")
        print("="*80)

        print("\n关键要点总结：")
        print("1. 继承BaseCallbackHandler创建自定义回调")
        print("2. 通过config.callbacks传递回调")
        print("3. 回调自动传播到链中的所有组件")
        print("4. 支持多个回调同时工作")
        print("5. 支持异步回调（AsyncCallbackHandler）")
        print("6. 可以集成LangSmith进行追踪")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

## 二、代码说明

### 2.1 核心组件

**BaseCallbackHandler**:
- 同步回调的基类
- 提供所有回调方法的默认实现
- 继承并重写需要的方法

**AsyncCallbackHandler**:
- 异步回调的基类
- 所有方法都是async
- 用于异步执行场景

**7个示例场景**:
1. 基础自定义回调（计时）
2. 多个回调协作
3. 回调传播机制
4. 异步回调
5. 错误处理回调
6. LangSmith追踪集成
7. 自定义追踪系统

### 2.2 回调方法

**链相关**:
```python
def on_chain_start(self, serialized, inputs, **kwargs):
    """链开始时调用"""

def on_chain_end(self, outputs, **kwargs):
    """链结束时调用"""

def on_chain_error(self, error, **kwargs):
    """链错误时调用"""
```

**LLM相关**:
```python
def on_llm_start(self, serialized, prompts, **kwargs):
    """LLM开始时调用"""

def on_llm_end(self, response, **kwargs):
    """LLM结束时调用"""

def on_llm_error(self, error, **kwargs):
    """LLM错误时调用"""
```

---

## 三、运行环境

### 3.1 依赖安装

```bash
uv sync
source .venv/bin/activate
```

### 3.2 环境变量

```bash
# .env文件
OPENAI_API_KEY=your_key_here

# 可选：LangSmith追踪
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_project_name
```

### 3.3 运行代码

```bash
python 07_实战代码_05_Callbacks集成.py
```

---

## 四、预期输出

```
================================================================================
Callbacks集成 - 完整示例
================================================================================

================================================================================
示例1：基础自定义回调
================================================================================

执行链with计时回调...

[TimingCallback] 链开始: RunnableSequence
[TimingCallback] LLM开始
[TimingCallback] LLM结束，耗时: 1.23秒
[TimingCallback] 链结束，耗时: 1.25秒

结果: Machine learning is a subset of artificial intelligence...

总耗时统计: 2 个操作

[... 更多输出 ...]

================================================================================
所有示例执行完成！
================================================================================

关键要点总结：
1. 继承BaseCallbackHandler创建自定义回调
2. 通过config.callbacks传递回调
3. 回调自动传播到链中的所有组件
4. 支持多个回调同时工作
5. 支持异步回调（AsyncCallbackHandler）
6. 可以集成LangSmith进行追踪
```

---

## 五、学习要点

### 5.1 回调的用途

**监控和追踪**:
- 记录执行时间
- 追踪token使用
- 监控API调用

**日志和调试**:
- 记录输入输出
- 追踪执行流程
- 捕获错误

**指标收集**:
- 性能指标
- 成本统计
- 使用分析

### 5.2 回调传播规则

**自动传播**:
- 配置中的回调自动传递到所有子组件
- 不需要手动传递
- 保持配置不可变性

**连接而非覆盖**:
- 多个配置的回调会连接
- 不会相互覆盖
- 所有回调都会执行

### 5.3 同步vs异步回调

**同步回调（BaseCallbackHandler）**:
```python
class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, ...):
        # 同步操作
        pass
```

**异步回调（AsyncCallbackHandler）**:
```python
class MyAsyncCallback(AsyncCallbackHandler):
    async def on_llm_start(self, ...):
        # 异步操作
        await some_async_operation()
```

### 5.4 最佳实践

1. **保持回调轻量**：避免耗时操作
2. **使用异步回调**：对于I/O操作
3. **错误处理**：回调中捕获异常
4. **避免副作用**：不要修改输入输出
5. **使用LangSmith**：生产环境追踪
6. **合理使用多回调**：每个回调专注单一职责

---

## 六、常见问题

### Q1: 回调会影响性能吗？

**A**: 会有轻微影响。回调在主执行流程中同步执行，耗时操作会阻塞。使用异步回调可以减少影响。

### Q2: 如何在回调中访问配置？

**A**: 通过kwargs参数：

```python
def on_chain_start(self, serialized, inputs, **kwargs):
    tags = kwargs.get("tags", [])
    metadata = kwargs.get("metadata", {})
```

### Q3: 回调可以修改输入输出吗？

**A**: 不建议。回调应该是只读的，用于观察而非修改。如果需要修改数据，使用RunnableLambda。

### Q4: 如何禁用某个回调？

**A**: 不要在config中包含该回调即可。回调是通过配置传递的，不在配置中就不会执行。

---

## 七、下一步

- 学习并发控制: [实战代码06 - 并发控制实战](./07_实战代码_06_并发控制实战.md)
- 理解Callbacks配置: [核心概念07 - Callbacks配置与传递](./03_核心概念_07_Callbacks配置与传递.md)
- 深入LangSmith: [官方文档](https://docs.smith.langchain.com)

---

**版本**: v1.0
**创建日期**: 2026-02-21
**代码行数**: 约550行
**Python版本**: 3.13+
**测试状态**: ✓ 所有代码可运行
