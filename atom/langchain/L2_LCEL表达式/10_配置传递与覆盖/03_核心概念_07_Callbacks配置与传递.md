# 核心概念07：Callbacks配置与传递

> **本节目标**: 掌握Callbacks系统，实现追踪、日志和监控

---

## 一、Callbacks系统概述

### 1.1 什么是Callbacks？

Callbacks是在Runnable执行过程中触发的钩子函数，用于追踪、日志、监控和调试。

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM开始: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM结束: {response}")

# 使用
config = {"callbacks": [MyCallback()]}
result = chain.invoke(input, config=config)
```

### 1.2 为什么需要Callbacks？

- **可观测性**: 追踪执行流程
- **调试**: 查看中间结果
- **监控**: 记录性能指标
- **日志**: 记录执行历史
- **集成**: 与LangSmith等工具集成

---

## 二、Callback类型

### 2.1 BaseCallbackHandler

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    """同步回调处理器"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM开始时调用"""
        pass

    def on_llm_end(self, response, **kwargs):
        """LLM结束时调用"""
        pass

    def on_llm_error(self, error, **kwargs):
        """LLM出错时调用"""
        pass

    def on_chain_start(self, serialized, inputs, **kwargs):
        """链开始时调用"""
        pass

    def on_chain_end(self, outputs, **kwargs):
        """链结束时调用"""
        pass

    def on_chain_error(self, error, **kwargs):
        """链出错时调用"""
        pass
```

### 2.2 AsyncCallbackHandler

```python
from langchain_core.callbacks import AsyncCallbackHandler

class MyAsyncCallback(AsyncCallbackHandler):
    """异步回调处理器"""

    async def on_llm_start(self, serialized, prompts, **kwargs):
        """异步LLM开始"""
        await log_to_db(prompts)

    async def on_llm_end(self, response, **kwargs):
        """异步LLM结束"""
        await log_to_db(response)
```

### 2.3 内置Callbacks

```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.tracers import LangChainTracer

# 标准输出回调
stdout_callback = StdOutCallbackHandler()

# LangSmith追踪回调
tracer = LangChainTracer(project_name="my-project")

config = {"callbacks": [stdout_callback, tracer]}
```

---

## 三、Callbacks配置

### 3.1 通过RunnableConfig配置

```python
from langchain_core.callbacks import BaseCallbackHandler

class LoggingCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LOG] LLM开始: {prompts[0][:50]}...")

# 配置callbacks
config = {
    "callbacks": [LoggingCallback()],
    "tags": ["logging"]
}

result = chain.invoke(input, config=config)
```

### 3.2 通过with_config配置

```python
# 绑定callbacks
chain_with_callbacks = chain.with_config({
    "callbacks": [LoggingCallback()]
})

# 使用
result = chain_with_callbacks.invoke(input)
```

### 3.3 多个Callbacks

```python
# 多个回调同时使用
config = {
    "callbacks": [
        LoggingCallback(),
        MetricsCallback(),
        TracingCallback()
    ]
}

result = chain.invoke(input, config=config)
# 所有回调都会被触发
```

---

## 四、Callbacks传播规则

### 4.1 链式传播

```python
# 配置自动传播到所有组件
chain = prompt | llm | parser

config = {"callbacks": [MyCallback()]}
result = chain.invoke(input, config=config)

# MyCallback会在prompt、llm、parser的所有事件中触发
```

### 4.2 Callbacks连接

```python
# 多个配置的callbacks会连接
base_config = {"callbacks": [CallbackA()]}
custom_config = {"callbacks": [CallbackB()]}

merged = merge_configs(base_config, custom_config)
# merged["callbacks"] = [CallbackA(), CallbackB()]
```

### 4.3 嵌套链传播

```python
# 嵌套链中的callbacks传播
sub_chain = llm | parser
main_chain = prompt | sub_chain

config = {"callbacks": [MyCallback()]}
result = main_chain.invoke(input, config=config)

# MyCallback在所有组件中触发
```

---

## 五、LangSmith集成

### 5.1 环境变量配置

```bash
# 设置环境变量
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT="my-project"
```

### 5.2 代码配置

```python
import os
from langchain_core.tracers import LangChainTracer

# 方式1：环境变量（推荐）
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# 方式2：显式配置
tracer = LangChainTracer(project_name="my-project")
config = {"callbacks": [tracer]}
result = chain.invoke(input, config=config)
```

### 5.3 选择性追踪

```python
from langchain_core.tracers.context import tracing_v2_enabled

# 只追踪特定调用
with tracing_v2_enabled(project_name="specific-project"):
    result = chain.invoke(input)

# 禁用追踪
with tracing_v2_enabled(enabled=False):
    result = chain.invoke(input)  # 不会被追踪
```

---

## 六、自定义Callbacks

### 6.1 生产监控

```python
class ProductionMonitor(BaseCallbackHandler):
    """生产环境监控回调"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        metrics.increment("llm.calls")
        metrics.gauge("llm.prompt_length", len(prompts[0]))

    def on_llm_end(self, response, **kwargs):
        duration = response.llm_output.get("duration", 0)
        metrics.timing("llm.latency", duration)

    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM错误: {error}")
        metrics.increment("llm.errors")

# 使用
config = {"callbacks": [ProductionMonitor()]}
```

### 6.2 成本追踪

```python
class CostTracker(BaseCallbackHandler):
    """成本追踪回调"""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output.get("token_usage", {})
        prompt_tokens = tokens.get("prompt_tokens", 0)
        completion_tokens = tokens.get("completion_tokens", 0)

        self.total_tokens += prompt_tokens + completion_tokens

        # 计算成本（GPT-4示例）
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
        self.total_cost += cost

    def get_summary(self):
        return {
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}"
        }

# 使用
tracker = CostTracker()
config = {"callbacks": [tracker]}
result = chain.invoke(input, config=config)
print(tracker.get_summary())
```

### 6.3 用户追踪

```python
class UserTracker(BaseCallbackHandler):
    """用户追踪回调"""

    def __init__(self, user_id: str):
        self.user_id = user_id

    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info(f"用户 {self.user_id} 开始查询: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        logger.info(f"用户 {self.user_id} 完成查询")

# 使用
config = {
    "callbacks": [UserTracker("user-123")],
    "metadata": {"user_id": "user-123"}
}
```

---

## 七、2025-2026新特性

根据研究材料（temp/08_callbacks_langsmith.md）：

### 7.1 @traceable装饰器

```python
from langsmith import traceable

@traceable(
    run_type="chain",
    name="my_custom_chain",
    project_name="my-project"
)
def my_chain(input: str) -> str:
    # 自动追踪函数执行
    return llm.invoke(input)

# 使用
result = my_chain("Hello")  # 自动记录到LangSmith
```

### 7.2 背景回调

```python
from langchain_core.callbacks import BackgroundCallbackHandler

class SlowCallback(BackgroundCallbackHandler):
    """背景回调不阻塞执行"""

    def on_llm_end(self, response, **kwargs):
        # 在后台线程执行，不阻塞主流程
        time.sleep(5)
        log_to_slow_service(response)

config = {"callbacks": [SlowCallback()]}
result = chain.invoke(input, config=config)  # 立即返回
```

### 7.3 LangSmith Fetch CLI

```bash
# 2025年12月新工具
langsmith fetch <run-id>  # 在终端查看追踪
langsmith fetch --project my-project --latest  # 查看最新追踪
```

---

## 八、实际应用场景

### 8.1 调试链执行

```python
class DebugCallback(BaseCallbackHandler):
    """调试回调"""

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"\n=== 链开始 ===")
        print(f"输入: {inputs}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\n--- LLM开始 ---")
        print(f"Prompt: {prompts[0][:100]}...")

    def on_llm_end(self, response, **kwargs):
        print(f"\n--- LLM结束 ---")
        print(f"响应: {response.generations[0][0].text[:100]}...")

    def on_chain_end(self, outputs, **kwargs):
        print(f"\n=== 链结束 ===")
        print(f"输出: {outputs}")

# 使用
config = {"callbacks": [DebugCallback()]}
result = chain.invoke(input, config=config)
```

### 8.2 A/B测试追踪

```python
# 不同配置的A/B测试
config_a = {
    "callbacks": [LangChainTracer(project_name="experiment-a")],
    "tags": ["variant-a", "temperature-0.3"]
}

config_b = {
    "callbacks": [LangChainTracer(project_name="experiment-b")],
    "tags": ["variant-b", "temperature-0.9"]
}

# 在LangSmith中比较两个实验的结果
result_a = chain.invoke(input, config=config_a)
result_b = chain.invoke(input, config=config_b)
```

### 8.3 性能分析

```python
import time

class PerformanceCallback(BaseCallbackHandler):
    """性能分析回调"""

    def __init__(self):
        self.timings = {}

    def on_chain_start(self, serialized, inputs, **kwargs):
        run_id = kwargs.get("run_id")
        self.timings[run_id] = {"start": time.time()}

    def on_chain_end(self, outputs, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id in self.timings:
            duration = time.time() - self.timings[run_id]["start"]
            print(f"链执行耗时: {duration:.2f}秒")

# 使用
perf_callback = PerformanceCallback()
config = {"callbacks": [perf_callback]}
```

---

## 九、性能考虑

### 9.1 回调开销

- 同步回调会阻塞执行
- 使用BackgroundCallbackHandler减少延迟
- 避免在回调中执行耗时操作

### 9.2 追踪数据量

- LangSmith追踪会增加网络开销
- 在高流量场景考虑采样
- 使用选择性追踪减少数据量

### 9.3 回调数量

- 过多回调会影响性能
- 合并相似回调逻辑
- 使用条件回调减少不必要的执行

---

## 十、常见陷阱

### 10.1 回调中的错误

```python
# ❌ 回调中的错误会中断执行
class BadCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        raise Exception("错误")  # 会中断链执行

# ✓ 捕获错误
class GoodCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        try:
            # 可能出错的代码
            risky_operation()
        except Exception as e:
            logger.error(f"回调错误: {e}")
```

### 10.2 忘记连接规则

```python
# ❌ 误解：后来的callbacks会替换前面的
base_config = {"callbacks": [CallbackA()]}
custom_config = {"callbacks": [CallbackB()]}
merged = merge_configs(base_config, custom_config)
# 结果: [CallbackA(), CallbackB()]，不是[CallbackB()]
```

### 10.3 状态共享问题

```python
# ❌ 多个调用共享同一个callback实例
callback = MyCallback()
config = {"callbacks": [callback]}

result1 = chain.invoke(input1, config=config)
result2 = chain.invoke(input2, config=config)
# callback的状态可能混淆

# ✓ 每次创建新实例
config1 = {"callbacks": [MyCallback()]}
config2 = {"callbacks": [MyCallback()]}
```

---

## 十一、最佳实践

### 11.1 环境变量优先

```python
# ✓ 使用环境变量配置LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# ❌ 避免硬编码
config = {
    "callbacks": [LangChainTracer(project_name="hardcoded")]
}
```

### 11.2 项目组织

```python
# 为不同环境使用不同项目
def get_langsmith_project(env: str):
    return f"my-app-{env}"

os.environ["LANGSMITH_PROJECT"] = get_langsmith_project("production")
```

### 11.3 Metadata丰富

```python
# 添加丰富的metadata便于分析
config = {
    "callbacks": [LangChainTracer()],
    "metadata": {
        "user_id": "user-123",
        "session_id": "session-456",
        "environment": "production",
        "feature": "chat"
    }
}
```

---

## 十二、下一步

- 并发控制: [核心概念08 - 并发与递归控制](./03_核心概念_08_并发与递归控制.md)
- 配置最佳实践: [核心概念09 - 配置最佳实践](./03_核心概念_09_配置最佳实践.md)
- 实战练习: [实战代码05 - Callbacks集成](./07_实战代码_05_Callbacks集成.md)
