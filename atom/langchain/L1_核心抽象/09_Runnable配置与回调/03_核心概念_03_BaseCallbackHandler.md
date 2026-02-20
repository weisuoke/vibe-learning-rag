# 核心概念 03：BaseCallbackHandler

> **BaseCallbackHandler 的完整方法列表、参数详解和实现模式**

---

## 概述

`BaseCallbackHandler` 是所有回调的基类，提供了 15+ 个事件方法，覆盖 LLM、Chain、Tool、Retriever、Agent 等所有组件。

**引用**：
> "BaseCallbackHandler provides hooks into the various stages of your LLM application."
> — [BaseCallbackHandler API](https://reference.langchain.com/v0.3/python/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html)

---

## 完整方法列表

### LLM 方法（5个）

```python
class BaseCallbackHandler:
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM 开始调用"""
        pass

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs) -> None:
        """Chat Model 开始调用"""
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """流式输出：每个 token 生成时"""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束"""
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """LLM 调用失败"""
        pass
```

### Chain 方法（3个）

```python
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Chain 开始执行"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Chain 执行结束"""
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Chain 执行失败"""
        pass
```

### Tool 方法（3个）

```python
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """工具开始调用"""
        pass

    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具调用结束"""
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """工具调用失败"""
        pass
```

### Retriever 方法（3个）

```python
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs) -> None:
        """检索开始"""
        pass

    def on_retriever_end(self, documents: List[Document], **kwargs) -> None:
        """检索结束"""
        pass

    def on_retriever_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """检索失败"""
        pass
```

### Agent 方法（2个）

```python
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Agent 执行动作"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Agent 完成任务"""
        pass
```

### Text 方法（1个）

```python
    def on_text(self, text: str, **kwargs) -> None:
        """任意文本输出"""
        pass
```

---

## 常用方法详解

### on_llm_end - 最常用

**参数详解**：

```python
def on_llm_end(self, response: LLMResult, **kwargs) -> None:
    # response.generations: 生成的文本列表
    text = response.generations[0][0].text

    # response.llm_output: LLM 输出信息
    token_usage = response.llm_output.get("token_usage", {})
    # - prompt_tokens: int
    # - completion_tokens: int
    # - total_tokens: int

    model_name = response.llm_output.get("model_name")

    # kwargs: 额外信息
    tags = kwargs.get("tags", [])
    metadata = kwargs.get("metadata", {})
    run_id = kwargs.get("run_id")
```

**完整示例**：

```python
class ComprehensiveMonitor(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        # 1. 获取生成文本
        text = response.generations[0][0].text

        # 2. 获取 token 使用量
        usage = response.llm_output.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # 3. 计算成本
        cost = prompt_tokens * 0.00003 + completion_tokens * 0.00006

        # 4. 获取元数据
        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id", "unknown")

        # 5. 记录日志
        logger.info("LLM 调用完成", extra={
            "user_id": user_id,
            "tokens": total_tokens,
            "cost": cost,
            "output_length": len(text)
        })
```

---

### on_retriever_end - RAG 必备

**参数详解**：

```python
def on_retriever_end(self, documents: List[Document], **kwargs) -> None:
    # documents: 检索到的文档列表
    for doc in documents:
        doc.page_content  # 文档内容
        doc.metadata      # 文档元数据
        # - source: 来源
        # - score: 相似度分数
        # - page: 页码
```

**完整示例**：

```python
class RetrievalQualityMonitor(BaseCallbackHandler):
    def on_retriever_end(self, documents, **kwargs):
        # 1. 统计文档数量
        num_docs = len(documents)

        # 2. 计算平均相似度
        scores = [d.metadata.get("score", 0) for d in documents]
        avg_score = sum(scores) / num_docs if num_docs > 0 else 0

        # 3. 分析文档来源
        sources = [d.metadata.get("source", "unknown") for d in documents]
        unique_sources = len(set(sources))

        # 4. 记录指标
        metrics.record("retrieval.num_docs", num_docs)
        metrics.record("retrieval.avg_score", avg_score)
        metrics.record("retrieval.unique_sources", unique_sources)

        # 5. 告警：检索质量低
        if avg_score < 0.5:
            alert("检索质量低", avg_score=avg_score, num_docs=num_docs)
```

---

## 实现模式

### 模式 1：成本追踪

```python
class CostTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_cost = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * 0.00003 +
            usage.get("completion_tokens", 0) * 0.00006
        )
        self.total_cost += cost
        self.total_tokens += usage.get("total_tokens", 0)
```

### 模式 2：性能监控

```python
class PerformanceMonitor(BaseCallbackHandler):
    def __init__(self):
        self.start_times = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        duration = time.time() - self.start_times.pop(run_id, time.time())
        metrics.record("llm.duration", duration)
```

### 模式 3：错误告警

```python
class ErrorAlerter(BaseCallbackHandler):
    def on_llm_error(self, error, **kwargs):
        metadata = kwargs.get("metadata", {})
        alert("LLM 调用失败", error=str(error), metadata=metadata)

    def on_chain_error(self, error, **kwargs):
        alert("Chain 执行失败", error=str(error))

    def on_retriever_error(self, error, **kwargs):
        alert("检索失败", error=str(error))
```

### 模式 4：流式监控

```python
class StreamMonitor(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        elapsed = time.time() - self.start_time
        speed = len(self.tokens) / elapsed
        print(f"速度: {speed:.1f} tokens/s")
```

---

## 过滤机制

### 使用 ignore_* 属性

```python
class ChainOnlyCallback(BaseCallbackHandler):
    ignore_llm = True        # 忽略 LLM 事件
    ignore_agent = True      # 忽略 Agent 事件
    ignore_retriever = True  # 忽略 Retriever 事件

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("只监控 Chain")
```

### 使用 tags 过滤

```python
class ProductionOnlyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        tags = kwargs.get("tags", [])
        if "production" not in tags:
            return  # 非生产环境不执行
        # 生产环境逻辑
```

---

## 异步回调

```python
from langchain_core.callbacks import AsyncCallbackHandler

class AsyncMonitor(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        await asyncio.sleep(0.1)
        print("异步回调")

    async def on_llm_end(self, response, **kwargs):
        await send_to_monitoring_async(response)
```

---

## 完整生产示例

```python
import time
import logging
from collections import defaultdict
from langchain_core.callbacks import BaseCallbackHandler

class ProductionCallback(BaseCallbackHandler):
    """生产环境完整监控"""

    def __init__(self):
        self.start_times = {}
        self.user_costs = defaultdict(float)

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()

        metadata = kwargs.get("metadata", {})
        logger.info("LLM 开始", extra={
            "user_id": metadata.get("user_id"),
            "model": serialized.get("name")
        })

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        duration = time.time() - self.start_times.pop(run_id, time.time())

        usage = response.llm_output.get("token_usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * 0.00003 +
            usage.get("completion_tokens", 0) * 0.00006
        )

        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id", "unknown")
        self.user_costs[user_id] += cost

        logger.info("LLM 完成", extra={
            "user_id": user_id,
            "duration": duration,
            "tokens": usage.get("total_tokens", 0),
            "cost": cost
        })

        if duration > 5.0:
            alert("延迟过高", duration=duration)

    def on_llm_error(self, error, **kwargs):
        metadata = kwargs.get("metadata", {})
        logger.error(f"LLM 失败: {error}", extra=metadata)
        alert("LLM 失败", error=str(error))

    def on_retriever_end(self, documents, **kwargs):
        num_docs = len(documents)
        scores = [d.metadata.get("score", 0) for d in documents]
        avg_score = sum(scores) / num_docs if num_docs > 0 else 0

        logger.info("检索完成", extra={
            "num_docs": num_docs,
            "avg_score": avg_score
        })

        if avg_score < 0.5:
            alert("检索质量低", avg_score=avg_score)
```

---

## 总结

### 方法速查表

| 组件 | 开始 | 结束 | 错误 | 其他 |
|------|------|------|------|------|
| LLM | `on_llm_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token` |
| Chain | `on_chain_start` | `on_chain_end` | `on_chain_error` | - |
| Tool | `on_tool_start` | `on_tool_end` | `on_tool_error` | - |
| Retriever | `on_retriever_start` | `on_retriever_end` | `on_retriever_error` | - |
| Agent | - | `on_agent_finish` | - | `on_agent_action` |

### 最常用的 3 个方法

1. **on_llm_end**：成本追踪、性能监控
2. **on_llm_error**：错误告警
3. **on_retriever_end**：RAG 检索质量监控

---

## 参考资料

- [BaseCallbackHandler API](https://reference.langchain.com/v0.3/python/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html)
- [Custom Callbacks Guide](https://python.langchain.com/docs/how_to/custom_callbacks/)
