# 实战代码 - 场景5：Token用量统计回调

## 场景说明

**应用场景**：实时追踪 LLM 调用的 token 用量和成本，适用于生产环境的成本控制和性能监控。

**核心价值**：
- 实时成本追踪，避免超支
- 性能监控，优化响应时间
- 用量分析，辅助决策
- 预算管理，控制开支

**RAG 开发价值**：
- RAG 系统通常涉及多次 LLM 调用（检索 + 生成）
- 成本累积快，需要精确追踪
- 帮助优化 Prompt 和 Chunk 策略
- 支持多租户计费

---

## 完整可运行代码

### 方案1：基础 Token 用量统计

```python
"""
Token 用量统计回调 - 基础方案
演示：自定义 CallbackHandler 追踪 token 用量和成本
适用场景：单次调用的成本追踪
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


# ===== 1. 自定义 Token 成本回调处理器 =====
class TokenCostCallback(BaseCallbackHandler):
    """
    追踪 token 用量和成本的回调处理器

    支持：
    - 实时追踪 prompt 和 completion tokens
    - 自动计算成本
    - 累积统计
    """

    def __init__(
        self,
        prompt_cost_per_1m: float = 5.00,      # GPT-4o: $5/1M input tokens
        completion_cost_per_1m: float = 15.00  # GPT-4o: $15/1M output tokens
    ):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.prompt_cost_per_1m = prompt_cost_per_1m
        self.completion_cost_per_1m = completion_cost_per_1m
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        LLM 调用结束时触发

        Args:
            response: LLM 响应，包含 token 用量信息
        """
        self.call_count += 1

        # 提取 token 用量
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            # 累积统计
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # 打印单次调用信息
            print(f"\n--- LLM 调用 #{self.call_count} ---")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Total tokens: {prompt_tokens + completion_tokens}")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要

        Returns:
            包含 token 用量和成本的字典
        """
        # 计算成本
        prompt_cost = (self.total_prompt_tokens / 1_000_000) * self.prompt_cost_per_1m
        completion_cost = (self.total_completion_tokens / 1_000_000) * self.completion_cost_per_1m
        total_cost = prompt_cost + completion_cost

        return {
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "prompt_cost_usd": round(prompt_cost, 6),
            "completion_cost_usd": round(completion_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }

    def reset(self):
        """重置统计"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0


# ===== 2. 测试 Token 统计 =====
print("=== Token 用量统计测试 ===\n")

# 创建回调处理器
callback = TokenCostCallback()

# 初始化 LLM（传入回调）
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 使用更便宜的模型测试
    temperature=0.7,
    callbacks=[callback]
)

# 测试1：简单问答
print("测试1：简单问答")
response1 = llm.invoke([HumanMessage(content="什么是 RAG？")])
print(f"回答: {response1.content[:100]}...")

# 测试2：复杂问答
print("\n测试2：复杂问答")
response2 = llm.invoke([
    HumanMessage(content="详细解释 RAG 系统的架构，包括检索、重排序和生成三个阶段。")
])
print(f"回答: {response2.content[:100]}...")

# 测试3：多轮对话
print("\n测试3：多轮对话")
response3 = llm.invoke([
    HumanMessage(content="如何优化 RAG 系统的检索性能？给出5个具体建议。")
])
print(f"回答: {response3.content[:100]}...")

# ===== 3. 输出统计摘要 =====
print("\n" + "="*50)
print("=== 统计摘要 ===")
summary = callback.get_summary()
for key, value in summary.items():
    print(f"{key}: {value}")
```

---

### 方案2：带延迟统计的增强版

```python
"""
Token 用量统计回调 - 增强版
演示：同时追踪 token 用量、成本和延迟
适用场景：性能监控和成本优化
"""

import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class PerformanceCallback(BaseCallbackHandler):
    """
    性能监控回调处理器

    追踪：
    - Token 用量和成本
    - 端到端延迟
    - 每次调用的详细信息
    """

    def __init__(
        self,
        prompt_cost_per_1m: float = 0.15,      # GPT-4o-mini: $0.15/1M input
        completion_cost_per_1m: float = 0.60   # GPT-4o-mini: $0.60/1M output
    ):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.prompt_cost_per_1m = prompt_cost_per_1m
        self.completion_cost_per_1m = completion_cost_per_1m

        # 延迟追踪
        self.start_time: Optional[float] = None
        self.total_latency = 0.0
        self.call_count = 0

        # 详细记录
        self.call_details: List[Dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """LLM 开始时记录时间"""
        self.start_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 结束时计算延迟和成本"""
        # 计算延迟
        latency = time.time() - self.start_time if self.start_time else 0
        self.total_latency += latency
        self.call_count += 1

        # 提取 token 用量
        prompt_tokens = 0
        completion_tokens = 0

        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

        # 计算单次成本
        call_cost = (
            (prompt_tokens / 1_000_000) * self.prompt_cost_per_1m +
            (completion_tokens / 1_000_000) * self.completion_cost_per_1m
        )

        # 记录详细信息
        detail = {
            "call_number": self.call_count,
            "latency_seconds": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(call_cost, 6)
        }
        self.call_details.append(detail)

        # 打印实时信息
        print(f"\n[调用 #{self.call_count}] "
              f"延迟: {latency:.2f}s | "
              f"Tokens: {prompt_tokens + completion_tokens} | "
              f"成本: ${call_cost:.6f}")

    def get_summary(self) -> Dict[str, Any]:
        """获取完整统计摘要"""
        total_cost = (
            (self.total_prompt_tokens / 1_000_000) * self.prompt_cost_per_1m +
            (self.total_completion_tokens / 1_000_000) * self.completion_cost_per_1m
        )

        avg_latency = self.total_latency / self.call_count if self.call_count > 0 else 0

        return {
            "summary": {
                "total_calls": self.call_count,
                "total_latency_seconds": round(self.total_latency, 3),
                "avg_latency_seconds": round(avg_latency, 3),
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost_usd": round(total_cost, 6)
            },
            "details": self.call_details
        }


# ===== 测试性能监控 =====
print("=== 性能监控测试 ===\n")

callback = PerformanceCallback()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    callbacks=[callback]
)

# 模拟 RAG 场景：多次调用
questions = [
    "什么是向量数据库？",
    "如何选择合适的 Embedding 模型？",
    "RAG 系统如何处理长文档？"
]

for i, question in enumerate(questions, 1):
    print(f"\n问题 {i}: {question}")
    response = llm.invoke([HumanMessage(content=question)])
    print(f"回答: {response.content[:80]}...")

# 输出完整统计
print("\n" + "="*60)
print("=== 完整统计报告 ===")
summary = callback.get_summary()

print("\n【总体统计】")
for key, value in summary["summary"].items():
    print(f"  {key}: {value}")

print("\n【详细记录】")
for detail in summary["details"]:
    print(f"  调用 #{detail['call_number']}: "
          f"{detail['latency_seconds']}s, "
          f"{detail['total_tokens']} tokens, "
          f"${detail['cost_usd']}")
```

---

### 方案3：RAG 场景的成本追踪

```python
"""
Token 用量统计回调 - RAG 场景
演示：在完整 RAG 管道中追踪成本
适用场景：生产环境的 RAG 系统
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGCostCallback(BaseCallbackHandler):
    """RAG 场景的成本追踪"""

    def __init__(self):
        super().__init__()
        # LLM 成本（GPT-4o-mini）
        self.llm_prompt_cost_per_1m = 0.15
        self.llm_completion_cost_per_1m = 0.60

        # Embedding 成本（text-embedding-3-small）
        self.embedding_cost_per_1m = 0.02

        # 统计
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.embedding_tokens = 0
        self.retrieval_count = 0
        self.generation_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """追踪 LLM 调用"""
        self.generation_count += 1

        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            self.llm_prompt_tokens += usage.get('prompt_tokens', 0)
            self.llm_completion_tokens += usage.get('completion_tokens', 0)

    def on_retriever_end(self, documents, **kwargs: Any) -> None:
        """追踪检索调用"""
        self.retrieval_count += 1
        # 估算 embedding tokens（假设每个文档平均 500 tokens）
        self.embedding_tokens += len(documents) * 500

    def get_rag_summary(self) -> Dict[str, Any]:
        """获取 RAG 场景的成本摘要"""
        # 计算各部分成本
        llm_cost = (
            (self.llm_prompt_tokens / 1_000_000) * self.llm_prompt_cost_per_1m +
            (self.llm_completion_tokens / 1_000_000) * self.llm_completion_cost_per_1m
        )

        embedding_cost = (self.embedding_tokens / 1_000_000) * self.embedding_cost_per_1m

        total_cost = llm_cost + embedding_cost

        return {
            "retrieval": {
                "count": self.retrieval_count,
                "estimated_tokens": self.embedding_tokens,
                "cost_usd": round(embedding_cost, 6)
            },
            "generation": {
                "count": self.generation_count,
                "prompt_tokens": self.llm_prompt_tokens,
                "completion_tokens": self.llm_completion_tokens,
                "cost_usd": round(llm_cost, 6)
            },
            "total": {
                "total_tokens": self.llm_prompt_tokens + self.llm_completion_tokens + self.embedding_tokens,
                "total_cost_usd": round(total_cost, 6)
            }
        }


# ===== 模拟 RAG 管道 =====
print("=== RAG 成本追踪测试 ===\n")

callback = RAGCostCallback()

# 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

# 模拟检索结果
retrieved_docs = [
    "RAG 系统结合了检索和生成...",
    "向量数据库用于存储文档的向量表示...",
    "Embedding 模型将文本转换为向量..."
]

# 模拟检索事件
callback.on_retriever_end(retrieved_docs)

# 构建 RAG 提示
prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题：

上下文：
{context}

问题：{question}

回答：
""")

# 执行生成
chain = (
    {"context": lambda x: "\n".join(retrieved_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "什么是 RAG 系统？"
print(f"问题: {question}\n")
answer = chain.invoke(question)
print(f"回答: {answer[:100]}...\n")

# 输出 RAG 成本摘要
print("="*60)
print("=== RAG 成本摘要 ===")
summary = callback.get_rag_summary()

print("\n【检索阶段】")
for key, value in summary["retrieval"].items():
    print(f"  {key}: {value}")

print("\n【生成阶段】")
for key, value in summary["generation"].items():
    print(f"  {key}: {value}")

print("\n【总计】")
for key, value in summary["total"].items():
    print(f"  {key}: {value}")
```

---

## 运行输出示例

### 方案1 输出：

```
=== Token 用量统计测试 ===

测试1：简单问答

--- LLM 调用 #1 ---
Prompt tokens: 15
Completion tokens: 87
Total tokens: 102
回答: RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的 AI 架构...

测试2：复杂问答

--- LLM 调用 #2 ---
Prompt tokens: 28
Completion tokens: 245
Total tokens: 273
回答: RAG 系统的架构包括三个核心阶段：1. 检索阶段：从向量数据库中检索相关文档...

测试3：多轮对话

--- LLM 调用 #3 ---
Prompt tokens: 22
Completion tokens: 198
Total tokens: 220
回答: 优化 RAG 系统检索性能的5个建议：1. 使用混合检索策略...

==================================================
=== 统计摘要 ===
total_calls: 3
total_prompt_tokens: 65
total_completion_tokens: 530
total_tokens: 595
prompt_cost_usd: 0.000098
completion_cost_usd: 0.000318
total_cost_usd: 0.000416
```

### 方案2 输出：

```
=== 性能监控测试 ===

问题 1: 什么是向量数据库？

[调用 #1] 延迟: 1.23s | Tokens: 156 | 成本: $0.000117
回答: 向量数据库是专门用于存储和检索高维向量的数据库系统...

问题 2: 如何选择合适的 Embedding 模型？

[调用 #2] 延迟: 1.45s | Tokens: 234 | 成本: $0.000168
回答: 选择 Embedding 模型需要考虑以下因素：1. 向量维度...

问题 3: RAG 系统如何处理长文档？

[调用 #3] 延迟: 1.67s | Tokens: 289 | 成本: $0.000201
回答: RAG 系统处理长文档的策略包括：1. 文本分块...

============================================================
=== 完整统计报告 ===

【总体统计】
  total_calls: 3
  total_latency_seconds: 4.35
  avg_latency_seconds: 1.45
  total_tokens: 679
  total_cost_usd: 0.000486

【详细记录】
  调用 #1: 1.23s, 156 tokens, $0.000117
  调用 #2: 1.45s, 234 tokens, $0.000168
  调用 #3: 1.67s, 289 tokens, $0.000201
```

### 方案3 输出：

```
=== RAG 成本追踪测试 ===

问题: 什么是 RAG 系统？

回答: 基于提供的上下文，RAG（Retrieval-Augmented Generation）系统是一种结合了检索和生成的 AI 架构...

============================================================
=== RAG 成本摘要 ===

【检索阶段】
  count: 1
  estimated_tokens: 1500
  cost_usd: 0.00003

【生成阶段】
  count: 1
  prompt_tokens: 145
  completion_tokens: 98
  cost_usd: 0.000081

【总计】
  total_tokens: 1743
  total_cost_usd: 0.000111
```

---

## 关键技术点

### 1. Token 用量提取

```python
# 从 LLMResult 中提取 token 用量
if response.llm_output and 'token_usage' in response.llm_output:
    usage = response.llm_output['token_usage']
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
```

### 2. 成本计算公式

```python
# 成本 = (tokens / 1,000,000) × 每百万 tokens 价格
prompt_cost = (prompt_tokens / 1_000_000) * prompt_cost_per_1m
completion_cost = (completion_tokens / 1_000_000) * completion_cost_per_1m
total_cost = prompt_cost + completion_cost
```

### 3. 延迟测量

```python
# 在 on_llm_start 记录开始时间
def on_llm_start(self, ...):
    self.start_time = time.time()

# 在 on_llm_end 计算延迟
def on_llm_end(self, ...):
    latency = time.time() - self.start_time
```

---

## 主流模型定价参考（2026年）

| 模型 | Input ($/1M tokens) | Output ($/1M tokens) | 适用场景 |
|------|---------------------|----------------------|---------|
| GPT-4o | $5.00 | $15.00 | 高质量生成 |
| GPT-4o-mini | $0.15 | $0.60 | 日常应用 |
| GPT-3.5-turbo | $0.50 | $1.50 | 简单任务 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 复杂推理 |
| text-embedding-3-small | $0.02 | - | Embedding |
| text-embedding-3-large | $0.13 | - | 高质量 Embedding |

---

## 与 RAG 开发的联系

### 1. RAG 系统的成本结构

```
总成本 = Embedding 成本 + 检索成本 + 生成成本

- Embedding 成本：文档向量化（一次性）
- 检索成本：查询向量化（每次查询）
- 生成成本：LLM 生成（每次查询）
```

### 2. 成本优化策略

```python
# 策略1：缓存 Embedding
# 避免重复向量化相同文档

# 策略2：优化 Chunk 大小
# 减少检索的文档数量

# 策略3：使用更便宜的模型
# GPT-4o-mini 而非 GPT-4o

# 策略4：Prompt 压缩
# 减少 prompt tokens
```

### 3. 预算管理

```python
# 设置成本上限
class BudgetCallback(BaseCallbackHandler):
    def __init__(self, max_cost_usd: float = 1.0):
        self.max_cost = max_cost_usd
        self.current_cost = 0.0

    def on_llm_end(self, response, **kwargs):
        # 计算成本
        cost = self.calculate_cost(response)
        self.current_cost += cost

        # 检查是否超预算
        if self.current_cost > self.max_cost:
            raise Exception(f"超出预算！当前成本: ${self.current_cost:.6f}")
```

---

## 常见问题

### 问题1：不同模型的 token 计数不一致

**原因**：不同模型使用不同的 tokenizer

**解决**：
```python
# 使用 tiktoken 精确计算
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode("你的文本")
token_count = len(tokens)
```

### 问题2：Embedding 成本难以追踪

**原因**：Embedding 调用通常不返回 token 用量

**解决**：
```python
# 估算方法
estimated_tokens = len(text.split()) * 1.3  # 英文
estimated_tokens = len(text) * 0.5  # 中文
```

### 问题3：流式输出的 token 统计

**原因**：流式输出时 token 逐个返回

**解决**：
```python
class StreamingCostCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        # 累积 token
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs):
        # 最终统计
        total_tokens = len(self.tokens)
```

---

## 总结

Token 用量统计回调是生产环境 RAG 系统的必备组件，通过自定义 CallbackHandler 可以实现精确的成本追踪和性能监控。关键是正确提取 token 用量、准确计算成本、合理设置预算上限。

---

## 下一步学习

掌握 Token 用量统计回调后，建议：

1. **延迟和成本计算**：阅读 **07_实战代码_场景6_延迟和成本计算.md**
   - 学习如何同时追踪延迟和成本
   - 实现端到端性能监控

2. **可观测性集成**：阅读 **07_实战代码_场景7_Langfuse集成实战.md**
   - 学习如何将成本数据发送到可观测性平台
   - 实现生产级的成本监控

3. **本地 LLM 对比**：阅读 **07_实战代码_场景4_本地LLM流式输出.md**
   - 对比本地 LLM 与云端 API 的成本
   - 学习如何选择合适的部署方案

4. **预算管理**：深入学习
   - 实现成本预警机制
   - 多租户成本分摊
   - 成本优化策略

---

## 学习检查清单

完成以下检查，确认你已掌握 Token 用量统计回调：

- [ ] 能够创建自定义 TokenCostCallback 类
- [ ] 理解如何从 LLMResult 中提取 token 用量
- [ ] 能够计算 prompt 和 completion 的成本
- [ ] 能够实现累积统计（多次调用）
- [ ] 能够追踪延迟（on_llm_start 和 on_llm_end）
- [ ] 理解主流模型的定价（GPT-4o、GPT-4o-mini、Claude）
- [ ] 能够在 RAG 场景中追踪成本（检索 + 生成）
- [ ] 能够估算 Embedding 的成本
- [ ] 知道如何优化成本（缓存、Prompt 压缩、模型选择）
- [ ] 能够实现预算管理（设置成本上限）
- [ ] 理解不同模型的 token 计数差异
- [ ] 知道如何处理流式输出的 token 统计

---

**记住**：成本追踪是生产环境 RAG 系统的必备能力，掌握 Token 用量统计后，你就能精确控制 AI 应用的运营成本！

---

**版本**：v1.0
**最后更新**：2026-02-25
**维护者**：Claude Code
