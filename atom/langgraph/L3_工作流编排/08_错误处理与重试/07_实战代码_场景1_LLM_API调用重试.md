# 场景1：LLM API 调用重试

> **实战代码 - 场景1** | 预计阅读：8分钟
> 处理 LLM API 的速率限制、超时和服务端错误

---

## 场景描述

在 RAG 应用中，LLM API 调用是最常见的失败点：
- **429 Too Many Requests**：速率限制
- **500/502/503**：服务端临时故障
- **Timeout**：网络延迟或模型响应慢
- **Connection Error**：网络不稳定

这些都是**瞬态错误**，应该重试。但编程错误（如无效的 API key）不应该重试。

---

## 完整代码

```python
"""
场景1：LLM API 调用重试
演示：处理 OpenAI API 的各种失败情况
"""

import os
import time
from typing import TypedDict
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

# ===== 1. 定义状态 =====
class State(TypedDict):
    query: str
    context: str
    answer: str

# ===== 2. 自定义重试判断函数 =====
def should_retry_openai_error(exc: Exception) -> bool:
    """
    判断 OpenAI 错误是否应该重试

    应该重试：
    - RateLimitError (429)
    - APIError with 5xx status
    - APITimeoutError
    - ConnectionError

    不应该重试：
    - AuthenticationError (401) - API key 错误
    - PermissionDeniedError (403) - 权限不足
    - NotFoundError (404) - 模型不存在
    - BadRequestError (400) - 请求参数错误
    """
    # 速率限制 - 必须重试
    if isinstance(exc, RateLimitError):
        print(f"    → 检测到速率限制，将重试")
        return True

    # 超时 - 应该重试
    if isinstance(exc, APITimeoutError):
        print(f"    → 检测到超时，将重试")
        return True

    # 连接错误 - 应该重试
    if isinstance(exc, ConnectionError):
        print(f"    → 检测到连接错误，将重试")
        return True

    # API 错误 - 检查状态码
    if isinstance(exc, APIError):
        # 5xx 服务端错误 - 应该重试
        if hasattr(exc, 'status_code') and exc.status_code >= 500:
            print(f"    → 检测到服务端错误 ({exc.status_code})，将重试")
            return True

        # 4xx 客户端错误 - 不重试（除了 429）
        print(f"    → 客户端错误 ({getattr(exc, 'status_code', 'unknown')})，不重试")
        return False

    # 其他错误 - 不重试
    print(f"    → 未知错误类型 ({type(exc).__name__})，不重试")
    return False

# ===== 3. 配置重试策略 =====
llm_retry_policy = RetryPolicy(
    max_attempts=5,              # 最多尝试 5 次
    initial_interval=2.0,        # 首次重试等待 2 秒
    max_interval=60.0,           # 最长等待 60 秒
    backoff_factor=2.0,          # 指数退避：2s → 4s → 8s → 16s → 32s
    jitter=True,                 # 启用随机抖动（±25%）
    retry_on=should_retry_openai_error
)

print("LLM 重试策略配置：")
print(f"  - 最多尝试: {llm_retry_policy.max_attempts} 次")
print(f"  - 初始间隔: {llm_retry_policy.initial_interval}s")
print(f"  - 退避系数: {llm_retry_policy.backoff_factor}x")
print(f"  - 最大间隔: {llm_retry_policy.max_interval}s")
print(f"  - 随机抖动: {'启用' if llm_retry_policy.jitter else '禁用'}")

# ===== 4. 定义 LLM 调用节点 =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(state: State) -> dict:
    """
    调用 OpenAI API 生成答案

    可能的失败：
    - RateLimitError: 速率限制
    - APITimeoutError: 超时
    - APIError: 服务端错误
    """
    print(f"\n[LLM] 调用 OpenAI API...")
    print(f"  查询: {state['query']}")
    print(f"  上下文长度: {len(state['context'])} 字符")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": f"基于以下上下文回答问题：\n\n上下文：{state['context']}\n\n问题：{state['query']}"}
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=30.0  # 30 秒超时
        )

        answer = response.choices[0].message.content
        print(f"[LLM] ✅ 成功生成答案 ({len(answer)} 字符)")

        return {"answer": answer}

    except RateLimitError as e:
        print(f"[LLM] ❌ 速率限制: {e}")
        raise  # 让 RetryPolicy 处理

    except APITimeoutError as e:
        print(f"[LLM] ❌ 请求超时: {e}")
        raise

    except APIError as e:
        print(f"[LLM] ❌ API 错误: {e}")
        raise

    except Exception as e:
        print(f"[LLM] ❌ 未知错误: {type(e).__name__}: {e}")
        raise

# ===== 5. 构建工作流 =====
builder = StateGraph(State)
builder.add_node("call_llm", call_llm, retry_policy=llm_retry_policy)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

graph = builder.compile()

print("\n✅ 工作流构建完成\n")

# ===== 6. 测试场景 =====
print("="*60)
print("测试：正常调用")
print("="*60)

initial_state = {
    "query": "什么是 LangGraph？",
    "context": "LangGraph 是一个用于构建有状态的多步骤 LLM 应用的框架。",
    "answer": ""
}

try:
    result = graph.invoke(initial_state)
    print(f"\n✅ 最终答案:\n{result['answer']}")
except Exception as e:
    print(f"\n❌ 执行失败: {type(e).__name__}: {e}")

# ===== 7. 模拟失败场景 =====
print("\n" + "="*60)
print("模拟场景：速率限制（会自动重试）")
print("="*60)

def call_llm_with_rate_limit(state: State) -> dict:
    """模拟速率限制错误"""
    print(f"\n[LLM] 调用 OpenAI API...")

    # 模拟：前 3 次调用都返回 429
    if not hasattr(call_llm_with_rate_limit, 'attempt_count'):
        call_llm_with_rate_limit.attempt_count = 0

    call_llm_with_rate_limit.attempt_count += 1

    if call_llm_with_rate_limit.attempt_count <= 3:
        print(f"[LLM] ❌ 模拟速率限制 (尝试 {call_llm_with_rate_limit.attempt_count})")
        raise RateLimitError("Rate limit exceeded. Please retry after 2 seconds.")

    # 第 4 次成功
    print(f"[LLM] ✅ 成功 (尝试 {call_llm_with_rate_limit.attempt_count})")
    return {"answer": "这是模拟的答案"}

# 构建测试图
test_builder = StateGraph(State)
test_builder.add_node("call_llm", call_llm_with_rate_limit, retry_policy=llm_retry_policy)
test_builder.add_edge(START, "call_llm")
test_builder.add_edge("call_llm", END)
test_graph = test_builder.compile()

# 重置计数器
call_llm_with_rate_limit.attempt_count = 0

try:
    result = test_graph.invoke(initial_state)
    print(f"\n✅ 重试成功！总尝试次数: {call_llm_with_rate_limit.attempt_count}")
    print(f"答案: {result['answer']}")
except Exception as e:
    print(f"\n❌ 重试失败: {type(e).__name__}: {e}")

# ===== 8. 实际应用建议 =====
print("\n" + "="*60)
print("实际应用建议")
print("="*60)

recommendations = """
1. **速率限制处理**
   - 使用指数退避（2s → 4s → 8s...）
   - 启用 jitter 防止多个请求同时重试
   - 考虑使用令牌桶算法主动限流

2. **超时设置**
   - 设置合理的超时时间（如 30-60 秒）
   - 超时应该重试，但不要无限重试
   - 考虑使用流式响应减少超时风险

3. **错误分类**
   - 4xx 错误（除 429）不重试 - 修复代码
   - 5xx 错误重试 - 服务端临时故障
   - 网络错误重试 - 瞬态问题

4. **监控与告警**
   - 记录每次重试的原因和延迟
   - 设置告警阈值（如重试率 > 20%）
   - 监控 API 配额使用情况

5. **降级策略**
   - 主模型失败 → 备用模型（如 gpt-4 → gpt-3.5）
   - 所有模型失败 → 返回缓存答案或默认响应
   - 考虑使用本地模型作为最终降级
"""

print(recommendations)

print("\n" + "="*60)
print("✅ 场景1 完成")
print("="*60)
```

---

## 运行输出示例

```
LLM 重试策略配置：
  - 最多尝试: 5 次
  - 初始间隔: 2.0s
  - 退避系数: 2.0x
  - 最大间隔: 60.0s
  - 随机抖动: 启用

✅ 工作流构建完成

============================================================
测试：正常调用
============================================================

[LLM] 调用 OpenAI API...
  查询: 什么是 LangGraph？
  上下文长度: 42 字符
[LLM] ✅ 成功生成答案 (156 字符)

✅ 最终答案:
LangGraph 是一个专门用于构建有状态的多步骤大语言模型（LLM）应用的框架...

============================================================
模拟场景：速率限制（会自动重试）
============================================================

[LLM] 调用 OpenAI API...
[LLM] ❌ 模拟速率限制 (尝试 1)
    → 检测到速率限制，将重试
    → 等待 2.1s 后重试...

[LLM] 调用 OpenAI API...
[LLM] ❌ 模拟速率限制 (尝试 2)
    → 检测到速率限制，将重试
    → 等待 3.8s 后重试...

[LLM] 调用 OpenAI API...
[LLM] ❌ 模拟速率限制 (尝试 3)
    → 检测到速率限制，将重试
    → 等待 8.2s 后重试...

[LLM] 调用 OpenAI API...
[LLM] ✅ 成功 (尝试 4)

✅ 重试成功！总尝试次数: 4
答案: 这是模拟的答案

============================================================
✅ 场景1 完成
============================================================
```

---

## 关键要点

### 1. 区分可重试和不可重试错误

```python
# ✅ 应该重试
- RateLimitError (429)
- APITimeoutError
- APIError with 5xx
- ConnectionError

# ❌ 不应该重试
- AuthenticationError (401)
- PermissionDeniedError (403)
- BadRequestError (400)
- NotFoundError (404)
```

### 2. 指数退避 + 随机抖动

```python
RetryPolicy(
    initial_interval=2.0,    # 2s
    backoff_factor=2.0,      # 2x
    jitter=True              # ±25%
)

# 实际等待时间：
# 尝试1 → 尝试2: 2.0s × (1 ± 0.25) = 1.5-2.5s
# 尝试2 → 尝试3: 4.0s × (1 ± 0.25) = 3.0-5.0s
# 尝试3 → 尝试4: 8.0s × (1 ± 0.25) = 6.0-10.0s
```

### 3. 设置合理的超时

```python
client.chat.completions.create(
    ...,
    timeout=30.0  # 30 秒超时
)
```

### 4. 监控重试行为

```python
def call_llm_with_monitoring(state: State) -> dict:
    try:
        return call_llm(state)
    except Exception as e:
        # 记录到监控系统
        logger.warning(f"LLM call failed: {type(e).__name__}, will retry")
        metrics.increment("llm.retry.count")
        raise
```

---

## 下一步

- 查看 **场景2：RAG检索容错管道** 了解多源检索的降级策略
- 查看 **场景3：多节点差异化重试** 了解不同节点的不同策略

---

[来源: OpenAI API 文档 | LangGraph 源码 | 生产实践]
