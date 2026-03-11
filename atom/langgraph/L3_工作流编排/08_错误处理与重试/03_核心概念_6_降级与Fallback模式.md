# 降级与 Fallback 模式

> **核心概念 6/6** | 预计阅读：7分钟
> **来源**：社区最佳实践 | 生产级 LLM 应用模式

---

## 什么是降级与 Fallback？

**降级 = 当正常功能不可用时，提供一个"够用但不完美"的替代方案。**
**Fallback = 主方案失败后的备选方案链。**

RetryPolicy 解决了"怎么重试"的问题，但重试耗尽后呢？Fallback 回答的是**"重试全失败了怎么办"**。

```
正常服务 → 重试 → 重试 → 重试 → 全失败了！
                                    ↓
                              Fallback 介入
                              ├── 备用模型
                              ├── 缓存结果
                              └── 静态默认值
```

---

## Fallback 链设计

### 典型的 4 层 Fallback

```
层级1：主模型（GPT-4）
  ↓ 失败
层级2：备用模型（Claude / Gemini）
  ↓ 失败
层级3：缓存结果（历史相似问题的答案）
  ↓ 失败
层级4：静态默认值（"抱歉，服务暂时不可用"）
```

**设计原则：每一层都是更低成本/更低质量的替代，但确保系统始终有响应。**

### 在 LangGraph 中实现 Fallback 链

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    query: str
    answer: str
    fallback_level: int
    error_info: str

def call_primary_llm(state: State) -> dict:
    """层级1：调用主模型。"""
    try:
        # 调用 GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": state["query"]}]
        )
        return {"answer": response.choices[0].message.content, "fallback_level": 0}
    except Exception as e:
        return {"error_info": str(e), "fallback_level": 1}

def call_backup_llm(state: State) -> dict:
    """层级2：调用备用模型。"""
    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": state["query"]}],
            max_tokens=1024,
        )
        return {"answer": response.content[0].text, "fallback_level": 1}
    except Exception as e:
        return {"error_info": str(e), "fallback_level": 2}

def use_cache(state: State) -> dict:
    """层级3：查找缓存。"""
    cached = cache.get_similar(state["query"])
    if cached:
        return {"answer": f"[缓存结果] {cached}", "fallback_level": 2}
    return {"answer": "抱歉，服务暂时不可用，请稍后再试。", "fallback_level": 3}

def route_by_fallback(state: State) -> str:
    """根据 fallback_level 路由。"""
    if state.get("answer") and state["fallback_level"] < 2:
        return END
    level = state.get("fallback_level", 0)
    if level == 0:
        return "call_primary"
    elif level == 1:
        return "call_backup"
    else:
        return "use_cache"

# 构建图
builder = StateGraph(State)
builder.add_node("call_primary", call_primary_llm,
                 retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("call_backup", call_backup_llm,
                 retry_policy=RetryPolicy(max_attempts=2))
builder.add_node("use_cache", use_cache)
```

---

## 断路器模式（Circuit Breaker）

### 什么是断路器？

**断路器 = 当一个服务持续失败时，自动停止向它发送请求。**

就像家里的保险丝：电流过大时自动断开，防止火灾。

### 三种状态

```
┌─────────┐   连续失败超过阈值   ┌──────────┐
│  关闭    │ ──────────────────→ │  打开     │
│ (正常)   │                     │ (阻断)    │
│ 所有请求 │                     │ 直接走    │
│ 正常发送 │                     │ Fallback  │
└─────────┘                     └──────────┘
     ↑                                │
     │  测试请求成功                    │ 超时后
     │                                ↓
     │                          ┌───────────┐
     └───────────────────────── │  半开      │
                                │ 允许少量   │
                                │ 测试请求   │
                                └───────────┘
```

### 状态转换规则

```
关闭 → 打开：连续失败次数 >= failure_threshold
打开 → 半开：等待 timeout_period 后自动转换
半开 → 关闭：测试请求连续成功 >= success_threshold
半开 → 打开：测试请求失败 → 重新打开
```

### 在 LangGraph 中模拟断路器

```python
import time
from dataclasses import dataclass, field

@dataclass
class CircuitBreaker:
    """简单的断路器实现。"""
    failure_threshold: int = 5      # 触发断路的失败次数
    timeout_period: float = 60.0    # 断路器打开持续时间（秒）
    success_threshold: int = 2      # 恢复需要的连续成功次数

    # 内部状态
    state: str = "closed"           # closed / open / half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        """判断是否允许执行请求。"""
        if self.state == "closed":
            return True
        if self.state == "open":
            # 检查是否已过超时期
            if time.time() - self.last_failure_time >= self.timeout_period:
                self.state = "half_open"
                self.success_count = 0
                return True
            return False  # 仍在断路状态
        if self.state == "half_open":
            return True
        return False

    def record_success(self):
        """记录成功执行。"""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
        else:
            self.failure_count = 0

    def record_failure(self):
        """记录失败执行。"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == "half_open":
            self.state = "open"
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"

# 使用断路器包装 LangGraph 节点
llm_breaker = CircuitBreaker(failure_threshold=3, timeout_period=30)

def call_llm_with_breaker(state: State) -> dict:
    """带断路器保护的 LLM 调用。"""
    if not llm_breaker.can_execute():
        # 断路器打开 → 直接走 fallback
        return {"fallback_level": 1, "error_info": "Circuit breaker is open"}

    try:
        result = call_llm(state["query"])
        llm_breaker.record_success()
        return {"answer": result, "fallback_level": 0}
    except Exception as e:
        llm_breaker.record_failure()
        return {"error_info": str(e), "fallback_level": 1}
```

---

## 状态驱动的错误追踪

### 在 State 中记录错误信息

```python
from typing import TypedDict, Annotated
from operator import add

class ResilientState(TypedDict):
    query: str
    answer: str
    error_count: Annotated[int, add]     # 累加错误次数
    error_types: list[str]               # 错误类型列表
    fallback_level: int                  # 当前降级层级
```

### 错误元数据的价值

```
错误追踪不仅是为了当下的处理，更是为了：

1. 监控告警 — 错误率超阈值时通知运维
2. 根因分析 — 哪类错误最多？哪个节点最不稳定？
3. 策略调优 — 根据历史数据优化 RetryPolicy 参数
4. 用户体验 — 告诉用户"使用了备用方案"而非"出错了"
```

---

## 三种模式协同工作

```
请求到达
  │
  ├──→ 断路器检查
  │     ├── 断路器打开 → 直接走 Fallback ──┐
  │     └── 断路器关闭 → 继续               │
  │                                         │
  ├──→ 主服务调用                            │
  │     ├── 成功 → 返回结果 ✅               │
  │     └── 失败 → RetryPolicy 重试          │
  │                                         │
  ├──→ 重试（指数退避 + 抖动）               │
  │     ├── 某次成功 → 返回结果 ✅           │
  │     └── 全部失败 → 断路器记录 ──────────┤
  │                                         │
  └──→ Fallback 链 ←───────────────────────┘
        ├── 备用模型 → 成功 → 返回 ✅
        ├── 缓存结果 → 成功 → 返回 ✅
        └── 默认值   → 返回 ⚠️
```

**三者的关系：**
- **RetryPolicy** — 短期应对（秒级），重试几次看看能不能成功
- **Fallback** — 中期应对（请求级），重试失败后找替代方案
- **断路器** — 长期应对（分钟级），持续失败时停止尝试，直接走 Fallback

---

## 实际应用建议

### 1. RAG 系统的 Fallback 策略

```
主检索：向量数据库语义检索
  ↓ 失败
Fallback 1：关键词检索（BM25）
  ↓ 失败
Fallback 2：预设 FAQ 匹配
  ↓ 无匹配
Fallback 3："抱歉，我暂时无法回答这个问题"
```

### 2. 多模型 Fallback

```python
# 模型降级链：高质量 → 快速 → 最小
models = [
    {"provider": "openai", "model": "gpt-4"},       # 最优质
    {"provider": "openai", "model": "gpt-3.5-turbo"},# 快速
    {"provider": "local", "model": "llama-3"},        # 本地备份
]
```

### 3. 监控指标建议

```
| 指标               | 含义                 | 告警阈值    |
|--------------------|--------------------|------------|
| 重试成功率          | 重试后成功的比例      | < 50%      |
| Fallback 激活率     | 走 Fallback 的请求比  | > 10%      |
| 断路器打开次数       | 断路器触发的频率      | > 3次/小时  |
| 平均降级层级         | 用户获得的服务质量     | > 1.5      |
```

---

## 前端类比

```
Fallback 链 ≈ 前端的图片加载降级

<picture>
    <source srcset="image.webp" type="image/webp">     <!-- 主方案 -->
    <source srcset="image.jpg" type="image/jpeg">       <!-- Fallback 1 -->
    <img src="placeholder.svg" alt="占位图">             <!-- Fallback 2 -->
</picture>

断路器 ≈ CDN 的健康检查

CDN 会持续监控源站健康状态：
- 源站正常 → 正常回源
- 源站连续失败 → 停止回源，使用缓存副本（断路器打开）
- 过一段时间 → 尝试回源（半开状态）
- 源站恢复 → 恢复正常回源（断路器关闭）
```

---

## 关键设计原则

```
1. 永远有兜底 — 最差情况下也要有友好的错误消息
2. 明确降级信号 — 让系统知道当前在哪个降级层级
3. 可观测性 — 记录每次降级，用于后续优化
4. 渐进恢复 — 通过断路器的半开状态逐步恢复
5. 质量预期管理 — 告诉用户"使用了备用方案"
```

---

[来源: search_生产模式_02.md | fetch_高级错误处理_01.md | fetch_断路器指南_02.md]
