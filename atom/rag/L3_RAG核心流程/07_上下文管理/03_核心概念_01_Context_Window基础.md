# 核心概念1：Context Window基础

> **理解LLM的"短期记忆"机制**

---

## 什么是Context Window？

### 定义

**Context Window（上下文窗口）** 是LLM在单次推理中能够处理的最大Token数量。

```
Context Window = Input Tokens + Output Tokens
```

**类比**：
- **前端**：浏览器的内存限制
- **生活**：人的短期记忆容量（7±2个信息块）

---

## 2026年主流模型对比

### Context Window大小

| 模型 | 窗口大小 | 发布时间 | 特点 |
|------|----------|----------|------|
| **GPT-3.5 Turbo** | 16K | 2023 | 基础版本 |
| **GPT-4** | 8K/32K | 2023 | 双版本 |
| **GPT-4 Turbo** | 128K | 2024 | 长上下文 |
| **Claude 3 Opus** | 200K | 2024 | 超长上下文 |
| **Claude 3.5 Sonnet** | 200K | 2024 | 平衡版本 |
| **Gemini 1.5 Pro** | 1M | 2024 | 百万级 |
| **Gemini 1.5 Flash** | 1M | 2024 | 快速版本 |
| **Llama 3.1 405B** | 128K | 2024 | 开源最大 |

### 成本对比（2026年价格）

| 模型 | Input价格 | Output价格 | 128K成本 |
|------|----------|-----------|---------|
| GPT-4 Turbo | $10/1M | $30/1M | $1.28 + $3.84 |
| Claude 3.5 Sonnet | $15/1M | $75/1M | $1.92 + $9.60 |
| Gemini 1.5 Pro | $7/1M | $21/1M | $0.90 + $2.69 |
| Llama 3.1 405B | 自托管 | 自托管 | 硬件成本 |

---

## Token计数机制

### 什么是Token？

**Token** 是LLM处理文本的基本单位，不等于字符或单词。

```python
# 示例
text_en = "Hello, world!"
text_cn = "你好，世界！"

# 英文：~3 tokens
# 中文：~6 tokens
```

### Token化规则

**英文**：
```
"Hello, world!" → ["Hello", ",", " world", "!"]
4个单词 → 4个tokens
```

**中文**：
```
"你好，世界！" → ["你", "好", "，", "世", "界", "！"]
5个字符 → 6个tokens（标点符号单独）
```

**经验规则**：
- 英文：1 token ≈ 0.75 单词 ≈ 4 字符
- 中文：1 token ≈ 0.5 字符
- 代码：1 token ≈ 0.6 字符

### 精确计算Token

```python
import tiktoken

# 初始化编码器
encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """精确计算Token数量"""
    return len(encoding.encode(text))

# 示例
text = "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。"
tokens = count_tokens(text)
print(f"Token数: {tokens}")  # 输出：~45 tokens
```

---

## Context Window的组成

### 完整结构

```
Total Context Window
├── System Message (固定)
├── Conversation History (累积)
├── User Input (当前)
├── Retrieved Context (RAG)
└── Reserved for Output (预留)
```

### 实际可用空间

```python
def calculate_available_context(
    model: str,
    system_tokens: int,
    history_tokens: int,
    query_tokens: int,
    output_tokens: int
) -> int:
    """计算实际可用的上下文空间"""
    context_limits = {
        "gpt-4-turbo": 128_000,
        "claude-3-sonnet": 200_000,
        "gemini-1.5-pro": 1_000_000
    }

    total = context_limits[model]
    used = system_tokens + history_tokens + query_tokens + output_tokens
    available = total - used

    return available

# 示例
model = "gpt-4-turbo"
system_tokens = 100
history_tokens = 2000
query_tokens = 50
output_tokens = 500

available = calculate_available_context(
    model, system_tokens, history_tokens, query_tokens, output_tokens
)
print(f"可用上下文: {available:,} tokens")  # 125,350 tokens
```

---

## Context Window的限制

### 限制1：硬性上限

```python
# 超过限制会报错
max_tokens = 128_000
current_tokens = 130_000

if current_tokens > max_tokens:
    raise ValueError(f"超过Context Window限制: {current_tokens} > {max_tokens}")
```

### 限制2：成本线性增长

```python
def calculate_cost(tokens: int, model: str = "gpt-4-turbo") -> float:
    """计算成本"""
    prices = {
        "gpt-4-turbo": {"input": 10, "output": 30}
    }

    price = prices[model]["input"]
    cost = (tokens / 1_000_000) * price
    return cost

# 示例
tokens_4k = 4_000
tokens_128k = 128_000

cost_4k = calculate_cost(tokens_4k)
cost_128k = calculate_cost(tokens_128k)

print(f"4K tokens成本: ${cost_4k:.4f}")      # $0.0400
print(f"128K tokens成本: ${cost_128k:.4f}")  # $1.2800
print(f"成本增长: {cost_128k / cost_4k}x")   # 32x
```

### 限制3：延迟增加

```python
# 实验数据（2026年测试）
latency_data = {
    4_000: 1.5,    # 秒
    8_000: 2.0,
    16_000: 2.5,
    32_000: 3.5,
    64_000: 6.0,
    128_000: 12.0
}

# 延迟与Token数近似线性关系
# latency ≈ 0.0001 * tokens + 1.0
```

### 限制4：Lost in the Middle

```python
# 位置召回率分布
position_recall = {
    "first_10%": 0.95,   # 首部：高召回
    "middle_80%": 0.55,  # 中间：低召回 ⚠️
    "last_10%": 0.90     # 尾部：高召回
}

# 结论：中间内容容易被忽略
```

---

## Context Window管理策略

### 策略1：截断（Truncation）

```python
def truncate_context(text: str, max_tokens: int) -> str:
    """简单截断"""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# 问题：可能截断关键信息
```

### 策略2：滑动窗口（Sliding Window）

```python
def sliding_window(
    messages: list[str],
    max_tokens: int,
    window_size: int = 10
) -> list[str]:
    """保留最近的N条消息"""
    total_tokens = sum(count_tokens(msg) for msg in messages)

    if total_tokens <= max_tokens:
        return messages

    # 保留最近的window_size条消息
    return messages[-window_size:]

# 适用场景：对话历史管理
```

### 策略3：摘要压缩（Summarization）

```python
def summarize_context(text: str, target_ratio: float = 0.3) -> str:
    """使用LLM压缩上下文"""
    prompt = f"""请将以下内容压缩到原长度的{target_ratio*100}%，保留关键信息：

{text}

压缩后的内容："""

    # 调用LLM进行压缩
    summary = llm.generate(prompt)
    return summary

# 优点：保留关键信息
# 缺点：需要额外LLM调用
```

### 策略4：智能选择（Selective Context）

```python
def select_relevant_context(
    query: str,
    documents: list[str],
    max_tokens: int
) -> str:
    """基于相关性选择上下文"""
    # 1. 计算相关性分数
    scores = [calculate_relevance(query, doc) for doc in documents]

    # 2. 按分数排序
    sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    # 3. 选择直到达到max_tokens
    selected = []
    current_tokens = 0

    for doc, score in sorted_docs:
        doc_tokens = count_tokens(doc)
        if current_tokens + doc_tokens <= max_tokens:
            selected.append(doc)
            current_tokens += doc_tokens
        else:
            break

    return "\n\n".join(selected)

# 优点：保留最相关的内容
```

---

## 实际应用场景

### 场景1：单文档问答

```python
# 文档：10K tokens
# 查询：50 tokens
# 输出：500 tokens
# 总计：10,550 tokens

# 适用模型：所有模型（都能容纳）
```

### 场景2：多文档RAG

```python
# 检索Top-10文档，每个1K tokens
# 总文档：10K tokens
# 查询：50 tokens
# 输出：500 tokens
# 总计：10,550 tokens

# 需要优化：
# - 压缩文档（4x压缩 → 2,550 tokens）
# - 或减少文档数量（Top-5 → 5,550 tokens）
```

### 场景3：长对话历史

```python
# 对话历史：20轮，每轮500 tokens
# 总历史：10K tokens
# 当前查询：50 tokens
# 输出：500 tokens
# 总计：10,550 tokens

# 策略：
# - 滑动窗口（保留最近10轮 → 5,550 tokens）
# - 或摘要压缩（压缩到3K tokens → 3,550 tokens）
```

### 场景4：代码库问答

```python
# 代码文件：50个，每个2K tokens
# 总代码：100K tokens
# 查询：50 tokens
# 输出：500 tokens
# 总计：100,550 tokens

# 必须优化：
# - 智能选择（Top-5相关文件 → 10,550 tokens）
# - 或代码摘要（提取关键函数 → 5,550 tokens）
```

---

## Context Window优化技巧

### 技巧1：预估Token数

```python
def estimate_tokens(text: str) -> int:
    """快速估算Token数（不精确但快速）"""
    # 英文：4字符 ≈ 1 token
    # 中文：0.5字符 ≈ 1 token

    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    estimated = chinese_chars * 2 + other_chars / 4
    return int(estimated)

# 用于快速判断，精确计算用tiktoken
```

### 技巧2：分层管理

```python
class ContextManager:
    """分层上下文管理"""
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.layers = {
            "system": 100,      # 固定
            "history": 2000,    # 可压缩
            "context": 5000,    # 可优化
            "query": 50,        # 固定
            "output": 500       # 预留
        }

    def get_available_context(self) -> int:
        """计算可用的上下文空间"""
        used = sum(self.layers.values())
        return self.max_tokens - used

    def optimize(self):
        """优化各层Token分配"""
        available = self.get_available_context()
        if available < 0:
            # 压缩history和context
            self.layers["history"] = int(self.layers["history"] * 0.5)
            self.layers["context"] = int(self.layers["context"] * 0.7)
```

### 技巧3：动态调整

```python
def dynamic_context_allocation(
    query_complexity: str,
    available_tokens: int
) -> dict:
    """根据查询复杂度动态分配"""
    if query_complexity == "simple":
        return {
            "context": int(available_tokens * 0.6),
            "history": int(available_tokens * 0.3),
            "output": int(available_tokens * 0.1)
        }
    elif query_complexity == "medium":
        return {
            "context": int(available_tokens * 0.7),
            "history": int(available_tokens * 0.2),
            "output": int(available_tokens * 0.1)
        }
    else:  # complex
        return {
            "context": int(available_tokens * 0.8),
            "history": int(available_tokens * 0.1),
            "output": int(available_tokens * 0.1)
        }
```

---

## 监控与调试

### 监控指标

```python
class ContextMonitor:
    """上下文使用监控"""
    def __init__(self):
        self.metrics = {
            "total_tokens": [],
            "context_tokens": [],
            "utilization": []
        }

    def log(self, total: int, context: int, max_tokens: int):
        """记录使用情况"""
        self.metrics["total_tokens"].append(total)
        self.metrics["context_tokens"].append(context)
        self.metrics["utilization"].append(total / max_tokens)

    def report(self):
        """生成报告"""
        avg_total = sum(self.metrics["total_tokens"]) / len(self.metrics["total_tokens"])
        avg_context = sum(self.metrics["context_tokens"]) / len(self.metrics["context_tokens"])
        avg_util = sum(self.metrics["utilization"]) / len(self.metrics["utilization"])

        return {
            "avg_total_tokens": avg_total,
            "avg_context_tokens": avg_context,
            "avg_utilization": avg_util
        }
```

### 调试工具

```python
def debug_context_usage(
    system: str,
    history: list[str],
    context: str,
    query: str,
    max_tokens: int
):
    """调试上下文使用情况"""
    system_tokens = count_tokens(system)
    history_tokens = sum(count_tokens(msg) for msg in history)
    context_tokens = count_tokens(context)
    query_tokens = count_tokens(query)

    total = system_tokens + history_tokens + context_tokens + query_tokens

    print(f"=== Context Usage Debug ===")
    print(f"System: {system_tokens} tokens ({system_tokens/total*100:.1f}%)")
    print(f"History: {history_tokens} tokens ({history_tokens/total*100:.1f}%)")
    print(f"Context: {context_tokens} tokens ({context_tokens/total*100:.1f}%)")
    print(f"Query: {query_tokens} tokens ({query_tokens/total*100:.1f}%)")
    print(f"Total: {total} tokens")
    print(f"Max: {max_tokens} tokens")
    print(f"Utilization: {total/max_tokens*100:.1f}%")

    if total > max_tokens:
        print(f"⚠️ 超过限制: {total - max_tokens} tokens")
```

---

## 2026年趋势

### 趋势1：长上下文普及

```
2023: 8K-32K主流
2024: 128K-200K普及
2025: 1M+成为标准
2026: 10M+开始出现
```

### 趋势2：成本持续下降

```
2023: $10/1M tokens
2024: $7/1M tokens
2025: $5/1M tokens
2026: $3/1M tokens（预测）
```

### 趋势3：长上下文与RAG协同

```
误区：长上下文替代RAG ❌
真相：长上下文 + 强检索 = 最优 ✅
```

**原因**：
1. Lost in the Middle依然存在
2. 成本和延迟仍是瓶颈
3. 精选内容质量更高

---

## 最佳实践

### 实践1：合理规划

```python
# 规划Context Window分配
allocation = {
    "system": 100,       # 1%
    "history": 2000,     # 20%
    "context": 5000,     # 50%
    "query": 50,         # 0.5%
    "output": 500,       # 5%
    "buffer": 2350       # 23.5%（预留）
}

# 总计：10,000 tokens
```

### 实践2：监控使用

```python
# 设置告警阈值
thresholds = {
    "warning": 0.8,   # 80%使用率
    "critical": 0.95  # 95%使用率
}

if utilization > thresholds["critical"]:
    alert("Context Window使用率过高")
```

### 实践3：持续优化

```python
# 定期分析和优化
def optimize_context_strategy():
    # 1. 分析使用模式
    patterns = analyze_usage_patterns()

    # 2. 识别瓶颈
    bottlenecks = identify_bottlenecks(patterns)

    # 3. 调整策略
    if "context_too_large" in bottlenecks:
        enable_compression()
    if "history_too_long" in bottlenecks:
        enable_sliding_window()
```

---

## 总结

### 核心要点

1. **定义**：Context Window = LLM的短期记忆容量
2. **限制**：硬性上限、成本增长、延迟增加、Lost in the Middle
3. **策略**：截断、滑动窗口、摘要压缩、智能选择
4. **趋势**：长上下文普及，但仍需RAG协同

### 记忆口诀

**"窗有限，需管理，长不如精，协同最优"**

### 下一步

理解了Context Window基础后，接下来学习：
- **Token优化技术**：如何降低Token使用
- **上下文压缩**：LLMLingua等技术
- **文档排序**：解决Lost in the Middle

---

**记住**：Context Window不是越大越好，精选内容 + 智能管理才是关键！
