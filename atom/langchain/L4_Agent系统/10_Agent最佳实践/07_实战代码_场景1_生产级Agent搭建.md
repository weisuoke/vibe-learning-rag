# Agent最佳实践 - 实战代码：场景 1 生产级 Agent 搭建

> Middleware 组合实战：用重试+降级+限制+摘要四层 Middleware 搭建一个可上线的智能客服 Agent。

---

## 场景描述

搭建一个**智能客服 Agent**，要求：

1. **功能**：搜索知识库、查询订单、创建工单
2. **可靠性**：API 超时自动重试、模型宕机自动降级
3. **安全性**：防止无限循环、控制工具调用频率
4. **上下文管理**：长对话自动摘要，不超出 Token 限制
5. **多轮对话**：支持对话记忆

---

## 第 1 步：定义工具

### 工具描述的黄金公式

```
description = "用途（做什么）+ 输入（需要什么）+ 输出（返回什么）+ 限制（什么时候不该用）"
```

### 三个业务工具

```python
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field


# --- 工具 1：搜索知识库 ---

class SearchKBInput(BaseModel):
    """知识库搜索的输入参数"""
    query: str = Field(description="搜索关键词，如'退货政策'、'配送时间'")

@tool(args_schema=SearchKBInput)
def search_kb(query: str) -> str:
    """搜索客服知识库获取产品信息和政策文档。
    输入：用户问题的关键词。
    输出：最相关的知识库文章摘要。
    限制：只能搜索已有知识库内容，不能回答实时信息（如物流状态）。
    """
    # 模拟知识库搜索
    kb = {
        "退货": "7天无理由退货，需保持商品完好。退货运费由买家承担。",
        "配送": "标准配送3-5个工作日，加急配送1-2个工作日（额外收费20元）。",
        "保修": "电子产品保修1年，人为损坏不在保修范围内。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return "未找到相关信息，建议联系人工客服。"


# --- 工具 2：查询订单 ---

class QueryOrderInput(BaseModel):
    """订单查询的输入参数"""
    order_id: str = Field(description="订单编号，格式为 ORD-开头，如 ORD-20240101-001")

@tool(args_schema=QueryOrderInput)
def query_order(order_id: str) -> str:
    """查询指定订单的状态和详情。
    输入：订单编号（ORD-开头）。
    输出：订单状态、金额、物流信息。
    限制：只能查询，不能修改订单。如需修改请引导用户联系人工。
    """
    try:
        # 模拟数据库查询
        orders = {
            "ORD-20240101-001": {
                "status": "已发货",
                "amount": 299.00,
                "logistics": "顺丰快递 SF123456789，预计明天送达",
            },
        }
        if order_id not in orders:
            raise ToolException(
                f"订单 {order_id} 不存在。请确认订单编号格式为 ORD-开头。"
            )
        order = orders[order_id]
        return (
            f"订单 {order_id}：状态={order['status']}，"
            f"金额={order['amount']}元，物流={order['logistics']}"
        )
    except ToolException:
        raise  # ToolException 要向上传播
    except Exception as e:
        raise ToolException(f"查询订单失败: {str(e)}") from e


# --- 工具 3：创建工单 ---

class CreateTicketInput(BaseModel):
    """创建工单的输入参数"""
    title: str = Field(description="工单标题，简要描述问题")
    description: str = Field(description="问题详细描述")
    priority: str = Field(
        default="medium",
        description="优先级：low（一般咨询）、medium（需要处理）、high（紧急问题）",
    )

@tool(args_schema=CreateTicketInput)
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """为用户创建客服工单，将问题转交给人工客服处理。
    输入：工单标题、问题描述、优先级。
    输出：工单编号。
    适用场景：知识库无法解答的问题，或用户明确要求人工服务。
    注意：此工具会创建工单（写操作），请确认用户确实需要人工帮助后再调用。
    """
    import uuid
    ticket_id = f"TK-{uuid.uuid4().hex[:8].upper()}"
    return f"工单已创建：编号={ticket_id}，标题='{title}'，优先级={priority}。人工客服将在1小时内跟进。"


tools = [search_kb, query_order, create_ticket]
```

### 工具设计要点总结

| 要点 | 体现 |
|------|------|
| **描述清晰** | 每个 `description` 都包含用途、输入、输出、限制 |
| **Schema 严格** | 使用 Pydantic BaseModel 定义参数类型和描述 |
| **错误可恢复** | `ToolException` 返回友好提示，Agent 可以自行纠正 |
| **原子化** | 每个工具只做一件事（搜索/查询/创建） |
| **副作用标注** | `create_ticket` 明确标注为写操作 |

---

## 第 2 步：配置 Middleware

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
)

agent = create_agent(
    "gpt-4.1",
    tools=tools,
    system_prompt="""你是「小助」，一个专业的电商客服助手。

核心规则：
1. 先搜索知识库，再考虑其他工具
2. 需要订单号时主动询问用户
3. 知识库无法解答的问题，创建工单转人工
4. 保持礼貌专业，回答简洁清晰
5. 不要编造信息，不确定就说不确定""",
    middleware=[
        # --- 第 1 层：全局限制保护（最外层）---
        ModelCallLimitMiddleware(
            run_limit=25,          # 单次对话最多 25 次 LLM 调用
            thread_limit=100,      # 整个会话最多 100 次
            exit_behavior="end",   # 超限时优雅结束
        ),
        ToolCallLimitMiddleware(
            allowed={
                "search_kb": 10,       # 搜索最多 10 次
                "query_order": 5,      # 查询最多 5 次
                "create_ticket": 3,    # 创建工单最多 3 次（写操作更严格）
            },
        ),

        # --- 第 2 层：模型可靠性（重试 + 降级）---
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,
            retry_on=("ConnectionError", "TimeoutError", "RateLimitError"),
        ),
        ModelFallbackMiddleware(
            fallbacks=["gpt-4.1-mini"],   # 主模型挂了用小模型
        ),

        # --- 第 3 层：工具可靠性 ---
        ToolRetryMiddleware(
            max_retries=2,
            tools=["search_kb", "query_order"],  # 只对查询类工具重试
            # 注意：create_ticket 不重试（防止重复创建）
        ),

        # --- 第 4 层：上下文管理 ---
        SummarizationMiddleware(
            trigger=("fraction", 0.6),   # 上下文超过 60% 时触发
            keep=5,                       # 保留最近 5 条消息
        ),
    ],
)
```

### Middleware 顺序解读

```
用户请求 ──→ [限制保护] ──→ [重试] ──→ [降级] ──→ LLM
               ↑                                    │
              外层                                  内层

洋葱模型执行顺序：
1. ModelCallLimitMiddleware: 检查是否超限 → 是则结束
2. ToolCallLimitMiddleware: 检查工具是否超限
3. ModelRetryMiddleware: 包裹 LLM 调用，失败时重试
4. ModelFallbackMiddleware: 重试都失败时切换模型
5. ToolRetryMiddleware: 包裹工具调用，失败时重试
6. SummarizationMiddleware: LLM 调用前检查上下文大小

关键：限制在最外层，重试都计入限制！
      → 即使重试 3 次，也不会超过 25 次总限制
```

---

## 第 3 步：支持多轮对话

```python
from langgraph.checkpoint.memory import MemorySaver

# 使用 MemorySaver 实现对话记忆
agent_with_memory = create_agent(
    "gpt-4.1",
    tools=tools,
    system_prompt="你是「小助」，一个专业的电商客服助手。...",
    middleware=[
        ModelCallLimitMiddleware(run_limit=25),
        ModelRetryMiddleware(max_retries=3),
        ModelFallbackMiddleware(fallbacks=["gpt-4.1-mini"]),
        SummarizationMiddleware(trigger=("fraction", 0.6), keep=5),
    ],
    checkpointer=MemorySaver(),   # 内存存储（开发用）
)
```

### 多轮对话示例

```python
config = {"configurable": {"thread_id": "user-session-001"}}

# 第 1 轮
result1 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "你好，我想了解退货政策"}]},
    config=config,
)
# Agent: 调用 search_kb("退货") → "7天无理由退货..."

# 第 2 轮（Agent 记得上下文）
result2 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "那运费谁出？"}]},
    config=config,
)
# Agent: 知道你在问退货运费 → "退货运费由买家承担"

# 第 3 轮
result3 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "帮我查一下 ORD-20240101-001"}]},
    config=config,
)
# Agent: 调用 query_order("ORD-20240101-001") → 返回订单详情
```

---

## 第 4 步：保护机制验证

### 场景 1：API 超时 → 自动重试

```
用户: "查一下退货政策"

执行过程：
  1. LLM 调用 gpt-4.1 → TimeoutError!
  2. ModelRetryMiddleware: 等 1.1s，重试 #1
  3. LLM 调用 gpt-4.1 → 成功！
  4. 调用 search_kb → 返回结果
  → 用户无感知，自动恢复 ✅
```

### 场景 2：模型宕机 → 自动降级

```
用户: "帮我查订单 ORD-20240101-001"

执行过程：
  1. LLM 调用 gpt-4.1 → ConnectionError!
  2. ModelRetryMiddleware: 重试 3 次都失败
  3. ModelFallbackMiddleware: 切换到 gpt-4.1-mini
  4. LLM 调用 gpt-4.1-mini → 成功！
  5. 调用 query_order → 返回订单详情
  → 降级到小模型，功能正常 ✅
```

### 场景 3：无限循环 → 自动终止

```
用户: "帮我分析一下所有商品的退货率"

执行过程：
  1. LLM 调用 → 搜索 "退货率"
  2. search_kb 返回 "未找到相关信息"
  3. LLM 调用 → 再搜 "商品退货数据"
  4. search_kb 返回 "未找到相关信息"
  5. ... 反复搜索 ...
  10. search_kb 达到 10 次限制 → ToolCallLimitMiddleware 拦截
  11. LLM 收到 "工具调用已达上限" → 创建工单转人工
  → 自动止损，不会无限循环 ✅
```

### 场景 4：长对话 → 自动摘要

```
对话进行到第 20 轮（上下文占用 65%）：

SummarizationMiddleware 触发：
  1. 保留最近 5 条消息（完整保留）
  2. 前 15 条消息压缩为摘要：
     "用户咨询了退货政策（7天无理由）、查询了订单
      ORD-20240101-001（已发货），询问了配送时间。"
  3. 摘要 + 最近 5 条 → 新的上下文（占用 30%）
  → 上下文不会溢出 ✅
```

---

## 完整代码汇总

```python
"""
生产级智能客服 Agent - 完整代码
Middleware 组合：限制 + 重试 + 降级 + 摘要
"""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
)
from langchain_core.tools import tool, ToolException
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


# ========== 1. 工具定义 ==========

class SearchKBInput(BaseModel):
    query: str = Field(description="搜索关键词")

@tool(args_schema=SearchKBInput)
def search_kb(query: str) -> str:
    """搜索客服知识库获取产品信息和政策文档。
    输入：关键词。输出：相关文章摘要。
    限制：只能搜索已有内容。"""
    kb = {
        "退货": "7天无理由退货，需保持商品完好。退货运费由买家承担。",
        "配送": "标准配送3-5个工作日，加急配送1-2个工作日。",
        "保修": "电子产品保修1年，人为损坏不在保修范围内。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return "未找到相关信息，建议联系人工客服。"


class QueryOrderInput(BaseModel):
    order_id: str = Field(description="订单编号，ORD-开头")

@tool(args_schema=QueryOrderInput)
def query_order(order_id: str) -> str:
    """查询订单状态和详情。输入：订单编号。
    输出：状态、金额、物流。限制：只能查询不能修改。"""
    orders = {
        "ORD-20240101-001": {
            "status": "已发货", "amount": 299.00,
            "logistics": "顺丰快递 SF123456789",
        },
    }
    if order_id not in orders:
        raise ToolException(f"订单 {order_id} 不存在，请确认编号格式。")
    o = orders[order_id]
    return f"订单{order_id}：{o['status']}，{o['amount']}元，{o['logistics']}"


class CreateTicketInput(BaseModel):
    title: str = Field(description="工单标题")
    description: str = Field(description="问题描述")
    priority: str = Field(default="medium", description="优先级: low/medium/high")

@tool(args_schema=CreateTicketInput)
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建客服工单转交人工处理。注意：此为写操作。"""
    import uuid
    tid = f"TK-{uuid.uuid4().hex[:8].upper()}"
    return f"工单已创建：{tid}，标题='{title}'，优先级={priority}"


# ========== 2. Agent 配置 ==========

agent = create_agent(
    "gpt-4.1",
    tools=[search_kb, query_order, create_ticket],
    system_prompt="""你是「小助」，专业电商客服助手。
规则：1.先搜知识库 2.需要订单号时主动询问
3.解决不了就创建工单 4.不要编造信息""",
    middleware=[
        # 限制保护
        ModelCallLimitMiddleware(run_limit=25, thread_limit=100),
        ToolCallLimitMiddleware(
            allowed={"search_kb": 10, "query_order": 5, "create_ticket": 3},
        ),
        # 模型可靠性
        ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, jitter=True),
        ModelFallbackMiddleware(fallbacks=["gpt-4.1-mini"]),
        # 工具可靠性（写操作不重试）
        ToolRetryMiddleware(max_retries=2, tools=["search_kb", "query_order"]),
        # 上下文管理
        SummarizationMiddleware(trigger=("fraction", 0.6), keep=5),
    ],
    checkpointer=MemorySaver(),
)


# ========== 3. 使用 ==========

config = {"configurable": {"thread_id": "user-001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我想了解退货政策"}]},
    config=config,
)
print(result["messages"][-1].content)
```

---

## 核心收获

| 维度 | 做法 | 效果 |
|------|------|------|
| **工具设计** | description 黄金公式 + Pydantic Schema + ToolException | LLM 准确选择工具，错误可自愈 |
| **限制保护** | ModelCallLimit(25) + ToolCallLimit(按工具) | 防循环、防滥用、控成本 |
| **模型可靠性** | Retry(3) + Fallback(mini) | API 超时自动恢复，宕机自动降级 |
| **工具可靠性** | ToolRetry(2) 只对读操作 | 查询重试，写操作不重复 |
| **上下文管理** | Summarization(60%, keep=5) | 长对话不溢出 |
| **对话记忆** | MemorySaver + thread_id | 多轮对话上下文连续 |

**关键原则：写操作不重试！** `create_ticket` 不在 `ToolRetryMiddleware` 的 `tools` 列表中，防止重试导致重复创建工单。

---

**下一步**: 阅读 [07_实战代码_场景2_安全合规Agent.md](./07_实战代码_场景2_安全合规Agent.md)，学习 PII + 人工审核的综合实战

---

**数据来源**：
- [源码分析] langchain_v1/agents/middleware/ — Middleware 组合与执行顺序
- [源码分析] langchain_core/tools/base.py — ToolException 错误恢复机制
- [Context7] LangChain Agent 工具设计最佳实践 — description 黄金公式
- [网络搜索] Agent 生产部署最佳实践 — 可靠性设计模式
