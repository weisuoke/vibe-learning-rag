# Agent最佳实践 - 实战代码：场景 3 高可靠 Agent

> 重试 + 降级 + 上下文管理完整实战：搭建一个长时间运行、面对各种故障都能自愈的数据分析 Agent。

---

## 场景描述

搭建一个**数据分析 Agent**，要求：

1. **任务复杂**：SQL 查询 + 数据计算 + 报告生成，经常需要 10+ 步
2. **长对话**：用户会在一个会话中连续分析多个数据集
3. **外部依赖多**：数据库、LLM API、计算服务都可能超时
4. **高可用**：即使部分服务不可用，Agent 也不能崩溃

---

## 第 1 步：定义数据分析工具

```python
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field


# --- 工具 1：执行 SQL 查询 ---

class SQLQueryInput(BaseModel):
    query: str = Field(description="要执行的 SQL 查询语句（只支持 SELECT）")
    database: str = Field(
        default="analytics",
        description="目标数据库：analytics（分析库）或 reporting（报表库）",
    )

@tool(args_schema=SQLQueryInput)
def execute_sql(query: str, database: str = "analytics") -> str:
    """在指定数据库上执行 SQL 查询并返回结果。
    输入：SQL 语句（仅支持 SELECT）和目标数据库。
    输出：查询结果（表格格式）。
    限制：1) 只支持 SELECT 查询 2) 结果最多返回 100 行 3) 查询超时 30 秒。"""
    # 安全检查：禁止非 SELECT 语句
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ToolException(
            "安全限制：只支持 SELECT 查询。"
            "INSERT/UPDATE/DELETE 等修改操作请联系 DBA。"
        )
    # 模拟查询
    if "sales" in query.lower():
        return (
            "| 月份    | 销售额      | 同比增长 |\n"
            "|---------|------------|--------|\n"
            "| 2024-01 | ¥1,250,000 | +12%   |\n"
            "| 2024-02 | ¥980,000   | -5%    |\n"
            "| 2024-03 | ¥1,500,000 | +20%   |"
        )
    if "users" in query.lower():
        return "| 总用户数 | 活跃用户 | 新增用户 |\n|---------|---------|--------|\n| 50,000  | 12,000  | 3,500  |"
    return "查询成功，返回 0 行结果。"


# --- 工具 2：数据计算 ---

class CalculateInput(BaseModel):
    expression: str = Field(
        description="Python 数学表达式，如 '(1250000 + 980000 + 1500000) / 3'"
    )
    description: str = Field(
        description="计算说明，如 '计算Q1平均月销售额'"
    )

@tool(args_schema=CalculateInput)
def calculate(expression: str, description: str) -> str:
    """执行数学计算并返回结果。
    输入：Python 数学表达式和计算说明。
    输出：计算结果。
    限制：只支持纯数学运算，不支持函数调用或导入。"""
    # 安全：只允许数字和基本运算符
    import re
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,]+$', expression.replace(',', '')):
        raise ToolException(
            "表达式包含不允许的字符。只支持数字和 +-*/() 运算符。"
        )
    try:
        result = eval(expression.replace(',', ''))  # 生产中应用 ast.literal_eval
        return f"{description}：{result:,.2f}"
    except Exception as e:
        raise ToolException(f"计算失败: {str(e)}。请检查表达式格式。")


# --- 工具 3：生成报告 ---

class GenerateReportInput(BaseModel):
    title: str = Field(description="报告标题")
    sections: str = Field(description="报告包含的数据摘要（用换行分隔各节）")
    format: str = Field(
        default="markdown",
        description="输出格式：markdown 或 text",
    )

@tool(args_schema=GenerateReportInput)
def generate_report(title: str, sections: str, format: str = "markdown") -> str:
    """将分析结果整理为结构化报告。
    输入：报告标题和各节数据摘要。
    输出：格式化的报告文本。
    适用：数据分析的最后一步，汇总所有发现。"""
    if format == "markdown":
        report = f"# {title}\n\n"
        for i, section in enumerate(sections.split("\n"), 1):
            if section.strip():
                report += f"## {i}. {section.strip()}\n\n"
        report += f"\n---\n*报告生成时间: 2024-01-15*"
        return report
    return f"{title}\n{'='*len(title)}\n{sections}"


tools = [execute_sql, calculate, generate_report]
```

---

## 第 2 步：高可靠 Middleware 配置

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
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    "gpt-4.1",
    tools=tools,
    system_prompt="""你是一个专业的数据分析助手。

工作流程：
1. 理解用户的分析需求
2. 编写并执行 SQL 查询获取数据
3. 使用 calculate 工具进行数据计算
4. 使用 generate_report 整理最终报告

规则：
- SQL 查询只使用 SELECT 语句
- 对数据做分析前先确认数据质量
- 计算结果要有明确的单位和说明
- 如果查询无结果，告知用户可能的原因""",
    middleware=[
        # ====== 第 1 层：全局限制 ======
        # 数据分析任务步骤较多，适当放宽限制
        ModelCallLimitMiddleware(
            run_limit=50,           # 单次分析最多 50 步
            thread_limit=200,       # 长会话最多 200 步
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            allowed={
                "execute_sql": 15,        # SQL 查询较多
                "calculate": 10,          # 计算次数
                "generate_report": 3,     # 报告生成较少
            },
        ),

        # ====== 第 2 层：模型可靠性 ======
        # 数据分析需要推理能力，降级链更长
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=30.0,
            jitter=True,
            retry_on=("ConnectionError", "TimeoutError", "RateLimitError"),
        ),
        ModelFallbackMiddleware(
            fallbacks=[
                "gpt-4.1-mini",          # 降级 1：同供应商小模型
                "claude-3.5-sonnet",     # 降级 2：跨供应商（供应商级容灾）
            ],
        ),

        # ====== 第 3 层：工具可靠性 ======
        # SQL 查询可能因数据库负载超时，需要重试
        ToolRetryMiddleware(
            max_retries=3,               # 工具重试次数也多一些
            tools=["execute_sql"],       # 只对 SQL 查询重试
            # calculate 不重试（幂等但无需）
            # generate_report 不重试（幂等但无需）
        ),

        # ====== 第 4 层：上下文管理 ======
        # 数据分析对话通常很长，上下文管理至关重要
        SummarizationMiddleware(
            trigger=("fraction", 0.5),   # 50% 就触发（比客服场景更积极）
            keep=3,                       # 保留最近 3 条（数据分析需要紧凑上下文）
        ),
    ],
    checkpointer=MemorySaver(),
)
```

### 各层参数选择理由

| 参数 | 客服 Agent（场景1） | 数据分析 Agent（场景3） | 原因 |
|------|-------|--------|------|
| `run_limit` | 25 | 50 | 数据分析步骤多（查询→计算→汇总） |
| `thread_limit` | 100 | 200 | 用户可能在一个会话分析多个数据集 |
| `execute_sql` 限制 | — | 15 | 可能需要多次查询不同表 |
| `ModelFallback` | 1 个备用 | 2 个备用 | 分析任务更重要，需要更多降级选项 |
| `ToolRetry` | 2 次 | 3 次 | 数据库超时更常见 |
| `Summarization trigger` | 60% | 50% | 数据分析产生大量中间结果 |
| `Summarization keep` | 5 | 3 | 数据分析更需要精简上下文 |

---

## 第 3 步：故障场景模拟

### 场景 1：数据库超时 → 工具自动重试

```
用户: "查一下今年Q1的月度销售数据"

执行过程：
  1. LLM → 决定执行 SQL 查询
  2. execute_sql("SELECT month, sales FROM sales WHERE year=2024 AND quarter=1")
     → 数据库超时！TimeoutError
  3. ToolRetryMiddleware 介入：
     → 重试 #1：等 1.1s → 再次查询 → 超时！
     → 重试 #2：等 2.3s → 再次查询 → 成功！✅
     → 返回月度销售数据表格
  4. LLM → 分析数据 → 返回分析结果

  总计重试 2 次，用户等待多了约 3.4s，但获得了结果
  → 比直接报错好得多 ✅
```

### 场景 2：主模型宕机 → 双重降级

```
用户: "帮我分析用户增长趋势"

执行过程：
  1. LLM 调用 gpt-4.1 → ConnectionError!
  2. ModelRetryMiddleware: 重试 3 次都失败
  3. ModelFallbackMiddleware 介入：
     → 尝试 gpt-4.1-mini → ConnectionError!（同供应商都挂了）
     → 尝试 claude-3.5-sonnet → 成功！✅
  4. claude-3.5-sonnet → 执行 SQL 查询 → 分析 → 返回结果

  整个 OpenAI 服务宕机，但 Agent 通过跨供应商降级继续工作
  → 供应商级别的容灾 ✅
```

### 场景 3：长对话 → 上下文自动管理

```
数据分析会话（20+ 轮对话）：

轮 1-5: 查询销售数据（5 次 SQL + 5 次 LLM 推理 = 大量中间结果）
轮 6-10: 查询用户数据（又是 5 次 SQL + 计算）
轮 11: SummarizationMiddleware 触发（上下文达 50%）

摘要过程：
  原始上下文：前 10 轮的 SQL 查询、结果、分析
  ↓
  摘要：
    "用户要求分析 Q1 数据：
    - 月度销售额分别为 125 万、98 万、150 万
    - 总用户 5 万，活跃用户 1.2 万
    - Q1 平均月销售额 124.3 万"
  +
  保留：最近 3 条消息（完整保留最新对话）
  ↓
  新上下文大小：从 50% 降到约 15%

轮 12-20: 继续分析，有足够的空间处理新数据
  → 不会因为上下文溢出而丢失信息 ✅
```

### 场景 4：复杂任务 → 多步骤协作

```
用户: "帮我做一个 Q1 销售分析报告"

执行过程（约 8 步）：
  步骤 1: [LLM] 理解需求，规划分析方案
  步骤 2: [SQL] 查询月度销售数据
  步骤 3: [SQL] 查询同比数据
  步骤 4: [计算] 平均月销售额 = (125+98+150)/3 = 124.3 万
  步骤 5: [计算] Q1 总销售额 = 125+98+150 = 373 万
  步骤 6: [计算] 同比增长率 = (+12-5+20)/3 = +9%
  步骤 7: [LLM] 分析趋势和洞察
  步骤 8: [报告] 生成结构化报告

  → 8 步完成，远低于 run_limit=50
  → 每一步都有重试和降级保护
  → 上下文始终在可控范围内
```

---

## 第 4 步：可靠性配置清单

### 开发环境 vs 生产环境

```python
# === 开发环境 ===
dev_agent = create_agent(
    "gpt-4.1",
    tools=tools,
    middleware=[
        # 宽松限制（方便调试）
        ModelCallLimitMiddleware(run_limit=100),
        # 不重试（看到原始错误更方便调试）
        # 不降级（确认主模型的行为）
        # 不摘要（看到完整上下文）
    ],
)

# === 生产环境 ===
prod_agent = create_agent(
    "gpt-4.1",
    tools=tools,
    middleware=[
        # 严格限制
        ModelCallLimitMiddleware(run_limit=50, thread_limit=200),
        ToolCallLimitMiddleware(
            allowed={"execute_sql": 15, "calculate": 10, "generate_report": 3},
        ),
        # 完整可靠性
        ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, jitter=True),
        ModelFallbackMiddleware(fallbacks=["gpt-4.1-mini", "claude-3.5-sonnet"]),
        ToolRetryMiddleware(max_retries=3, tools=["execute_sql"]),
        # 上下文管理
        SummarizationMiddleware(trigger=("fraction", 0.5), keep=3),
    ],
    checkpointer=MemorySaver(),  # 生产环境用 PostgresSaver
)
```

### 不同场景的参数调优指南

| 场景特征 | run_limit | 工具重试 | 降级深度 | 摘要触发 |
|---------|-----------|---------|---------|---------|
| **简单问答** | 15-25 | 2 次 | 1 级 | 70% |
| **客服对话** | 25-30 | 2 次 | 1 级 | 60% |
| **数据分析** | 40-60 | 3 次 | 2 级 | 50% |
| **代码生成** | 30-50 | 2 次 | 2 级 | 55% |
| **研究报告** | 50-80 | 3 次 | 2 级 | 45% |

---

## 核心收获

| 可靠性维度 | 机制 | 本场景配置 |
|-----------|------|----------|
| **防循环** | ModelCallLimit | run=50, thread=200 |
| **防滥用** | ToolCallLimit | SQL≤15, 计算≤10, 报告≤3 |
| **API 超时** | ModelRetry + ToolRetry | 模型 3 次 + SQL 3 次 |
| **服务宕机** | ModelFallback | mini → claude（跨供应商） |
| **上下文溢出** | Summarization | 50% 触发，保留 3 条 |
| **对话记忆** | MemorySaver | 持久化对话状态 |

**关键原则：** 可靠性配置要根据任务特征调整——数据分析需要更多步骤、更积极的摘要、更深的降级链。不存在放之四海而皆准的参数，理解每个参数的含义比记住默认值更重要。

---

**下一步**: 阅读 [08_面试必问.md](./08_面试必问.md)，掌握 Agent 最佳实践的面试高频问题

---

**数据来源**：
- [源码分析] langchain_v1/agents/middleware/ — 全部 Middleware 参数详解
- [源码分析] langchain_v1/agents/middleware/model_retry.py — 指数退避 + 抖动
- [源码分析] langchain_v1/agents/middleware/summarization.py — 摘要触发策略
- [网络搜索] Agent 可靠性设计 — 降级策略、超时管理、长对话处理
