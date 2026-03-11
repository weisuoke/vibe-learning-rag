# Agent最佳实践 - 核心概念 5：工具设计与 Agent 架构最佳实践

> 工具设计是 Agent 能力的基础，架构选择决定 Agent 的可维护性——好的工具 description 让 LLM 选对工具，好的架构让系统从单 Agent 平滑扩展到多 Agent。

---

## 一、工具设计六大原则

### 原则 1：清晰的描述（最重要！）

工具的 `description` 是 LLM 决定用哪个工具的**唯一依据**。

```python
# ❌ 模糊描述 — LLM 不知道该不该用
class SearchTool(BaseTool):
    name = "search"
    description = "A useful search tool"  # 太模糊！

# ✅ 清晰描述 — LLM 能准确判断
class SearchTool(BaseTool):
    name = "search_knowledge_base"
    description = (
        "在公司内部知识库中搜索文档。"
        "输入：搜索关键词（中文或英文）。"
        "输出：最相关的 5 篇文档摘要。"
        "注意：只能搜索已索引的文档，不支持实时网页搜索。"
        "适用场景：用户询问公司产品、政策、流程相关问题时使用。"
    )
```

**description 黄金公式：**

```
description = 做什么 + 输入是什么 + 输出是什么 + 限制/注意事项 + 适用场景
```

### 原则 2：明确的 Schema

使用 Pydantic 定义严格的输入输出类型，每个参数都要有 description。

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(
        description="搜索关键词，支持中英文，建议 2-10 个字"
    )
    max_results: int = Field(
        default=5,
        description="返回结果数量，默认 5，最大 20",
        ge=1, le=20,
    )
    category: str | None = Field(
        default=None,
        description="可选的文档分类过滤，如 'product', 'policy', 'faq'",
    )

class SearchTool(BaseTool):
    name = "search_kb"
    args_schema = SearchInput  # 注意：类型注解应为 Type[BaseModel]
```

### 原则 3：原子化操作

每个工具只做一件事，避免"瑞士军刀"式工具。

```python
# ❌ 一个工具做太多事
class DatabaseTool(BaseTool):
    name = "database"
    description = "查询、插入、更新、删除数据库记录"
    # LLM 需要判断太多参数，容易出错

# ✅ 拆分为原子操作
class QueryRecordTool(BaseTool):
    name = "query_record"
    description = "按 ID 查询用户记录（只读）"

class UpdateRecordTool(BaseTool):
    name = "update_record"
    description = "更新用户记录的指定字段（需要人工审核）"
```

### 原则 4：幂等性（Idempotency）

```python
# ✅ 读操作天然幂等
def search(query: str) -> list:
    return db.search(query)  # 多次调用结果相同

# ⚠️ 写操作需要设计为幂等
def create_or_update_user(user_id: str, data: dict):
    # 使用 upsert 而不是 insert，多次调用不会重复创建
    db.upsert({"id": user_id, **data})

# ❌ 非幂等操作需要额外保护
def transfer_money(from_account, to_account, amount):
    # 每次调用都会转账！Agent 重试会导致重复转账！
    # 解决方案：添加 idempotency_key 参数
    pass
```

### 原则 5：友好的错误处理

```python
from langchain.tools import ToolException

class SearchTool(BaseTool):
    def _run(self, query: str) -> str:
        try:
            results = api.search(query)
            if not results:
                return "未找到相关结果，建议：1) 换个关键词 2) 检查拼写"
            return format_results(results)
        except ConnectionError:
            # ✅ 返回 LLM 能理解的错误，而不是原始异常
            raise ToolException(
                "知识库服务暂时不可用，请稍后再试或使用其他方式回答用户。"
            )
        except Exception as e:
            raise ToolException(
                f"搜索出错: {str(e)}。建议：尝试简化搜索关键词。"
            )
```

### 原则 6：标注副作用

```python
# ✅ 在 description 中明确标注读/写
class ReadOrderTool(BaseTool):
    name = "read_order"
    description = "【只读】查询订单详情，不会修改任何数据"

class CancelOrderTool(BaseTool):
    name = "cancel_order"
    description = "【写操作】取消指定订单。注意：此操作不可撤销！"
```

### 工具设计反模式

| 反模式 | 问题 | 解决方案 |
|--------|------|----------|
| **工具太多**（>15个） | LLM 选择困难，准确率下降 | 按任务分组，使用子代理 |
| **描述模糊** | LLM 无法正确选择工具 | 使用黄金公式写 description |
| **无错误处理** | 工具失败导致代理崩溃 | 使用 ToolException 返回友好错误 |
| **副作用未标注** | LLM 不知道操作会修改数据 | 在 description 中标注【只读】/【写操作】 |
| **参数过多**（>5个） | LLM 难以正确填充所有参数 | 拆分工具或使用默认值 |

---

## 二、Agent 架构选择

### 从单 Agent 到多 Agent 的决策

```
是否需要多 Agent？

问题 1: 工具数量超过 10-15 个？
  └── 否 → 单 Agent 够用
  └── 是 → 继续

问题 2: 任务需要不同专业领域？
  └── 否 → 单 Agent + 工具分组
  └── 是 → 继续

问题 3: 需要不同的权限级别？
  └── 否 → 单 Agent + 工具限制
  └── 是 → 多 Agent

结论: 从单 Agent 开始，验证后再扩展！
```

### 四种多 Agent 模式

#### 1. Subagents（子代理模式）

```python
# 主代理将子代理包装为工具
from langchain.agents import create_agent

# 子代理 1: 研究员
researcher = create_agent(
    "gpt-4.1",
    tools=[search_web, search_papers],
    system_prompt="你是一个研究助手，负责信息检索和整理。",
)

# 子代理 2: 写手
writer = create_agent(
    "gpt-4.1",
    tools=[write_document, format_text],
    system_prompt="你是一个写作助手，负责内容创作。",
)

# 主代理：编排子代理
main_agent = create_agent(
    "gpt-4.1",
    tools=[researcher.as_tool(), writer.as_tool()],
    system_prompt="你是项目经理，协调研究员和写手完成任务。",
)
```

**适用场景：** 任务分解明确，子任务相互独立。

#### 2. Handoffs（切换模式）

```python
# 基于状态动态切换代理行为
# 适合客服系统中的角色切换
```

**适用场景：** 对话式系统，需要动态切换代理角色（如售前→售后→技术支持）。

#### 3. Skills（技能模式）

```python
# 单代理保持控制权，按需加载不同"技能"
# 适合任务类型多但逻辑相似的场景
```

**适用场景：** 通用助手根据问题类型加载不同专业知识。

#### 4. Router（路由模式）

```python
# 对输入分类，导向不同专家处理
# 适合输入类型多样的场景
```

**适用场景：** 多种输入类型需要不同专家处理。

### 何时使用/不使用多 Agent

| 信号 | 决策 |
|------|------|
| 工具 > 15 个 | 考虑多 Agent |
| 需要不同权限级别 | 考虑多 Agent |
| 需要并行处理 | 考虑多 Agent |
| 上下文窗口不够 | 考虑多 Agent |
| 单 Agent + 少量工具能解决 | **不要**多 Agent |
| 任务是线性的 | **不要**多 Agent |
| 增加的复杂度不值得 | **不要**多 Agent |

---

## 三、测试策略（测试金字塔）

```
        /\
       /  \  端到端测试 (E2E)
      /    \  - 完整对话流程测试
     /------\  - 多步骤任务验证
    /        \
   /  集成测试  \  - 工具调用验证
  /            \  - Agent 与外部系统交互
 /--------------\
/   单元测试      \  - 工具函数测试
/                  \  - 状态转换逻辑
/--------------------\  - 条件路由测试
```

### 五层测试

| 层级 | 测试什么 | 怎么测 | 频率 |
|------|---------|--------|------|
| **单元测试** | 工具函数输入输出 | Mock LLM，只测工具逻辑 | 每次提交 |
| **集成测试** | Agent + 真实工具交互 | 使用测试环境的 API | 每日 |
| **端到端测试** | 完整对话流程 | 真实 LLM + 真实工具 | 每周 |
| **红队测试** | 安全漏洞 | 提示注入、越狱尝试 | 每月 |
| **评估** | 准确率、完成率 | LangSmith 系统评估 | 持续 |

### 关键测试场景

```python
# 1. 工具调用正确性
def test_agent_calls_right_tool():
    """验证 Agent 对特定问题选择正确的工具"""
    response = agent.invoke("今天北京天气怎么样？")
    assert "weather_tool" in get_called_tools(response)

# 2. 领域边界测试
def test_agent_ignores_out_of_scope():
    """验证 Agent 不处理超出范围的问题"""
    response = agent.invoke("帮我写一首诗")  # 超出客服范围
    assert "无法处理" in response or "建议您" in response

# 3. 多步骤复杂任务
def test_agent_multi_step_task():
    """验证 Agent 能完成多步骤任务"""
    response = agent.invoke("查找订单 12345 并申请退款")
    # 应该先查订单，再申请退款
    tools_called = get_called_tools(response)
    assert tools_called == ["query_order", "apply_refund"]
```

---

## 四、版本迁移路径

### 从 AgentExecutor 迁移到 create_agent

```python
# ========== 旧方式 (AgentExecutor, LangChain 0.x) ==========
from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,
    handle_parsing_errors=True,
)
result = executor.invoke({"input": "..."})

# ========== 新方式 (create_agent, LangChain 1.0) ==========
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
)

agent = create_agent(
    "gpt-4.1",                          # 直接传模型名
    tools=tools,
    system_prompt="You are a helpful assistant.",  # 不再需要 ChatPromptTemplate
    middleware=[
        ModelCallLimitMiddleware(run_limit=15),   # 替代 max_iterations
        ModelRetryMiddleware(max_retries=3),       # 替代 handle_parsing_errors
    ],
)
result = agent.invoke({"messages": [...]})
```

### 关键变化

| 维度 | AgentExecutor (旧) | create_agent (新) |
|------|--------------------|--------------------|
| **入口函数** | `create_tool_calling_agent` + `AgentExecutor` | `create_agent` 一步到位 |
| **提示词** | `ChatPromptTemplate` | `system_prompt` 字符串 |
| **错误处理** | `handle_parsing_errors`, `handle_tool_error` | Middleware（Retry, Fallback） |
| **迭代限制** | `max_iterations` | `ModelCallLimitMiddleware` |
| **扩展能力** | 有限（固定参数） | Middleware 可无限扩展 |
| **底层实现** | 自定义循环 | LangGraph StateGraph |

---

## 五、生产部署检查清单

### 设计阶段 ✅

- [ ] 从单 Agent 开始，验证后再扩展
- [ ] 为每个工具编写清晰、具体的 description
- [ ] 使用 Pydantic 定义工具 schema
- [ ] 设置 ModelCallLimitMiddleware 防止无限循环
- [ ] 选择合适的 Agent 架构（单 Agent vs 多 Agent）

### 安全阶段 🔒

- [ ] 实施最小权限原则（每个 Agent 只给必需工具）
- [ ] 添加 PIIMiddleware 防止数据泄露
- [ ] 关键操作添加 HumanInTheLoopMiddleware
- [ ] 代码执行使用 ShellToolMiddleware(sandbox="docker")
- [ ] 对用户输入进行消毒

### 可靠性阶段 🔄

- [ ] 添加 ModelRetryMiddleware（指数退避 + 抖动）
- [ ] 配置 ModelFallbackMiddleware（跨供应商降级）
- [ ] 设置 ToolCallLimitMiddleware（按工具限制）
- [ ] 配置 SummarizationMiddleware（长对话压缩）

### 测试阶段 🧪

- [ ] 工具函数单元测试
- [ ] Agent 工作流集成测试
- [ ] 端到端对话测试
- [ ] 红队对抗测试（提示注入、越狱）
- [ ] LangSmith 系统评估

### 部署与运维阶段 📊

- [ ] LangSmith 全链路跟踪
- [ ] 监控告警（工具调用频率、错误率、成本）
- [ ] 成本预算和限制
- [ ] 版本管理和灰度发布
- [ ] 定期安全审计

---

## 在 RAG 和 AI Agent 开发中的应用

| 场景 | 工具设计要点 | 架构选择 |
|------|------------|---------|
| **RAG 文档问答** | 检索工具 description 要包含知识库范围 | 单 Agent |
| **智能客服** | 按业务拆分工具（查询/操作/投诉） | Skills 或 Router |
| **代码助手** | 读/写/运行工具分离，写操作需审核 | 单 Agent + HITL |
| **数据分析** | SQL 工具 + 可视化工具，标注只读 | 单 Agent |
| **企业自动化** | 多部门工具按权限分组 | Subagents |

---

**下一步**: 阅读 [04_最小可用.md](./04_最小可用.md)，了解最小可用的 Agent 最佳实践配置

---

**数据来源**：
- [Context7 文档] LangChain 多代理模式 — Subagents, Handoffs, Skills, Router
- [Context7 文档] create_agent 迁移路径 — create_react_agent → create_agent
- [网络搜索] Agent 工具设计 6 大原则 + 5 大反模式
- [网络搜索] Agent 测试金字塔 — 单元/集成/E2E/红队/评估
- [网络搜索] 生产部署检查清单 — 设计/安全/测试/部署/运维 5 阶段
- [源码分析] langchain_core/tools/base.py — args_schema 验证 + ToolException
