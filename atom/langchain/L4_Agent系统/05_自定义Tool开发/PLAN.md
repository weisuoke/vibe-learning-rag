# 05_自定义Tool开发 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_tool_development_01.md - LangChain Tool 核心机制分析
  - 分析文件：base.py, convert.py, structured.py
  - 关键发现：三种工具创建方式、Schema 推断、参数注入、异步支持

### Context7 官方文档
- ✓ reference/context7_langchain_tools_01.md - LangChain 官方文档
  - 库：LangChain (2026-02-26)
  - 内容：@tool 装饰器用法、StructuredTool、BaseTool 接口、最佳实践

### 网络搜索
- ✓ reference/search_tool_development_01.md - 2025-2026 社区实践
  - 平台：GitHub Issues, Reddit, Twitter/X
  - 关键话题：@tool vs BaseTool、参数描述、版本兼容、大规模工具管理

### 待抓取链接
- [ ] https://github.com/MervinPraison/PraisonAI-Tools - 工具库源码和示例
- [ ] https://www.reddit.com/r/LangChain/comments/1k0adul/custom_tools_with_multiple_parameters/ - 多参数工具讨论
- [ ] https://www.reddit.com/r/LangChain/comments/1k6oi7j/langgraph_ollama_agent_using_local_model_qwen25/ - 本地模型集成

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_@tool装饰器方法.md - 现代推荐方式，自动schema推断 [来源: 源码+Context7+网络]
- [ ] 03_核心概念_2_StructuredTool创建.md - 灵活方式，显式配置 [来源: 源码+Context7]
- [ ] 03_核心概念_3_BaseTool类继承.md - 最大控制，复杂状态管理 [来源: 源码+Context7+网络]
- [ ] 03_核心概念_4_Schema定义与验证.md - Pydantic schema、类型推断 [来源: 源码+Context7]
- [ ] 03_核心概念_5_参数注入机制.md - InjectedToolArg、RunnableConfig [来源: 源码+网络]
- [ ] 03_核心概念_6_异步工具支持.md - coroutine、_arun、线程池回退 [来源: 源码+Context7]
- [ ] 03_核心概念_7_响应格式控制.md - content vs content_and_artifact [来源: 源码]
- [ ] 03_核心概念_8_错误处理模式.md - ToolException、handle_tool_error [来源: 源码]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_基础计算器工具.md - @tool装饰器入门 [来源: Context7]
- [ ] 07_实战代码_场景2_异步天气API工具.md - 外部API集成、异步处理 [来源: Context7+网络]
- [ ] 07_实战代码_场景3_数据库查询工具.md - 复杂验证、连接管理 [来源: 源码+网络]
- [ ] 07_实战代码_场景4_有状态会话工具.md - BaseTool继承、状态管理 [来源: 源码+网络]
- [ ] 07_实战代码_场景5_多格式响应工具.md - content_and_artifact、元数据 [来源: 源码]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 知识点拆解详情

### 核心概念 1：@tool 装饰器方法

**内容要点**：
- 装饰器的三种使用模式（无参、带参、函数调用）
- 自动 schema 推断机制（create_schema_from_function）
- Google-style docstring 解析（parse_docstring=True）
- 类型提示要求和最佳实践
- 与 StructuredTool 的关系（底层委托）

**数据来源**：
- 源码：convert.py 完整实现
- Context7：多个使用示例
- 网络：2025-2026 最佳实践

**关键代码片段**：
```python
# 无参装饰器
@tool
def simple_tool(query: str) -> str:
    """Simple tool description."""
    return f"Result: {query}"

# 带参装饰器
@tool("custom_name", parse_docstring=True)
def advanced_tool(param: int) -> str:
    """Advanced tool.

    Args:
        param: Parameter description.
    """
    return str(param)
```

### 核心概念 2：StructuredTool 创建

**内容要点**：
- StructuredTool.from_function() 类方法
- 显式 schema 定义（args_schema 参数）
- 同步和异步函数支持（func 和 coroutine）
- response_format 配置
- 适用场景（包装现有 API、显式控制）

**数据来源**：
- 源码：structured.py 完整实现
- Context7：Infobip 邮件工具示例

**关键代码片段**：
```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    to: str = Field(description="Email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")

email_tool = StructuredTool.from_function(
    name="send_email",
    description="Send email via API",
    func=api_wrapper.send,
    args_schema=EmailInput,
)
```

### 核心概念 3：BaseTool 类继承

**内容要点**：
- BaseTool 抽象基类接口
- 必须实现的方法（_run）
- 可选实现的方法（_arun）
- 状态管理和初始化
- 与 RunnableSerializable 的集成
- LangGraph 中的状态访问限制（2025-2026 issue）

**数据来源**：
- 源码：base.py 基类定义
- 网络：GitHub #1777 状态访问问题

**关键代码片段**：
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="Query string")

class MyTool(BaseTool):
    name = "my_tool"
    description = "My custom tool"
    args_schema = MyToolInput

    # 工具状态
    api_client: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = initialize_client()

    def _run(self, query: str, config: RunnableConfig) -> str:
        """Synchronous implementation."""
        return self.api_client.query(query)

    async def _arun(self, query: str, config: RunnableConfig) -> str:
        """Asynchronous implementation."""
        return await self.api_client.aquery(query)
```

### 核心概念 4：Schema 定义与验证

**内容要点**：
- Pydantic BaseModel 定义
- Field 描述和约束
- Literal 类型限制
- 默认值设置
- JSON Schema 格式（字典）
- 自动推断 vs 显式定义

**数据来源**：
- 源码：create_schema_from_function 实现
- Context7：WeatherInput 复杂示例

**关键代码片段**：
```python
from typing import Literal
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )
```

### 核心概念 5：参数注入机制

**内容要点**：
- InjectedToolArg 标记
- InjectedToolCallId 注入
- RunnableConfig 注入
- callbacks 参数注入
- 过滤机制（FILTERED_ARGS）
- _is_injected_arg_type 检测

**数据来源**：
- 源码：base.py 注入逻辑
- 网络：ToolRuntime 参数问题（GitHub #6318）

**关键代码片段**：
```python
from langchain_core.tools import InjectedToolArg
from langchain_core.runnables import RunnableConfig
from typing import Annotated

@tool
def tool_with_injection(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Tool with config injection."""
    user_id = config.get("configurable", {}).get("user_id")
    return f"Query: {query}, User: {user_id}"
```

### 核心概念 6：异步工具支持

**内容要点**：
- coroutine 参数（StructuredTool）
- _arun 方法（BaseTool）
- run_in_executor 自动回退
- ainvoke 实现
- 性能优化建议

**数据来源**：
- 源码：structured.py ainvoke 实现
- Context7：异步工具示例

**关键代码片段**：
```python
import asyncio
from langchain.tools import tool

@tool
async def async_api_tool(query: str) -> str:
    """Async API tool."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com?q={query}")
        return response.text
```

### 核心概念 7：响应格式控制

**内容要点**：
- "content" 模式（默认）
- "content_and_artifact" 模式
- ToolMessage 结构
- 使用场景对比
- 最佳实践

**数据来源**：
- 源码：response_format 参数
- Context7：content_and_artifact 示例

**关键代码片段**：
```python
@tool(response_format="content_and_artifact")
def search_api(query: str) -> tuple[str, dict]:
    """Search API with rich response."""
    full_results = {"items": [...], "metadata": {...}}
    summary = f"Found {len(full_results['items'])} results"
    return summary, full_results
```

### 核心概念 8：错误处理模式

**内容要点**：
- ToolException 特殊异常
- handle_tool_error 配置
- 错误传播 vs 错误捕获
- Agent 循环中的错误处理
- 最佳实践

**数据来源**：
- 源码：base.py ToolException 定义

**关键代码片段**：
```python
from langchain.tools import ToolException, tool

@tool
def risky_tool(param: str) -> str:
    """Tool that may fail."""
    try:
        result = external_api_call(param)
        return result
    except APIError as e:
        # 抛出 ToolException 不会中断 Agent
        raise ToolException(f"API failed: {e}")
```

## 实战场景详情

### 场景 1：基础计算器工具（Beginner）

**目标**：使用 @tool 装饰器创建简单工具

**技术点**：
- @tool 无参装饰器
- 类型提示
- Docstring 编写
- 基础验证

**代码大纲**：
```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions.

    Args:
        expression: Math expression like "2 + 2" or "10 * 5"

    Returns:
        Calculation result as string
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# 测试
print(calculator.invoke("2 + 2"))
print(calculator.args_schema.model_json_schema())
```

### 场景 2：异步天气 API 工具（Intermediate）

**目标**：集成外部 API，处理异步调用

**技术点**：
- 异步函数装饰
- 复杂 Pydantic schema
- 错误处理
- API 密钥管理

**代码大纲**：
```python
import httpx
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    location: str = Field(description="City name")
    units: Literal["metric", "imperial"] = Field(default="metric")

@tool(args_schema=WeatherInput)
async def get_weather(location: str, units: str = "metric") -> str:
    """Get current weather for a location."""
    api_key = os.getenv("WEATHER_API_KEY")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.weather.com/v1/current",
            params={"q": location, "units": units, "key": api_key}
        )
        data = response.json()
        return f"Weather in {location}: {data['temp']}°, {data['condition']}"

# 测试
import asyncio
result = asyncio.run(get_weather.ainvoke({"location": "Beijing"}))
```

### 场景 3：数据库查询工具（Advanced）

**目标**：复杂验证、连接管理、SQL 注入防护

**技术点**：
- StructuredTool.from_function
- 复杂 schema 验证
- 连接池管理
- 安全性考虑

**代码大纲**：
```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field, validator
import sqlalchemy

class QueryInput(BaseModel):
    table: str = Field(description="Table name")
    columns: list[str] = Field(description="Columns to select")
    limit: int = Field(default=10, ge=1, le=100)

    @validator('table')
    def validate_table(cls, v):
        # 白名单验证
        allowed_tables = ['users', 'products', 'orders']
        if v not in allowed_tables:
            raise ValueError(f"Table must be one of {allowed_tables}")
        return v

class DatabaseTool:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)

    def query(self, table: str, columns: list[str], limit: int) -> str:
        # 使用参数化查询防止 SQL 注入
        cols = ", ".join(columns)
        query = f"SELECT {cols} FROM {table} LIMIT :limit"
        with self.engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            return str(result.fetchall())

db_tool_instance = DatabaseTool("postgresql://...")
db_query_tool = StructuredTool.from_function(
    name="database_query",
    description="Query database tables safely",
    func=db_tool_instance.query,
    args_schema=QueryInput,
)
```

### 场景 4：有状态会话工具（Enterprise）

**目标**：BaseTool 继承，状态管理，初始化/清理

**技术点**：
- BaseTool 子类化
- __init__ 初始化
- 状态存储
- 资源清理

**代码大纲**：
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

class SessionInput(BaseModel):
    action: str = Field(description="Action: 'get', 'set', 'delete'")
    key: str = Field(description="Session key")
    value: str = Field(default="", description="Value for 'set' action")

class SessionTool(BaseTool):
    name = "session_manager"
    description = "Manage user session data"
    args_schema = SessionInput

    # 状态
    session_store: dict[str, Any] = {}
    user_id: str = ""

    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.session_store = {}

    def _run(self, action: str, key: str, value: str = "", config: RunnableConfig = None) -> str:
        session_key = f"{self.user_id}:{key}"

        if action == "get":
            return self.session_store.get(session_key, "Not found")
        elif action == "set":
            self.session_store[session_key] = value
            return f"Set {key} = {value}"
        elif action == "delete":
            self.session_store.pop(session_key, None)
            return f"Deleted {key}"
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _arun(self, *args, **kwargs) -> str:
        # 异步版本（可选）
        return self._run(*args, **kwargs)

# 使用
user_tool = SessionTool(user_id="user123")
user_tool.invoke({"action": "set", "key": "name", "value": "Alice"})
user_tool.invoke({"action": "get", "key": "name"})
```

### 场景 5：多格式响应工具（Production）

**目标**：content_and_artifact 模式，丰富元数据

**技术点**：
- response_format="content_and_artifact"
- ToolMessage 结构
- 元数据和 extras
- 生产级错误处理

**代码大纲**：
```python
from langchain.tools import tool
from typing import Tuple, Dict, Any
import json

@tool(response_format="content_and_artifact")
def advanced_search(query: str, max_results: int = 10) -> Tuple[str, Dict[str, Any]]:
    """Advanced search with rich metadata.

    Returns:
        Tuple of (summary, full_results)
    """
    # 模拟搜索
    results = [
        {"title": f"Result {i}", "url": f"https://example.com/{i}", "score": 0.9 - i*0.1}
        for i in range(max_results)
    ]

    # 摘要（给 LLM 看）
    summary = f"Found {len(results)} results for '{query}'. Top result: {results[0]['title']}"

    # 完整数据（给程序用）
    artifact = {
        "query": query,
        "total_results": len(results),
        "results": results,
        "metadata": {
            "search_time_ms": 123,
            "api_version": "v2",
        }
    }

    return summary, artifact

# 使用
from langchain_core.messages import ToolMessage

result = advanced_search.invoke({"query": "LangChain", "max_results": 5})
# result 是 ToolMessage
print(result.content)  # 摘要
print(result.artifact)  # 完整数据
```

## 生成进度

- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [ ] 阶段二：补充调研（针对需要更多资料的部分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）

## 文档生成策略

### 批次划分（使用 subagent）

**批次 1：基础维度（前3个）**
- 00_概览.md
- 01_30字核心.md
- 02_第一性原理.md

**批次 2：核心概念（1-4）**
- 03_核心概念_1_@tool装饰器方法.md
- 03_核心概念_2_StructuredTool创建.md
- 03_核心概念_3_BaseTool类继承.md
- 03_核心概念_4_Schema定义与验证.md

**批次 3：核心概念（5-8）**
- 03_核心概念_5_参数注入机制.md
- 03_核心概念_6_异步工具支持.md
- 03_核心概念_7_响应格式控制.md
- 03_核心概念_8_错误处理模式.md

**批次 4：基础维度（中3个）**
- 04_最小可用.md
- 05_双重类比.md
- 06_反直觉点.md

**批次 5：实战代码（1-3）**
- 07_实战代码_场景1_基础计算器工具.md
- 07_实战代码_场景2_异步天气API工具.md
- 07_实战代码_场景3_数据库查询工具.md

**批次 6：实战代码（4-5）**
- 07_实战代码_场景4_有状态会话工具.md
- 07_实战代码_场景5_多格式响应工具.md

**批次 7：基础维度（后3个）**
- 08_面试必问.md
- 09_化骨绵掌.md
- 10_一句话总结.md

### 每批次执行方式
- 开启 3 个 subagent
- 每个 subagent 生成 2 个文档（批次 1-4, 7）
- 每个 subagent 生成 1-2 个文档（批次 5-6，实战代码较长）

## 质量控制

### 引用规范
- 源码引用：`[来源: sourcecode/langchain/libs/core/langchain_core/tools/<文件名>]`
- Context7 引用：`[来源: reference/context7_langchain_tools_01.md | LangChain 官方文档]`
- 网络引用：`[来源: reference/search_tool_development_01.md | GitHub #<issue号> / Reddit]`

### 代码质量
- 所有代码必须完整可运行
- 包含必要的 import 语句
- 提供测试示例
- 注释关键逻辑

### 文档长度
- 目标：300-500 行/文件
- 超过 500 行自动拆分
- 实战代码可适当放宽（600 行以内）

## 下一步操作

1. **用户确认 Plan**：是否需要调整知识点拆解或实战场景？
2. **阶段二（可选）**：是否需要抓取待抓取链接获取更多细节？
3. **阶段三**：开始批量生成文档（使用 subagent）
