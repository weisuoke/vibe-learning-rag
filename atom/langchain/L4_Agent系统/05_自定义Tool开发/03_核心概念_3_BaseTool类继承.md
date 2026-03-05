# 核心概念 3：BaseTool 类继承

> 通过继承 BaseTool 实现最大控制权的自定义工具开发

---

## 概述

BaseTool 是 LangChain 工具系统的抽象基类，提供了最大的灵活性和控制权。当你需要管理复杂状态、实现自定义初始化逻辑、或需要完全控制工具的生命周期时，继承 BaseTool 是最佳选择。

**核心价值**：
- 完全控制工具的初始化和清理
- 支持复杂的状态管理
- 可以重写更多生命周期方法
- 与 LCEL 深度集成（继承自 RunnableSerializable）

---

## BaseTool 抽象基类接口

### 类定义

```python
# 源码位置: langchain_core/tools/base.py
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

class BaseTool(RunnableSerializable[Union[str, Dict, ToolCall], Any]):
    """Base class for tools."""

    # 必需属性
    name: str  # 工具名称
    description: str  # 工具描述
    args_schema: type[BaseModel]  # 输入 schema

    # 可选属性
    return_direct: bool = False  # 是否直接返回结果
    verbose: bool = False  # 是否打印详细日志
    handle_tool_error: bool | str | Callable = False  # 错误处理策略
    response_format: Literal["content", "content_and_artifact"] = "content"

    # 必须实现的方法
    @abstractmethod
    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        **kwargs: Any
    ) -> Any:
        """Use the tool synchronously."""

    # 可选实现的方法
    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        **kwargs: Any
    ) -> Any:
        """Use the tool asynchronously."""
```

[来源: sourcecode/langchain/libs/core/langchain_core/tools/base.py]

### 关键设计点

1. **继承自 RunnableSerializable**
   - 完全集成 LCEL 表达式
   - 支持管道操作符 `|`
   - 自动支持 `invoke()`, `ainvoke()`, `batch()`, `stream()` 等方法

2. **抽象方法 `_run()`**
   - 必须实现的同步执行方法
   - 接收 `config: RunnableConfig` 参数
   - 返回任意类型的结果

3. **可选方法 `_arun()`**
   - 异步执行方法
   - 如果不实现，会自动回退到线程池执行 `_run()`
   - 推荐为 I/O 密集型工具实现异步版本

---

## 必须实现的方法

### _run() 方法

```python
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate mathematical expressions"
    args_schema = CalculatorInput

    def _run(
        self,
        expression: str,
        config: RunnableConfig | None = None
    ) -> str:
        """同步执行计算."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# 使用
tool = CalculatorTool()
result = tool.invoke({"expression": "2 + 2"})
print(result)  # "Result: 4"
```

**关键点**：
- 方法签名必须包含 `config: RunnableConfig` 参数
- 参数名称必须与 `args_schema` 中定义的字段一致
- 返回值类型可以是任意类型（字符串、字典、列表等）

### _arun() 方法（可选）

```python
import httpx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="City name")

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a location"
    args_schema = WeatherInput

    # 同步版本
    def _run(
        self,
        location: str,
        config: RunnableConfig | None = None
    ) -> str:
        """同步获取天气（不推荐用于 I/O 操作）."""
        # 使用同步 HTTP 库
        import requests
        response = requests.get(f"https://api.weather.com?q={location}")
        return response.text

    # 异步版本（推荐）
    async def _arun(
        self,
        location: str,
        config: RunnableConfig | None = None
    ) -> str:
        """异步获取天气（推荐）."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.weather.com?q={location}")
            return response.text

# 使用
tool = WeatherTool()

# 同步调用
result = tool.invoke({"location": "Beijing"})

# 异步调用
import asyncio
result = asyncio.run(tool.ainvoke({"location": "Beijing"}))
```

**最佳实践**：
- I/O 密集型工具（API 调用、数据库查询）应实现 `_arun()`
- CPU 密集型工具可以只实现 `_run()`
- 如果不实现 `_arun()`，框架会自动使用线程池执行 `_run()`

[来源: sourcecode/langchain/libs/core/langchain_core/tools/structured.py | ainvoke 实现]

---

## 状态管理和初始化

### 工具状态

BaseTool 支持在类中定义状态字段，这是它相比 `@tool` 装饰器的主要优势。

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any
import redis

class CacheInput(BaseModel):
    key: str = Field(description="Cache key")
    value: str = Field(default="", description="Value to set (for set operation)")
    operation: str = Field(description="Operation: 'get' or 'set'")

class RedisCacheTool(BaseTool):
    name = "redis_cache"
    description = "Get or set values in Redis cache"
    args_schema = CacheInput

    # 工具状态
    redis_client: Any = None
    connection_string: str = ""

    def __init__(self, connection_string: str, **kwargs):
        """初始化工具并建立 Redis 连接."""
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self.redis_client = redis.from_url(connection_string)

    def _run(
        self,
        key: str,
        value: str = "",
        operation: str = "get",
        config: RunnableConfig | None = None
    ) -> str:
        """执行 Redis 操作."""
        if operation == "get":
            result = self.redis_client.get(key)
            return result.decode() if result else "Key not found"
        elif operation == "set":
            self.redis_client.set(key, value)
            return f"Set {key} = {value}"
        else:
            return f"Unknown operation: {operation}"

    def __del__(self):
        """清理资源."""
        if self.redis_client:
            self.redis_client.close()

# 使用
tool = RedisCacheTool(connection_string="redis://localhost:6379")
tool.invoke({"key": "user:123", "value": "Alice", "operation": "set"})
tool.invoke({"key": "user:123", "operation": "get"})
```

**状态管理要点**：
- 状态字段定义为类属性
- 在 `__init__()` 中初始化状态
- 必须调用 `super().__init__(**kwargs)` 初始化基类
- 可以在 `__del__()` 中清理资源

---

## 与 RunnableSerializable 的集成

BaseTool 继承自 `RunnableSerializable`，这意味着它完全集成了 LCEL 表达式系统。

### LCEL 管道操作

```python
from langchain.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

class SearchTool(BaseTool):
    name = "search"
    description = "Search the web"
    args_schema = SearchInput

    def _run(self, query: str, config: RunnableConfig | None = None) -> str:
        return f"Search results for: {query}"

# 创建工具
search_tool = SearchTool()

# 与 LCEL 组合
llm = ChatOpenAI()
chain = search_tool | llm | StrOutputParser()

# 执行
result = chain.invoke({"query": "LangChain tutorials"})
```

### Runnable 方法支持

```python
# invoke() - 单次调用
result = tool.invoke({"query": "test"})

# ainvoke() - 异步调用
result = await tool.ainvoke({"query": "test"})

# batch() - 批量调用
results = tool.batch([
    {"query": "query1"},
    {"query": "query2"},
    {"query": "query3"}
])

# stream() - 流式调用（如果工具支持）
for chunk in tool.stream({"query": "test"}):
    print(chunk)
```

[来源: sourcecode/langchain/libs/core/langchain_core/tools/base.py | RunnableSerializable 继承]

---

## LangGraph 状态访问限制（2025-2026）

### 已知问题

在 LangGraph 中使用 BaseTool 子类时，存在状态访问限制。

```python
# ❌ 问题：BaseTool 无法通过 run_manager 访问 LangGraph 状态
from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode

class StatefulTool(BaseTool):
    name = "stateful_tool"
    description = "Tool that needs graph state"

    def _run(self, query: str, config: RunnableConfig | None = None) -> str:
        # 尝试访问 LangGraph 状态
        # ❌ 这在 BaseTool 中不工作
        state = config.get("configurable", {}).get("state")
        return f"State: {state}"

# ✅ 解决方案：使用 @tool 装饰器
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def stateful_tool(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Tool with state access."""
    state = config.get("configurable", {}).get("state")
    return f"State: {state}"
```

[来源: reference/search_tool_development_01.md | GitHub #1777]

### 何时使用 BaseTool vs @tool

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 简单无状态工具 | `@tool` | 代码更简洁 |
| 需要访问 LangGraph 状态 | `@tool` | 支持 InjectedToolArg |
| 复杂初始化逻辑 | `BaseTool` | 可以自定义 `__init__()` |
| 需要管理连接池 | `BaseTool` | 状态管理更灵活 |
| 需要资源清理 | `BaseTool` | 可以实现 `__del__()` |
| 需要重写多个方法 | `BaseTool` | 更多控制权 |

---

## 完整的有状态工具示例

### 数据库连接池工具

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, validator
from typing import Any, List
import sqlalchemy
from sqlalchemy.pool import QueuePool

class QueryInput(BaseModel):
    """数据库查询输入."""
    table: str = Field(description="Table name to query")
    columns: List[str] = Field(description="Columns to select")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")

    @validator('table')
    def validate_table(cls, v):
        """验证表名（白名单）."""
        allowed_tables = ['users', 'products', 'orders']
        if v not in allowed_tables:
            raise ValueError(f"Table must be one of {allowed_tables}")
        return v

    @validator('columns')
    def validate_columns(cls, v):
        """验证列名."""
        allowed_columns = ['id', 'name', 'email', 'created_at', 'price', 'quantity']
        for col in v:
            if col not in allowed_columns:
                raise ValueError(f"Column {col} not allowed")
        return v

class DatabaseQueryTool(BaseTool):
    """安全的数据库查询工具."""

    name = "database_query"
    description = """Query database tables safely.

    Use this tool to retrieve data from the database.
    Only allowed tables: users, products, orders.
    """
    args_schema = QueryInput

    # 工具状态
    engine: Any = None
    connection_string: str = ""
    max_connections: int = 5

    def __init__(
        self,
        connection_string: str,
        max_connections: int = 5,
        **kwargs
    ):
        """初始化数据库连接池."""
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self.max_connections = max_connections

        # 创建连接池
        self.engine = sqlalchemy.create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=max_connections,
            max_overflow=10,
            pool_pre_ping=True  # 自动检测断开的连接
        )

    def _run(
        self,
        table: str,
        columns: List[str],
        limit: int = 10,
        config: RunnableConfig | None = None
    ) -> str:
        """同步执行查询."""
        try:
            # 构建安全的查询（使用参数化查询）
            cols = ", ".join(columns)
            query = sqlalchemy.text(f"SELECT {cols} FROM {table} LIMIT :limit")

            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(query, {"limit": limit})
                rows = result.fetchall()

                # 格式化结果
                if not rows:
                    return "No results found"

                formatted = []
                for row in rows:
                    formatted.append(dict(zip(columns, row)))

                return str(formatted)

        except Exception as e:
            return f"Query error: {str(e)}"

    async def _arun(
        self,
        table: str,
        columns: List[str],
        limit: int = 10,
        config: RunnableConfig | None = None
    ) -> str:
        """异步执行查询."""
        # 对于数据库操作，可以使用异步驱动
        # 这里简化为调用同步版本
        return self._run(table, columns, limit, config)

    def __del__(self):
        """清理连接池."""
        if self.engine:
            self.engine.dispose()

# 使用示例
tool = DatabaseQueryTool(
    connection_string="postgresql://user:pass@localhost/mydb",
    max_connections=10
)

# 查询用户表
result = tool.invoke({
    "table": "users",
    "columns": ["id", "name", "email"],
    "limit": 5
})
print(result)

# 与 Agent 集成
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(temperature=0)
prompt = hub.pull("langchain-ai/openai-functions-template")
agent = create_openai_functions_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

agent_executor.invoke({
    "input": "Show me the first 3 users in the database"
})
```

[来源: reference/context7_langchain_tools_01.md | StructuredTool 示例改编]

---

## 错误处理模式

### handle_tool_error 配置

```python
from langchain.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

class APIInput(BaseModel):
    endpoint: str = Field(description="API endpoint to call")

class APITool(BaseTool):
    name = "api_call"
    description = "Call external API"
    args_schema = APIInput

    # 错误处理策略
    handle_tool_error = True  # 捕获错误并返回错误消息

    def _run(
        self,
        endpoint: str,
        config: RunnableConfig | None = None
    ) -> str:
        """调用 API."""
        try:
            # 模拟 API 调用
            if endpoint == "/error":
                raise Exception("API error")
            return f"Success: {endpoint}"

        except Exception as e:
            # 抛出 ToolException 不会中断 Agent
            raise ToolException(f"API call failed: {str(e)}")

# 使用
tool = APITool()

# 正常调用
result = tool.invoke({"endpoint": "/users"})
print(result)  # "Success: /users"

# 错误调用（不会中断程序）
result = tool.invoke({"endpoint": "/error"})
print(result)  # "API call failed: API error"
```

### 自定义错误处理函数

```python
def custom_error_handler(error: Exception) -> str:
    """自定义错误处理."""
    if isinstance(error, TimeoutError):
        return "Request timed out. Please try again."
    elif isinstance(error, ConnectionError):
        return "Connection failed. Check your network."
    else:
        return f"Unexpected error: {str(error)}"

class RobustAPITool(BaseTool):
    name = "robust_api"
    description = "API tool with custom error handling"
    args_schema = APIInput

    # 使用自定义错误处理函数
    handle_tool_error = custom_error_handler

    def _run(self, endpoint: str, config: RunnableConfig | None = None) -> str:
        # 实现...
        pass
```

[来源: sourcecode/langchain/libs/core/langchain_core/tools/base.py | ToolException 定义]

---

## 最佳实践

### 1. 何时继承 BaseTool

**适合场景**：
- 需要管理数据库连接池
- 需要初始化外部客户端（Redis、Elasticsearch 等）
- 需要在工具间共享状态
- 需要实现资源清理逻辑
- 需要重写多个生命周期方法

**不适合场景**：
- 简单的无状态函数（用 `@tool`）
- 需要访问 LangGraph 状态（用 `@tool` + `InjectedToolArg`）
- 快速原型开发（用 `@tool`）

### 2. 初始化模式

```python
class MyTool(BaseTool):
    # 状态字段
    client: Any = None
    config: dict = {}

    def __init__(self, api_key: str, **kwargs):
        # 1. 先调用父类初始化
        super().__init__(**kwargs)

        # 2. 再初始化自己的状态
        self.client = ExternalClient(api_key)
        self.config = {"timeout": 30}
```

### 3. 资源清理

```python
class ResourceTool(BaseTool):
    connection: Any = None

    def __del__(self):
        """清理资源."""
        if self.connection:
            self.connection.close()
```

### 4. 类型安全

```python
from typing import TypedDict

class ToolState(TypedDict):
    """工具状态类型定义."""
    client: Any
    cache: dict
    retry_count: int

class TypeSafeTool(BaseTool):
    state: ToolState

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = {
            "client": None,
            "cache": {},
            "retry_count": 3
        }
```

---

## 与其他创建方式对比

| 特性 | @tool | StructuredTool | BaseTool |
|------|-------|----------------|----------|
| 代码简洁度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 状态管理 | ❌ | ❌ | ✅ |
| 初始化控制 | ❌ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| LangGraph 状态访问 | ✅ | ✅ | ❌ |
| 资源清理 | ❌ | ❌ | ✅ |
| 学习曲线 | 低 | 中 | 高 |
| 适用场景 | 简单工具 | 中等复杂度 | 复杂工具 |

[来源: reference/search_tool_development_01.md | @tool vs BaseTool 对比]

---

## 常见陷阱

### 1. 忘记调用 super().__init__()

```python
# ❌ 错误
class BadTool(BaseTool):
    def __init__(self, api_key: str):
        self.api_key = api_key  # 忘记调用 super().__init__()

# ✅ 正确
class GoodTool(BaseTool):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)  # 必须调用
        self.api_key = api_key
```

### 2. 参数名称不匹配

```python
# ❌ 错误
class BadInput(BaseModel):
    query_text: str  # schema 中的字段名

class BadTool(BaseTool):
    args_schema = BadInput

    def _run(self, query: str, config=None):  # 参数名不匹配
        pass

# ✅ 正确
class GoodInput(BaseModel):
    query: str

class GoodTool(BaseTool):
    args_schema = GoodInput

    def _run(self, query: str, config=None):  # 参数名匹配
        pass
```

### 3. 在 LangGraph 中访问状态

```python
# ❌ 不推荐：BaseTool 在 LangGraph 中状态访问受限
class StatefulBaseTool(BaseTool):
    def _run(self, query: str, config=None):
        state = config.get("configurable", {}).get("state")  # 可能不工作
        return state

# ✅ 推荐：使用 @tool + InjectedToolArg
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def stateful_tool(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    state = config.get("configurable", {}).get("state")
    return state
```

---

## 总结

BaseTool 类继承提供了最大的灵活性和控制权，适合需要复杂状态管理、资源初始化和清理的场景。虽然代码量比 `@tool` 装饰器多，但在企业级应用中，这种额外的控制权是值得的。

**关键要点**：
1. 必须实现 `_run()` 方法
2. 可选实现 `_arun()` 方法提升性能
3. 在 `__init__()` 中初始化状态
4. 在 `__del__()` 中清理资源
5. 注意 LangGraph 状态访问限制
6. 参数名称必须与 `args_schema` 一致

**何时使用**：
- ✅ 需要管理连接池或外部客户端
- ✅ 需要在工具间共享状态
- ✅ 需要实现复杂的初始化逻辑
- ❌ 简单的无状态函数（用 `@tool`）
- ❌ 需要访问 LangGraph 状态（用 `@tool` + `InjectedToolArg`）
