# 核心概念 2：StructuredTool 创建

## 一句话定义

**StructuredTool 是 LangChain 提供的灵活工具创建方式，通过 `from_function()` 类方法将现有函数包装为工具，支持显式 schema 定义和同步/异步双模式。**

---

## 详细解释

### 什么是 StructuredTool？

StructuredTool 是 LangChain 工具体系中的**中间层实现**，它继承自 `BaseTool`，提供了比 `@tool` 装饰器更多的控制权，同时又比直接继承 `BaseTool` 更简单。

**核心特点**：
1. **显式配置**：可以精确控制工具的名称、描述、schema
2. **函数包装**：将现有函数（同步或异步）包装为工具
3. **灵活性**：支持复杂的输入验证和响应格式控制
4. **API 集成友好**：特别适合包装第三方 API 客户端

### 为什么需要 StructuredTool？

在以下场景中，`@tool` 装饰器可能不够用：

1. **包装现有 API**：你有一个已经写好的 API 客户端类，不想修改它
2. **显式控制 schema**：需要精确定义输入验证规则
3. **动态创建工具**：运行时根据配置生成工具
4. **复杂响应格式**：需要使用 `content_and_artifact` 模式

**类比**：
- **前端类比**：就像 React 的 `React.createElement()` vs JSX，前者更灵活但更冗长
- **日常生活类比**：就像定制西装 vs 成衣，前者可以精确控制每个细节

---

## 代码示例

### 示例 1：基础用法 - 包装简单函数

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 1. 定义输入 schema
class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '2 + 2' 或 '10 * 5'")

# 2. 定义工具函数
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

# 3. 创建 StructuredTool
calculator_tool = StructuredTool.from_function(
    name="calculator",
    description="执行数学计算，支持加减乘除和括号",
    func=calculate,
    args_schema=CalculatorInput,
)

# 4. 使用工具
result = calculator_tool.invoke({"expression": "2 + 2"})
print(result)  # 输出: 计算结果: 4
```

**关键点**：
- `from_function()` 是类方法，用于创建实例
- `args_schema` 参数定义输入验证规则
- `func` 参数接收同步函数

---

### 示例 2：包装现有 API 客户端

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal

# 假设你有一个现有的 API 客户端
class WeatherAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_weather(self, location: str, units: str = "metric") -> dict:
        """调用天气 API"""
        # 实际 API 调用逻辑
        return {
            "location": location,
            "temperature": 22,
            "condition": "晴天",
            "units": units
        }

# 1. 定义输入 schema
class WeatherInput(BaseModel):
    """天气查询输入"""
    location: str = Field(description="城市名称，如 '北京' 或 '上海'")
    units: Literal["metric", "imperial"] = Field(
        default="metric",
        description="温度单位：metric (摄氏度) 或 imperial (华氏度)"
    )

# 2. 初始化 API 客户端
weather_client = WeatherAPIClient(api_key="your_api_key")

# 3. 包装为工具
weather_tool = StructuredTool.from_function(
    name="get_weather",
    description="获取指定城市的当前天气信息",
    func=weather_client.get_weather,  # 直接使用客户端方法
    args_schema=WeatherInput,
)

# 4. 使用工具
result = weather_tool.invoke({"location": "北京", "units": "metric"})
print(result)
```

**优势**：
- 不需要修改现有的 `WeatherAPIClient` 类
- 通过 `args_schema` 添加输入验证
- 工具描述独立于原始函数

---

### 示例 3：异步工具支持

```python
import asyncio
import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 1. 定义输入 schema
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=20, description="最大结果数")

# 2. 定义异步函数
async def async_search(query: str, max_results: int = 5) -> str:
    """异步搜索 API"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/search",
            params={"q": query, "limit": max_results}
        )
        data = response.json()
        return f"找到 {len(data['results'])} 条结果"

# 3. 创建异步工具
search_tool = StructuredTool.from_function(
    name="web_search",
    description="在网络上搜索信息",
    coroutine=async_search,  # 使用 coroutine 参数而非 func
    args_schema=SearchInput,
)

# 4. 异步调用
async def main():
    result = await search_tool.ainvoke({"query": "LangChain", "max_results": 10})
    print(result)

asyncio.run(main())
```

**关键区别**：
- 异步函数使用 `coroutine` 参数
- 调用时使用 `ainvoke()` 而非 `invoke()`
- 自动支持并发执行

---

### 示例 4：content_and_artifact 响应格式

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Tuple, Dict, Any

# 1. 定义输入 schema
class DatabaseQueryInput(BaseModel):
    table: str = Field(description="表名")
    limit: int = Field(default=10, ge=1, le=100, description="返回行数")

# 2. 定义返回 tuple 的函数
def query_database(table: str, limit: int = 10) -> Tuple[str, Dict[str, Any]]:
    """查询数据库并返回摘要和完整数据"""
    # 模拟数据库查询
    full_data = {
        "rows": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ],
        "metadata": {
            "query_time_ms": 45,
            "total_rows": 2,
        }
    }

    # 摘要（给 LLM 看）
    summary = f"从表 {table} 查询到 {len(full_data['rows'])} 行数据"

    # 返回 (摘要, 完整数据)
    return summary, full_data

# 3. 创建工具（指定 response_format）
db_tool = StructuredTool.from_function(
    name="database_query",
    description="查询数据库表",
    func=query_database,
    args_schema=DatabaseQueryInput,
    response_format="content_and_artifact",  # 关键配置
)

# 4. 使用工具
result = db_tool.invoke({"table": "users", "limit": 10})
print(f"摘要: {result.content}")
print(f"完整数据: {result.artifact}")
```

**使用场景**：
- **摘要给 LLM**：简洁的文本描述，帮助 LLM 理解结果
- **完整数据给程序**：结构化数据，供后续处理使用
- **元数据保留**：查询时间、版本信息等

---

## 在实际应用中的使用

### 场景 1：RAG 系统中的向量检索工具

```python
from langchain.tools import StructuredTool
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

class RetrievalInput(BaseModel):
    query: str = Field(description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="返回文档数")

# 假设已有向量数据库
vectorstore = Chroma(...)

def retrieve_documents(query: str, top_k: int = 3) -> str:
    """从向量数据库检索相关文档"""
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_tool = StructuredTool.from_function(
    name="knowledge_base_search",
    description="从知识库中检索相关文档",
    func=retrieve_documents,
    args_schema=RetrievalInput,
)
```

---

### 场景 2：Agent 系统中的多工具集成

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# 创建多个工具
tools = [
    calculator_tool,
    weather_tool,
    search_tool,
    retrieval_tool,
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

# 执行任务
result = agent_executor.invoke({
    "input": "北京今天天气如何？如果温度超过 25 度，计算 25 * 1.8 + 32"
})
```

---

## 与 @tool 装饰器的对比

| 特性 | @tool 装饰器 | StructuredTool.from_function() |
|------|-------------|-------------------------------|
| **代码简洁性** | ✅ 非常简洁 | ⚠️ 稍显冗长 |
| **显式控制** | ⚠️ 有限 | ✅ 完全控制 |
| **包装现有函数** | ⚠️ 需要修改函数 | ✅ 无需修改 |
| **动态创建** | ❌ 不支持 | ✅ 支持 |
| **schema 推断** | ✅ 自动推断 | ⚠️ 需要显式定义 |
| **适用场景** | 简单工具 | 复杂工具、API 包装 |

**选择建议**：
- **优先使用 @tool**：如果你在写新函数，且逻辑简单
- **使用 StructuredTool**：如果你需要包装现有 API 或需要精确控制

---

## 源码实现原理

### StructuredTool 类定义

```python
# 源码位置: langchain_core/tools/structured.py
class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: ArgsSchema = Field(...)  # 必需
    func: Callable[..., Any] | None = None  # 同步函数
    coroutine: Callable[..., Awaitable[Any]] | None = None  # 异步函数

    @classmethod
    def from_function(
        cls,
        func: Callable | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
        name: str | None = None,
        description: str | None = None,
        args_schema: ArgsSchema | None = None,
        response_format: Literal["content", "content_and_artifact"] = "content",
        **kwargs: Any,
    ) -> StructuredTool:
        """从函数创建工具"""
        # 实现逻辑...
```

**关键设计**：
1. 同时支持 `func` 和 `coroutine`，自动处理同步/异步
2. `args_schema` 是必需的，确保输入验证
3. `response_format` 控制返回格式

---

## 最佳实践

### 1. Schema 设计原则

```python
from pydantic import BaseModel, Field, validator
from typing import Literal

class GoodSchema(BaseModel):
    """✅ 好的 schema 设计"""

    # 1. 使用 Field 提供详细描述
    location: str = Field(description="城市名称，如 '北京' 或 '上海'")

    # 2. 使用 Literal 限制枚举值
    units: Literal["metric", "imperial"] = Field(default="metric")

    # 3. 使用验证器添加业务规则
    @validator('location')
    def validate_location(cls, v):
        if len(v) < 2:
            raise ValueError("城市名称至少 2 个字符")
        return v

    # 4. 提供合理的默认值
    timeout: int = Field(default=30, ge=1, le=300)
```

### 2. 错误处理

```python
from langchain.tools import ToolException

def safe_api_call(param: str) -> str:
    """带错误处理的 API 调用"""
    try:
        result = external_api.call(param)
        return result
    except APIError as e:
        # 使用 ToolException 不会中断 Agent
        raise ToolException(f"API 调用失败: {e}")
    except Exception as e:
        # 其他异常会中断 Agent
        raise

tool = StructuredTool.from_function(
    func=safe_api_call,
    name="api_tool",
    description="调用外部 API",
    args_schema=...,
)
```

### 3. 异步优先

```python
# ✅ 推荐：使用异步函数
async def async_tool_func(param: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com?q={param}")
        return response.text

tool = StructuredTool.from_function(
    coroutine=async_tool_func,  # 使用 coroutine
    name="async_tool",
    description="异步工具",
    args_schema=...,
)

# ❌ 不推荐：同步函数会阻塞
def sync_tool_func(param: str) -> str:
    response = requests.get(f"https://api.example.com?q={param}")
    return response.text
```

---

## 常见陷阱

### 陷阱 1：忘记定义 args_schema

```python
# ❌ 错误：缺少 args_schema
tool = StructuredTool.from_function(
    func=my_function,
    name="my_tool",
    description="我的工具",
    # 缺少 args_schema！
)
# 结果：LLM 不知道如何调用工具

# ✅ 正确：始终提供 args_schema
tool = StructuredTool.from_function(
    func=my_function,
    name="my_tool",
    description="我的工具",
    args_schema=MySchema,  # 必需
)
```

### 陷阱 2：schema 与函数签名不匹配

```python
class MySchema(BaseModel):
    param1: str
    param2: int

# ❌ 错误：函数参数名不匹配
def my_function(arg1: str, arg2: int) -> str:
    return f"{arg1} {arg2}"

# ✅ 正确：参数名必须一致
def my_function(param1: str, param2: int) -> str:
    return f"{param1} {param2}"
```

### 陷阱 3：混用 func 和 coroutine

```python
# ❌ 错误：同时提供 func 和 coroutine
tool = StructuredTool.from_function(
    func=sync_function,
    coroutine=async_function,  # 冲突！
    name="my_tool",
    description="我的工具",
    args_schema=MySchema,
)

# ✅ 正确：只提供一个
tool = StructuredTool.from_function(
    coroutine=async_function,  # 只用异步
    name="my_tool",
    description="我的工具",
    args_schema=MySchema,
)
```

---

## 参考资料

**来源标注**：
- [源码分析] `langchain_core/tools/structured.py` - StructuredTool 实现
- [Context7 文档] LangChain 官方文档 - StructuredTool.from_function() 用法
- [Context7 文档] Infobip 邮件工具示例 - API 包装最佳实践

---

## 学习检查清单

- [ ] 理解 StructuredTool 与 @tool 的区别
- [ ] 掌握 `from_function()` 的基本用法
- [ ] 能够定义复杂的 Pydantic schema
- [ ] 理解同步/异步工具的创建方式
- [ ] 掌握 `content_and_artifact` 响应格式
- [ ] 能够包装现有 API 客户端为工具
- [ ] 理解常见陷阱并能避免

---

## 下一步学习

- **核心概念 3**：BaseTool 类继承 - 最大控制权的工具创建方式
- **核心概念 4**：Schema 定义与验证 - 深入 Pydantic 验证机制
- **实战代码 3**：数据库查询工具 - StructuredTool 的生产级应用
