# 核心概念 1：@tool 装饰器方法

> 现代推荐的工具创建方式，自动 schema 推断，代码简洁优雅

---

## 概述

`@tool` 装饰器是 LangChain 中创建自定义工具的**首选方式**，它通过装饰器语法将普通 Python 函数转换为 Agent 可调用的工具。这种方式结合了简洁性和强大功能，是 2025-2026 年社区推荐的主流方法。

**核心价值**：
- 自动从函数签名推断 schema
- 支持 Google-style docstring 解析
- 代码量最少，可读性最高
- 与 LangGraph 状态管理无缝集成

---

## 三种使用模式

### 模式 1：无参数装饰器（最简单）

**适用场景**：简单工具，函数名即工具名，docstring 即描述

```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word.

    Args:
        word: The word to measure

    Returns:
        The number of characters in the word
    """
    return len(word)

# 自动生成的工具属性
print(get_word_length.name)         # "get_word_length"
print(get_word_length.description)  # "Returns the length of a word."
print(get_word_length.args)         # {"word": {"type": "string"}}
```

**工作原理**：
1. 函数名自动转换为工具名（snake_case 保持不变）
2. Docstring 第一行作为工具描述
3. 类型提示自动推断为 JSON Schema
4. 参数描述从 docstring 的 Args 部分提取

[来源: reference/source_tool_development_01.md | convert.py 源码分析]

---

### 模式 2：带参数装饰器（自定义配置）

**适用场景**：需要自定义工具名、描述或启用高级特性

```python
from langchain.tools import tool

@tool(
    "calculator",  # 自定义工具名
    description="Performs arithmetic calculations. Use this for any math problems.",
    parse_docstring=True,  # 启用 docstring 解析
    return_direct=False,   # 是否直接返回结果给用户
)
def calc(expression: str) -> str:
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

# 使用自定义配置
print(calc.name)         # "calculator" (自定义名称)
print(calc.description)  # "Performs arithmetic calculations..."
```

**关键参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name_or_callable` | str \| Callable | None | 工具名称或被装饰的函数 |
| `description` | str | None | 工具描述（覆盖 docstring） |
| `parse_docstring` | bool | False | 是否解析 Google-style docstring |
| `args_schema` | BaseModel | None | 显式指定输入 schema |
| `return_direct` | bool | False | 是否直接返回结果给用户 |
| `response_format` | Literal | "content" | 响应格式（content 或 content_and_artifact） |
| `infer_schema` | bool | True | 是否自动推断 schema |

[来源: reference/source_tool_development_01.md | convert.py 参数定义]

---

### 模式 3：函数调用模式（动态创建）

**适用场景**：运行时动态创建工具，或包装现有函数

```python
from langchain.tools import tool

def existing_function(query: str, max_results: int = 10) -> str:
    """Search function that already exists."""
    return f"Found {max_results} results for '{query}'"

# 动态包装为工具
search_tool = tool(
    "web_search",
    description="Search the web for information",
)(existing_function)

# 或者一步完成
search_tool = tool(
    "web_search",
    description="Search the web for information",
    func=existing_function
)
```

**使用场景**：
- 包装第三方库函数
- 运行时根据配置创建工具
- 批量创建相似工具

---

## 自动 Schema 推断机制

### 核心函数：create_schema_from_function

`@tool` 装饰器底层使用 `create_schema_from_function` 自动推断 schema：

```python
# 源码简化版（来自 base.py）
def create_schema_from_function(
    model_name: str,
    func: Callable,
    *,
    filter_args: Sequence[str] | None = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> type[BaseModel]:
    """Create a Pydantic schema from a function's signature."""

    # 1. 获取函数签名
    sig = inspect.signature(func)

    # 2. 检测 Pydantic v1/v2 注解
    # 3. 使用 validate_arguments() 创建 Pydantic 模型
    # 4. 过滤特殊参数（run_manager, callbacks, self, cls）
    # 5. 解析 docstring 提取参数描述（可选）
    # 6. 创建子集模型

    return schema_model
```

**推断规则**：

| Python 类型 | JSON Schema 类型 | 示例 |
|-------------|------------------|------|
| `str` | string | `"hello"` |
| `int` | integer | `42` |
| `float` | number | `3.14` |
| `bool` | boolean | `true` |
| `list[str]` | array of strings | `["a", "b"]` |
| `dict[str, Any]` | object | `{"key": "value"}` |
| `Literal["a", "b"]` | enum | `"a"` or `"b"` |

[来源: reference/source_tool_development_01.md | create_schema_from_function 分析]

---

### 类型提示要求

**必须提供类型提示**：

```python
# ✅ 正确：有类型提示
@tool
def good_tool(query: str, limit: int = 10) -> str:
    """Good tool with type hints."""
    return f"Query: {query}, Limit: {limit}"

# ❌ 错误：缺少类型提示
@tool
def bad_tool(query, limit=10):
    """Bad tool without type hints."""
    return f"Query: {query}, Limit: {limit}"
# 会导致 schema 推断失败或不准确
```

**复杂类型支持**：

```python
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

@tool
def advanced_tool(
    items: List[str],
    metadata: Dict[str, Any],
    mode: Literal["fast", "accurate"] = "fast",
    optional_param: Optional[int] = None
) -> str:
    """Tool with complex types."""
    return f"Processed {len(items)} items in {mode} mode"
```

---

## Google-style Docstring 解析

### 启用 parse_docstring=True

当启用 `parse_docstring=True` 时，装饰器会解析 Google-style docstring 提取参数描述：

```python
@tool(parse_docstring=True)
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location.

    This tool fetches real-time weather data from an external API.

    Args:
        location: City name or coordinates (e.g., "Beijing" or "39.9,116.4")
        units: Temperature unit, either "celsius" or "fahrenheit"

    Returns:
        Weather description including temperature and conditions

    Raises:
        ValueError: If location is invalid
    """
    # Implementation
    return f"Weather in {location}: 22°{units[0].upper()}"

# 生成的 schema 包含参数描述
print(get_weather.args_schema.model_json_schema())
# {
#   "properties": {
#     "location": {
#       "type": "string",
#       "description": "City name or coordinates (e.g., \"Beijing\" or \"39.9,116.4\")"
#     },
#     "units": {
#       "type": "string",
#       "description": "Temperature unit, either \"celsius\" or \"fahrenheit\"",
#       "default": "celsius"
#     }
#   }
# }
```

**Docstring 格式要求**：

1. **必须有空行**分隔 summary 和 Args 部分
2. **Args 格式**：`arg_name: description`
3. **支持多行描述**（缩进对齐）
4. **验证参数名**与函数签名一致

[来源: reference/source_tool_development_01.md | _parse_google_docstring 分析]

---

### 错误处理：error_on_invalid_docstring

```python
@tool(
    parse_docstring=True,
    error_on_invalid_docstring=True  # 严格模式
)
def strict_tool(param: str) -> str:
    """Tool with strict docstring validation.

    Args:
        param: Parameter description
        invalid_param: This will cause an error!  # ❌ 不在函数签名中
    """
    return param
# 抛出 ValueError: Docstring args do not match function signature
```

**最佳实践**：
- 开发阶段：`error_on_invalid_docstring=True`（及早发现错误）
- 生产环境：`error_on_invalid_docstring=False`（容错性）

---

## 显式 Schema 定义

### 使用 args_schema 参数

当自动推断不够精确时，可以显式提供 Pydantic schema：

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(
        description="City name or coordinates",
        examples=["Beijing", "New York", "39.9,116.4"]
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp}°{units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny, Cloudy, Rainy, Sunny, Cloudy"
    return result
```

**显式 schema 的优势**：
- 更详细的字段描述
- 使用 `Field()` 添加约束（min, max, regex）
- 提供示例值（examples）
- 使用 `Literal` 限制枚举值
- 自定义验证器（@validator）

[来源: reference/context7_langchain_tools_01.md | WeatherInput 示例]

---

### Pydantic Field 高级用法

```python
from pydantic import BaseModel, Field, validator

class SearchInput(BaseModel):
    query: str = Field(
        description="Search query",
        min_length=1,
        max_length=200,
        examples=["LangChain tutorial", "Python async"]
    )
    max_results: int = Field(
        default=10,
        ge=1,  # greater than or equal
        le=100,  # less than or equal
        description="Maximum number of results to return"
    )
    language: Literal["en", "zh", "es"] = Field(
        default="en",
        description="Result language"
    )

    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 10, language: str = "en") -> str:
    """Search with advanced validation."""
    return f"Searching '{query}' (max {max_results} results, lang={language})"
```

---

## 与 StructuredTool 的关系

### 底层委托机制

`@tool` 装饰器实际上是 `StructuredTool.from_function()` 的语法糖：

```python
# 源码简化版（来自 convert.py）
def tool(*args, **kwargs):
    """Tool decorator."""
    def _make_tool(func: Callable) -> BaseTool:
        # 最终委托给 StructuredTool.from_function()
        return StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema,
            # ... 其他参数
        )
    return _make_tool
```

**等价写法**：

```python
# 使用 @tool 装饰器
@tool("my_tool", description="My tool")
def my_func(x: str) -> str:
    return x

# 等价于 StructuredTool.from_function()
from langchain.tools import StructuredTool

my_tool = StructuredTool.from_function(
    func=my_func,
    name="my_tool",
    description="My tool"
)
```

**选择建议**：
- 优先使用 `@tool`：代码更简洁，可读性更高
- 使用 `StructuredTool.from_function()`：需要更多控制（如 coroutine 参数）

[来源: reference/source_tool_development_01.md | 架构设计分析]

---

## 2025-2026 最佳实践

### 1. 优先使用 @tool 装饰器

**社区共识**（来自 GitHub #1777）：
- ✅ 简单工具：使用 `@tool`
- ✅ 中等复杂度：使用 `@tool(args_schema=...)`
- ⚠️ 复杂状态管理：考虑 `BaseTool` 继承

```python
# ✅ 推荐：大多数场景
@tool
def simple_tool(x: str) -> str:
    """Simple tool."""
    return x

# ✅ 推荐：需要显式 schema
@tool(args_schema=MySchema)
def medium_tool(x: str) -> str:
    """Medium complexity tool."""
    return x

# ⚠️ 仅在必要时使用
class ComplexTool(BaseTool):
    """Complex tool with state."""
    # ...
```

[来源: reference/search_tool_development_01.md | GitHub #1777 讨论]

---

### 2. 启用 parse_docstring 避免参数描述丢失

**问题**（来自 GitHub #31070）：
- `create_react_agent` 中工具参数描述可能丢失
- 影响 LLM 理解工具用法

**解决方案**：

```python
# ❌ 可能丢失参数描述
@tool
def search(query: str, limit: int) -> str:
    """Search tool."""
    return f"Results for {query}"

# ✅ 确保参数描述传递给 LLM
@tool(parse_docstring=True)
def search(query: str, limit: int) -> str:
    """Search tool.

    Args:
        query: Search query string
        limit: Maximum number of results
    """
    return f"Results for {query}"
```

[来源: reference/search_tool_development_01.md | GitHub #31070 解决方案]

---

### 3. 本地模型工具调用优化

**Ollama + LangGraph 最佳实践**（来自 Reddit）：

```python
@tool(parse_docstring=True)
def local_model_tool(query: str) -> str:
    """Search local knowledge base.

    Use this tool when the user asks about internal documents or company data.

    Args:
        query: Search query (keep it simple and specific)

    Returns:
        Relevant information from the knowledge base
    """
    # 本地模型对工具调用格式更敏感
    # 描述要清晰简洁
    return search_local_kb(query)
```

**关键点**：
- 描述清晰简洁（本地模型理解能力有限）
- 明确说明何时使用该工具
- 参数描述简单直接
- 测试工具在本地模型上的表现

[来源: reference/search_tool_development_01.md | Reddit 本地模型讨论]

---

### 4. 版本兼容性注意事项

**LangGraph 1.0.2+ 破坏性变更**（来自 GitHub #6363）：

```python
# 固定依赖版本避免兼容问题
# pyproject.toml 或 requirements.txt
langchain-core==0.3.15
langgraph==0.2.45  # 使用 == 而非 >=
```

**最佳实践**：
- 固定主要依赖版本
- 测试工具在新版本中的兼容性
- 关注官方 changelog 和 GitHub issues
- 使用 CI/CD 自动化测试

[来源: reference/search_tool_development_01.md | GitHub #6363 讨论]

---

## 完整示例：生产级工具

### 场景：异步天气 API 工具

```python
import os
import httpx
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolException

class WeatherInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(
        description="City name (e.g., 'Beijing', 'New York')",
        min_length=1,
        max_length=100,
        examples=["Beijing", "New York", "London"]
    )
    units: Literal["metric", "imperial"] = Field(
        default="metric",
        description="Temperature unit: 'metric' (Celsius) or 'imperial' (Fahrenheit)"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(
    "get_weather",
    description="Get current weather and optional forecast for any city worldwide. Use this when users ask about weather conditions.",
    args_schema=WeatherInput,
    parse_docstring=True,
    response_format="content"
)
async def get_weather(
    location: str,
    units: str = "metric",
    include_forecast: bool = False
) -> str:
    """Get current weather for a location.

    This tool fetches real-time weather data from OpenWeatherMap API.

    Args:
        location: City name
        units: Temperature unit
        include_forecast: Whether to include forecast

    Returns:
        Weather description with temperature and conditions

    Raises:
        ToolException: If API call fails or location is invalid
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ToolException("OPENWEATHER_API_KEY not set")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 获取当前天气
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "units": units,
                    "appid": api_key
                }
            )
            response.raise_for_status()
            data = response.json()

            temp_unit = "°C" if units == "metric" else "°F"
            result = (
                f"Weather in {data['name']}, {data['sys']['country']}:\n"
                f"Temperature: {data['main']['temp']}{temp_unit}\n"
                f"Feels like: {data['main']['feels_like']}{temp_unit}\n"
                f"Conditions: {data['weather'][0]['description']}\n"
                f"Humidity: {data['main']['humidity']}%"
            )

            # 可选：获取预报
            if include_forecast:
                forecast_response = await client.get(
                    "https://api.openweathermap.org/data/2.5/forecast",
                    params={
                        "q": location,
                        "units": units,
                        "appid": api_key,
                        "cnt": 5
                    }
                )
                forecast_data = forecast_response.json()
                result += "\n\nNext 5 forecasts:\n"
                for item in forecast_data['list']:
                    result += f"- {item['dt_txt']}: {item['main']['temp']}{temp_unit}, {item['weather'][0]['description']}\n"

            return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise ToolException(f"Location '{location}' not found")
        raise ToolException(f"Weather API error: {e}")
    except httpx.TimeoutException:
        raise ToolException("Weather API timeout")
    except Exception as e:
        raise ToolException(f"Unexpected error: {e}")

# 测试工具
if __name__ == "__main__":
    import asyncio

    # 测试基本调用
    result = asyncio.run(get_weather.ainvoke({
        "location": "Beijing",
        "units": "metric"
    }))
    print(result)

    # 测试 schema
    print("\nTool Schema:")
    print(get_weather.args_schema.model_json_schema())

    # 测试错误处理
    try:
        result = asyncio.run(get_weather.ainvoke({
            "location": "InvalidCityName12345"
        }))
    except ToolException as e:
        print(f"\nExpected error: {e}")
```

**代码亮点**：
1. ✅ 使用 Pydantic schema 提供详细描述
2. ✅ 异步实现（性能更好）
3. ✅ 完善的错误处理（ToolException）
4. ✅ 环境变量管理（API key）
5. ✅ 超时控制（timeout=10.0）
6. ✅ 可选参数（include_forecast）
7. ✅ 完整的测试代码

---

## 常见问题与解决方案

### Q1: 参数描述没有传递给 LLM？

**问题**：LLM 不知道如何使用工具参数

**解决方案**：
```python
# ❌ 问题代码
@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return param

# ✅ 解决方案 1：启用 parse_docstring
@tool(parse_docstring=True)
def my_tool(param: str) -> str:
    """Tool description.

    Args:
        param: Parameter description here
    """
    return param

# ✅ 解决方案 2：显式 schema
class MyInput(BaseModel):
    param: str = Field(description="Parameter description here")

@tool(args_schema=MyInput)
def my_tool(param: str) -> str:
    """Tool description."""
    return param
```

---

### Q2: 如何支持可选参数？

**解决方案**：
```python
from typing import Optional

@tool
def search(query: str, limit: Optional[int] = None) -> str:
    """Search with optional limit.

    Args:
        query: Search query
        limit: Optional maximum results (default: no limit)
    """
    if limit is None:
        limit = 100  # 默认值
    return f"Searching '{query}' (limit={limit})"
```

---

### Q3: 如何处理列表和字典参数？

**解决方案**：
```python
from typing import List, Dict, Any

@tool
def batch_process(
    items: List[str],
    config: Dict[str, Any]
) -> str:
    """Process multiple items with configuration.

    Args:
        items: List of items to process
        config: Configuration dictionary
    """
    return f"Processed {len(items)} items with config {config}"

# 调用示例
result = batch_process.invoke({
    "items": ["item1", "item2", "item3"],
    "config": {"mode": "fast", "parallel": True}
})
```

---

### Q4: 如何在 LangGraph 中使用？

**解决方案**：
```python
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# 创建工具
@tool
def my_tool(x: str) -> str:
    """My tool."""
    return x

# 绑定到 LLM
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([my_tool])

# 在 LangGraph 中使用
tool_node = ToolNode([my_tool])

# 在图中添加节点
graph.add_node("tools", tool_node)
```

---

## 性能优化建议

### 1. 使用异步工具

```python
# ❌ 同步版本（阻塞）
@tool
def slow_api_call(query: str) -> str:
    """Slow API call."""
    response = requests.get(f"https://api.example.com?q={query}")
    return response.text

# ✅ 异步版本（非阻塞）
@tool
async def fast_api_call(query: str) -> str:
    """Fast API call."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com?q={query}")
        return response.text
```

---

### 2. 缓存工具结果

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_computation(x: str) -> str:
    """Expensive computation."""
    # ... 耗时操作
    return result

@tool
def cached_tool(x: str) -> str:
    """Tool with caching."""
    return _cached_computation(x)
```

---

### 3. 超时控制

```python
import asyncio

@tool
async def timeout_tool(query: str) -> str:
    """Tool with timeout."""
    try:
        async with asyncio.timeout(5.0):  # 5 秒超时
            result = await slow_operation(query)
            return result
    except asyncio.TimeoutError:
        raise ToolException("Operation timed out")
```

---

## 总结

### 核心要点

1. **@tool 是首选方式**：简洁、强大、社区推荐
2. **类型提示必不可少**：schema 推断依赖类型提示
3. **启用 parse_docstring**：确保参数描述传递给 LLM
4. **显式 schema 更精确**：使用 Pydantic 提供详细描述
5. **异步优先**：性能更好，特别是 I/O 密集型工具
6. **错误处理**：使用 ToolException 不中断 Agent 循环

### 选择决策树

```
需要创建工具？
├─ 简单函数（无状态）
│  └─ 使用 @tool 装饰器 ✅
├─ 需要详细 schema
│  └─ 使用 @tool(args_schema=...) ✅
├─ 包装现有 API
│  └─ 使用 @tool 或 StructuredTool.from_function() ✅
└─ 复杂状态管理
   └─ 考虑 BaseTool 继承 ⚠️
```

### 下一步学习

- **核心概念 2**：StructuredTool 创建方法
- **核心概念 3**：BaseTool 类继承
- **实战代码**：异步天气 API 工具完整实现

---

**参考资料**：
- [来源: reference/source_tool_development_01.md | LangChain Tool 核心机制源码分析]
- [来源: reference/context7_langchain_tools_01.md | LangChain 官方文档]
- [来源: reference/search_tool_development_01.md | 2025-2026 社区最佳实践]
