# 核心概念 4：Schema 定义与验证

> 使用 Pydantic 定义工具输入 schema，实现类型安全和自动验证

---

## 概述

Schema 定义是 LangChain 工具系统的核心机制，它决定了 LLM 如何理解和调用你的工具。通过 Pydantic BaseModel，你可以定义严格的类型约束、参数描述、默认值和验证规则，确保工具调用的安全性和可靠性。

**核心价值**：
- 类型安全：编译时捕获类型错误
- 自动验证：运行时验证输入参数
- 自动文档：生成 JSON Schema 供 LLM 理解
- 开发体验：IDE 自动补全和类型提示

---

## Pydantic BaseModel 定义

### 基础 Schema 定义

```python
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """计算器工具的输入 schema."""
    expression: str = Field(description="Mathematical expression to evaluate")

# 使用
from langchain.tools import tool

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

# 查看生成的 JSON Schema
print(calculator.args_schema.model_json_schema())
```

**输出的 JSON Schema**：
```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Mathematical expression to evaluate"
    }
  },
  "required": ["expression"]
}
```

[来源: reference/context7_langchain_tools_01.md | Pydantic Schema 示例]

### 多字段 Schema

```python
from pydantic import BaseModel, Field
from typing import List

class SearchInput(BaseModel):
    """搜索工具的输入 schema."""
    query: str = Field(description="Search query string")
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return (1-100)"
    )
    filters: List[str] = Field(
        default_factory=list,
        description="Optional filters to apply"
    )

@tool(args_schema=SearchInput)
def search(query: str, num_results: int = 10, filters: List[str] = None) -> str:
    """Search the web with optional filters."""
    filters = filters or []
    return f"Searching for '{query}', returning {num_results} results, filters: {filters}"

# 使用
result = search.invoke({
    "query": "LangChain tutorials",
    "num_results": 5,
    "filters": ["python", "beginner"]
})
```

**关键点**：
- 使用 `Field()` 提供详细描述
- 使用 `default` 设置默认值
- 使用 `ge`, `le` 设置数值范围
- 使用 `default_factory` 处理可变默认值（如列表）

---

## Field 描述和约束

### Field 参数详解

```python
from pydantic import BaseModel, Field
from typing import Optional

class AdvancedInput(BaseModel):
    """高级 schema 示例."""

    # 必需字段
    required_field: str = Field(
        description="This field is required"
    )

    # 可选字段
    optional_field: Optional[str] = Field(
        default=None,
        description="This field is optional"
    )

    # 带默认值的字段
    with_default: str = Field(
        default="default_value",
        description="Field with default value"
    )

    # 数值约束
    age: int = Field(
        ge=0,  # greater than or equal
        le=150,  # less than or equal
        description="Age must be between 0 and 150"
    )

    # 字符串长度约束
    username: str = Field(
        min_length=3,
        max_length=20,
        description="Username must be 3-20 characters"
    )

    # 正则表达式约束
    email: str = Field(
        pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$',
        description="Valid email address"
    )

    # 列表约束
    tags: List[str] = Field(
        default_factory=list,
        min_items=0,
        max_items=10,
        description="Up to 10 tags"
    )
```

**常用约束参数**：

| 参数 | 适用类型 | 说明 |
|------|----------|------|
| `description` | 所有 | 字段描述（LLM 会读取） |
| `default` | 所有 | 默认值 |
| `default_factory` | 所有 | 默认值工厂函数 |
| `ge` | 数值 | 大于等于 |
| `le` | 数值 | 小于等于 |
| `gt` | 数值 | 大于 |
| `lt` | 数值 | 小于 |
| `min_length` | 字符串/列表 | 最小长度 |
| `max_length` | 字符串/列表 | 最大长度 |
| `pattern` | 字符串 | 正则表达式 |
| `min_items` | 列表 | 最小元素数 |
| `max_items` | 列表 | 最大元素数 |

---

## Literal 类型限制

### 枚举值约束

```python
from typing import Literal
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """天气查询输入."""
    location: str = Field(description="City name or coordinates")

    # 使用 Literal 限制可选值
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )

    # 多个可选值
    format: Literal["json", "text", "xml"] = Field(
        default="json",
        description="Response format"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str,
    units: str = "celsius",
    format: str = "json"
) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}, format: {format}"

# 使用
result = get_weather.invoke({
    "location": "Beijing",
    "units": "celsius",  # ✅ 有效
    "format": "json"     # ✅ 有效
})

# 无效值会被 Pydantic 拒绝
try:
    result = get_weather.invoke({
        "location": "Beijing",
        "units": "kelvin"  # ❌ 无效，不在 Literal 中
    })
except Exception as e:
    print(f"Validation error: {e}")
```

**生成的 JSON Schema**：
```json
{
  "properties": {
    "location": {"type": "string"},
    "units": {
      "type": "string",
      "enum": ["celsius", "fahrenheit"],
      "default": "celsius"
    },
    "format": {
      "type": "string",
      "enum": ["json", "text", "xml"],
      "default": "json"
    }
  }
}
```

[来源: reference/context7_langchain_tools_01.md | WeatherInput 示例]

### 复杂枚举场景

```python
from typing import Literal
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    """数据库查询输入."""

    # 表名枚举
    table: Literal["users", "products", "orders"] = Field(
        description="Table to query"
    )

    # 操作类型枚举
    operation: Literal["select", "count", "aggregate"] = Field(
        default="select",
        description="Query operation type"
    )

    # 排序方向枚举
    sort_order: Literal["asc", "desc"] = Field(
        default="asc",
        description="Sort order"
    )

@tool(args_schema=DatabaseQueryInput)
def query_database(
    table: str,
    operation: str = "select",
    sort_order: str = "asc"
) -> str:
    """Query database with type-safe parameters."""
    return f"Querying {table} with {operation}, sorted {sort_order}"
```

---

## 默认值设置

### 简单默认值

```python
from pydantic import BaseModel, Field

class SimpleDefaults(BaseModel):
    """简单默认值示例."""

    # 字符串默认值
    name: str = Field(default="Anonymous", description="User name")

    # 数值默认值
    age: int = Field(default=0, description="User age")

    # 布尔默认值
    active: bool = Field(default=True, description="Is active")

    # None 默认值
    email: str | None = Field(default=None, description="Optional email")
```

### 可变默认值（default_factory）

```python
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime

class ComplexDefaults(BaseModel):
    """复杂默认值示例."""

    # ❌ 错误：不要直接使用可变对象作为默认值
    # tags: List[str] = Field(default=[], description="Tags")

    # ✅ 正确：使用 default_factory
    tags: List[str] = Field(
        default_factory=list,
        description="Tags"
    )

    # 字典默认值
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata"
    )

    # 自定义工厂函数
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp"
    )

    # 复杂对象默认值
    config: Dict[str, int] = Field(
        default_factory=lambda: {"timeout": 30, "retries": 3},
        description="Configuration"
    )
```

**为什么需要 default_factory**：
```python
# ❌ 危险：所有实例共享同一个列表
class BadSchema(BaseModel):
    items: List[str] = Field(default=[])

# 问题演示
schema1 = BadSchema()
schema1.items.append("item1")

schema2 = BadSchema()
print(schema2.items)  # ['item1'] - 意外共享了数据！

# ✅ 安全：每个实例有独立的列表
class GoodSchema(BaseModel):
    items: List[str] = Field(default_factory=list)

schema1 = GoodSchema()
schema1.items.append("item1")

schema2 = GoodSchema()
print(schema2.items)  # [] - 正确的空列表
```

---

## JSON Schema 格式（字典）

### 使用字典定义 Schema

除了 Pydantic BaseModel，LangChain 也支持直接使用 JSON Schema 字典。

```python
from langchain.tools import tool

# 字典格式的 schema
weather_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name or coordinates"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius",
            "description": "Temperature unit"
        },
        "include_forecast": {
            "type": "boolean",
            "default": False,
            "description": "Include 5-day forecast"
        }
    },
    "required": ["location"]
}

@tool(args_schema=weather_schema)
def get_weather(
    location: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    """Get weather using JSON Schema."""
    return f"Weather in {location}: 22°{units[0].upper()}"
```

[来源: reference/context7_langchain_tools_01.md | JSON Schema 示例]

### Pydantic vs JSON Schema 对比

| 特性 | Pydantic BaseModel | JSON Schema 字典 |
|------|-------------------|------------------|
| 类型安全 | ✅ 编译时检查 | ❌ 运行时检查 |
| IDE 支持 | ✅ 自动补全 | ❌ 无支持 |
| 验证器 | ✅ 支持自定义 | ❌ 不支持 |
| 可读性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 推荐场景 | 大多数情况 | 动态生成 schema |

**推荐**：优先使用 Pydantic BaseModel，除非需要动态生成 schema。

---

## 自动推断 vs 显式定义

### 自动推断（默认行为）

```python
from langchain.tools import tool

@tool
def simple_tool(query: str, limit: int = 10) -> str:
    """Simple tool with automatic schema inference.

    Args:
        query: Search query string
        limit: Maximum number of results
    """
    return f"Query: {query}, Limit: {limit}"

# 自动推断的 schema
print(simple_tool.args_schema.model_json_schema())
```

**自动推断的 JSON Schema**：
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "limit": {"type": "integer", "default": 10}
  },
  "required": ["query"]
}
```

**自动推断的限制**：
- 无法添加详细的字段描述
- 无法设置数值范围约束
- 无法使用 Literal 限制枚举值
- 依赖 docstring 解析（需要 `parse_docstring=True`）

### 显式定义（推荐）

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchInput(BaseModel):
    """显式定义的 schema."""
    query: str = Field(description="Search query string")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results (1-100)"
    )
    sort_by: Literal["relevance", "date", "popularity"] = Field(
        default="relevance",
        description="Sort order"
    )

@tool(args_schema=SearchInput)
def advanced_search(
    query: str,
    limit: int = 10,
    sort_by: str = "relevance"
) -> str:
    """Advanced search with explicit schema."""
    return f"Query: {query}, Limit: {limit}, Sort: {sort_by}"
```

**显式定义的优势**：
- 完全控制 schema 结构
- 详细的字段描述
- 严格的类型约束
- 更好的 LLM 理解

---

## create_schema_from_function 机制

### 自动 Schema 推断原理

LangChain 使用 `create_schema_from_function` 从函数签名自动生成 Pydantic schema。

```python
# 源码位置: langchain_core/tools/base.py:289-387
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
```

[来源: sourcecode/langchain/libs/core/langchain_core/tools/base.py]

### 工作流程

1. **提取函数签名**：使用 `inspect.signature()` 获取参数信息
2. **过滤特殊参数**：移除 `run_manager`, `callbacks`, `self`, `cls`
3. **解析 docstring**：提取参数描述（如果 `parse_docstring=True`）
4. **创建 Pydantic 模型**：使用 `validate_arguments()` 生成模型
5. **生成子集模型**：只包含非过滤参数

### 使用 parse_docstring

```python
from langchain.tools import tool

@tool(parse_docstring=True)
def documented_tool(query: str, limit: int = 10) -> str:
    """Search tool with Google-style docstring.

    Args:
        query: The search query string to execute
        limit: Maximum number of results to return (default: 10)

    Returns:
        Search results as a formatted string
    """
    return f"Query: {query}, Limit: {limit}"

# 查看推断的 schema
schema = documented_tool.args_schema.model_json_schema()
print(schema)
```

**生成的 JSON Schema**：
```json
{
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query string to execute"
    },
    "limit": {
      "type": "integer",
      "default": 10,
      "description": "Maximum number of results to return (default: 10)"
    }
  }
}
```

**Docstring 格式要求**：
- 必须使用 Google-style docstring
- Args 部分格式：`arg_name: description`
- 必须有空行分隔 summary 和 Args 部分

[来源: sourcecode/langchain/libs/core/langchain_core/tools/base.py | _parse_google_docstring]

---

## 复杂 Schema 示例

### 嵌套对象 Schema

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class Address(BaseModel):
    """地址信息."""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    postal_code: str = Field(description="Postal code")

class UserProfile(BaseModel):
    """用户资料."""
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150, description="Age")
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$', description="Email")
    addresses: List[Address] = Field(
        default_factory=list,
        description="List of addresses"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

@tool(args_schema=UserProfile)
def create_user(
    name: str,
    age: int,
    email: str,
    addresses: List[Address] = None,
    metadata: Dict[str, str] = None
) -> str:
    """Create a new user profile."""
    addresses = addresses or []
    metadata = metadata or {}
    return f"Created user: {name}, {age}, {email}"
```

### 联合类型 Schema

```python
from pydantic import BaseModel, Field
from typing import Union, Literal

class TextQuery(BaseModel):
    """文本查询."""
    type: Literal["text"] = "text"
    query: str = Field(description="Text query")

class ImageQuery(BaseModel):
    """图像查询."""
    type: Literal["image"] = "image"
    image_url: str = Field(description="Image URL")

class MultiModalInput(BaseModel):
    """多模态查询输入."""
    query: Union[TextQuery, ImageQuery] = Field(
        description="Query can be text or image"
    )
    max_results: int = Field(default=10, description="Max results")

@tool(args_schema=MultiModalInput)
def multimodal_search(query: Union[TextQuery, ImageQuery], max_results: int = 10) -> str:
    """Search using text or image."""
    if isinstance(query, TextQuery):
        return f"Text search: {query.query}"
    else:
        return f"Image search: {query.image_url}"
```

### 自定义验证器

```python
from pydantic import BaseModel, Field, validator
from typing import List

class AdvancedSearchInput(BaseModel):
    """高级搜索输入."""
    query: str = Field(description="Search query")
    tags: List[str] = Field(default_factory=list, description="Filter tags")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Min score")

    @validator('query')
    def validate_query(cls, v):
        """验证查询字符串."""
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")
        return v.strip()

    @validator('tags')
    def validate_tags(cls, v):
        """验证标签列表."""
        if len(v) > 10:
            raise ValueError("Too many tags (max 10)")
        # 去重并转小写
        return list(set(tag.lower() for tag in v))

    @validator('min_score')
    def validate_score(cls, v):
        """验证分数范围."""
        if v < 0 or v > 1:
            raise ValueError("Score must be between 0 and 1")
        return v

@tool(args_schema=AdvancedSearchInput)
def advanced_search(
    query: str,
    tags: List[str] = None,
    min_score: float = 0.0
) -> str:
    """Advanced search with validation."""
    tags = tags or []
    return f"Searching: {query}, Tags: {tags}, Min score: {min_score}"

# 使用
try:
    result = advanced_search.invoke({
        "query": "",  # ❌ 会触发验证错误
        "tags": ["python"],
        "min_score": 0.5
    })
except Exception as e:
    print(f"Validation error: {e}")
```

---

## 最佳实践

### 1. Schema 设计原则

**清晰的描述**：
```python
# ❌ 不好：描述不清晰
class BadInput(BaseModel):
    q: str = Field(description="query")

# ✅ 好：描述清晰详细
class GoodInput(BaseModel):
    query: str = Field(
        description="Search query string. Use natural language to describe what you're looking for."
    )
```

**合理的默认值**：
```python
# ✅ 提供合理的默认值
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Number of results")
    include_metadata: bool = Field(default=False, description="Include metadata")
```

**使用 Literal 限制枚举**：
```python
# ✅ 使用 Literal 而不是字符串
class ConfigInput(BaseModel):
    mode: Literal["fast", "accurate", "balanced"] = Field(
        default="balanced",
        description="Processing mode"
    )
```

### 2. 验证策略

**输入验证**：
```python
from pydantic import BaseModel, Field, validator

class SafeInput(BaseModel):
    url: str = Field(description="URL to fetch")

    @validator('url')
    def validate_url(cls, v):
        """验证 URL 格式和安全性."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        if 'localhost' in v or '127.0.0.1' in v:
            raise ValueError("Localhost URLs not allowed")
        return v
```

**业务逻辑验证**：
```python
class DateRangeInput(BaseModel):
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")

    @validator('end_date')
    def validate_date_range(cls, v, values):
        """验证日期范围."""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v
```

### 3. 性能优化

**避免过度验证**：
```python
# ❌ 不好：每次都进行昂贵的验证
class SlowInput(BaseModel):
    data: str

    @validator('data')
    def expensive_validation(cls, v):
        # 昂贵的验证逻辑
        time.sleep(1)
        return v

# ✅ 好：只在必要时验证
class FastInput(BaseModel):
    data: str = Field(max_length=1000)  # 使用内置约束
```

### 4. 错误处理

**友好的错误消息**：
```python
from pydantic import BaseModel, Field, validator

class UserInput(BaseModel):
    age: int = Field(ge=0, le=150, description="User age")

    @validator('age')
    def validate_age(cls, v):
        """提供友好的错误消息."""
        if v < 0:
            raise ValueError("Age cannot be negative. Please provide a valid age.")
        if v > 150:
            raise ValueError("Age seems unrealistic. Please check your input.")
        return v
```

---

## 常见陷阱

### 1. 可变默认值

```python
# ❌ 错误：共享可变对象
class BadSchema(BaseModel):
    items: List[str] = Field(default=[])

# ✅ 正确：使用 default_factory
class GoodSchema(BaseModel):
    items: List[str] = Field(default_factory=list)
```

### 2. 参数名称不匹配

```python
# ❌ 错误：schema 和函数参数名不一致
class BadInput(BaseModel):
    search_query: str

@tool(args_schema=BadInput)
def bad_tool(query: str) -> str:  # 参数名不匹配
    return query

# ✅ 正确：参数名一致
class GoodInput(BaseModel):
    query: str

@tool(args_schema=GoodInput)
def good_tool(query: str) -> str:
    return query
```

### 3. 忘记 Field 描述

```python
# ❌ 不好：没有描述，LLM 难以理解
class BadInput(BaseModel):
    q: str
    n: int

# ✅ 好：清晰的描述
class GoodInput(BaseModel):
    query: str = Field(description="Search query string")
    num_results: int = Field(description="Number of results to return")
```

### 4. 过度复杂的 Schema

```python
# ❌ 不好：过于复杂，LLM 难以理解
class OverlyComplexInput(BaseModel):
    config: Dict[str, Dict[str, List[Dict[str, Any]]]]

# ✅ 好：简化结构
class SimpleInput(BaseModel):
    mode: Literal["fast", "accurate"] = Field(description="Processing mode")
    options: Dict[str, str] = Field(default_factory=dict, description="Additional options")
```

---

## 总结

Schema 定义是工具开发的基础，直接影响 LLM 对工具的理解和调用准确性。通过 Pydantic BaseModel，你可以实现类型安全、自动验证和清晰的文档。

**关键要点**：
1. 优先使用 Pydantic BaseModel 而非 JSON Schema 字典
2. 使用 `Field()` 提供详细的字段描述
3. 使用 `Literal` 限制枚举值
4. 使用 `default_factory` 处理可变默认值
5. 使用自定义验证器实现复杂验证逻辑
6. 保持 schema 简单清晰，避免过度复杂

**最佳实践**：
- ✅ 清晰详细的字段描述
- ✅ 合理的默认值
- ✅ 严格的类型约束
- ✅ 友好的错误消息
- ❌ 避免可变默认值
- ❌ 避免过度复杂的嵌套结构
- ❌ 避免昂贵的验证逻辑
