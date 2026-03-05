# 核心概念 4：Tool Schema 与参数验证

> Schema 是工具的"说明书"——它告诉 LLM 需要什么参数、什么类型、什么含义，LLM 据此生成正确的 JSON 参数，Pydantic 再验证参数合法性。没有 Schema，LLM 就是在盲猜。

---

## 概述

**Tool Schema** 解决的核心问题是：LLM 怎么知道一个工具需要什么参数？

1. **工具需要哪些参数？** → Schema 定义参数名称和类型
2. **每个参数是什么含义？** → Schema 中的 `description` 字段
3. **参数合法性怎么保证？** → Pydantic 自动验证

```
Schema 的完整生命周期：

定义 Schema → 转换为 JSON Schema → 发送给 LLM → LLM 生成参数 → Pydantic 验证 → 执行工具
    ↑              ↑                    ↑              ↑              ↑
 三种方式      convert_to_openai_tool  bind_tools    tool_calls   args_schema.validate
```

Schema 的质量直接决定 LLM 调用工具的准确率。一个描述模糊的 Schema，LLM 可能传错参数；一个描述精确的 Schema，LLM 几乎不会出错。这就像 API 文档写得好不好，直接决定前端开发者能不能正确调用接口。

---

## 1. 三种 Schema 定义方式

| 维度 | 自动推断（默认） | Pydantic BaseModel | parse_docstring |
|------|-----------------|-------------------|-----------------|
| **配置量** | 零配置 | 中等 | 少量 |
| **参数描述** | 无（仅类型） | 精确控制每个字段 | 从 docstring 提取 |
| **验证能力** | 基础类型检查 | 完整 Pydantic 验证 | 基础类型检查 |
| **适用场景** | 简单工具、快速原型 | 生产环境、复杂参数 | 中等复杂度 |
| **源码入口** | `create_schema_from_function()` | `args_schema` 参数 | `_parse_google_docstring()` |

---

### 方式 A：自动推断（从函数签名）

`@tool` 装饰器默认 `infer_schema=True`，从 Python 类型注解自动推断 JSON Schema。

[来源: `langchain_core/tools/convert.py`]

```python
from langchain_core.tools import tool

@tool
def search(query: str, max_results: int = 10) -> str:
    """在知识库中搜索文档"""
    return f"搜索 '{query}'，返回 {max_results} 条结果"

print(search.args_schema.model_json_schema())
# → {"properties": {"query": {"type": "string"}, "max_results": {"default": 10, "type": "integer"}},
#    "required": ["query"], "type": "object"}
```

#### 类型映射规则

| Python 类型 | JSON Schema 类型 | 示例 |
|------------|-----------------|------|
| `str` | `{"type": "string"}` | `query: str` |
| `int` | `{"type": "integer"}` | `count: int` |
| `float` | `{"type": "number"}` | `score: float` |
| `bool` | `{"type": "boolean"}` | `verbose: bool` |
| `List[str]` | `{"type": "array", "items": {"type": "string"}}` | `tags: List[str]` |
| `Optional[str]` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` | `title: Optional[str]` |

#### 核心函数：create_schema_from_function()

[来源: `langchain_core/tools/base.py`]

```python
# 源码简化版逻辑
def create_schema_from_function(model_name, func, *, filter_args=None):
    sig = inspect.signature(func)           # 1. 获取函数签名
    type_hints = get_type_hints(func)       # 2. 获取类型注解
    fields = {}
    for name, param in sig.parameters.items():
        if name in (filter_args or []):
            continue                        # 3. 跳过被过滤的参数
        annotation = type_hints.get(name, Any)
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[name] = (annotation, default)
    return create_model(model_name, **fields)  # 4. 动态创建 Pydantic Model
```

**局限**：自动推断只能获取参数**类型**，无法获取**含义描述**。LLM 只能从参数名猜测含义。

---

### 方式 B：Pydantic BaseModel（生产推荐）

通过 `args_schema` 参数传入 Pydantic BaseModel，精确控制每个参数的类型、描述、默认值和验证规则。

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    """知识库搜索参数"""
    query: str = Field(description="搜索关键词，支持自然语言")
    max_results: int = Field(default=10, ge=1, le=100, description="最大返回结果数")
    category: str = Field(default="all", description="搜索分类：all/tech/business/legal")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 10, category: str = "all") -> str:
    """在知识库中搜索文档"""
    return f"在 {category} 分类中搜索 '{query}'，返回 {max_results} 条结果"
```

输出的 JSON Schema 会包含 `description`、`minimum`、`maximum` 等约束信息，LLM 能精确理解每个参数。

#### 为什么是生产首选？

1. **参数描述精确**：`Field(description=...)` 让 LLM 准确理解每个参数
2. **内置验证**：`ge=1, le=100` 自动拒绝非法值
3. **类型安全**：Pydantic 自动做类型转换和校验
4. **可复用**：同一个 Schema 可以被多个 Tool 共享

#### Field 常用验证参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `description` | 参数描述（LLM 可见） | `Field(description="搜索关键词")` |
| `default` | 默认值 | `Field(default=10)` |
| `ge` / `le` | 数值范围 | `Field(ge=1, le=100)` |
| `min_length` / `max_length` | 字符串长度 | `Field(min_length=1)` |
| `pattern` | 正则匹配 | `Field(pattern=r"^\d{4}-\d{2}-\d{2}$")` |

---

### 方式 C：parse_docstring（从 Google-style docstring 提取）

设置 `parse_docstring=True`，LangChain 解析 Google-style docstring 的 `Args:` 部分提取参数描述。

[来源: `langchain_core/utils/function_calling.py` — `_parse_google_docstring()`]

```python
@tool(parse_docstring=True)
def search(query: str, max_results: int = 10) -> str:
    """在知识库中搜索文档。

    根据用户的自然语言查询，在向量知识库中进行语义搜索。

    Args:
        query: 搜索关键词，支持自然语言查询
        max_results: 最大返回结果数，默认10条
    """
    return f"搜索 '{query}'，返回 {max_results} 条结果"
```

解析规则：第一行作为简短描述，`Args:` 之前的段落作为详细描述，`Args:` 中每行解析为 `参数名: 描述`。

**注意**：必须严格遵循 Google-style 格式，`Args:` 顶格写，参数缩进 4 空格，参数名后用冒号 `:`。

#### 三种方式效果对比

```python
# 方式 A：自动推断 — 无参数描述
@tool
def search_v1(query: str, max_results: int = 10) -> str:
    """搜索知识库"""
    ...
# → query(string), max_results(integer) — LLM 只能猜

# 方式 B：Pydantic — 有描述 + 有验证
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=100, description="最大结果数")

@tool(args_schema=SearchInput)
def search_v2(query: str, max_results: int = 10) -> str:
    """搜索知识库"""
    ...
# → query("搜索关键词"), max_results(1-100, "最大结果数") — 精确

# 方式 C：parse_docstring — 有描述，无验证
@tool(parse_docstring=True)
def search_v3(query: str, max_results: int = 10) -> str:
    """搜索知识库。

    Args:
        query: 搜索关键词
        max_results: 最大结果数
    """
    ...
# → query("搜索关键词"), max_results("最大结果数") — 中等
```

---

## 2. Schema 转换链

Tool Schema 从定义到发送给 LLM API，经历一条转换链：

```
args_schema (Pydantic Model)
    │
    ▼
tool_call_schema (过滤 InjectedToolArg 后的 Schema)
    │
    ▼
convert_to_openai_tool()
    │
    ▼
OpenAI API JSON 格式 → 发送给 LLM
```

### args_schema → tool_call_schema

`tool_call_schema` 在 `args_schema` 基础上**过滤掉 InjectedToolArg 标记的参数**——这些参数由系统注入，不应暴露给 LLM。

[来源: `langchain_core/tools/base.py`]

```python
@property
def tool_call_schema(self) -> type[BaseModel]:
    full_schema = self.args_schema
    return _get_filtered_args(full_schema)  # 过滤 InjectedToolArg 字段
```

### tool_call_schema → OpenAI 格式

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

openai_format = convert_to_openai_tool(search)
# → {"type": "function", "function": {"name": "search", "description": "...",
#     "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
```

其他提供商（Anthropic、Bedrock）有各自格式，`bind_tools()` 内部自动处理转换。

---

## 3. 参数验证机制

### Pydantic 自动验证

LLM 返回 `tool_calls` 后，LangChain 用 `args_schema` 验证参数再执行工具：

```python
class TemperatureInput(BaseModel):
    city: str = Field(min_length=1, description="城市名称")
    unit: str = Field(default="celsius", pattern=r"^(celsius|fahrenheit)$",
                      description="温度单位")

@tool(args_schema=TemperatureInput)
def get_temperature(city: str, unit: str = "celsius") -> str:
    """查询城市温度"""
    return f"{city}: 25°{'C' if unit == 'celsius' else 'F'}"

# 合法调用 → 通过
get_temperature.invoke({"city": "北京", "unit": "celsius"})  # "北京: 25°C"

# 非法调用 → ValidationError
get_temperature.invoke({"city": "", "unit": "kelvin"})  # 抛异常
```

### handle_validation_error

控制验证失败时的行为：

| 值 | 行为 |
|---|------|
| `False`（默认） | 抛出 `ValidationError` 异常 |
| `True` | 返回错误信息字符串，不抛异常 |
| `str` | 返回指定的字符串 |
| `Callable` | 调用函数处理错误 |

**Agent 最佳实践**：设置 `handle_validation_error=True`，让 Agent 看到错误后自动修正参数重试，而不是崩溃。

---

## 4. InjectedToolArg — 系统注入参数

有些参数不应由 LLM 决定（如用户 ID、数据库连接），而是由系统自动注入。

[来源: `langchain_core/tools/base.py`]

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def search_user_docs(
    query: str,
    user_id: Annotated[str, InjectedToolArg],  # 不暴露给 LLM
) -> str:
    """搜索用户的私有文档"""
    return f"用户 {user_id} 搜索: {query}"

# Schema 中 user_id 被过滤掉，LLM 看不到
print(search_user_docs.get_input_schema().model_json_schema())
# → {"properties": {"query": {"type": "string"}}, "required": ["query"]}

# 调用时由系统注入
result = search_user_docs.invoke({"query": "RAG 文档", "user_id": "user_12345"})
```

### FILTERED_ARGS — 自动过滤的内部参数

| 参数 | 说明 |
|------|------|
| `run_manager` | 回调管理器 |
| `callbacks` | 回调函数列表 |

这些参数即使出现在函数签名中，也不会出现在 Schema 里。

---

## 5. 高级 Schema 特性

### 枚举类型 — 限制可选值

```python
from enum import Enum
from pydantic import BaseModel, Field

class SearchCategory(str, Enum):
    TECH = "tech"
    BUSINESS = "business"
    LEGAL = "legal"

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    category: SearchCategory = Field(default=SearchCategory.TECH, description="搜索分类")

# Schema 生成 enum 约束 → "category": {"enum": ["tech", "business", "legal"]}
```

### 嵌套对象

```python
from typing import List, Optional

class DateRange(BaseModel):
    start: str = Field(description="开始日期 YYYY-MM-DD")
    end: str = Field(description="结束日期 YYYY-MM-DD")

class AdvancedSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    date_range: Optional[DateRange] = Field(default=None, description="日期范围")
    tags: List[str] = Field(default_factory=list, description="标签过滤")
```

LLM 能理解嵌套结构并生成正确的嵌套 JSON 参数。

---

## 6. 在 RAG 中的应用

### 生产级检索工具 Schema 设计

```python
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class RAGSearchInput(BaseModel):
    """RAG 知识库检索参数"""
    query: str = Field(description="自然语言搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回文档数量")
    search_mode: SearchMode = Field(default=SearchMode.HYBRID, description="检索模式")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")

@tool(args_schema=RAGSearchInput)
def rag_search(query: str, top_k: int = 5, search_mode: str = "hybrid",
               score_threshold: float = 0.7) -> str:
    """在 RAG 知识库中检索相关文档片段"""
    return f"[{search_mode}] 搜索 '{query}' | top_k={top_k}, threshold={score_threshold}"
```

**设计要点**：`top_k` 限制 1-20 防止不合理值，`search_mode` 用枚举限制可选值，`score_threshold` 限制 0-1 范围。

### Schema 质量对调用准确率的影响

| Schema 质量 | LLM 行为 | 准确率 |
|------------|----------|--------|
| 无描述（自动推断） | 从参数名猜测 | ~70% |
| 有描述（parse_docstring） | 理解参数含义 | ~85% |
| 精确描述 + 约束（Pydantic） | 精确理解 + 受限选择 | ~95% |
| 精确描述 + 枚举 + 示例 | 几乎不会出错 | ~98% |

---

## 关键源码映射

| 概念 | 源码位置 | 说明 |
|------|----------|------|
| `create_schema_from_function()` | `langchain_core/tools/base.py` | 从函数签名推断 Schema |
| `args_schema` | `langchain_core/tools/base.py` | Tool 的参数 Schema |
| `tool_call_schema` | `langchain_core/tools/base.py` | 过滤注入参数后的 Schema |
| `_parse_google_docstring()` | `langchain_core/utils/function_calling.py` | 解析 docstring |
| `convert_to_openai_tool()` | `langchain_core/utils/function_calling.py` | 转换为 API 格式 |
| `InjectedToolArg` | `langchain_core/tools/base.py` | 标记系统注入参数 |
| `handle_validation_error` | `langchain_core/tools/base.py` | 控制验证失败行为 |

---

## 总结

Tool Schema 是 LLM 调用工具的"说明书"，质量直接决定调用准确率：

1. **三种定义方式**：自动推断（原型）→ parse_docstring（文档驱动）→ Pydantic BaseModel（生产首选）
2. **转换链**：`args_schema` → `tool_call_schema`（过滤注入参数）→ `convert_to_openai_tool()`（API 格式）
3. **参数验证**：Pydantic 自动验证，`handle_validation_error` 控制失败行为
4. **InjectedToolArg**：系统注入参数对 LLM 不可见，实现权限控制
5. **生产建议**：始终用 Pydantic BaseModel + 精确 Field description，这是提升调用准确率最有效的手段

---

**上一篇**: [03_核心概念_3_ToolCall与ToolMessage消息流.md](./03_核心概念_3_ToolCall与ToolMessage消息流.md) — 请求→执行→响应的完整消息链路
**下一篇**: [03_核心概念_5_工具选择策略.md](./03_核心概念_5_工具选择策略.md) — 动态工具过滤与 Middleware 模式
