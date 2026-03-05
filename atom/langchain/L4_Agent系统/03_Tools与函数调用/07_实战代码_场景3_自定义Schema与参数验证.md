# 实战代码 场景3：自定义 Schema 与参数验证

> 演示如何用 Pydantic 模型定义复杂的工具参数 Schema（枚举、嵌套对象、范围约束），以及参数验证机制（ValidationError、handle_validation_error）和 InjectedToolArg 系统注入参数

---

## 场景概述

场景1 和场景2 中的 Tool 参数都比较简单——几个 `str` 和 `int` 就够了。但生产环境中，工具参数往往更复杂：枚举限制可选值、嵌套对象表达结构化条件、范围约束防止非法输入。

本场景聚焦 **Schema 设计** 和 **参数验证**，所有代码纯本地运行，不调用 LLM。

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain-core pydantic

# 本场景不需要 API Key，不调用 LLM
# 所有代码纯本地运行
```

---

## 示例1：基础 Pydantic Schema — 给每个参数加描述和约束

**解决的问题：** 自动推断的 Schema 没有参数描述，LLM 只能从参数名猜含义。用 Pydantic BaseModel + `Field()` 精确控制每个参数的描述、默认值和验证规则。

```python
"""
基础 Pydantic Schema：通过 args_schema 参数传入 BaseModel
Field() 的 description 会写入 JSON Schema，LLM 据此理解参数含义
Field() 的 ge/le 会在调用时自动验证参数合法性
"""
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SearchParams(BaseModel):
    """文档搜索参数"""
    query: str = Field(description="搜索关键词，支持自然语言")
    top_k: int = Field(default=5, ge=1, le=100, description="返回结果数量")
    category: str = Field(default="all", description="搜索分类: all/tech/business/legal")


@tool(args_schema=SearchParams)
def search_docs(query: str, top_k: int = 5, category: str = "all") -> str:
    """在知识库中搜索文档"""
    return f"在 [{category}] 分类中搜索 '{query}'，返回 {top_k} 条结果"


# ===== 直接调用验证 =====
result = search_docs.invoke({"query": "RAG 架构", "top_k": 10, "category": "tech"})
print(f"调用结果: {result}")

# 使用默认值
result2 = search_docs.invoke({"query": "向量数据库"})
print(f"默认参数: {result2}")

# ===== 查看 Tool 属性 =====
print(f"\n名称: {search_docs.name}")
print(f"描述: {search_docs.description}")

# ===== 查看完整 JSON Schema =====
schema = search_docs.args_schema.model_json_schema()
print(f"\n完整 JSON Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
调用结果: 在 [tech] 分类中搜索 'RAG 架构'，返回 10 条结果
默认参数: 在 [all] 分类中搜索 '向量数据库'，返回 5 条结果

名称: search_docs
描述: 在知识库中搜索文档

完整 JSON Schema:
{
  "description": "文档搜索参数",
  "properties": {
    "query": {
      "description": "搜索关键词，支持自然语言",
      "title": "Query",
      "type": "string"
    },
    "top_k": {
      "default": 5,
      "description": "返回结果数量",
      "maximum": 100,
      "minimum": 1,
      "title": "Top K",
      "type": "integer"
    },
    "category": {
      "default": "all",
      "description": "搜索分类: all/tech/business/legal",
      "title": "Category",
      "type": "string"
    }
  },
  "required": [
    "query"
  ],
  "title": "SearchParams",
  "type": "object"
}
```

> **关键点：** `ge=1, le=100` 在 JSON Schema 中变成了 `minimum` 和 `maximum`。LLM 能看到这些约束，Pydantic 在调用时也会强制验证。

---

## 示例2：parse_docstring 自动提取参数描述

**解决的问题：** 不想写 Pydantic BaseModel，但又需要参数描述？`parse_docstring=True` 从 Google-style docstring 自动提取。

```python
"""
parse_docstring=True：从 docstring 的 Args 部分自动提取参数描述
优点：不用写 BaseModel，代码更紧凑
缺点：无法添加 ge/le 等验证规则
"""
import json
from langchain_core.tools import tool


@tool(parse_docstring=True)
def advanced_search(query: str, filters: str, sort_by: str = "relevance") -> str:
    """高级搜索工具。

    根据关键词和过滤条件在知识库中执行高级搜索。

    Args:
        query: 搜索关键词，支持自然语言查询
        filters: 过滤条件，如 "date>2024-01-01,category=tech"
        sort_by: 排序方式，可选 relevance/date/popularity
    """
    return f"高级搜索: query='{query}', filters='{filters}', sort_by='{sort_by}'"


# ===== 直接调用验证 =====
result = advanced_search.invoke({
    "query": "LangChain 教程",
    "filters": "date>2024-06-01",
    "sort_by": "date",
})
print(f"调用结果: {result}")

# ===== 查看 Schema =====
schema = advanced_search.args_schema.model_json_schema()
print(f"\n完整 JSON Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
调用结果: 高级搜索: query='LangChain 教程', filters='date>2024-06-01', sort_by='date'

完整 JSON Schema:
{
  "description": "高级搜索工具。\n\n根据关键词和过滤条件在知识库中执行高级搜索。",
  "properties": {
    "query": {
      "description": "搜索关键词，支持自然语言查询",
      "title": "Query",
      "type": "string"
    },
    "filters": {
      "description": "过滤条件，如 \"date>2024-01-01,category=tech\"",
      "title": "Filters",
      "type": "string"
    },
    "sort_by": {
      "default": "relevance",
      "description": "排序方式，可选 relevance/date/popularity",
      "title": "Sort By",
      "type": "string"
    }
  },
  "required": [
    "query",
    "filters"
  ],
  "title": "advanced_searchSchema",
  "type": "object"
}
```

> **parse_docstring vs Pydantic 的取舍：** docstring 方式更简洁，但无法加 `ge/le/pattern` 等验证规则。如果参数只需要描述不需要验证，用 `parse_docstring`；如果需要约束，用 Pydantic。

---

## 示例3：枚举类型约束 — 限制参数可选值

**解决的问题：** `sort_by` 参数只允许 `relevance/date/popularity` 三个值，用字符串描述 LLM 可能传错。用 `Enum` 类型在 Schema 中生成 `enum` 约束，LLM 只能从固定选项中选择。

```python
"""
枚举类型约束：用 Enum 限制参数的可选值
Schema 中会生成 "enum": ["relevance", "date", "popularity"]
LLM 看到 enum 后只会从这几个值中选择，大幅降低出错率
"""
import json
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SortOrder(str, Enum):
    """排序方式"""
    RELEVANCE = "relevance"
    DATE = "date"
    POPULARITY = "popularity"


class SearchCategory(str, Enum):
    """搜索分类"""
    ALL = "all"
    TECH = "tech"
    BUSINESS = "business"
    LEGAL = "legal"


class EnumSearchParams(BaseModel):
    """带枚举约束的搜索参数"""
    query: str = Field(description="搜索关键词")
    sort_by: SortOrder = Field(
        default=SortOrder.RELEVANCE,
        description="排序方式",
    )
    category: SearchCategory = Field(
        default=SearchCategory.ALL,
        description="搜索分类",
    )


@tool(args_schema=EnumSearchParams)
def enum_search(
    query: str,
    sort_by: str = "relevance",
    category: str = "all",
) -> str:
    """带枚举约束的知识库搜索"""
    return f"搜索 '{query}' | 排序={sort_by} | 分类={category}"


# ===== 直接调用验证 =====
result = enum_search.invoke({
    "query": "向量检索",
    "sort_by": "date",
    "category": "tech",
})
print(f"调用结果: {result}")

# ===== 查看 Schema 中的 enum 约束 =====
schema = enum_search.args_schema.model_json_schema()
print(f"\n完整 JSON Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
调用结果: 搜索 '向量检索' | 排序=date | 分类=tech

完整 JSON Schema:
{
  "description": "带枚举约束的搜索参数",
  "properties": {
    "query": {
      "description": "搜索关键词",
      "title": "Query",
      "type": "string"
    },
    "sort_by": {
      "$ref": "#/$defs/SortOrder",
      "default": "relevance",
      "description": "排序方式"
    },
    "category": {
      "$ref": "#/$defs/SearchCategory",
      "default": "all",
      "description": "搜索分类"
    }
  },
  "$defs": {
    "SortOrder": {
      "enum": ["relevance", "date", "popularity"],
      "title": "SortOrder",
      "type": "string"
    },
    "SearchCategory": {
      "enum": ["all", "tech", "business", "legal"],
      "title": "SearchCategory",
      "type": "string"
    }
  },
  "required": ["query"],
  "title": "EnumSearchParams",
  "type": "object"
}
```

> **为什么用 Enum 而不是字符串描述？** 字符串描述 `"可选 relevance/date/popularity"` 只是建议，LLM 可能传 `"newest"` 或 `"by_date"`。Enum 在 Schema 中生成硬约束 `"enum": [...]`，LLM 只能从列表中选。

---

## 示例4：嵌套对象 Schema — 表达结构化参数

**解决的问题：** 有些参数本身是一个结构体（如日期范围包含 start 和 end），用嵌套的 BaseModel 表达，LLM 能理解并生成正确的嵌套 JSON。

```python
"""
嵌套对象 Schema：用 BaseModel 嵌套表达复杂参数结构
LLM 能理解嵌套结构并生成正确的嵌套 JSON 参数
"""
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class DateRange(BaseModel):
    """日期范围"""
    start: str = Field(description="开始日期，格式 YYYY-MM-DD")
    end: str = Field(description="结束日期，格式 YYYY-MM-DD")


class FilterConfig(BaseModel):
    """过滤配置"""
    date_range: DateRange | None = Field(default=None, description="日期范围过滤")
    categories: list[str] = Field(default_factory=list, description="分类过滤列表")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="最低相似度分数")


class NestedSearchParams(BaseModel):
    """嵌套结构的搜索参数"""
    query: str = Field(description="搜索关键词")
    filters: FilterConfig = Field(
        default_factory=FilterConfig,
        description="过滤配置",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量")


@tool(args_schema=NestedSearchParams)
def nested_search(query: str, filters: dict | None = None, top_k: int = 5) -> str:
    """支持嵌套过滤条件的知识库搜索"""
    if filters is None:
        filters = {}
    return json.dumps({
        "action": "nested_search",
        "query": query,
        "filters": filters,
        "top_k": top_k,
    }, ensure_ascii=False)


# ===== 带嵌套参数调用 =====
result = nested_search.invoke({
    "query": "RAG 优化策略",
    "filters": {
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "categories": ["tech", "tutorial"],
        "min_score": 0.8,
    },
    "top_k": 10,
})
print(f"嵌套调用结果:\n{json.loads(result)}")

# ===== 使用默认 filters 调用 =====
result2 = nested_search.invoke({"query": "Embedding 模型"})
print(f"\n默认调用结果:\n{json.loads(result2)}")

# ===== 查看嵌套 Schema =====
schema = nested_search.args_schema.model_json_schema()
print(f"\n嵌套 JSON Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
嵌套调用结果:
{'action': 'nested_search', 'query': 'RAG 优化策略', 'filters': {'date_range': {'start': '2024-01-01', 'end': '2024-12-31'}, 'categories': ['tech', 'tutorial'], 'min_score': 0.8}, 'top_k': 10}

默认调用结果:
{'action': 'nested_search', 'query': 'Embedding 模型', 'filters': {}, 'top_k': 5}

嵌套 JSON Schema:
{
  "$defs": {
    "DateRange": {
      "description": "日期范围",
      "properties": {
        "start": {
          "description": "开始日期，格式 YYYY-MM-DD",
          "title": "Start",
          "type": "string"
        },
        "end": {
          "description": "结束日期，格式 YYYY-MM-DD",
          "title": "End",
          "type": "string"
        }
      },
      "required": ["start", "end"],
      "title": "DateRange",
      "type": "object"
    },
    "FilterConfig": {
      "description": "过滤配置",
      "properties": {
        "date_range": {
          "anyOf": [{"$ref": "#/$defs/DateRange"}, {"type": "null"}],
          "default": null,
          "description": "日期范围过滤"
        },
        "categories": {
          "default": [],
          "description": "分类过滤列表",
          "items": {"type": "string"},
          "title": "Categories",
          "type": "array"
        },
        "min_score": {
          "default": 0.0,
          "description": "最低相似度分数",
          "maximum": 1.0,
          "minimum": 0.0,
          "title": "Min Score",
          "type": "number"
        }
      },
      "title": "FilterConfig",
      "type": "object"
    }
  },
  "description": "嵌套结构的搜索参数",
  "properties": {
    "query": {
      "description": "搜索关键词",
      "title": "Query",
      "type": "string"
    },
    "filters": {
      "$ref": "#/$defs/FilterConfig",
      "default": {},
      "description": "过滤配置"
    },
    "top_k": {
      "default": 5,
      "description": "返回结果数量",
      "maximum": 50,
      "minimum": 1,
      "title": "Top K",
      "type": "integer"
    }
  },
  "required": ["query"],
  "title": "NestedSearchParams",
  "type": "object"
}
```

> **嵌套 Schema 的实际价值：** RAG 系统中，检索条件往往是结构化的——日期范围、分类列表、分数阈值。嵌套 BaseModel 让 LLM 能生成结构化的过滤条件，而不是把所有条件塞进一个字符串里手动解析。

---

## 示例5：参数验证演示 — 正确、错误与错误处理

**解决的问题：** LLM 生成的参数可能不合法（超出范围、类型错误）。Pydantic 自动验证参数，`handle_validation_error` 控制验证失败时的行为——是抛异常还是返回错误信息让 Agent 自动修正。

```python
"""
参数验证演示：
1. 正确参数 → 正常执行
2. 错误参数 → ValidationError
3. handle_validation_error=True → 返回错误信息而不是抛异常
"""
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool


# ===== 定义带严格约束的 Schema =====
class StrictSearchParams(BaseModel):
    """严格参数验证的搜索"""
    query: str = Field(min_length=1, max_length=200, description="搜索关键词")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量，1-20")
    score_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="相似度阈值，0.0-1.0",
    )


def _strict_search(query: str, top_k: int = 5, score_threshold: float = 0.7) -> str:
    """严格参数验证的搜索工具"""
    return f"搜索 '{query}' | top_k={top_k} | threshold={score_threshold}"


# ========================================
# 场景 A：默认行为 — 验证失败抛异常
# ========================================
strict_tool = StructuredTool.from_function(
    func=_strict_search,
    name="strict_search",
    description="严格参数验证的搜索",
    args_schema=StrictSearchParams,
)

# 正确参数 → 成功
result = strict_tool.invoke({"query": "RAG", "top_k": 10, "score_threshold": 0.8})
print(f"正确参数: {result}")

# 错误参数 → ValidationError
print("\n--- 测试错误参数 ---")
try:
    strict_tool.invoke({"query": "", "top_k": 200, "score_threshold": 2.0})
except Exception as e:
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {e}")

# ========================================
# 场景 B：handle_validation_error=True — 返回错误信息
# ========================================
safe_tool = StructuredTool.from_function(
    func=_strict_search,
    name="safe_search",
    description="安全搜索（验证失败返回错误信息）",
    args_schema=StrictSearchParams,
    handle_validation_error=True,  # 关键配置
)

print("\n--- handle_validation_error=True ---")
result = safe_tool.invoke({"query": "", "top_k": 200})
print(f"返回结果（不抛异常）: {result}")

# ========================================
# 场景 C：handle_validation_error=自定义字符串
# ========================================
custom_msg_tool = StructuredTool.from_function(
    func=_strict_search,
    name="custom_error_search",
    description="自定义错误消息的搜索",
    args_schema=StrictSearchParams,
    handle_validation_error="参数不合法，请检查：query 不能为空，top_k 范围 1-20，score_threshold 范围 0-1",
)

print("\n--- handle_validation_error=自定义字符串 ---")
result = custom_msg_tool.invoke({"query": "", "top_k": 999})
print(f"返回结果: {result}")
```

### 预期输出

```
正确参数: 搜索 'RAG' | top_k=10 | threshold=0.8

--- 测试错误参数 ---
异常类型: ValidationError
异常信息: 3 validation errors for StrictSearchParams
query
  String should have at least 1 character [type=string_too_short, input_value='', input_type=str]
top_k
  Input should be less than or equal to 20 [type=less_than_equal, input_value=200, input_type=int]
score_threshold
  Input should be less than or equal to 1.0 [type=less_than_equal, input_value=2.0, input_type=float]

--- handle_validation_error=True ---
返回结果（不抛异常）: 2 validation errors for StrictSearchParams
query
  String should have at least 1 character [type=string_too_short, input_value='', input_type=str]
top_k
  Input should be less than or equal to 20 [type=less_than_equal, input_value=200, input_type=int]

--- handle_validation_error=自定义字符串 ---
返回结果: 参数不合法，请检查：query 不能为空，top_k 范围 1-20，score_threshold 范围 0-1
```

### handle_validation_error 选择指南

| 值 | 行为 | 适用场景 |
|---|------|----------|
| `False`（默认） | 抛出 `ValidationError` | 开发调试阶段 |
| `True` | 返回 Pydantic 原始错误信息 | Agent 自动重试场景 |
| `"自定义字符串"` | 返回固定提示信息 | 需要友好错误提示 |

> **Agent 最佳实践：** 设置 `handle_validation_error=True`。Agent 收到错误信息后会自动修正参数重试，而不是因为异常中断整个对话。

---

## 示例6：InjectedToolArg — 系统注入参数（对 LLM 不可见）

**解决的问题：** 有些参数不应由 LLM 决定——用户 ID、API 密钥、数据库连接。`InjectedToolArg` 标记这些参数，它们会从 LLM 可见的 Schema 中过滤掉，由系统在调用时注入。

```python
"""
InjectedToolArg：标记系统注入参数
- LLM 看到的 Schema 中不包含这些参数
- 调用时由系统代码注入
- 典型场景：user_id、db_connection、api_key
"""
import json
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg


@tool
def user_search(
    query: str,
    user_id: Annotated[str, InjectedToolArg],
) -> str:
    """搜索用户相关文档。LLM 只看到 query 参数。"""
    return f"用户 [{user_id}] 搜索: '{query}'"


@tool
def admin_operation(
    action: str,
    target: str,
    operator_id: Annotated[str, InjectedToolArg],
    permission_level: Annotated[int, InjectedToolArg],
) -> str:
    """执行管理操作。LLM 只看到 action 和 target。"""
    return (
        f"操作员 [{operator_id}](权限={permission_level}) "
        f"执行 '{action}' → 目标: '{target}'"
    )


# ===== 查看 LLM 可见的 Schema（已过滤注入参数） =====
print("=== user_search: LLM 可见的 Schema ===")
# get_input_schema() 返回过滤后的 Schema（tool_call_schema）
visible_schema = user_search.get_input_schema().model_json_schema()
print(json.dumps(visible_schema, indent=2, ensure_ascii=False))

print("\n=== admin_operation: LLM 可见的 Schema ===")
visible_schema2 = admin_operation.get_input_schema().model_json_schema()
print(json.dumps(visible_schema2, indent=2, ensure_ascii=False))

# ===== 调用时注入系统参数 =====
print("\n=== 调用验证 ===")
result1 = user_search.invoke({"query": "RAG 文档", "user_id": "user_12345"})
print(f"user_search: {result1}")

result2 = admin_operation.invoke({
    "action": "删除",
    "target": "过期文档",
    "operator_id": "admin_001",
    "permission_level": 3,
})
print(f"admin_operation: {result2}")
```

### 预期输出

```
=== user_search: LLM 可见的 Schema ===
{
  "description": "搜索用户相关文档。LLM 只看到 query 参数。",
  "properties": {
    "query": {
      "title": "Query",
      "type": "string"
    }
  },
  "required": [
    "query"
  ],
  "title": "user_search",
  "type": "object"
}

=== admin_operation: LLM 可见的 Schema ===
{
  "description": "执行管理操作。LLM 只看到 action 和 target。",
  "properties": {
    "action": {
      "title": "Action",
      "type": "string"
    },
    "target": {
      "title": "Target",
      "type": "string"
    }
  },
  "required": [
    "action",
    "target"
  ],
  "title": "admin_operation",
  "type": "object"
}

=== 调用验证 ===
user_search: 用户 [user_12345] 搜索: 'RAG 文档'
admin_operation: 操作员 [admin_001](权限=3) 执行 '删除' → 目标: '过期文档'
```

> **安全意义：** `user_id` 和 `operator_id` 对 LLM 完全不可见。LLM 无法伪造用户身份或提升权限。这是 RAG 系统中实现多租户隔离的关键机制。

---

## 示例7：完整 Schema 对比 — args_schema vs tool_call_schema

**解决的问题：** 理解 `args_schema`（完整参数）和 `tool_call_schema`（LLM 可见参数）的区别。

```python
"""
对比 args_schema 和 tool_call_schema
- args_schema: 包含所有参数（含 InjectedToolArg）
- tool_call_schema: 过滤掉 InjectedToolArg 后的 Schema（发送给 LLM）
"""
import json
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolArg


class TenantSearchParams(BaseModel):
    """多租户搜索参数"""
    query: str = Field(description="搜索关键词")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    tenant_id: Annotated[str, InjectedToolArg] = Field(description="租户 ID（系统注入）")


@tool(args_schema=TenantSearchParams)
def tenant_search(query: str, top_k: int = 5, tenant_id: str = "") -> str:
    """多租户知识库搜索"""
    return f"租户[{tenant_id}] 搜索 '{query}' top_k={top_k}"


# ===== args_schema: 完整 Schema（包含 tenant_id） =====
print("=== args_schema（完整，含注入参数） ===")
full_schema = tenant_search.args_schema.model_json_schema()
print(json.dumps(full_schema, indent=2, ensure_ascii=False))

# ===== tool_call_schema: LLM 可见 Schema（不含 tenant_id） =====
print("\n=== tool_call_schema（LLM 可见，已过滤） ===")
visible_schema = tenant_search.tool_call_schema.model_json_schema()
print(json.dumps(visible_schema, indent=2, ensure_ascii=False))

# ===== 对比差异 =====
full_props = set(full_schema.get("properties", {}).keys())
visible_props = set(visible_schema.get("properties", {}).keys())
hidden_props = full_props - visible_props
print(f"\n完整参数: {full_props}")
print(f"LLM 可见: {visible_props}")
print(f"被隐藏的: {hidden_props}")
```

### 预期输出

```
=== args_schema（完整，含注入参数） ===
{
  "description": "多租户搜索参数",
  "properties": {
    "query": {
      "description": "搜索关键词",
      "title": "Query",
      "type": "string"
    },
    "top_k": {
      "default": 5,
      "description": "返回结果数量",
      "maximum": 20,
      "minimum": 1,
      "title": "Top K",
      "type": "integer"
    },
    "tenant_id": {
      "description": "租户 ID（系统注入）",
      "title": "Tenant Id",
      "type": "string"
    }
  },
  "required": ["query", "tenant_id"],
  "title": "TenantSearchParams",
  "type": "object"
}

=== tool_call_schema（LLM 可见，已过滤） ===
{
  "description": "多租户搜索参数",
  "properties": {
    "query": {
      "description": "搜索关键词",
      "title": "Query",
      "type": "string"
    },
    "top_k": {
      "default": 5,
      "description": "返回结果数量",
      "maximum": 20,
      "minimum": 1,
      "title": "Top K",
      "type": "integer"
    }
  },
  "required": ["query"],
  "title": "TenantSearchParams",
  "type": "object"
}

完整参数: {'query', 'top_k', 'tenant_id'}
LLM 可见: {'query', 'top_k'}
被隐藏的: {'tenant_id'}
```

> **`tenant_id` 在 `args_schema` 中存在但在 `tool_call_schema` 中消失了。** 这就是 `InjectedToolArg` 的过滤机制——`bind_tools()` 发送给 LLM 的是 `tool_call_schema`，而不是 `args_schema`。

---

## Schema 设计决策速查

```
需要设计 Tool 的参数 Schema？
    │
    ├─ 参数简单（2-3 个 str/int）？
    │   └─ @tool(parse_docstring=True) — 从 docstring 提取描述
    │
    ├─ 需要范围约束（ge/le）或正则验证（pattern）？
    │   └─ Pydantic BaseModel + Field() — 精确控制
    │
    ├─ 参数有固定可选值？
    │   └─ Enum 类型 — 生成 "enum": [...] 硬约束
    │
    ├─ 参数是结构化对象（日期范围、过滤条件）？
    │   └─ 嵌套 BaseModel — 表达层级结构
    │
    └─ 有些参数不应让 LLM 看到（user_id、api_key）？
        └─ Annotated[str, InjectedToolArg] — 系统注入
```

---

## 关键要点总结

1. **Pydantic BaseModel 是生产首选**：`Field(description=..., ge=..., le=...)` 同时提供 LLM 可读的描述和运行时验证。
2. **parse_docstring 是轻量替代**：不需要验证规则时，从 docstring 自动提取参数描述，代码更紧凑。
3. **Enum 约束可选值**：Schema 中生成 `"enum": [...]`，LLM 只能从固定选项中选择，大幅降低出错率。
4. **嵌套 BaseModel 表达复杂结构**：日期范围、过滤条件等结构化参数用嵌套对象，比字符串拼接更可靠。
5. **handle_validation_error=True**：Agent 场景必备，让验证失败返回错误信息而不是抛异常，Agent 可以自动修正重试。
6. **InjectedToolArg 实现权限隔离**：`user_id`、`tenant_id` 等敏感参数对 LLM 不可见，由系统注入，防止身份伪造。

---

**上一篇**: [07_实战代码_场景2_函数调用完整流程.md](./07_实战代码_场景2_函数调用完整流程.md) — bind_tools → LLM 调用 → 结果回传的完整链路
**下一篇**: [07_实战代码_场景4_多工具Agent实战.md](./07_实战代码_场景4_多工具Agent实战.md) — 多工具绑定与 Agent 自动选择调用
