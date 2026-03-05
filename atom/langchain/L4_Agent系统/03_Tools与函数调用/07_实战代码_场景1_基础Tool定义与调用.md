# 实战代码 场景1：基础 Tool 定义与调用

> 演示三种 Tool 定义方式——@tool 装饰器、StructuredTool.from_function()、BaseTool 子类——并直接调用每个 Tool 验证功能，不涉及模型绑定

---

## 场景概述

LangChain 提供三种定义 Tool 的方式，复杂度递增，灵活度递增。本文不涉及 LLM 调用，纯粹聚焦于 **Tool 的定义和直接调用**，帮你搞清楚每种方式的写法、自动推断机制和输出结构。

每个示例都是完整可运行的 Python 代码，复制即用。

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain-core pydantic

# 本场景不需要 API Key，不调用 LLM
# 所有代码纯本地运行
```

---

## 示例1：@tool 装饰器方式（推荐，最简洁）

**解决的问题：** 用最少的代码把一个普通 Python 函数变成 LangChain Tool，自动提取名称、描述和参数 Schema。

### 1.1 基础用法：无参数装饰器

```python
"""
@tool 装饰器基础用法
规则：函数名 → Tool 名称，docstring → Tool 描述，类型注解 → 参数 Schema
"""
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """搜索网页获取信息。"""
    return f"搜索结果: {query} 的相关信息..."


# ===== 直接调用验证 =====
# Tool 继承自 Runnable，支持 .invoke() 方法
# 注意：参数必须以 dict 形式传入
result = search_web.invoke({"query": "LangChain Tools 教程"})
print(f"调用结果: {result}")

# ===== 查看 Tool 属性 =====
print(f"名称: {search_web.name}")
print(f"描述: {search_web.description}")
print(f"参数: {search_web.args}")
```

### 预期输出

```
调用结果: 搜索结果: LangChain Tools 教程 的相关信息...
名称: search_web
描述: 搜索网页获取信息。
参数: {'query': {'title': 'Query', 'type': 'string'}}
```

### 1.2 进阶用法：parse_docstring 自动提取参数描述

当工具有多个参数时，开启 `parse_docstring=True`，LangChain 会从 Google-style docstring 自动提取每个参数的描述写入 Schema。这些描述会发送给 LLM，帮助它正确填写参数。

```python
"""
@tool + parse_docstring：从 docstring 自动提取参数描述
要求：docstring 必须是 Google-style 格式（Args: 开头）
"""
import json
from langchain_core.tools import tool


@tool(parse_docstring=True)
def calculate(expression: str, precision: int) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式字符串
        precision: 小数精度
    """
    result = eval(expression)  # 示例用，生产环境应使用安全的解析器
    return f"{result:.{precision}f}"


# ===== 直接调用验证 =====
result = calculate.invoke({"expression": "3.14159 * 2", "precision": 3})
print(f"调用结果: {result}")

# ===== 查看 Tool 属性 =====
print(f"名称: {calculate.name}")
print(f"描述: {calculate.description}")
print(f"参数: {calculate.args}")

# ===== 查看完整 JSON Schema =====
schema = calculate.args_schema.model_json_schema()
print(f"\n完整 Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
调用结果: 6.283
名称: calculate
描述: 计算数学表达式。
参数: {'expression': {'description': '数学表达式字符串', 'title': 'Expression', 'type': 'string'}, 'precision': {'description': '小数精度', 'title': 'Precision', 'type': 'integer'}}

完整 Schema:
{
  "description": "计算数学表达式。",
  "properties": {
    "expression": {
      "description": "数学表达式字符串",
      "title": "Expression",
      "type": "string"
    },
    "precision": {
      "description": "小数精度",
      "title": "Precision",
      "type": "integer"
    }
  },
  "required": [
    "expression",
    "precision"
  ],
  "title": "calculateSchema",
  "type": "object"
}
```

> **关键记忆点：** `parse_docstring=True` 让你不用写 Pydantic BaseModel 就能给每个参数加描述。
> docstring 格式必须是 Google-style：`Args:` 开头，每行 `参数名: 描述`。

---

## 示例2：StructuredTool.from_function() 方式（自定义 Schema）

**解决的问题：** 当你需要对参数进行精确控制——添加默认值、枚举约束、范围验证——用 Pydantic BaseModel 定义 `args_schema`，再通过 `StructuredTool.from_function()` 创建 Tool。

```python
"""
StructuredTool.from_function() 方式
核心优势：通过 Pydantic BaseModel 精确控制参数描述和验证规则
"""
import json
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ===== 第一步：定义参数 Schema =====
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称")
    unit: str = Field(default="celsius", description="温度单位: celsius 或 fahrenheit")


# ===== 第二步：定义执行函数 =====
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取城市天气信息。"""
    temp = 25
    symbol = "C" if unit == "celsius" else "F"
    if unit == "fahrenheit":
        temp = int(temp * 9 / 5 + 32)
    return f"{city}天气: 晴，{temp}°{symbol}，湿度 45%"


# ===== 第三步：组装 Tool =====
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="获取指定城市的天气信息",
    args_schema=WeatherInput,
)

# ===== 直接调用验证 =====
# 使用默认参数
result1 = weather_tool.invoke({"city": "北京"})
print(f"默认单位: {result1}")

# 指定参数
result2 = weather_tool.invoke({"city": "上海", "unit": "fahrenheit"})
print(f"华氏度:   {result2}")

# ===== 查看 Tool 属性 =====
print(f"\n名称: {weather_tool.name}")
print(f"描述: {weather_tool.description}")
print(f"参数: {weather_tool.args}")

# ===== 查看完整 JSON Schema =====
schema = weather_tool.args_schema.model_json_schema()
print(f"\n完整 Schema:")
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

### 预期输出

```
默认单位: 北京天气: 晴，25°C，湿度 45%
华氏度:   上海天气: 晴，77°F，湿度 45%

名称: get_weather
描述: 获取指定城市的天气信息
参数: {'city': {'description': '城市名称', 'title': 'City', 'type': 'string'}, 'unit': {'default': 'celsius', 'description': '温度单位: celsius 或 fahrenheit', 'title': 'Unit', 'type': 'string'}}

完整 Schema:
{
  "description": "获取城市天气信息。",
  "properties": {
    "city": {
      "description": "城市名称",
      "title": "City",
      "type": "string"
    },
    "unit": {
      "default": "celsius",
      "description": "温度单位: celsius 或 fahrenheit",
      "title": "Unit",
      "type": "string"
    }
  },
  "required": [
    "city"
  ],
  "title": "WeatherInput",
  "type": "object"
}
```

### @tool vs StructuredTool 的区别

| 维度 | @tool | StructuredTool |
|------|-------|----------------|
| 参数描述来源 | docstring（需 parse_docstring） | Pydantic Field(description=...) |
| 默认值 | 函数签名中定义 | Field(default=...) |
| 验证规则 | 不支持 | Field(ge=1, le=20) 等 |
| 名称来源 | 函数名（或 @tool("name")） | name 参数显式指定 |
| 适用场景 | 简单工具 | 需要精确参数控制 |

> **什么时候用 StructuredTool？** 当你需要给参数加默认值、范围约束、枚举限制，或者函数名不适合做 Tool 名称时。

---

## 示例3：BaseTool 子类方式（最灵活）

**解决的问题：** 当你需要维护内部状态（缓存、计数器、连接池）、自定义错误处理、或复杂的初始化逻辑时，继承 `BaseTool` 是唯一的选择。

```python
"""
BaseTool 子类方式
核心优势：支持状态管理、自定义错误处理、完全控制执行逻辑
"""
import json
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ===== 第一步：定义参数 Schema（可选，不定义则从 _run 签名推断） =====
class DatabaseQueryInput(BaseModel):
    query: str = Field(description="查询关键词")
    limit: int = Field(default=5, description="返回结果数量上限")


# ===== 第二步：继承 BaseTool =====
class DatabaseQueryTool(BaseTool):
    """查询数据库获取用户信息"""

    name: str = "query_database"
    description: str = "查询数据库获取用户信息，支持按关键词搜索"
    args_schema: Type[BaseModel] = DatabaseQueryInput

    # 内部状态：记录查询历史（这是 @tool 和 StructuredTool 做不到的）
    query_history: list[str] = []
    call_count: int = 0

    def _run(self, query: str, limit: int = 5) -> str:
        """同步执行逻辑（必须实现）"""
        self.call_count += 1
        self.query_history.append(query)

        # 模拟数据库查询
        mock_data = {
            "用户": [
                {"id": 1, "name": "张三", "role": "开发者"},
                {"id": 2, "name": "李四", "role": "设计师"},
                {"id": 3, "name": "王五", "role": "产品经理"},
            ],
            "订单": [
                {"id": 101, "amount": 299, "status": "已完成"},
                {"id": 102, "amount": 599, "status": "进行中"},
            ],
        }

        for key, records in mock_data.items():
            if key in query:
                results = records[:limit]
                return json.dumps(
                    {
                        "query": query,
                        "count": len(results),
                        "data": results,
                        "call_number": self.call_count,
                    },
                    ensure_ascii=False,
                )

        return json.dumps(
            {"query": query, "count": 0, "data": [], "call_number": self.call_count},
            ensure_ascii=False,
        )

    async def _arun(self, query: str, limit: int = 5) -> str:
        """异步执行逻辑（可选，不实现则 ainvoke 会回退到 _run）"""
        return self._run(query, limit)


# ===== 实例化 =====
db_tool = DatabaseQueryTool()

# ===== 直接调用验证 =====
result1 = db_tool.invoke({"query": "用户", "limit": 2})
print(f"查询1: {result1}")

result2 = db_tool.invoke({"query": "订单"})
print(f"查询2: {result2}")

result3 = db_tool.invoke({"query": "不存在的表"})
print(f"查询3: {result3}")

# ===== 查看内部状态（BaseTool 独有能力） =====
print(f"\n总调用次数: {db_tool.call_count}")
print(f"查询历史: {db_tool.query_history}")

# ===== 查看 Tool 属性 =====
print(f"\n名称: {db_tool.name}")
print(f"描述: {db_tool.description}")
print(f"参数: {db_tool.args}")
```

### 预期输出

```
查询1: {"query": "用户", "count": 2, "data": [{"id": 1, "name": "张三", "role": "开发者"}, {"id": 2, "name": "李四", "role": "设计师"}], "call_number": 1}
查询2: {"query": "订单", "count": 2, "data": [{"id": 101, "amount": 299, "status": "已完成"}, {"id": 102, "amount": 599, "status": "进行中"}], "call_number": 2}
查询3: {"query": "不存在的表", "count": 0, "data": [], "call_number": 3}

总调用次数: 3
查询历史: ['用户', '订单', '不存在的表']

名称: query_database
描述: 查询数据库获取用户信息，支持按关键词搜索
参数: {'query': {'description': '查询关键词', 'title': 'Query', 'type': 'string'}, 'limit': {'default': 5, 'description': '返回结果数量上限', 'title': 'Limit', 'type': 'integer'}}
```

> **BaseTool 的核心优势是状态管理。** 上面的 `query_history` 和 `call_count` 在多次调用之间持久保存。
> 这在实际项目中非常有用：连接池复用、查询缓存、调用频率限制等场景都需要状态。

---

## 示例4：三种方式并排对比

**解决的问题：** 把三种方式放在一起，直观对比它们的写法差异和 Schema 输出。

```python
"""
三种 Tool 定义方式并排对比
同一个功能（搜索文档），用三种方式实现，对比差异
"""
import json
from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ========================================
# 方式 1：@tool 装饰器
# ========================================
@tool(parse_docstring=True)
def search_docs_v1(query: str, top_k: int = 5) -> str:
    """搜索文档知识库。

    Args:
        query: 搜索查询文本
        top_k: 返回结果数量
    """
    return f"[v1] 搜索 '{query}'，返回 {top_k} 条结果"


# ========================================
# 方式 2：StructuredTool.from_function()
# ========================================
class SearchDocsInput(BaseModel):
    query: str = Field(description="搜索查询文本")
    top_k: int = Field(default=5, description="返回结果数量")


def _search_docs_impl(query: str, top_k: int = 5) -> str:
    """搜索文档"""
    return f"[v2] 搜索 '{query}'，返回 {top_k} 条结果"


search_docs_v2 = StructuredTool.from_function(
    func=_search_docs_impl,
    name="search_docs_v2",
    description="搜索文档知识库",
    args_schema=SearchDocsInput,
)


# ========================================
# 方式 3：BaseTool 子类
# ========================================
class SearchDocsV3(BaseTool):
    name: str = "search_docs_v3"
    description: str = "搜索文档知识库"
    args_schema: Type[BaseModel] = SearchDocsInput

    def _run(self, query: str, top_k: int = 5) -> str:
        return f"[v3] 搜索 '{query}'，返回 {top_k} 条结果"


search_docs_v3 = SearchDocsV3()


# ========================================
# 对比调用结果
# ========================================
tools = [search_docs_v1, search_docs_v2, search_docs_v3]
test_input = {"query": "RAG 架构设计", "top_k": 3}

print("=" * 50)
print("三种方式调用同一输入的结果对比")
print("=" * 50)

for t in tools:
    result = t.invoke(test_input)
    print(f"\n【{t.name}】")
    print(f"  结果: {result}")
    print(f"  类型: {type(t).__name__}")
    print(f"  描述: {t.description}")
```

### 预期输出

```
==================================================
三种方式调用同一输入的结果对比
==================================================

【search_docs_v1】
  结果: [v1] 搜索 'RAG 架构设计'，返回 3 条结果
  类型: StructuredTool
  描述: 搜索文档知识库。

【search_docs_v2】
  结果: [v2] 搜索 'RAG 架构设计'，返回 3 条结果
  类型: StructuredTool
  描述: 搜索文档知识库

【search_docs_v3】
  结果: [v3] 搜索 'RAG 架构设计'，返回 3 条结果
  类型: SearchDocsV3
  描述: 搜索文档知识库
```

> **注意 `search_docs_v1` 的类型是 `StructuredTool`。** 这验证了一个关键事实：`@tool` 装饰器内部就是调用 `StructuredTool.from_function()`，它只是一层语法糖。

---

## 示例5：查看 Tool 的完整 Schema（发送给 LLM 的内容）

**解决的问题：** 理解 Tool 的 Schema 长什么样——这就是通过 `bind_tools()` 发送给 LLM 的参数描述，LLM 据此决定如何调用工具。

```python
"""
查看 Tool 的完整属性和 Schema
理解 LLM 看到的工具描述是什么样的
"""
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool(parse_docstring=True)
def rag_search(query: str, top_k: int = 5, source: str = "all") -> str:
    """在 RAG 知识库中执行语义搜索。

    Args:
        query: 用户的搜索查询文本
        top_k: 返回结果数量，默认 5 条
        source: 数据来源过滤，可选 all/docs/faq/wiki
    """
    return f"搜索 '{query}'，来源={source}，top_k={top_k}"


# ===== 基础属性 =====
print("=== 基础属性 ===")
print(f"name:        {rag_search.name}")
print(f"description: {rag_search.description}")
print(f"args:        {rag_search.args}")
print(f"return_direct: {rag_search.return_direct}")

# ===== 完整 JSON Schema（LLM 看到的内容） =====
print("\n=== args_schema.model_json_schema() ===")
schema = rag_search.args_schema.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))

# ===== 验证参数描述是否正确提取 =====
print("\n=== 参数描述验证 ===")
props = schema["properties"]
for param_name, param_info in props.items():
    desc = param_info.get("description", "（无描述）")
    param_type = param_info.get("type", "unknown")
    default = param_info.get("default", "（无默认值）")
    print(f"  {param_name}: type={param_type}, default={default}, desc={desc}")
```

### 预期输出

```
=== 基础属性 ===
name:        rag_search
description: 在 RAG 知识库中执行语义搜索。
args:        {'query': {'description': '用户的搜索查询文本', 'title': 'Query', 'type': 'string'}, 'top_k': {'default': 5, 'description': '返回结果数量，默认 5 条', 'title': 'Top K', 'type': 'integer'}, 'source': {'default': 'all', 'description': '数据来源过滤，可选 all/docs/faq/wiki', 'title': 'Source', 'type': 'string'}}
return_direct: False

=== args_schema.model_json_schema() ===
{
  "description": "在 RAG 知识库中执行语义搜索。",
  "properties": {
    "query": {
      "description": "用户的搜索查询文本",
      "title": "Query",
      "type": "string"
    },
    "top_k": {
      "default": 5,
      "description": "返回结果数量，默认 5 条",
      "title": "Top K",
      "type": "integer"
    },
    "source": {
      "default": "all",
      "description": "数据来源过滤，可选 all/docs/faq/wiki",
      "title": "Source",
      "type": "string"
    }
  },
  "required": [
    "query"
  ],
  "title": "rag_searchSchema",
  "type": "object"
}

=== 参数描述验证 ===
  query: type=string, default=（无默认值）, desc=用户的搜索查询文本
  top_k: type=integer, default=5, desc=返回结果数量，默认 5 条
  source: type=string, default=all, desc=数据来源过滤，可选 all/docs/faq/wiki
```

### Schema 结构解读

```
args_schema.model_json_schema() 输出的 JSON Schema：

{
  "description": "...",     ← Tool 描述（来自 docstring 第一行）
  "properties": {           ← 参数列表
    "query": {
      "description": "...", ← 参数描述（来自 docstring Args 部分）
      "type": "string"      ← 参数类型（来自类型注解）
    },
    "top_k": {
      "default": 5,         ← 默认值（来自函数签名）
      "type": "integer"
    }
  },
  "required": ["query"]     ← 必填参数（没有默认值的参数）
}

这个 Schema 就是 bind_tools() 发送给 LLM 的内容。
LLM 根据 description 理解工具用途，根据 properties 填写参数。
```

---

## 三种方式选择速查

```
需要定义一个 Tool？
    │
    ├─ 简单函数，参数不超过 3 个？
    │   └─ YES → @tool（+ parse_docstring=True）
    │
    ├─ 需要参数默认值、范围约束、枚举限制？
    │   └─ YES → StructuredTool.from_function() + Pydantic BaseModel
    │
    └─ 需要维护状态（缓存/计数器/连接池）？
        └─ YES → BaseTool 子类
```

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| RAG 检索工具 | `@tool(parse_docstring=True)` | 参数简单，docstring 自动提取描述 |
| 天气/汇率 API | `StructuredTool` | 需要枚举单位、范围约束 |
| 数据库连接池 | `BaseTool` 子类 | 需要维护连接池状态 |
| 带权限检查的工具 | `BaseTool` 子类 | 需要自定义错误处理 |

---

## 关键要点总结

1. **三种方式本质相同**：都是填充 `name` + `description` + `args_schema` + 执行函数这四个要素，只是写法不同。
2. **@tool 是语法糖**：内部调用的就是 `StructuredTool.from_function()`，所以 `@tool` 创建的对象类型是 `StructuredTool`。
3. **Tool 是 Runnable**：支持 `.invoke()`、`.ainvoke()`、`.batch()` 等标准方法，参数以 dict 形式传入。
4. **Schema 是给 LLM 看的**：`description` 和参数的 `description` 决定了 LLM 能否正确理解和调用工具。写得越清晰，调用越准确。
5. **BaseTool 独有状态管理**：只有继承 BaseTool 才能在多次调用之间保持状态（缓存、计数器等）。

---

**下一步：** 阅读 [07_实战代码_场景2_函数调用完整流程.md](./07_实战代码_场景2_函数调用完整流程.md)，学习如何用 `bind_tools()` 将工具绑定到模型，完成 Tool 定义 → 模型调用 → 结果回传的完整链路

