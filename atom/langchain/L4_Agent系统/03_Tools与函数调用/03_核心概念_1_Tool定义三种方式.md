# 核心概念 1：Tool 定义三种方式

> LangChain 提供三种定义 Tool 的方式：`@tool` 装饰器（最简洁）、`StructuredTool.from_function()`（自定义 Schema）、`BaseTool` 子类（最灵活）。理解三者的适用场景，是高效构建 Agent 工具集的基础。

---

## 三种方式总览

在开始详细讲解之前，先看一张对比表，建立全局认知：

| 维度 | `@tool` 装饰器 | `StructuredTool.from_function()` | `BaseTool` 子类 |
|------|---------------|----------------------------------|-----------------|
| **适用场景** | 简单函数包装 | 需要自定义参数 Schema | 复杂工具、需要状态管理 |
| **代码量** | 最少（3-5 行） | 中等（10-15 行） | 最多（15-30 行） |
| **灵活度** | 中等 | 较高 | 最高 |
| **Schema 来源** | 自动从函数签名推断 | 手动指定 Pydantic BaseModel | 手动指定 Pydantic BaseModel |
| **异步支持** | 自动识别 async 函数 | 通过 coroutine 参数传入 | 实现 `_arun()` 方法 |
| **状态管理** | 不支持 | 不支持 | 支持（类属性） |
| **推荐度** | ⭐⭐⭐⭐⭐ 首选 | ⭐⭐⭐⭐ 需要时用 | ⭐⭐⭐ 复杂场景用 |
| **源码位置** | `tools/convert.py` | `tools/structured.py` | `tools/base.py` |

**选择原则**：能用 `@tool` 就用 `@tool`，需要精确控制参数描述用 `StructuredTool`，需要维护状态或自定义执行逻辑用 `BaseTool`。

---

## 方式 1：@tool 装饰器（推荐，最简洁）

### 核心思想

`@tool` 装饰器是 LangChain 中定义 Tool 最推荐的方式。它的设计哲学是：**你只需要写一个普通的 Python 函数，加上类型注解和 docstring，装饰器帮你搞定剩下的一切。**

[来源: `langchain_core/tools/convert.py`（477 行）]

### 基本原理

`@tool` 装饰器做了三件事：
1. **提取函数名** → 作为 Tool 的 `name`
2. **提取 docstring** → 作为 Tool 的 `description`
3. **从类型注解推断 Schema** → 作为 Tool 的 `args_schema`

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息"""
    return f"{city}今天晴，25°C"

# 装饰器自动提取的信息：
print(get_weather.name)          # "get_weather"（函数名）
print(get_weather.description)   # "查询指定城市的实时天气信息"（docstring）
print(get_weather.args_schema.model_json_schema())
# → {"properties": {"city": {"type": "string"}}, "required": ["city"], ...}
```

### 4 种使用模式

源码中 `@tool` 有 4 种重载签名，覆盖不同场景：

#### 模式 1：无参数装饰器（最常用）

直接用 `@tool` 装饰函数，名称取函数名，描述取 docstring。

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """在向量知识库中搜索与查询相关的文档片段"""
    # 实际实现中会调用向量数据库
    return f"搜索结果：关于 '{query}' 的相关文档..."

# 验证
print(search_knowledge_base.name)         # "search_knowledge_base"
print(search_knowledge_base.description)  # "在向量知识库中搜索与查询相关的文档片段"

# 直接调用（Tool 也是 Runnable，支持 invoke）
result = search_knowledge_base.invoke({"query": "RAG 架构"})
print(result)  # "搜索结果：关于 'RAG 架构' 的相关文档..."
```

#### 模式 2：指定自定义名称

用 `@tool("custom_name")` 覆盖默认的函数名。适合函数名不适合作为 Tool 名称的场景。

```python
from langchain_core.tools import tool

@tool("web_search")
def _internal_search_impl(query: str) -> str:
    """搜索互联网获取最新信息"""
    return f"网络搜索结果：{query}"

# 名称是自定义的，不是函数名
print(_internal_search_impl.name)  # "web_search"（不是 "_internal_search_impl"）
```

#### 模式 3：带参数装饰器（精细控制）

传入关键字参数，控制 Tool 的各种行为。

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 自定义参数 Schema
class CalculatorInput(BaseModel):
    expression: str = Field(description="要计算的数学表达式，如 '2 + 3 * 4'")

@tool(
    description="精确计算数学表达式的结果",  # 覆盖 docstring
    args_schema=CalculatorInput,              # 自定义 Schema
    return_direct=False,                      # 结果是否直接返回用户
    response_format="content",                # 返回格式
    parse_docstring=False,                    # 不从 docstring 解析参数描述
)
def calculator(expression: str) -> str:
    """这个 docstring 会被 description 参数覆盖"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

print(calculator.name)         # "calculator"
print(calculator.description)  # "精确计算数学表达式的结果"
print(calculator.invoke({"expression": "2 + 3 * 4"}))
# → "计算结果：2 + 3 * 4 = 14"
```

**@tool 装饰器支持的全部参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name_or_callable` | str / Callable | 函数名 | Tool 名称或被装饰的函数 |
| `description` | str | docstring | Tool 描述（覆盖 docstring） |
| `return_direct` | bool | False | True 时结果直接返回用户，跳过 LLM |
| `args_schema` | BaseModel | 自动推断 | 自定义参数 Schema |
| `infer_schema` | bool | True | 是否从函数签名推断 Schema |
| `response_format` | str | "content" | "content" 或 "content_and_artifact" |
| `parse_docstring` | bool | False | 从 Google-style docstring 提取参数描述 |
| `error_on_invalid_docstring` | bool | True | docstring 格式错误时是否报错 |
| `extras` | dict | None | 提供商特定字段（如 Anthropic cache_control） |

#### 模式 4：函数调用方式（包装 Runnable）

不用装饰器语法，直接调用 `tool()` 函数，可以包装 Runnable 对象。

```python
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

# 包装一个已有的 Runnable
my_runnable = RunnableLambda(lambda x: f"处理结果：{x}")
my_tool = tool("process_data", my_runnable, description="处理输入数据")

print(my_tool.name)         # "process_data"
print(my_tool.description)  # "处理输入数据"
```

### parse_docstring：从 docstring 自动提取参数描述

这是一个非常实用的特性。开启 `parse_docstring=True` 后，`@tool` 会解析 Google-style docstring，自动提取每个参数的描述，写入 Schema。

```python
from langchain_core.tools import tool

@tool(parse_docstring=True)
def create_document(title: str, content: str, tags: list[str]) -> str:
    """创建一个新文档并保存到知识库。

    Args:
        title: 文档标题，不超过100个字符
        content: 文档正文内容
        tags: 文档标签列表，用于分类检索
    """
    return f"文档 '{title}' 已创建，包含 {len(tags)} 个标签"

# 查看自动提取的参数描述
schema = create_document.args_schema.model_json_schema()
print(schema["properties"]["title"]["description"])
# → "文档标题，不超过100个字符"
print(schema["properties"]["content"]["description"])
# → "文档正文内容"
print(schema["properties"]["tags"]["description"])
# → "文档标签列表，用于分类检索"
```

**注意**：docstring 必须是 Google-style 格式（`Args:` 开头，每个参数用 `name: description` 格式）。

### 异步函数支持

`@tool` 自动识别 `async` 函数，生成的 Tool 同时支持同步和异步调用。

```python
from langchain_core.tools import tool
import asyncio

@tool
async def async_search(query: str) -> str:
    """异步搜索互联网"""
    await asyncio.sleep(0.1)  # 模拟网络请求
    return f"异步搜索结果：{query}"

# 异步调用
result = await async_search.ainvoke({"query": "LangChain tools"})
print(result)  # "异步搜索结果：LangChain tools"
```

---

## 方式 2：StructuredTool.from_function()（自定义 Schema）

### 核心思想

当你需要对参数 Schema 进行精确控制——比如添加参数描述、设置默认值、添加验证规则——`StructuredTool.from_function()` 是更好的选择。

[来源: `langchain_core/tools/structured.py`（272 行）]

### 基本用法

```python
from langchain_core.tools import StructuredTool

def search_documents(query: str, top_k: int = 5) -> str:
    """搜索文档"""
    return f"搜索 '{query}'，返回前 {top_k} 条结果"

# 从函数创建 Tool
search_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_docs",
    description="在知识库中搜索相关文档，返回最相关的结果",
)

print(search_tool.name)         # "search_docs"
print(search_tool.description)  # "在知识库中搜索相关文档，返回最相关的结果"
print(search_tool.invoke({"query": "向量数据库", "top_k": 3}))
# → "搜索 '向量数据库'，返回前 3 条结果"
```

### 自定义 Pydantic Schema（核心优势）

`StructuredTool` 最大的优势是支持传入自定义的 Pydantic BaseModel 作为 `args_schema`，实现精确的参数描述和验证。

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class SearchType(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class RAGSearchInput(BaseModel):
    """RAG 检索参数 Schema"""
    query: str = Field(
        description="用户的搜索查询文本"
    )
    search_type: SearchType = Field(
        default=SearchType.VECTOR,
        description="检索类型：vector（向量检索）、keyword（关键词检索）、hybrid（混合检索）"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回结果数量，范围 1-20"
    )
    filter_source: Optional[str] = Field(
        default=None,
        description="按数据来源过滤，如 'wiki'、'docs'、'faq'"
    )

def rag_search(
    query: str,
    search_type: str = "vector",
    top_k: int = 5,
    filter_source: Optional[str] = None,
) -> str:
    """执行 RAG 检索"""
    result = f"[{search_type}] 搜索 '{query}'，top_k={top_k}"
    if filter_source:
        result += f"，来源过滤={filter_source}"
    return result

# 使用自定义 Schema 创建 Tool
rag_search_tool = StructuredTool.from_function(
    func=rag_search,
    name="rag_search",
    description="在 RAG 知识库中执行检索，支持向量检索、关键词检索和混合检索",
    args_schema=RAGSearchInput,
)

# 查看生成的 JSON Schema（这就是发送给 LLM 的参数描述）
import json
schema = rag_search_tool.args_schema.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

输出的 JSON Schema 会包含每个参数的 description、类型、默认值、枚举值等信息，LLM 据此生成正确的参数。

### parse_docstring 支持

`StructuredTool.from_function()` 同样支持 `parse_docstring`，从 Google-style docstring 提取参数描述：

```python
from langchain_core.tools import StructuredTool

def send_notification(user_id: str, message: str, priority: str = "normal") -> str:
    """发送通知给指定用户。

    Args:
        user_id: 目标用户的唯一标识符
        message: 通知内容文本
        priority: 优先级，可选 low/normal/high
    """
    return f"已向用户 {user_id} 发送 {priority} 优先级通知"

notification_tool = StructuredTool.from_function(
    func=send_notification,
    name="send_notification",
    description="发送通知给指定用户",
    parse_docstring=True,  # 自动从 docstring 提取参数描述
)

# 参数描述已自动填充
schema = notification_tool.args_schema.model_json_schema()
print(schema["properties"]["user_id"]["description"])
# → "目标用户的唯一标识符"
```

### 异步函数支持

通过 `coroutine` 参数传入异步函数：

```python
from langchain_core.tools import StructuredTool
import asyncio

async def async_db_query(sql: str) -> str:
    """异步执行数据库查询"""
    await asyncio.sleep(0.1)
    return f"查询结果：{sql}"

def sync_db_query(sql: str) -> str:
    """同步版本（fallback）"""
    return f"同步查询结果：{sql}"

db_tool = StructuredTool.from_function(
    func=sync_db_query,          # 同步函数
    coroutine=async_db_query,    # 异步函数
    name="db_query",
    description="执行数据库查询",
)

# 同步调用走 func，异步调用走 coroutine
result = db_tool.invoke({"sql": "SELECT * FROM docs"})
result_async = await db_tool.ainvoke({"sql": "SELECT * FROM docs"})
```

---

## 方式 3：BaseTool 子类（最灵活）

### 核心思想

当你需要完全控制 Tool 的行为——维护内部状态、自定义错误处理、复杂的初始化逻辑——继承 `BaseTool` 是唯一的选择。

[来源: `langchain_core/tools/base.py`（55KB）]

### 基本结构

继承 `BaseTool` 需要：
1. 定义 `name`、`description` 类属性
2. 定义 `args_schema`（Pydantic BaseModel）
3. 实现 `_run()` 方法（同步执行逻辑）
4. 可选实现 `_arun()` 方法（异步执行逻辑）

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询文本")
    top_k: int = Field(default=5, description="返回结果数量")

class KnowledgeBaseSearchTool(BaseTool):
    """知识库搜索工具 — 继承 BaseTool 实现"""

    name: str = "knowledge_search"
    description: str = "在知识库中搜索相关文档，返回最匹配的结果"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 5) -> str:
        """同步执行搜索"""
        return f"搜索 '{query}'，返回前 {top_k} 条结果"

    async def _arun(self, query: str, top_k: int = 5) -> str:
        """异步执行搜索（可选）"""
        return f"[异步] 搜索 '{query}'，返回前 {top_k} 条结果"

# 实例化并使用
search_tool = KnowledgeBaseSearchTool()
print(search_tool.name)         # "knowledge_search"
print(search_tool.description)  # "在知识库中搜索相关文档，返回最匹配的结果"
print(search_tool.invoke({"query": "RAG 优化", "top_k": 3}))
# → "搜索 'RAG 优化'，返回前 3 条结果"
```

### 核心优势：状态管理

`BaseTool` 子类可以维护内部状态，这是另外两种方式做不到的。

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
class QueryInput(BaseModel):
    query: str = Field(description="搜索查询")

class StatefulSearchTool(BaseTool):
    """带状态的搜索工具：记录搜索历史，支持去重和统计"""

    name: str = "stateful_search"
    description: str = "搜索知识库，自动去重并记录搜索历史"
    args_schema: Type[BaseModel] = QueryInput

    # 内部状态（不会暴露给 LLM）
    search_history: list[str] = []
    call_count: int = 0

    def _run(self, query: str) -> str:
        self.call_count += 1

        # 去重：如果已经搜索过相同的 query，直接返回缓存提示
        if query in self.search_history:
            return f"[缓存命中] '{query}' 已搜索过（第 {self.call_count} 次调用）"

        self.search_history.append(query)
        return f"搜索结果：'{query}'（第 {self.call_count} 次调用，历史 {len(self.search_history)} 条）"

# 使用示例
search = StatefulSearchTool()
print(search.invoke({"query": "RAG 架构"}))
# → "搜索结果：'RAG 架构'（第 1 次调用，历史 1 条）"
print(search.invoke({"query": "RAG 架构"}))
# → "[缓存命中] 'RAG 架构' 已搜索过（第 2 次调用）"
print(search.invoke({"query": "向量数据库"}))
# → "搜索结果：'向量数据库'（第 3 次调用，历史 2 条）"
```

### 自定义错误处理

`BaseTool` 允许你精细控制错误处理逻辑：

```python
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from typing import Type

class DBQueryInput(BaseModel):
    sql: str = Field(description="SQL 查询语句")

class SafeDatabaseTool(BaseTool):
    """安全的数据库查询工具，带权限检查和错误处理"""

    name: str = "safe_db_query"
    description: str = "安全地执行只读数据库查询"
    args_schema: Type[BaseModel] = DBQueryInput
    handle_tool_error: bool = True  # 启用错误处理

    # 配置属性
    allowed_tables: list[str] = ["documents", "users", "logs"]

    def _run(self, sql: str) -> str:
        # 安全检查：禁止写操作
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ToolException(
                    f"安全拦截：禁止执行 {keyword} 操作，本工具仅支持只读查询"
                )

        # 表名检查
        for table in self.allowed_tables:
            if table in sql.lower():
                return f"查询结果：执行 '{sql}' 成功"

        raise ToolException(
            f"权限不足：只允许查询 {self.allowed_tables} 表"
        )

# 使用
db_tool = SafeDatabaseTool()
print(db_tool.invoke({"sql": "SELECT * FROM documents WHERE id = 1"}))
# → "查询结果：执行 'SELECT * FROM documents WHERE id = 1' 成功"

# 危险操作会被拦截
print(db_tool.invoke({"sql": "DROP TABLE documents"}))
# → "安全拦截：禁止执行 DROP 操作，本工具仅支持只读查询"
```

---

## 三种方式的选择指南

### 决策流程图

```
需要定义一个 Tool
    │
    ├─ 函数逻辑简单，参数类型注解清晰？
    │   └─ YES → 用 @tool 装饰器 ✅
    │
    ├─ 需要精确控制参数描述/验证规则？
    │   └─ YES → 用 StructuredTool.from_function() ✅
    │
    ├─ 需要维护内部状态？（如缓存、计数器、连接池）
    │   └─ YES → 用 BaseTool 子类 ✅
    │
    ├─ 需要自定义错误处理逻辑？
    │   └─ YES → 用 BaseTool 子类 ✅
    │
    └─ 需要包装一个已有的 Runnable？
        └─ YES → 用 tool("name", runnable) ✅
```

### 实际场景对应

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 简单的搜索/计算工具 | `@tool` | 代码最少，自动推断 Schema |
| RAG 检索工具（多参数） | `@tool(parse_docstring=True)` | docstring 自动提取参数描述 |
| 需要枚举值/范围约束的参数 | `StructuredTool` | Pydantic Field 支持丰富的验证 |
| 数据库连接池工具 | `BaseTool` 子类 | 需要维护连接池状态 |
| 带权限检查的 API 工具 | `BaseTool` 子类 | 需要自定义错误处理 |
| 包装第三方 SDK | `StructuredTool` 或 `BaseTool` | 需要适配已有接口 |

---

## 在 Agent 开发中的实际应用

### 典型 RAG Agent 工具集

```python
from langchain_core.tools import tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field
from typing import Type, Optional

# ===== 工具 1：用 @tool（简单场景） =====
@tool
def search_web(query: str) -> str:
    """当知识库中没有答案时，搜索互联网获取最新信息"""
    return f"网络搜索结果：{query}..."

# ===== 工具 2：用 StructuredTool（需要精确参数控制） =====
class VectorSearchInput(BaseModel):
    query: str = Field(description="语义搜索查询文本")
    collection: str = Field(
        default="default",
        description="知识库集合名称，如 'docs'、'faq'、'wiki'"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")

def vector_search(query: str, collection: str = "default", top_k: int = 5) -> str:
    return f"从 {collection} 集合中搜索 '{query}'，返回 {top_k} 条结果"

vector_search_tool = StructuredTool.from_function(
    func=vector_search,
    name="vector_search",
    description="在向量知识库中执行语义搜索",
    args_schema=VectorSearchInput,
)

# ===== 工具 3：用 BaseTool（需要状态管理） =====
class ConversationMemoryInput(BaseModel):
    action: str = Field(description="操作类型：'save' 保存记忆，'recall' 回忆相关内容")
    content: str = Field(description="要保存或查询的内容")

class ConversationMemoryTool(BaseTool):
    """对话记忆工具，维护对话上下文"""

    name: str = "conversation_memory"
    description: str = "保存和回忆对话中的关键信息"
    args_schema: Type[BaseModel] = ConversationMemoryInput

    memories: list[str] = []

    def _run(self, action: str, content: str) -> str:
        if action == "save":
            self.memories.append(content)
            return f"已保存记忆（共 {len(self.memories)} 条）"
        elif action == "recall":
            relevant = [m for m in self.memories if content.lower() in m.lower()]
            if relevant:
                return f"相关记忆：{'; '.join(relevant)}"
            return "没有找到相关记忆"
        return f"未知操作：{action}"

# ===== 绑定到模型 =====
# tools = [search_web, vector_search_tool, ConversationMemoryTool()]
# model_with_tools = chat_model.bind_tools(tools)
```

### 三种方式混合使用

在实际项目中，三种方式经常混合使用。选择标准始终是：**用最简单的方式满足需求**。

```
Agent 工具集
├── @tool: search_web          （简单，无需自定义）
├── @tool: calculator          （简单，无需自定义）
├── StructuredTool: rag_search （需要精确的参数描述和验证）
├── BaseTool: db_query         （需要连接池状态管理）
└── BaseTool: file_manager     （需要权限检查和错误处理）
```

---

## 源码架构映射

三种方式在源码中的继承关系：

```
RunnableSerializable
    └── BaseTool                    # tools/base.py（抽象基类）
            ├── StructuredTool      # tools/structured.py（多参数 Tool）
            │       └── from_function()  # 类方法，从函数创建
            └── Tool                # tools/simple.py（单参数 Tool，向后兼容）

@tool 装饰器                        # tools/convert.py
    └── 内部调用 StructuredTool.from_function() 或 Tool() 创建实例
```

**关键洞察**：`@tool` 装饰器并不是一种独立的 Tool 类型，它内部调用的就是 `StructuredTool.from_function()`。装饰器只是一层语法糖，让创建过程更简洁。

---

## 关键要点总结

1. **@tool 装饰器**是首选，覆盖 80% 以上的场景。它从函数签名自动推断 Schema，代码量最少。
2. **StructuredTool.from_function()** 适合需要精确控制参数描述和验证规则的场景，通过 Pydantic BaseModel 定义 `args_schema`。
3. **BaseTool 子类**适合需要状态管理、自定义错误处理、复杂初始化逻辑的场景，提供最大灵活性。
4. 三种方式的本质相同：都是填充 `name` + `description` + `args_schema` + 执行函数这四个要素。
5. 在实际 Agent 项目中，三种方式经常混合使用，选择标准是"用最简单的方式满足需求"。

---

**下一步**: 阅读 [03_核心概念_2_函数调用协议.md](./03_核心概念_2_函数调用协议.md)，了解 Tool 定义好之后如何通过 `bind_tools()` 绑定到模型并完成函数调用
