# 核心概念 6：高级 Tool 特性

> Tool 不只是"函数包装器"——LangChain 在 `BaseTool` 上提供了 response_format、InjectedToolArg、extras、return_direct、handle_tool_error 等高级特性，解决内容与工件分离、参数安全注入、提供商适配、错误容错等生产级问题。

---

## 概述

前面几篇讲了 Tool 的定义、Schema、消息流和选择策略。但在生产环境中，你还会遇到这些问题：

1. **搜索结果太大，全部发给模型太浪费 token** → `response_format="content_and_artifact"`
2. **用户 ID、session 等参数不能让 LLM 决定** → `InjectedToolArg`
3. **需要关联工具调用的请求和响应** → `InjectedToolCallId`
4. **Anthropic 等提供商有特殊配置** → `extras`
5. **某些工具结果不需要模型二次加工** → `return_direct`
6. **工具执行失败不能让 Agent 崩溃** → `handle_tool_error`
7. **参数验证失败也要优雅处理** → `handle_validation_error`
8. **需要区分"工具异常"和"系统异常"** → `ToolException`

```
BaseTool 高级特性全景：

                    ┌─────────────────────────────────────┐
                    │            BaseTool                  │
                    ├─────────────────────────────────────┤
                    │  response_format    → 输出格式控制   │
                    │  return_direct      → 跳过模型润色   │
                    │  handle_tool_error  → 执行错误容错   │
                    │  handle_validation_error → 验证容错  │
                    │  extras             → 提供商扩展配置 │
                    ├─────────────────────────────────────┤
                    │  InjectedToolArg    → 隐藏参数注入   │
                    │  InjectedToolCallId → 调用ID注入     │
                    │  ToolException      → 专用异常类     │
                    └─────────────────────────────────────┘
```

这些特性单独看都不复杂，但组合起来能解决 Agent 系统中大量的生产级问题。

---

## 1. response_format — 内容与工件分离

### 一句话定义

**`response_format` 控制工具输出的格式——默认全部作为消息内容，设为 `"content_and_artifact"` 后可以把摘要发给模型、完整数据保留给用户。**

### 为什么需要？

一个 RAG 搜索工具返回了 20 条文档片段，每条 500 字。如果全部塞进 `ToolMessage.content` 发给模型：

- 10000 字 ≈ 3000+ tokens，成本高
- 模型可能被大量上下文干扰，回答质量下降
- 用户其实需要看完整结果，但模型只需要摘要

`response_format="content_and_artifact"` 的解决方案：工具返回一个元组 `(content, artifact)`，content 是摘要发给模型，artifact 是完整数据保存在 `ToolMessage.artifact` 中供用户使用。

### 两种模式对比

| 模式 | response_format 值 | 返回值要求 | ToolMessage 结果 |
|------|-------------------|-----------|-----------------|
| 默认 | `"content"` | 返回字符串 | `content=返回值`, `artifact=None` |
| 分离 | `"content_and_artifact"` | 返回 `(str, Any)` 元组 | `content=元组[0]`, `artifact=元组[1]` |

### 代码示例

```python
from langchain_core.tools import tool

# ===== 默认模式：全部作为 content =====
@tool
def search_default(query: str) -> str:
    """搜索文档（默认模式）"""
    results = [
        {"title": "RAG 架构", "content": "RAG 由检索和生成两部分组成...", "score": 0.95},
        {"title": "向量检索", "content": "向量检索基于 embedding 相似度...", "score": 0.87},
    ]
    # 全部序列化为字符串 → 全部发给模型（浪费 token）
    return str(results)


# ===== content_and_artifact 模式：摘要给模型，完整数据给用户 =====
@tool(response_format="content_and_artifact")
def search_smart(query: str) -> tuple[str, list[dict]]:
    """搜索文档（分离模式）"""
    results = [
        {"title": "RAG 架构", "content": "RAG 由检索和生成两部分组成...", "score": 0.95},
        {"title": "向量检索", "content": "向量检索基于 embedding 相似度...", "score": 0.87},
    ]
    # 摘要给模型（省 token）
    summary = f"找到 {len(results)} 个相关文档：" + "、".join(r["title"] for r in results)
    # 完整结果给用户（不发给模型）
    return summary, results
```

### 调用结果对比

```python
from langchain_core.messages import ToolMessage

# 默认模式
msg1 = search_default.invoke({"query": "RAG"})
# → "[ {'title': 'RAG 架构', 'content': '...长文本...', 'score': 0.95}, ...]"
# 全部内容都在 content 里，发给模型

# 分离模式
msg2 = search_smart.invoke({"query": "RAG"})
# msg2 是 ToolMessage 对象:
#   content = "找到 2 个相关文档：RAG 架构、向量检索"  ← 发给模型（短）
#   artifact = [{"title": "RAG 架构", ...}, ...]       ← 保留给用户（完整）
```

### 源码原理

[来源: `langchain_core/tools/base.py`]

```python
# BaseTool._run_and_format() 简化逻辑
def _run_and_format(self, *args, **kwargs):
    result = self._run(*args, **kwargs)  # 执行工具

    if self.response_format == "content_and_artifact":
        # 期望返回 (content, artifact) 元组
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("content_and_artifact 模式必须返回 (content, artifact) 元组")
        content, artifact = result
        return ToolMessage(content=content, artifact=artifact, ...)
    else:
        # 默认模式：全部作为 content
        return ToolMessage(content=str(result), ...)
```

### RAG 实战：搜索结果分离

```python
@tool(response_format="content_and_artifact")
def rag_search(query: str) -> tuple[str, list[dict]]:
    """在知识库中搜索相关文档片段。"""
    # 模拟向量检索
    results = [
        {"chunk_id": "doc1_p3", "text": "RAG 系统由三个核心组件构成...", "score": 0.95},
        {"chunk_id": "doc2_p1", "text": "Embedding 模型将文本转换为向量...", "score": 0.89},
        {"chunk_id": "doc3_p7", "text": "检索器负责从向量库中找到相关文档...", "score": 0.82},
    ]

    # 给模型的摘要：只包含关键信息
    summary_lines = [f"- {r['chunk_id']}（相似度 {r['score']}）: {r['text'][:30]}..." for r in results]
    summary = f"找到 {len(results)} 个相关片段：\n" + "\n".join(summary_lines)

    # 给用户的完整数据：包含全文、元数据等
    return summary, results

# 在 Agent 中使用
# 模型看到的：简短摘要（~100 tokens）
# 用户拿到的：完整搜索结果（可展示在 UI 中）
```

---

## 2. InjectedToolArg — 自动注入参数

### 一句话定义

**`InjectedToolArg` 标记某些参数为"系统注入"——这些参数从 Schema 中隐藏，LLM 看不到也不需要生成，由运行时系统自动提供。**

### 为什么需要？

Agent 调用工具时，有些参数不应该由 LLM 决定：

| 参数类型 | 示例 | 为什么不能让 LLM 决定 |
|----------|------|---------------------|
| 用户身份 | `user_id` | LLM 可能伪造身份，绕过权限 |
| 会话信息 | `session_id` | LLM 不知道当前会话 |
| 数据库连接 | `db_connection` | 运行时资源，不是文本参数 |
| API 密钥 | `api_key` | 安全敏感信息 |
| 配置对象 | `config` | 系统级配置 |

### 代码示例

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def search_user_docs(
    query: str,                                      # LLM 可见，需要生成
    user_id: Annotated[str, InjectedToolArg],        # LLM 不可见，系统注入
    db_name: Annotated[str, InjectedToolArg] = "default",  # 不可见，有默认值
) -> str:
    """搜索用户的私有文档库。"""
    return f"用户 {user_id} 在 {db_name} 中搜索: {query}"

# LLM 看到的 Schema（tool_call_schema）：
print(search_user_docs.tool_call_schema.model_json_schema())
# → {"properties": {"query": {"type": "string"}}, "required": ["query"]}
# user_id 和 db_name 被过滤掉了！

# 完整 Schema（args_schema）：
print(search_user_docs.args_schema.model_json_schema())
# → {"properties": {"query": {...}, "user_id": {...}, "db_name": {...}}, ...}
# 内部仍然保留所有参数
```

### 运行时注入方式

```python
# 方式 1: 直接在 invoke 时传入
result = search_user_docs.invoke({
    "query": "RAG 最佳实践",
    "user_id": "user_12345",       # 由系统代码注入
    "db_name": "knowledge_base_v2",
})

# 方式 2: 在 Agent 的 ToolNode 中通过 injected_state 注入
# （LangGraph 场景，ToolNode 自动从 state 中提取注入参数）
```

### 源码原理

[来源: `langchain_core/tools/base.py`]

```python
# InjectedToolArg 本质是一个标记类
class InjectedToolArg:
    """标记参数为注入参数，不暴露给 LLM。"""
    pass

# tool_call_schema 属性会过滤掉 InjectedToolArg 标记的字段
@property
def tool_call_schema(self) -> type[BaseModel]:
    full_schema = self.args_schema
    # _get_filtered_args 检查每个字段的 metadata 中是否有 InjectedToolArg
    return _get_filtered_args(full_schema)
```

关键点：`Annotated[str, InjectedToolArg]` 利用 Python 的 `Annotated` 类型，在类型注解的 metadata 中附加 `InjectedToolArg` 标记。`_get_filtered_args()` 遍历所有字段，检查 metadata 中是否包含 `InjectedToolArg` 实例，有则从 `tool_call_schema` 中移除。

---

## 3. InjectedToolCallId — 注入工具调用 ID

### 一句话定义

**`InjectedToolCallId` 自动将当前 `tool_call` 的 ID 注入到工具参数中，用于关联请求和响应。**

### 使用场景

当工具需要知道"这次调用的 ID 是什么"时——比如构造 `ToolMessage` 响应、日志追踪、或异步回调。

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId

@tool
def async_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],  # 自动注入调用 ID
) -> str:
    """异步搜索，返回带调用 ID 的结果。"""
    # 可以用 tool_call_id 做日志追踪
    print(f"[{tool_call_id}] 开始搜索: {query}")
    result = f"搜索结果: {query}"
    print(f"[{tool_call_id}] 搜索完成")
    return result

# LLM 看到的 Schema 中没有 tool_call_id
# 运行时自动从 ToolCall 对象中提取 id 并注入
```

### InjectedToolArg vs InjectedToolCallId

| 特性 | InjectedToolArg | InjectedToolCallId |
|------|----------------|-------------------|
| 注入内容 | 任意值（由调用方提供） | 当前 tool_call 的 ID |
| 注入来源 | 系统代码 / state | ToolCall 对象 |
| 典型用途 | user_id, session, config | 日志追踪, 响应关联 |
| LLM 可见 | 否 | 否 |

---

## 4. extras — 提供商特定配置

### 一句话定义

**`extras` 是一个字典，用于传递不适合标准 Tool 字段的提供商特定配置——比如 Anthropic 的缓存控制、输入示例等。**

### 为什么需要？

不同 LLM 提供商对工具有不同的扩展能力：

| 提供商 | 扩展配置 | 用途 |
|--------|---------|------|
| Anthropic | `cache_control` | 控制工具定义的缓存策略 |
| Anthropic | `input_examples` | 提供参数调用示例 |

这些配置不属于 OpenAI 标准的 `function` 格式，放在 `extras` 中透传给对应提供商。

### 代码示例

```python
from langchain_core.tools import tool

# Anthropic 缓存控制：让工具定义被缓存，减少重复传输成本
@tool(extras={"cache_control": {"type": "ephemeral"}})
def search_kb(query: str) -> str:
    """在知识库中搜索相关文档。"""
    return f"搜索结果: {query}"

# 带输入示例：帮助模型理解如何调用
@tool(extras={
    "input_examples": [
        {"query": "什么是 RAG？"},
        {"query": "如何优化检索质量？"},
    ]
})
def rag_search(query: str) -> str:
    """RAG 知识库检索。"""
    return f"结果: {query}"
```

### 源码原理

[来源: `langchain_core/tools/base.py`]

```python
class BaseTool(RunnableSerializable):
    extras: Optional[dict] = None  # 提供商特定配置

    # 在序列化为 API 格式时，extras 会被合并到工具定义中
    # 具体如何合并取决于各提供商的 bind_tools() 实现
```

`extras` 的值在 `convert_to_openai_tool()` 等转换函数中被透传，各提供商的 ChatModel 实现负责解析和应用。

---

## 5. return_direct — 跳过模型二次处理

### 一句话定义

**`return_direct=True` 让工具结果直接返回给用户，不再经过模型的二次推理和润色。**

### 两种模式对比

```
return_direct=False（默认）：
用户提问 → 模型决定调用工具 → 工具执行 → 结果传回模型 → 模型润色后回答用户

return_direct=True：
用户提问 → 模型决定调用工具 → 工具执行 → 结果直接返回用户（跳过模型）
```

### 什么时候用？

| 场景 | return_direct | 原因 |
|------|--------------|------|
| 计算器 | `True` | 计算结果精确，不需要模型润色 |
| 格式化输出 | `True` | 工具已经生成了最终格式 |
| 数据库查询 | `True` | 结构化数据直接展示 |
| RAG 检索 | `False` | 需要模型基于检索结果生成回答 |
| 天气查询 | `False` | 需要模型用自然语言组织回答 |

### 代码示例

```python
from langchain_core.tools import tool

# 计算器：结果精确，不需要模型加工
@tool(return_direct=True)
def calculator(expression: str) -> str:
    """计算数学表达式，返回精确结果。"""
    try:
        result = eval(expression)  # 生产环境请用 ast.literal_eval 或 sympy
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

# RAG 检索：需要模型基于结果生成回答
@tool(return_direct=False)  # 默认值，可以不写
def search_kb(query: str) -> str:
    """搜索知识库获取相关文档片段。"""
    return "RAG 系统由检索器、向量库和生成器三部分组成..."
```

### 源码原理

[来源: `langchain_core/tools/base.py`]

```python
class BaseTool(RunnableSerializable):
    return_direct: bool = False
    # Agent 执行循环中检查此标志：
    # if tool.return_direct:
    #     return tool_result  # 直接返回，不再调用模型
    # else:
    #     messages.append(ToolMessage(content=tool_result))
    #     continue  # 继续调用模型
```

---

## 6. handle_tool_error — 执行错误容错

### 一句话定义

**`handle_tool_error` 控制工具执行失败时的行为——是抛异常让 Agent 崩溃，还是返回错误信息让 Agent 自行修正。**

### 四种取值

| 值 | 类型 | 行为 | 适用场景 |
|---|------|------|---------|
| `False` | bool | 抛出异常，Agent 停止 | 开发调试 |
| `True` | bool | 返回异常信息字符串 | 生产环境（推荐） |
| `"自定义消息"` | str | 返回指定的错误消息 | 用户友好提示 |
| `callable` | Callable | 调用自定义函数处理 | 复杂错误处理逻辑 |

### 代码示例

```python
from langchain_core.tools import tool, ToolException

# ===== False（默认）：抛异常 =====
@tool
def risky_tool_v1(query: str) -> str:
    """可能失败的工具"""
    raise ToolException("API 超时")
# → 直接抛出 ToolException，Agent 崩溃

# ===== True：返回错误信息字符串 =====
@tool(handle_tool_error=True)
def risky_tool_v2(query: str) -> str:
    """可能失败的工具"""
    raise ToolException("API 超时")
# → 返回 "API 超时"，Agent 看到错误后可以重试或换工具

# ===== 自定义字符串：友好提示 =====
@tool(handle_tool_error="搜索服务暂时不可用，请稍后重试或换个问法。")
def risky_tool_v3(query: str) -> str:
    """可能失败的工具"""
    raise ToolException("API 超时")
# → 返回 "搜索服务暂时不可用，请稍后重试或换个问法。"

# ===== Callable：自定义处理逻辑 =====
def custom_error_handler(error: ToolException) -> str:
    """根据错误类型返回不同提示"""
    error_msg = str(error)
    if "超时" in error_msg:
        return "服务响应超时，建议缩小搜索范围后重试。"
    elif "权限" in error_msg:
        return "没有访问权限，请联系管理员。"
    else:
        return f"工具执行出错: {error_msg}"

@tool(handle_tool_error=custom_error_handler)
def risky_tool_v4(query: str) -> str:
    """可能失败的工具"""
    raise ToolException("API 超时")
# → 返回 "服务响应超时，建议缩小搜索范围后重试。"
```

### 源码原理

[来源: `langchain_core/tools/base.py`]

```python
class BaseTool(RunnableSerializable):
    handle_tool_error: Union[bool, str, Callable] = False

    def _run_and_format(self, *args, **kwargs):
        try:
            result = self._run(*args, **kwargs)
        except ToolException as e:
            if not self.handle_tool_error:
                raise e                              # False → 抛异常
            elif isinstance(self.handle_tool_error, bool):
                return str(e)                        # True → 返回错误字符串
            elif isinstance(self.handle_tool_error, str):
                return self.handle_tool_error        # str → 返回自定义消息
            elif callable(self.handle_tool_error):
                return self.handle_tool_error(e)     # Callable → 调用处理函数
```

**关键点**：只有 `ToolException` 会被 `handle_tool_error` 捕获。普通的 `Exception`、`ValueError` 等不会被捕获——它们仍然会导致 Agent 崩溃。这是有意为之的设计：工具开发者应该主动抛出 `ToolException` 来表示"可恢复的工具错误"。

---

## 7. handle_validation_error — 验证错误容错

### 一句话定义

**`handle_validation_error` 控制参数验证失败时的行为——当 LLM 生成的参数不符合 Schema 约束时，是崩溃还是返回提示让 LLM 修正。**

### 与 handle_tool_error 的区别

| 特性 | handle_validation_error | handle_tool_error |
|------|----------------------|-------------------|
| 触发时机 | 参数验证阶段（执行前） | 工具执行阶段（执行中） |
| 捕获异常 | `ValidationError` | `ToolException` |
| 典型原因 | LLM 传了非法参数 | 外部服务失败、数据异常 |
| 四种取值 | 同 handle_tool_error | 同 |

### 代码示例

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(min_length=1, description="搜索关键词")
    top_k: int = Field(ge=1, le=20, description="返回数量")

@tool(args_schema=SearchInput, handle_validation_error=True)
def search(query: str, top_k: int = 5) -> str:
    """搜索知识库"""
    return f"搜索 '{query}'，返回 {top_k} 条"

# LLM 传了非法参数
result = search.invoke({"query": "", "top_k": 100})
# → 返回验证错误信息（而不是崩溃）：
# "1 validation error for SearchInput\nquery\n  String should have at least 1 character..."
# Agent 看到这个错误后，会修正参数重试
```

### 生产最佳实践

```python
# 推荐：同时开启两种错误处理
@tool(
    args_schema=SearchInput,
    handle_validation_error=True,   # 参数错误 → 返回提示
    handle_tool_error=True,         # 执行错误 → 返回提示
)
def production_search(query: str, top_k: int = 5) -> str:
    """生产级搜索工具"""
    try:
        results = vector_store.search(query, top_k=top_k)
        return str(results)
    except Exception as e:
        # 将普通异常转为 ToolException，才能被 handle_tool_error 捕获
        from langchain_core.tools import ToolException
        raise ToolException(f"搜索失败: {e}")
```

---

## 8. ToolException — 工具专用异常

### 一句话定义

**`ToolException` 是 LangChain 定义的专用异常类，只有它会被 `handle_tool_error` 捕获，用于区分"可恢复的工具错误"和"不可恢复的系统错误"。**

### 为什么不直接用 Exception？

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def demo_tool(query: str) -> str:
    """演示异常处理"""
    if query == "tool_error":
        raise ToolException("工具错误")    # ✅ 被 handle_tool_error 捕获
    if query == "system_error":
        raise ValueError("系统错误")       # ❌ 不被捕获，Agent 崩溃
    return "正常结果"
```

设计意图：

| 异常类型 | 含义 | 处理方式 |
|----------|------|---------|
| `ToolException` | 可恢复的业务错误（API 超时、无结果） | `handle_tool_error` 捕获，返回提示 |
| `Exception` 等 | 不可恢复的系统错误（代码 bug、配置错误） | 直接抛出，需要开发者修复 |

### 实战模式：异常转换

```python
from langchain_core.tools import tool, ToolException

@tool(handle_tool_error=True)
def robust_search(query: str) -> str:
    """健壮的搜索工具"""
    try:
        # 可能抛出各种异常的外部调用
        response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
        response.raise_for_status()
        return response.json()["results"]
    except requests.Timeout:
        raise ToolException("搜索服务响应超时，请缩小查询范围后重试")
    except requests.HTTPError as e:
        raise ToolException(f"搜索服务返回错误: {e.response.status_code}")
    except Exception as e:
        # 未知错误也转为 ToolException，避免 Agent 崩溃
        raise ToolException(f"搜索出现未知错误: {str(e)}")
```

---

## 9. 全特性对比表

| 特性 | 作用 | 默认值 | 典型场景 |
|------|------|--------|---------|
| `response_format` | 控制输出格式 | `"content"` | 搜索结果分离（摘要给模型，全文给用户） |
| `InjectedToolArg` | 隐藏参数注入 | — | user_id、session、db 连接 |
| `InjectedToolCallId` | 注入调用 ID | — | 日志追踪、异步回调 |
| `extras` | 提供商扩展配置 | `None` | Anthropic cache_control |
| `return_direct` | 跳过模型润色 | `False` | 计算器、格式化输出 |
| `handle_tool_error` | 执行错误容错 | `False` | 生产环境必开 |
| `handle_validation_error` | 验证错误容错 | `False` | 生产环境必开 |
| `ToolException` | 专用异常类 | — | 配合 handle_tool_error |

---

## 10. 综合实战：生产级 RAG 搜索工具

把所有高级特性组合在一起，构建一个生产级的 RAG 搜索工具：

```python
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolArg, ToolException


class RAGSearchInput(BaseModel):
    """RAG 知识库检索参数"""
    query: str = Field(min_length=1, max_length=500, description="自然语言搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回文档数量")


@tool(
    args_schema=RAGSearchInput,
    response_format="content_and_artifact",   # 摘要给模型，完整结果给用户
    handle_tool_error=True,                   # 执行错误不崩溃
    handle_validation_error=True,             # 验证错误不崩溃
)
def rag_search(
    query: str,
    top_k: int = 5,
    user_id: Annotated[str, InjectedToolArg] = "anonymous",  # 系统注入，LLM 不可见
) -> tuple[str, list[dict]]:
    """在 RAG 知识库中检索相关文档片段。

    根据用户的自然语言查询，在向量知识库中进行语义搜索，
    返回最相关的文档片段及其相似度分数。
    """
    try:
        # 模拟向量检索（实际替换为真实的向量库调用）
        results = [
            {"chunk_id": f"doc_{i}", "text": f"文档片段 {i} 的内容...",
             "score": round(0.95 - i * 0.05, 2), "user_id": user_id}
            for i in range(top_k)
        ]

        # 摘要给模型（省 token）
        summary_lines = []
        for r in results:
            summary_lines.append(f"- [{r['chunk_id']}] (相似度 {r['score']}): {r['text'][:50]}")
        summary = f"为用户 {user_id} 找到 {len(results)} 个相关片段：\n" + "\n".join(summary_lines)

        # 完整结果给用户
        return summary, results

    except ConnectionError:
        raise ToolException("向量数据库连接失败，请检查 Milvus/ChromaDB 服务状态")
    except Exception as e:
        raise ToolException(f"检索出错: {str(e)}")


# 验证：LLM 看到的 Schema
print("LLM 可见参数:", list(rag_search.tool_call_schema.model_json_schema()["properties"].keys()))
# → ['query', 'top_k']  （user_id 被隐藏）

# 验证：调用
result = rag_search.invoke({
    "query": "什么是 RAG？",
    "top_k": 3,
    "user_id": "user_12345",  # 由系统注入
})
print(type(result))  # ToolMessage
```

### 这个工具用到了哪些高级特性？

```
1. args_schema=RAGSearchInput        → Pydantic 参数验证（min_length, ge, le）
2. response_format="content_and_artifact" → 摘要给模型，完整结果给用户
3. handle_tool_error=True             → 执行错误返回提示，不崩溃
4. handle_validation_error=True       → 参数错误返回提示，不崩溃
5. InjectedToolArg (user_id)          → 用户 ID 由系统注入，LLM 不可见
6. ToolException                      → 区分可恢复错误和系统错误
```

---

## 关键源码映射

| 概念 | 源码位置 | 说明 |
|------|----------|------|
| `response_format` | `langchain_core/tools/base.py` | BaseTool 属性，控制输出格式 |
| `return_direct` | `langchain_core/tools/base.py` | BaseTool 属性，跳过模型润色 |
| `handle_tool_error` | `langchain_core/tools/base.py` | BaseTool 属性，执行错误处理 |
| `handle_validation_error` | `langchain_core/tools/base.py` | BaseTool 属性，验证错误处理 |
| `extras` | `langchain_core/tools/base.py` | BaseTool 属性，提供商扩展 |
| `InjectedToolArg` | `langchain_core/tools/base.py` | 标记类，隐藏参数 |
| `InjectedToolCallId` | `langchain_core/tools/base.py` | 标记类，注入调用 ID |
| `ToolException` | `langchain_core/tools/base.py` | 专用异常类 |
| `tool_call_schema` | `langchain_core/tools/base.py` | 过滤注入参数后的 Schema |
| `_get_filtered_args()` | `langchain_core/tools/base.py` | 过滤 InjectedToolArg 字段 |

---

## 总结

高级 Tool 特性解决的是从"能用"到"生产可用"的最后一公里：

1. **`response_format="content_and_artifact"`** — 内容与工件分离，摘要给模型省 token，完整数据给用户展示
2. **`InjectedToolArg`** — 安全参数注入，user_id 等敏感参数对 LLM 不可见
3. **`InjectedToolCallId`** — 自动注入调用 ID，用于日志追踪和响应关联
4. **`extras`** — 提供商特定配置的透传通道（如 Anthropic 缓存控制）
5. **`return_direct`** — 精确结果直接返回用户，跳过模型不必要的润色
6. **`handle_tool_error` + `ToolException`** — 可恢复错误优雅降级，Agent 不崩溃
7. **`handle_validation_error`** — 参数验证失败返回提示，LLM 可自行修正重试

生产环境的最低配置：`handle_tool_error=True` + `handle_validation_error=True`。这两个开关能让 Agent 从"遇到错误就崩溃"变成"遇到错误会自我修正"。

---

**上一篇**: [03_核心概念_5_工具选择策略.md](./03_核心概念_5_工具选择策略.md) — 动态工具过滤与 Middleware 模式
**下一篇**: [04_最小可用.md](./04_最小可用.md) — 最小可运行的 Tool 示例
