---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/tools/
analyzed_files:
  - base.py
  - convert.py
  - structured.py
  - simple.py
analyzed_at: 2026-03-02
knowledge_point: 自定义Tool开发
---

# 源码分析：LangChain Tool 开发核心机制

## 分析的文件

- `langchain_core/tools/base.py` - BaseTool 基类和工具基础设施
- `langchain_core/tools/convert.py` - @tool 装饰器实现
- `langchain_core/tools/structured.py` - StructuredTool 实现
- `langchain_core/tools/__init__.py` - 公共 API 导出

## 关键发现

### 1. 三种工具创建方式的架构设计

#### @tool 装饰器（推荐方式）
```python
# 源码位置: convert.py:76-390
def tool(
    name_or_callable: str | Callable | None = None,
    runnable: Runnable | None = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable | Runnable], BaseTool]:
```

**设计亮点**：
- 支持多种调用模式（装饰器无参、装饰器带参、函数调用）
- 自动推断 schema（通过 `create_schema_from_function`）
- 支持 Google-style docstring 解析
- 底层委托给 `StructuredTool.from_function()`

#### StructuredTool（灵活方式）
```python
# 源码位置: structured.py:40-272
class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Annotated[ArgsSchema, SkipValidation()] = Field(...)
    func: Callable[..., Any] | None = None
    coroutine: Callable[..., Awaitable[Any]] | None = None
```

**设计亮点**：
- 同时支持同步和异步函数
- 通过 `from_function()` 类方法创建
- 自动处理 callback 和 config 注入
- 支持 `response_format` 控制返回格式

#### BaseTool（最大控制）
```python
# 源码位置: base.py（抽象基类）
class BaseTool(RunnableSerializable[Union[str, Dict, ToolCall], Any]):
    """Base class for tools."""

    @abstractmethod
    def _run(self, *args, config: RunnableConfig, **kwargs) -> Any:
        """Use the tool."""

    async def _arun(self, *args, config: RunnableConfig, **kwargs) -> Any:
        """Use the tool asynchronously."""
```

**设计亮点**：
- 继承自 `RunnableSerializable`，完全集成 LCEL
- 必须实现 `_run()` 方法
- 可选实现 `_arun()` 方法（异步）
- 适合需要状态管理的复杂工具

### 2. Schema 自动推断机制

#### create_schema_from_function 核心逻辑
```python
# 源码位置: base.py:289-387
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

**工作流程**：
1. 使用 `inspect.signature()` 获取函数签名
2. 检测 Pydantic v1/v2 注解（不允许混用）
3. 使用 `validate_arguments()` 创建 Pydantic 模型
4. 过滤特殊参数（`run_manager`, `callbacks`, `self`, `cls`）
5. 解析 docstring 提取参数描述（可选）
6. 创建子集模型（`_create_subset_model`）

**关键常量**：
```python
FILTERED_ARGS = ("run_manager", "callbacks")
```

### 3. 参数注入机制（InjectedToolArg）

#### 支持的注入类型
```python
# 从源码推断的注入参数类型
- InjectedToolArg: 标记需要注入的参数
- InjectedToolCallId: 注入工具调用 ID
- RunnableConfig: 注入运行配置
```

**注入检测逻辑**：
```python
# 源码位置: base.py:126-152
def _get_filtered_args(
    inferred_model: type[BaseModel],
    func: Callable,
    *,
    filter_args: Sequence[str],
    include_injected: bool = True,
) -> dict:
    """Get filtered arguments from a function's signature."""
    # 检查参数是否为注入类型
    if not include_injected and _is_injected_arg_type(param.annotation):
        filter_args_.append(existing_param)
```

### 4. Response Format 机制

#### 两种响应格式
```python
response_format: Literal["content", "content_and_artifact"] = "content"
```

**"content" 模式**：
- 工具返回值直接作为 `ToolMessage.content`
- 适合简单的字符串或 JSON 返回

**"content_and_artifact" 模式**：
- 工具返回 `tuple[str, Any]`
- 第一个元素：`ToolMessage.content`（给 LLM 看的摘要）
- 第二个元素：`ToolMessage.artifact`（完整数据，给程序用）

### 5. Docstring 解析机制

#### Google-style Docstring 支持
```python
# 源码位置: base.py:155-175
def _parse_python_function_docstring(
    function: Callable,
    annotations: dict,
    *,
    error_on_invalid_docstring: bool = False
) -> tuple[str, dict]:
    """Parse function and argument descriptions from a docstring."""
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(
        docstring,
        list(annotations),
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
```

**解析规则**：
- 必须有空行分隔 summary 和 Args 部分
- Args 部分格式：`arg_name: description`
- 验证 docstring 参数与函数签名一致

### 6. 异步支持设计

#### StructuredTool 的异步处理
```python
# 源码位置: structured.py:60-70
async def ainvoke(
    self,
    input: str | dict | ToolCall,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Any:
    if not self.coroutine:
        # 如果没有异步实现，回退到线程池执行同步版本
        return await run_in_executor(config, self.invoke, input, config, **kwargs)

    return await super().ainvoke(input, config, **kwargs)
```

**设计哲学**：
- 优先使用 `coroutine`（如果提供）
- 自动回退到 `run_in_executor`（线程池）
- 保证 API 一致性（同步/异步都可调用）

### 7. 错误处理机制

#### ToolException 特殊异常
```python
# 源码位置: base.py:390-397
class ToolException(Exception):
    """Exception thrown when a tool execution error occurs.

    This exception allows tools to signal errors without stopping the agent.

    The error is handled according to the tool's `handle_tool_error` setting,
    and the result is returned as an observation to the agent.
    """
```

**使用场景**：
- 工具执行失败但不想中断 Agent 循环
- 错误信息作为观察返回给 Agent
- 配合 `handle_tool_error` 配置使用

### 8. Extras 机制（Provider-specific）

#### 扩展字段支持
```python
extras: dict[str, Any] | None = None
```

**用途**：
- 传递特定 LLM 提供商的配置
- 例如 Anthropic 的 `cache_control`, `defer_loading`
- 不影响标准工具字段

## 架构设计总结

### 分层设计
```
@tool 装饰器 (用户友好层)
    ↓
StructuredTool.from_function() (中间层)
    ↓
BaseTool (核心抽象层)
    ↓
RunnableSerializable (LCEL 集成层)
```

### 关键设计模式

1. **工厂模式**：`StructuredTool.from_function()` 和 `@tool` 装饰器
2. **策略模式**：`response_format` 控制返回格式
3. **模板方法模式**：`BaseTool._run()` 和 `_arun()`
4. **装饰器模式**：`@tool` 的多种使用方式
5. **依赖注入**：`InjectedToolArg` 和 `RunnableConfig`

### 向后兼容性

- 同时支持 Pydantic v1 和 v2
- 自动检测并使用对应版本的 API
- 不允许在同一函数中混用 v1/v2 注解

## 实现建议

### 推荐使用顺序
1. **简单工具**：使用 `@tool` 装饰器
2. **需要显式 schema**：使用 `StructuredTool.from_function()`
3. **复杂状态管理**：继承 `BaseTool`

### 最佳实践（从源码推断）
- 始终提供类型提示（schema 推断依赖）
- 使用 Google-style docstring（启用 `parse_docstring=True`）
- 异步工具优先提供 `coroutine`
- 使用 `response_format="content_and_artifact"` 分离摘要和数据
- 使用 `ToolException` 处理可恢复错误
