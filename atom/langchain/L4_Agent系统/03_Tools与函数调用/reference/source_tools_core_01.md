---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/tools/base.py
  - libs/core/langchain_core/tools/convert.py
  - libs/core/langchain_core/tools/structured.py
  - libs/core/langchain_core/tools/simple.py
  - libs/core/langchain_core/tools/render.py
  - libs/core/langchain_core/tools/retriever.py
  - libs/core/langchain_core/tools/__init__.py
  - libs/core/langchain_core/messages/tool.py
  - libs/core/langchain_core/utils/function_calling.py
analyzed_at: 2026-02-28
knowledge_point: 03_Tools与函数调用
---

# 源码分析：LangChain Tools 核心系统

## 分析的文件

### 1. `libs/core/langchain_core/tools/base.py` (55KB)
- **BaseTool**: 所有 Tool 的抽象基类，继承自 `RunnableSerializable`
- **BaseToolkit**: 工具集合的基类
- **ToolException**: 工具错误异常类
- **InjectedToolArg**: 注入参数的类型注解
- **InjectedToolCallId**: 工具调用 ID 的类型注解
- **SchemaAnnotationError**: Schema 注解错误
- **关键属性**: name, description, args_schema, return_direct, response_format, extras
- **关键方法**: invoke(), ainvoke(), _run(), _arun(), tool_call_schema, args
- **Schema 推断**: 从函数签名自动推断参数 Schema
- **Pydantic v1/v2 双支持**

### 2. `libs/core/langchain_core/tools/convert.py` (477行)
- **@tool 装饰器**: 将函数/Runnable 转换为 Tool
  - 支持有参数和无参数两种装饰器用法
  - 参数: description, return_direct, args_schema, infer_schema, response_format, parse_docstring, extras
  - 自动从函数签名推断 Schema
  - 支持同步和异步函数
  - 支持 Runnable 对象（需要字符串名称）
- **convert_runnable_to_tool()**: 将 Runnable 转换为 Tool
- **4种 @tool 重载签名**:
  1. `@tool` - 无参数装饰器
  2. `@tool("name")` - 带名称装饰器
  3. `@tool(parse_docstring=True)` - 带参数装饰器
  4. `tool("name", runnable)` - 函数调用方式

### 3. `libs/core/langchain_core/tools/structured.py` (272行)
- **StructuredTool**: 支持多参数的 Tool 实现
  - 必须有 args_schema (Pydantic BaseModel 或 dict)
  - `from_function()` 类方法：从函数创建 Tool
  - 支持 parse_docstring 从 Google-style docstring 提取参数描述
  - 支持 response_format ("content" 或 "content_and_artifact")

### 4. `libs/core/langchain_core/tools/simple.py` (205行)
- **Tool**: 简单的单参数 Tool 实现
  - 包装单个函数或协程
  - 默认接受单个字符串输入
  - 向后兼容旧代码

### 5. `libs/core/langchain_core/tools/render.py` (68行)
- **render_text_description()**: 渲染工具名称和描述
- **render_text_description_and_args()**: 渲染工具名称、描述和参数 Schema
- **ToolsRenderer**: 渲染函数的类型别名

### 6. `libs/core/langchain_core/messages/tool.py`
- **ToolCall**: TypedDict，表示模型请求调用工具 (name, args, id)
- **ToolMessage**: 工具执行结果消息 (content, tool_call_id, artifact, status)
- **ToolCallChunk**: 流式传输时的工具调用块
- **InvalidToolCall**: 无效的工具调用
- **ToolOutputMixin**: 工具可直接返回的对象混入

### 7. `libs/core/langchain_core/utils/function_calling.py` (813行)
- **convert_to_openai_tool()**: 将 Tool 转换为 OpenAI 格式
- **convert_to_openai_function()**: 转换为 OpenAI function 格式
- **convert_to_json_schema()**: 转换为 JSON Schema
- **tool_example_to_messages()**: 将工具示例转换为消息列表
- **支持多种输入格式**: dict, BaseModel, TypedDict, BaseTool, Callable
- **支持 strict 模式**: OpenAI 严格模式
- **支持多种工具格式**: OpenAI, Anthropic, Amazon Bedrock Converse

## 关键发现

### Tool 定义三种模式
1. **@tool 装饰器** (推荐): 最简洁，自动推断 Schema
2. **StructuredTool.from_function()**: 适合需要自定义 Schema 的场景
3. **BaseTool 子类**: 最灵活，适合复杂工具

### 函数调用协议流程
1. 定义 Tool → 2. bind_tools() 绑定到模型 → 3. 模型返回 AIMessage(tool_calls) → 4. 执行 Tool → 5. 返回 ToolMessage → 6. 模型生成最终回答

### Schema 转换链
Tool → tool_call_schema → convert_to_openai_tool() → OpenAI API 格式

### 2026 新特性
- **extras**: 提供商特定的额外字段（如 Anthropic cache_control）
- **Middleware 工具选择**: LLMToolSelectorMiddleware 动态选择工具
- **response_format**: content_and_artifact 支持返回内容和工件
- **InjectedToolArg/InjectedToolCallId**: 自动注入参数
