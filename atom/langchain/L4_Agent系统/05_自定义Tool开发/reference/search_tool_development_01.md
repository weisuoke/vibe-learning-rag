---
type: search_result
search_query: LangChain custom tool development 2025 2026 best practices @tool decorator BaseTool
search_engine: grok-mcp
searched_at: 2026-03-02
knowledge_point: 自定义Tool开发
---

# 搜索结果：LangChain 自定义工具开发最佳实践

## 搜索摘要
搜索关键词：LangChain custom tool development 2025 2026 best practices @tool decorator BaseTool
平台：GitHub, Reddit, Twitter/X
结果数量：8 个高质量资源

## 相关链接

### GitHub Issues 和源码

1. [PraisonAI-Tools BaseTool和@tool装饰器](https://github.com/MervinPraison/PraisonAI-Tools)
   - 提供BaseTool基类与@tool装饰器创建自定义工具的包
   - 包含元数据设置和复杂工具示例
   - 适用于LangChain风格开发

2. [LangChain react agent工具参数描述问题](https://github.com/langchain-ai/langchain/issues/31070)
   - 2025年4月issue
   - 探讨BaseTool和工具装饰器在create_react_agent中丢失参数描述的问题
   - 提供解决方案

3. [LangGraph自定义工具状态访问问题](https://github.com/langchain-ai/langgraph/issues/1777)
   - BaseTool子类自定义工具无法通过run_manager访问状态
   - @tool装饰器支持状态访问
   - 社区讨论对比两种方法

4. [LangGraph prebuilt 1.0.2破坏性变更](https://github.com/langchain-ai/langgraph/issues/6363)
   - 2025年10月更新影响自定义工具执行逻辑
   - 推荐固定依赖版本以避免兼容问题的最佳实践

5. [LangGraph ToolRuntime自定义工具支持](https://github.com/langchain-ai/langgraph/issues/6318)
   - 2025年issue
   - 讨论@tool装饰器中ToolRuntime参数在内置ToolNode中的支持问题
   - 提供workaround方案

### Reddit 社区讨论

6. [LangChain自定义多参数工具开发](https://www.reddit.com/r/LangChain/comments/1k0adul/custom_tools_with_multiple_parameters/)
   - 社区分享使用@tool或BaseTool实现多参数自定义工具
   - 代码示例和最佳方法

7. [LangGraph Ollama Agent工具设置](https://www.reddit.com/r/LangChain/comments/1k6oi7j/langgraph_ollama_agent_using_local_model_qwen25/)
   - 强调正确使用@tool装饰器创建工具
   - 确保本地模型LangGraph agent正常工作

### Twitter/X 官方动态

8. [LangGraph BigTool大规模工具库](https://x.com/LangChain/status/1897021019203711338)
   - 2025年官方推文
   - 介绍支持数百工具的LangGraph库
   - 自定义工具管理和检索的最佳实践

## 关键信息提取

### 1. @tool 装饰器 vs BaseTool 对比（来自 GitHub #1777）

**@tool 装饰器优势**：
- 自动支持状态访问（通过 InjectedToolArg）
- 代码更简洁
- 自动 schema 推断
- 推荐用于大多数场景

**BaseTool 子类优势**：
- 更多控制权
- 适合复杂状态管理
- 可以重写更多方法
- 适合需要初始化/清理的工具

**社区共识**：
- 简单工具优先使用 @tool
- 复杂工具考虑 BaseTool
- LangGraph 中 @tool 对状态访问支持更好

### 2. 参数描述丢失问题（来自 GitHub #31070）

**问题描述**：
- create_react_agent 中工具参数描述可能丢失
- 影响 LLM 理解工具用法

**解决方案**：
- 使用 `parse_docstring=True` 参数
- 确保 docstring 格式正确（Google-style）
- 显式提供 `args_schema`

### 3. 2025-2026 破坏性变更（来自 GitHub #6363）

**LangGraph 1.0.2 变更**：
- 影响自定义工具执行逻辑
- 可能导致现有工具失效

**最佳实践**：
- 固定依赖版本（使用 `==` 而非 `>=`）
- 测试工具在新版本中的兼容性
- 关注官方 changelog

### 4. ToolRuntime 参数支持（来自 GitHub #6318）

**问题**：
- @tool 装饰器中的 ToolRuntime 参数在 ToolNode 中不被识别

**Workaround**：
- 使用自定义 ToolNode
- 或使用 InjectedToolArg 替代

### 5. 多参数工具开发（来自 Reddit）

**最佳实践**：
```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class MultiParamInput(BaseModel):
    param1: str = Field(description="First parameter")
    param2: int = Field(description="Second parameter")
    param3: list[str] = Field(description="Third parameter")

@tool(args_schema=MultiParamInput)
def multi_param_tool(param1: str, param2: int, param3: list[str]) -> str:
    """Tool with multiple parameters."""
    return f"Processed: {param1}, {param2}, {param3}"
```

**关键点**：
- 使用 Pydantic BaseModel 定义 schema
- 每个参数都要有 Field 描述
- 类型提示必须与 schema 一致

### 6. 本地模型工具集成（来自 Reddit）

**Ollama + LangGraph 最佳实践**：
- 确保工具描述清晰简洁
- 使用 @tool 装饰器而非手动定义
- 测试工具在本地模型上的表现
- 本地模型对工具调用格式更敏感

### 7. 大规模工具管理（来自 Twitter/X）

**LangGraph BigTool 库特性**：
- 支持数百个工具
- 工具检索和过滤
- 工具元数据管理
- 适合企业级应用

**应用场景**：
- 多租户系统
- 工具市场
- 动态工具加载

### 8. PraisonAI-Tools 库（来自 GitHub）

**特性**：
- 提供 BaseTool 和 @tool 的统一接口
- 丰富的工具示例
- 元数据和配置管理

**适用场景**：
- 快速原型开发
- 学习工具开发模式
- 复用常见工具模板

## 2025-2026 趋势总结

### 技术趋势
1. **@tool 装饰器成为主流**：更简洁，更易用
2. **LangGraph 集成深化**：工具与状态管理紧密结合
3. **大规模工具管理**：企业级工具库和检索
4. **本地模型支持**：Ollama 等本地模型的工具调用优化

### 最佳实践演进
1. **优先使用 @tool**：除非需要复杂状态管理
2. **显式 schema 定义**：使用 Pydantic 提升可靠性
3. **版本固定**：避免破坏性变更影响
4. **充分测试**：特别是本地模型场景

### 常见陷阱
1. 参数描述丢失（需要 parse_docstring=True）
2. 状态访问问题（BaseTool 在 LangGraph 中的限制）
3. 版本兼容性（LangGraph 1.0.2+ 变更）
4. ToolRuntime 参数不被识别（需要 workaround）

## 待抓取链接（需要更多细节）

以下链接包含实践案例和代码示例，建议抓取完整内容：

1. https://github.com/MervinPraison/PraisonAI-Tools - 工具库源码和示例
2. https://www.reddit.com/r/LangChain/comments/1k0adul/custom_tools_with_multiple_parameters/ - 多参数工具完整讨论
3. https://www.reddit.com/r/LangChain/comments/1k6oi7j/langgraph_ollama_agent_using_local_model_qwen25/ - 本地模型集成案例
