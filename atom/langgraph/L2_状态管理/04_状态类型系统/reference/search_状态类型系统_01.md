---
type: search_result
search_query: LangGraph state type system TypedDict Pydantic 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-26
knowledge_point: 04_状态类型系统
---

# 搜索结果：LangGraph 状态类型系统

## 搜索摘要

搜索关键词：LangGraph state type system TypedDict Pydantic 2025 2026
搜索平台：GitHub, Reddit, Twitter
搜索结果数：10

## 相关链接

### 官方文档

1. [Graph API overview - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/graph-api)
   - LangGraph状态schema可使用TypedDict或Pydantic模型定义
   - TypedDict为主要推荐方式
   - 支持dataclass默认值
   - Pydantic用于需要递归验证的情况但性能较低

2. [Use the graph API - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
   - 详细说明LangGraph中状态定义支持TypedDict、Pydantic BaseModel或dataclass
   - TypedDict为默认轻量选择
   - Pydantic提供运行时输入验证

### 技术博客与教程

3. [Type Safety in LangGraph: When to Use TypedDict vs. Pydantic](https://shazaali.substack.com/p/type-safety-in-langgraph-when-to)
   - 比较TypedDict与Pydantic在LangGraph中的类型安全
   - TypedDict适合轻量无运行时验证的内部状态
   - Pydantic提供严格验证适用于需要可靠性的场景

4. [LangGraph: State with Pydantic BaseModel](https://medium.com/fundamentals-of-artificial-intelligence/langgraph-state-with-pydantic-basemodel-023a2158ab00)
   - 解释TypedDict与Pydantic BaseModel在LangGraph状态定义中的区别
   - TypedDict轻量灵活适合开发
   - Pydantic提供更严格的类型检查和验证

5. [LangGraph Best Practices](https://www.swarnendu.de/blog/langgraph-best-practices)
   - 推荐使用TypedDict或Pydantic定义最小化、类型化的状态
   - 仅在需要累积消息时添加reducer
   - 保持代码一致性

6. [langgraph-state skill by orchestkit](https://playbooks.com/skills/yonatangross/orchestkit/langgraph-state)
   - 指导LangGraph状态schema设计
   - 选择TypedDict用于内部轻量状态
   - Pydantic用于边界验证
   - 支持Annotated reducer实现累积不可变状态

7. [LangGraph state basics](https://medium.com/@koreymstafford/langgraph-state-basics-f2852b315849)
   - LangGraph状态通常使用TypedDict或Pydantic BaseModel
   - 支持reducer函数、messages channel及不同类型状态定义

### GitHub Issues

8. [typing.TypedDict is not supported with pydantic in Python < 3.12 - GitHub Issue](https://github.com/langchain-ai/langgraph/issues/2198)
   - 讨论Python版本低于3.12时TypedDict与Pydantic兼容问题
   - 建议使用typing_extensions.TypedDict避免错误

### 社交媒体讨论

9. [LangGraph just crossed 22M monthly downloads - X Post](https://x.com/pulsemarkai/status/2015416815959126158)
   - 介绍LangGraph构建agentic workflow基础步骤
   - 包括使用TypedDict定义状态、创建节点和添加checkpointing

10. [Running LangGraph in production - X Post](https://x.com/saen_dev/status/2024035657677488144)
    - 生产环境中LangGraph用于状态持久化
    - 结合Pydantic进行边缘验证
    - 适用于复杂stateful orchestration

## 关键信息提取

### 1. TypedDict vs Pydantic 选择建议

**TypedDict 适用场景**：
- 内部状态管理（轻量、快速）
- 不需要运行时验证的场景
- 开发阶段的快速迭代
- 性能敏感的应用

**Pydantic 适用场景**：
- 边界验证（API 输入、外部数据）
- 需要递归验证的复杂数据结构
- 需要严格类型检查的场景
- 生产环境的可靠性要求

### 2. 官方推荐

**LangChain 官方文档明确指出**：
- TypedDict 是主要推荐方式
- Pydantic 性能较低，仅在需要递归验证时使用
- 支持 dataclass 作为第三种选择

### 3. 最佳实践

**状态设计原则**：
- 最小化状态：只包含必要的字段
- 类型化：使用类型注解提高代码质量
- 一致性：在项目中保持统一的状态定义方式
- Reducer 使用：仅在需要累积数据时添加

**Reducer 函数**：
- 使用 `Annotated[type, reducer]` 绑定
- 支持累积不可变状态
- 常用于消息列表、历史记录等

### 4. 兼容性问题

**Python 版本兼容性**：
- Python < 3.12：使用 `typing_extensions.TypedDict` 避免与 Pydantic 的兼容问题
- Python >= 3.12：可以直接使用 `typing.TypedDict`

### 5. 生产环境实践

**状态持久化**：
- LangGraph 支持 checkpointing
- 结合 Pydantic 进行边缘验证
- 适用于复杂的 stateful orchestration

**性能考虑**：
- TypedDict 性能优于 Pydantic
- 在高并发场景下，TypedDict 是更好的选择
- Pydantic 的验证开销在边界处是值得的

### 6. 社区趋势

**采用情况**：
- LangGraph 月下载量超过 22M（2025 年数据）
- 社区普遍推荐 TypedDict 作为默认选择
- Pydantic 在需要严格验证的场景下使用

**常见模式**：
- 内部状态使用 TypedDict
- API 边界使用 Pydantic
- 混合使用两种方式

## 实践案例

### 案例 1：基础 TypedDict 状态

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph

def add_messages(left: list, right: list) -> list:
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

graph = StateGraph(State)
```

### 案例 2：Pydantic 边界验证

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

class InputState(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')

class InternalState(TypedDict):
    messages: list
    context: dict

# 使用 Pydantic 验证输入
# 使用 TypedDict 管理内部状态
```

### 案例 3：混合使用

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

# 外部 API 使用 Pydantic
class APIInput(BaseModel):
    query: str
    user_id: str

# 内部状态使用 TypedDict
class InternalState(TypedDict):
    messages: list
    context: dict

def process_input(api_input: APIInput) -> InternalState:
    # 验证后转换为内部状态
    return {
        "messages": [api_input.query],
        "context": {"user_id": api_input.user_id}
    }
```

## 常见问题

### Q1: 为什么 TypedDict 是推荐方式？

**A**:
- 性能：TypedDict 没有运行时验证开销
- 简单：语法简洁，易于理解
- 灵活：支持 Annotated reducer
- 兼容：与 Python 类型系统无缝集成

### Q2: 什么时候必须使用 Pydantic？

**A**:
- 需要递归验证复杂数据结构
- API 边界需要严格验证
- 需要自定义验证器
- 需要数据转换和清洗

### Q3: 如何处理 Python < 3.12 的兼容性问题？

**A**:
- 使用 `typing_extensions.TypedDict` 而非 `typing.TypedDict`
- 确保 `typing_extensions` 版本 >= 4.0.0
- 避免混合使用 `typing.TypedDict` 和 Pydantic

### Q4: Reducer 函数如何工作？

**A**:
- Reducer 函数接受两个参数：当前值和新值
- 返回合并后的值
- 使用 `Annotated[type, reducer]` 绑定到字段
- 常用于累积列表、合并字典等

## 参考资源

- LangChain 官方文档：https://docs.langchain.com/oss/python/langgraph/
- LangGraph GitHub：https://github.com/langchain-ai/langgraph
- Pydantic 文档：https://docs.pydantic.dev/
- typing_extensions 文档：https://github.com/python/typing_extensions
