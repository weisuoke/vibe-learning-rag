---
type: context7_documentation
library: langgraph
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 12_状态管理最佳实践
context7_query: state management best practices, state schema design, reducer patterns, performance optimization
---

# Context7 文档：LangGraph 状态管理

## 文档来源
- 库名称：LangGraph
- Context7 ID：/websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph

## 关键信息提取

### 1. State Schema 定义

**TypedDict 方式**：
```python
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int
```

**Pydantic 方式**：
```python
from pydantic import BaseModel

class ComplexState(BaseModel):
    text: str
    count: int
    nested: NestedModel
```

### 2. Reducer 模式

- 每个 key 可以有独立的 reducer 函数
- 无 reducer → 覆盖更新
- `Annotated[T, reducer]` → 自定义更新逻辑
- `add_messages` 是消息列表的标准 reducer

### 3. Pydantic 集成

- 节点接收验证后的 Pydantic 对象
- 返回字典更新（非完整对象）
- 支持嵌套模型序列化/反序列化
- 输出可转回 Pydantic 模型

### 4. 状态设计原则（官方文档）

- State 是图中所有节点和边的输入 schema
- 节点发出状态更新，通过 reducer 应用
- Schema 可以是 TypedDict 或 Pydantic model
- 每个 key 的 reducer 独立控制更新方式
