---
type: context7_documentation
library: langgraph
version: latest
fetched_at: 2026-02-27
knowledge_point: 06_状态验证
context7_query: state validation Pydantic BaseModel state schema validation runtime type checking
---

# Context7 文档：LangGraph 状态验证

## 文档来源
- 库名称：langgraph
- Context7 ID: /websites/langchain_oss_python_langgraph
- 官方文档链接：https://docs.langchain.com/oss/python/langgraph/use-graph-api

## 关键信息提取

### 1. Pydantic BaseModel 作为 State Schema

StateGraph 接受 `state_schema` 参数，支持 TypedDict、dataclass 和 Pydantic BaseModel。
使用 Pydantic BaseModel 可以添加**运行时输入验证**。

```python
from pydantic import BaseModel

class OverallState(BaseModel):
    a: str
```

### 2. 运行时类型强制转换

Pydantic 会自动进行类型强制转换：
- 字符串数字 → 整数
- 字符串布尔值 → 布尔值

```python
class CoercionExample(BaseModel):
    number: int
    flag: bool

# "42" → 42, "true" → True
result = graph.invoke({"number": "42", "flag": "true"})

# 无法转换时抛出 ValidationError
try:
    graph.invoke({"number": "not-a-number", "flag": "true"})
except Exception as e:
    print(f"Expected validation error: {e}")
```

### 3. 嵌套模型支持

```python
class NestedModel(BaseModel):
    value: str

class ComplexState(BaseModel):
    text: str
    count: int
    nested: NestedModel
```

### 4. 验证范围限制

**重要**：Pydantic 验证仅对**首个节点的输入**生效。
节点之间的状态传递不会重新触发 Pydantic 验证。

### 5. 无效输入异常处理

```python
try:
    graph.invoke({"a": 123})  # 应该是字符串
except Exception as e:
    print("Pydantic validation error")
```
