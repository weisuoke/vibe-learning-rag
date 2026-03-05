# 实战代码：场景2 - Pydantic模型状态

## 场景概述

本文档展示如何在 LangGraph 中使用 Pydantic BaseModel 定义状态，包括字段验证、类型转换、严格模式和 model_fields_set 的实际应用。

**适用场景**：
- API 边界验证
- 需要运行时类型检查
- 复杂数据结构验证
- 需要自定义验证器

## 核心代码示例

### 示例1：基础 Pydantic 状态定义

```python
"""
基础 Pydantic 状态定义示例
展示 BaseModel 的基本用法和自动类型转换
"""
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import Optional
from datetime import datetime

# 定义 Pydantic 状态模型
class UserState(BaseModel):
    """用户状态模型 - 使用 Pydantic 进行验证"""
    user_id: int = Field(..., gt=0, description="用户ID，必须大于0")
    username: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=150)
    created_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)

# 创建图
graph = StateGraph(UserState)

def validate_user(state: UserState) -> UserState:
    """验证用户信息"""
    print(f"验证用户: {state.username}")
    print(f"显式设置的字段: {state.model_fields_set}")

    # Pydantic 自动进行类型转换和验证
    return UserState(
        user_id=state.user_id,
        username=state.username.strip(),
        email=state.email.lower(),
        age=state.age,
        created_at=datetime.now(),
        tags=state.tags + ["validated"]
    )

graph.add_node("validate", validate_user)
graph.add_edge(START, "validate")
graph.add_edge("validate", END)

app = graph.compile()

# 测试：自动类型转换
result = app.invoke({
    "user_id": "123",  # 字符串会自动转换为整数
    "username": "  john_doe  ",
    "email": "JOHN@EXAMPLE.COM",
    "age": "25",  # 字符串会自动转换为整数
})

print(f"结果: {result}")
# 输出: user_id=123 (int), email='john@example.com' (小写)
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 示例2：字段验证和约束

```python
"""
字段验证和约束示例
展示 Pydantic 的丰富验证功能
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from langgraph.graph import StateGraph, START, END
from typing import Optional

class ProductState(BaseModel):
    """产品状态 - 带复杂验证规则"""
    product_id: int = Field(..., gt=0)
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="价格必须大于0")
    discount: float = Field(0.0, ge=0, le=1, description="折扣范围 0-1")
    stock: int = Field(0, ge=0)
    category: str = Field(..., pattern=r'^[a-zA-Z_]+$')

    # 字段级验证器
    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        """验证价格精度"""
        if round(v, 2) != v:
            raise ValueError('价格最多保留2位小数')
        return v

    # 模型级验证器
    @model_validator(mode='after')
    def validate_discount_price(self):
        """验证折扣后价格"""
        final_price = self.price * (1 - self.discount)
        if final_price < 0.01:
            raise ValueError('折扣后价格不能低于0.01')
        return self

# 创建图
graph = StateGraph(ProductState)

def process_product(state: ProductState) -> ProductState:
    """处理产品信息"""
    final_price = state.price * (1 - state.discount)
    print(f"产品: {state.name}, 原价: {state.price}, 折扣: {state.discount}, 最终价格: {final_price}")
    return state

graph.add_node("process", process_product)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

# 测试：验证成功
result = app.invoke({
    "product_id": 1,
    "name": "Laptop",
    "price": 999.99,
    "discount": 0.1,
    "stock": 50,
    "category": "electronics"
})
print(f"验证成功: {result.name}")

# 测试：验证失败
try:
    app.invoke({
        "product_id": 1,
        "name": "Laptop",
        "price": 10.0,
        "discount": 0.99,  # 折扣后价格 < 0.01，会失败
        "stock": 50,
        "category": "electronics"
    })
except Exception as e:
    print(f"验证失败: {e}")
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 示例3：严格模式

```python
"""
严格模式示例
禁用类型转换，要求精确匹配
"""
from pydantic import BaseModel, Field, StrictInt, StrictStr, StrictBool
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from uuid import UUID

class StrictState(BaseModel):
    """严格模式状态 - 不允许类型转换"""
    id: UUID  # 必须是 UUID 对象，不接受字符串
    count: StrictInt  # 必须是整数，不接受字符串 "123"
    name: StrictStr  # 必须是字符串
    active: StrictBool  # 必须是布尔值，不接受 1/0

# 创建图
graph = StateGraph(StrictState)

def process_strict(state: StrictState) -> StrictState:
    """处理严格类型状态"""
    print(f"ID类型: {type(state.id)}, Count类型: {type(state.count)}")
    return state

graph.add_node("process", process_strict)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

# 测试：正确的类型
from uuid import uuid4
result = app.invoke({
    "id": uuid4(),  # UUID 对象
    "count": 123,  # 整数
    "name": "test",  # 字符串
    "active": True  # 布尔值
})
print(f"严格模式验证成功")

# 测试：类型转换会失败
try:
    app.invoke({
        "id": "550e8400-e29b-41d4-a716-446655440000",  # 字符串，会失败
        "count": "123",  # 字符串，会失败
        "name": "test",
        "active": 1  # 整数，会失败
    })
except Exception as e:
    print(f"严格模式验证失败: {e}")
```

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 示例4：model_fields_set 应用

```python
"""
model_fields_set 应用示例
展示 LangGraph 如何使用 model_fields_set 跟踪显式设置的字段
"""
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import Optional

class ConfigState(BaseModel):
    """配置状态 - 演示 model_fields_set"""
    api_key: str = Field(default="default_key")
    timeout: int = Field(default=30)
    retries: int = Field(default=3)
    debug: bool = Field(default=False)
    custom_header: Optional[str] = None

# 创建图
graph = StateGraph(ConfigState)

def update_config(state: ConfigState) -> ConfigState:
    """更新配置 - 只更新显式设置的字段"""
    print(f"显式设置的字段: {state.model_fields_set}")
    print(f"所有字段值: api_key={state.api_key}, timeout={state.timeout}, retries={state.retries}")

    # LangGraph 内部使用 model_fields_set 判断哪些字段需要更新
    # 只有在 model_fields_set 中的字段才会被更新到状态
    return ConfigState(
        api_key=state.api_key,
        timeout=state.timeout * 2,  # 修改 timeout
        retries=state.retries,
        debug=True  # 显式设置 debug
    )

graph.add_node("update", update_config)
graph.add_edge(START, "update")
graph.add_edge("update", END)

app = graph.compile()

# 测试1：只设置部分字段
result1 = app.invoke({
    "api_key": "my_key",  # 显式设置
    "timeout": 60  # 显式设置
    # retries 和 debug 使用默认值
})
print(f"测试1 - model_fields_set: {result1.model_fields_set}")
# 输出: {'api_key', 'timeout', 'debug'}

# 测试2：设置所有字段
result2 = app.invoke({
    "api_key": "my_key",
    "timeout": 60,
    "retries": 5,
    "debug": False,
    "custom_header": "X-Custom"
})
print(f"测试2 - model_fields_set: {result2.model_fields_set}")
# 输出: {'api_key', 'timeout', 'retries', 'debug', 'custom_header'}
```

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]
[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

### 示例5：LangGraph 中的 Pydantic 特殊处理

```python
"""
LangGraph 中的 Pydantic 特殊处理
展示 LangGraph 如何处理 Pydantic 模型的更新
"""
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import Optional

class MessageState(BaseModel):
    """消息状态"""
    content: str = Field(default="")
    sender: str = Field(default="system")
    timestamp: Optional[int] = None
    metadata: dict = Field(default_factory=dict)

# 创建图
graph = StateGraph(MessageState)

def add_message(state: MessageState) -> MessageState:
    """添加消息"""
    # 返回新的状态
    # LangGraph 会使用 model_fields_set 判断哪些字段被更新
    return MessageState(
        content=state.content + " [processed]",
        sender="bot",
        # timestamp 不设置，使用默认值 None
        # metadata 不设置，使用默认值 {}
    )

def add_metadata(state: MessageState) -> MessageState:
    """添加元数据"""
    import time
    return MessageState(
        content=state.content,
        sender=state.sender,
        timestamp=int(time.time()),
        metadata={"processed": True}
    )

graph.add_node("add_message", add_message)
graph.add_node("add_metadata", add_metadata)
graph.add_edge(START, "add_message")
graph.add_edge("add_message", "add_metadata")
graph.add_edge("add_metadata", END)

app = graph.compile()

# 测试
result = app.invoke({
    "content": "Hello",
    "sender": "user"
})

print(f"最终状态:")
print(f"  content: {result.content}")
print(f"  sender: {result.sender}")
print(f"  timestamp: {result.timestamp}")
print(f"  metadata: {result.metadata}")
```

[来源: reference/source_状态类型系统_01.md | LangGraph 源码分析]

## 完整实战：RAG 系统的 Pydantic 状态管理

```python
"""
完整实战：RAG 系统的 Pydantic 状态管理
展示在实际 RAG 系统中如何使用 Pydantic 进行状态验证
"""
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END
from typing import Optional, Literal
from datetime import datetime

# 定义 RAG 系统状态
class RAGState(BaseModel):
    """RAG 系统状态 - 使用 Pydantic 进行严格验证"""

    # 输入字段
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询")
    user_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$', description="用户ID")

    # 检索字段
    retrieved_docs: list[str] = Field(default_factory=list, description="检索到的文档")
    relevance_scores: list[float] = Field(default_factory=list, description="相关性分数")

    # 生成字段
    response: str = Field(default="", description="生成的回答")
    confidence: float = Field(default=0.0, ge=0, le=1, description="置信度")

    # 元数据
    status: Literal["pending", "retrieved", "generated", "error"] = Field(default="pending")
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('relevance_scores')
    @classmethod
    def validate_scores(cls, v: list[float]) -> list[float]:
        """验证相关性分数"""
        for score in v:
            if not 0 <= score <= 1:
                raise ValueError(f'相关性分数必须在 0-1 之间: {score}')
        return v

    @field_validator('retrieved_docs', 'relevance_scores')
    @classmethod
    def validate_lengths_match(cls, v, info):
        """验证文档和分数数量匹配"""
        if info.field_name == 'relevance_scores':
            docs = info.data.get('retrieved_docs', [])
            if len(docs) != len(v):
                raise ValueError(f'文档数量({len(docs)})和分数数量({len(v)})不匹配')
        return v

# 创建 RAG 图
graph = StateGraph(RAGState)

def retrieve_documents(state: RAGState) -> RAGState:
    """检索文档"""
    print(f"检索查询: {state.query}")

    # 模拟检索
    docs = [
        f"文档1: 关于 {state.query} 的内容",
        f"文档2: 更多关于 {state.query} 的信息",
        f"文档3: {state.query} 的详细说明"
    ]
    scores = [0.95, 0.87, 0.76]

    return RAGState(
        query=state.query,
        user_id=state.user_id,
        retrieved_docs=docs,
        relevance_scores=scores,
        status="retrieved",
        timestamp=state.timestamp
    )

def generate_response(state: RAGState) -> RAGState:
    """生成回答"""
    print(f"生成回答，基于 {len(state.retrieved_docs)} 个文档")

    # 模拟生成
    context = "\n".join(state.retrieved_docs[:2])  # 使用前2个文档
    response = f"基于检索到的文档，关于 '{state.query}' 的回答是：{context}"

    # 计算置信度（基于相关性分数）
    confidence = sum(state.relevance_scores[:2]) / 2 if state.relevance_scores else 0.0

    return RAGState(
        query=state.query,
        user_id=state.user_id,
        retrieved_docs=state.retrieved_docs,
        relevance_scores=state.relevance_scores,
        response=response,
        confidence=confidence,
        status="generated",
        timestamp=state.timestamp
    )

# 构建图
graph.add_node("retrieve", retrieve_documents)
graph.add_node("generate", generate_response)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# 测试：正常流程
result = app.invoke({
    "query": "什么是 LangGraph",
    "user_id": "user_123"
})

print(f"\n=== RAG 系统执行结果 ===")
print(f"查询: {result.query}")
print(f"状态: {result.status}")
print(f"检索文档数: {len(result.retrieved_docs)}")
print(f"置信度: {result.confidence:.2f}")
print(f"回答: {result.response[:100]}...")
print(f"显式设置的字段: {result.model_fields_set}")

# 测试：验证失败
try:
    app.invoke({
        "query": "",  # 空查询，会失败
        "user_id": "user_123"
    })
except Exception as e:
    print(f"\n验证失败: {e}")
```

[来源: reference/search_状态类型系统_01.md | LangGraph 社区最佳实践]

## 性能对比：Pydantic vs TypedDict

```python
"""
性能对比示例
展示 Pydantic 和 TypedDict 的性能差异
"""
import time
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Pydantic 版本
class PydanticState(BaseModel):
    value: int
    name: str

# TypedDict 版本
class TypedDictState(TypedDict):
    value: int
    name: str

def benchmark_pydantic():
    """测试 Pydantic 性能"""
    graph = StateGraph(PydanticState)

    def process(state: PydanticState) -> PydanticState:
        return PydanticState(value=state.value + 1, name=state.name)

    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    start = time.time()
    for i in range(1000):
        app.invoke({"value": i, "name": f"test_{i}"})
    return time.time() - start

def benchmark_typeddict():
    """测试 TypedDict 性能"""
    graph = StateGraph(TypedDictState)

    def process(state: TypedDictState) -> TypedDictState:
        return {"value": state["value"] + 1, "name": state["name"]}

    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    start = time.time()
    for i in range(1000):
        app.invoke({"value": i, "name": f"test_{i}"})
    return time.time() - start

# 运行性能测试
pydantic_time = benchmark_pydantic()
typeddict_time = benchmark_typeddict()

print(f"\n=== 性能对比 (1000次迭代) ===")
print(f"Pydantic: {pydantic_time:.3f}秒")
print(f"TypedDict: {typeddict_time:.3f}秒")
print(f"性能差异: Pydantic 慢 {(pydantic_time / typeddict_time - 1) * 100:.1f}%")
```

[来源: reference/search_状态类型系统_01.md | LangGraph 社区最佳实践]

## 最佳实践总结

### 何时使用 Pydantic

**推荐场景**：
- API 边界验证（用户输入、外部数据）
- 需要复杂验证规则
- 需要自定义验证器
- 需要数据转换和清洗

**不推荐场景**：
- 内部状态管理（使用 TypedDict 更快）
- 性能敏感的应用
- 简单的数据结构

### 关键要点

1. **model_fields_set**：LangGraph 使用此属性判断哪些字段需要更新
2. **类型转换**：Pydantic 自动进行类型转换，但可以使用严格模式禁用
3. **验证器**：使用 `@field_validator` 和 `@model_validator` 添加自定义验证
4. **性能**：Pydantic 比 TypedDict 慢，但提供更强的类型安全

[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]
[来源: reference/search_状态类型系统_01.md | LangGraph 社区最佳实践]

## 参考资源

- Pydantic 官方文档：https://docs.pydantic.dev/
- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/
- LangGraph 源码：https://github.com/langchain-ai/langgraph
