# 实战代码 - 场景4：Context 注入实战

> **场景定位**：掌握 LangGraph 的运行时上下文注入机制，实现配置与数据的分离

---

## 场景描述

在实际的 LangGraph 应用中，我们经常需要在运行时传递一些**不可变的配置数据**，如：
- API 密钥和凭证
- 环境配置（开发/测试/生产）
- LLM 模型参数（模型名称、温度、最大 token）
- 用户会话信息（用户 ID、权限）
- 全局参数（超时设置、重试次数）

这些数据的特点是：
1. **不需要在节点间传递和修改**
2. **所有节点都可能需要访问**
3. **运行时才能确定具体值**

如果把这些数据放在 State 中，会导致：
- State 结构臃肿
- 节点需要不断传递这些不变的数据
- 容易被意外修改

**Context Schema 与 Runtime 注入机制**正是为了解决这个问题而设计的。

---

## 核心挑战

### 挑战 1：配置与数据分离

**问题**：如何将不可变的配置数据与可变的工作流数据分离？

**解决方案**：使用 `context_schema` 定义配置结构，通过 `Runtime[ContextT]` 注入到节点。

### 挑战 2：类型安全的上下文访问

**问题**：如何确保节点访问上下文数据时的类型安全？

**解决方案**：使用 TypedDict 定义 Context 类型，利用泛型 `Runtime[ContextT]` 提供类型检查。

### 挑战 3：多环境配置管理

**问题**：如何在不同环境（开发/测试/生产）中使用不同的配置？

**解决方案**：在调用 `invoke()` 时传入不同的 `context` 参数。

---

## 完整代码实现

### 场景：RAG 系统配置注入

我们将构建一个 RAG 系统，通过 Context 注入以下配置：
- LLM 配置（模型名称、温度、最大 token）
- Embedding 配置（模型名称、维度）
- 向量数据库配置（连接信息）
- 环境标识（dev/prod）

```python
"""
LangGraph Context 注入实战：RAG 系统配置管理

演示如何使用 Context Schema 和 Runtime 注入实现配置与数据分离
"""

from typing_extensions import TypedDict, Literal, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
import os


# ===== 1. 定义 Context Schema =====
class RAGContext(TypedDict):
    """RAG 系统运行时配置"""
    # 环境配置
    environment: Literal["dev", "staging", "prod"]

    # LLM 配置
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_api_key: NotRequired[str]  # 可选字段

    # Embedding 配置
    embedding_model: str
    embedding_dimension: int

    # 向量数据库配置
    vector_db_url: str
    vector_db_collection: str

    # 系统参数
    max_retries: int
    timeout: int


# ===== 2. 定义 State Schema =====
class RAGState(TypedDict):
    """RAG 工作流状态"""
    query: str                    # 用户查询
    documents: list[str]          # 检索到的文档
    context: str                  # 构建的上下文
    response: str                 # 最终响应
    metadata: dict                # 元数据


# ===== 3. 创建 StateGraph =====
graph = StateGraph(
    state_schema=RAGState,
    context_schema=RAGContext  # 指定 Context Schema
)


# ===== 4. 定义节点函数 =====

def retrieve_documents(
    state: RAGState,
    *,
    runtime: Runtime[RAGContext]  # 注入 Runtime
) -> dict:
    """
    检索节点：从向量数据库检索相关文档

    通过 runtime.context 访问配置信息
    """
    # 从 runtime.context 获取配置
    env = runtime.context.get("environment", "dev")
    embedding_model = runtime.context.get("embedding_model", "text-embedding-ada-002")
    embedding_dim = runtime.context.get("embedding_dimension", 1536)
    vector_db_url = runtime.context.get("vector_db_url", "http://localhost:19530")
    collection = runtime.context.get("vector_db_collection", "documents")

    query = state["query"]

    print(f"[{env}] Retrieving documents...")
    print(f"  Embedding Model: {embedding_model} (dim={embedding_dim})")
    print(f"  Vector DB: {vector_db_url}/{collection}")
    print(f"  Query: {query}")

    # 模拟检索过程
    # 实际应用中这里会调用向量数据库
    documents = [
        f"Document 1 about {query}",
        f"Document 2 related to {query}",
        f"Document 3 discussing {query}"
    ]

    return {
        "documents": documents,
        "metadata": {
            "embedding_model": embedding_model,
            "num_documents": len(documents)
        }
    }


def build_context(
    state: RAGState,
    *,
    runtime: Runtime[RAGContext]
) -> dict:
    """
    上下文构建节点：将检索到的文档组织成上下文

    根据环境使用不同的上下文构建策略
    """
    env = runtime.context.get("environment", "dev")
    max_tokens = runtime.context.get("llm_max_tokens", 4000)

    documents = state["documents"]

    print(f"[{env}] Building context...")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Documents: {len(documents)}")

    # 根据环境使用不同策略
    if env == "prod":
        # 生产环境：更严格的 token 控制
        context = "\n\n".join(documents[:2])  # 只使用前2个文档
    else:
        # 开发/测试环境：使用所有文档
        context = "\n\n".join(documents)

    return {"context": context}


def generate_response(
    state: RAGState,
    *,
    runtime: Runtime[RAGContext]
) -> dict:
    """
    生成节点：使用 LLM 生成最终响应

    使用 runtime.context 中的 LLM 配置
    """
    # 从 runtime.context 获取 LLM 配置
    env = runtime.context.get("environment", "dev")
    model = runtime.context.get("llm_model", "gpt-3.5-turbo")
    temperature = runtime.context.get("llm_temperature", 0.7)
    max_tokens = runtime.context.get("llm_max_tokens", 1000)
    timeout = runtime.context.get("timeout", 30)
    max_retries = runtime.context.get("max_retries", 3)

    query = state["query"]
    context = state["context"]

    print(f"[{env}] Generating response...")
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Timeout: {timeout}s")
    print(f"  Max retries: {max_retries}")

    # 模拟 LLM 调用
    # 实际应用中这里会调用 OpenAI API
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    # 根据环境生成不同的响应
    if env == "dev":
        response = f"[DEV] Mock response for: {query}"
    elif env == "staging":
        response = f"[STAGING] Test response for: {query}"
    else:
        response = f"Based on the provided context, here's the answer to '{query}'..."

    return {
        "response": response,
        "metadata": {
            **state.get("metadata", {}),
            "llm_model": model,
            "temperature": temperature
        }
    }


# ===== 5. 构建图 =====
graph.add_node("retrieve", retrieve_documents)
graph.add_node("build_context", build_context)
graph.add_node("generate", generate_response)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", END)


# ===== 6. 编译图 =====
compiled_graph = graph.compile()


# ===== 7. 测试不同环境配置 =====

def test_dev_environment():
    """测试开发环境配置"""
    print("=" * 60)
    print("测试 1：开发环境配置")
    print("=" * 60)

    result = compiled_graph.invoke(
        # State 初始值
        {
            "query": "What is LangGraph?",
            "documents": [],
            "context": "",
            "response": "",
            "metadata": {}
        },
        # Context 配置（开发环境）
        context={
            "environment": "dev",
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "vector_db_url": "http://localhost:19530",
            "vector_db_collection": "dev_documents",
            "max_retries": 1,
            "timeout": 10
        }
    )

    print("\n最终结果:")
    print(f"  Query: {result['query']}")
    print(f"  Documents: {len(result['documents'])} documents")
    print(f"  Response: {result['response']}")
    print(f"  Metadata: {result['metadata']}")
    print()


def test_prod_environment():
    """测试生产环境配置"""
    print("=" * 60)
    print("测试 2：生产环境配置")
    print("=" * 60)

    result = compiled_graph.invoke(
        # State 初始值
        {
            "query": "What is LangGraph?",
            "documents": [],
            "context": "",
            "response": "",
            "metadata": {}
        },
        # Context 配置（生产环境）
        context={
            "environment": "prod",
            "llm_model": "gpt-4",
            "llm_temperature": 0.3,
            "llm_max_tokens": 2000,
            "llm_api_key": os.getenv("OPENAI_API_KEY", "sk-xxx"),
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
            "vector_db_url": "https://prod-milvus.example.com",
            "vector_db_collection": "prod_documents",
            "max_retries": 5,
            "timeout": 60
        }
    )

    print("\n最终结果:")
    print(f"  Query: {result['query']}")
    print(f"  Documents: {len(result['documents'])} documents")
    print(f"  Response: {result['response']}")
    print(f"  Metadata: {result['metadata']}")
    print()


def test_staging_environment():
    """测试测试环境配置"""
    print("=" * 60)
    print("测试 3：测试环境配置")
    print("=" * 60)

    result = compiled_graph.invoke(
        {
            "query": "How to use Context in LangGraph?",
            "documents": [],
            "context": "",
            "response": "",
            "metadata": {}
        },
        context={
            "environment": "staging",
            "llm_model": "gpt-4-turbo",
            "llm_temperature": 0.5,
            "llm_max_tokens": 1500,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "vector_db_url": "http://staging-milvus.example.com",
            "vector_db_collection": "staging_documents",
            "max_retries": 3,
            "timeout": 30
        }
    )

    print("\n最终结果:")
    print(f"  Query: {result['query']}")
    print(f"  Documents: {len(result['documents'])} documents")
    print(f"  Response: {result['response']}")
    print(f"  Metadata: {result['metadata']}")
    print()


# ===== 8. 运行测试 =====
if __name__ == "__main__":
    test_dev_environment()
    test_prod_environment()
    test_staging_environment()
```

---

## 运行结果

```
============================================================
测试 1：开发环境配置
============================================================
[dev] Retrieving documents...
  Embedding Model: text-embedding-ada-002 (dim=1536)
  Vector DB: http://localhost:19530/dev_documents
  Query: What is LangGraph?
[dev] Building context...
  Max tokens: 1000
  Documents: 3
[dev] Generating response...
  Model: gpt-3.5-turbo
  Temperature: 0.7
  Max tokens: 1000
  Timeout: 10s
  Max retries: 1

最终结果:
  Query: What is LangGraph?
  Documents: 3 documents
  Response: [DEV] Mock response for: What is LangGraph?
  Metadata: {'embedding_model': 'text-embedding-ada-002', 'num_documents': 3, 'llm_model': 'gpt-3.5-turbo', 'temperature': 0.7}

============================================================
测试 2：生产环境配置
============================================================
[prod] Retrieving documents...
  Embedding Model: text-embedding-3-large (dim=3072)
  Vector DB: https://prod-milvus.example.com/prod_documents
  Query: What is LangGraph?
[prod] Building context...
  Max tokens: 2000
  Documents: 3
[prod] Generating response...
  Model: gpt-4
  Temperature: 0.3
  Max tokens: 2000
  Timeout: 60s
  Max retries: 5

最终结果:
  Query: What is LangGraph?
  Documents: 3 documents
  Response: Based on the provided context, here's the answer to 'What is LangGraph?'...
  Metadata: {'embedding_model': 'text-embedding-3-large', 'num_documents': 3, 'llm_model': 'gpt-4', 'temperature': 0.3}

============================================================
测试 3：测试环境配置
============================================================
[staging] Retrieving documents...
  Embedding Model: text-embedding-3-small (dim=1536)
  Vector DB: http://staging-milvus.example.com/staging_documents
  Query: How to use Context in LangGraph?
[staging] Building context...
  Max tokens: 1500
  Documents: 3
[staging] Generating response...
  Model: gpt-4-turbo
  Temperature: 0.5
  Max tokens: 1500
  Timeout: 30s
  Max retries: 3

最终结果:
  Query: How to use Context in LangGraph?
  Documents: 3 documents
  Response: [STAGING] Test response for: How to use Context in LangGraph?
  Metadata: {'embedding_model': 'text-embedding-3-small', 'num_documents': 3, 'llm_model': 'gpt-4-turbo', 'temperature': 0.5}
```

---

## 关键点解析

### 1. Context Schema 定义

```python
class RAGContext(TypedDict):
    environment: Literal["dev", "staging", "prod"]
    llm_model: str
    llm_temperature: float
    llm_api_key: NotRequired[str]  # 可选字段
```

**关键点**：
- 使用 `TypedDict` 定义结构，提供类型检查
- 使用 `Literal` 限制可选值（如环境类型）
- 使用 `NotRequired` 标记可选字段（如 API 密钥）
- Context 中的字段应该是**不可变的配置数据**

### 2. Runtime 注入

```python
def retrieve_documents(
    state: RAGState,
    *,
    runtime: Runtime[RAGContext]  # 关键字参数
) -> dict:
    env = runtime.context.get("environment", "dev")
    # ...
```

**关键点**：
- `runtime` 必须是**关键字参数**（使用 `*` 分隔）
- 泛型 `Runtime[RAGContext]` 提供类型安全
- 使用 `runtime.context.get()` 访问配置，提供默认值

### 3. 配置传递

```python
result = compiled_graph.invoke(
    {"query": "...", ...},  # State 初始值
    context={...}           # Context 配置
)
```

**关键点**：
- State 和 Context 分别传递
- Context 在运行时注入，不会被修改
- 不同的调用可以使用不同的 Context

### 4. 不可变性保证

Context 数据是**只读的**，节点函数无法修改：

```python
# ✅ 正确：读取 Context
env = runtime.context.get("environment")

# ❌ 错误：尝试修改 Context（会失败）
runtime.context["environment"] = "prod"  # TypeError
```

### 5. 多环境支持

通过传入不同的 Context 实现多环境配置：

```python
# 开发环境
dev_context = {
    "environment": "dev",
    "llm_model": "gpt-3.5-turbo",
    "timeout": 10
}

# 生产环境
prod_context = {
    "environment": "prod",
    "llm_model": "gpt-4",
    "timeout": 60
}
```

---

## 常见问题

### Q1: Context 和 State 有什么区别？

**A**: 核心区别在于**可变性**和**用途**：

| 特性 | State | Context |
|------|-------|---------|
| 可变性 | 可变（节点可以更新） | 不可变（只读） |
| 传递方式 | 节点间传递 | 运行时注入 |
| 用途 | 工作流数据 | 配置和环境 |
| 定义位置 | `state_schema` | `context_schema` |

### Q2: 什么数据应该放在 Context 中？

**A**: 适合放在 Context 中的数据：
- ✅ 环境配置（dev/staging/prod）
- ✅ API 密钥和凭证
- ✅ 全局参数（超时、重试次数）
- ✅ LLM 配置（模型名称、温度）
- ✅ 用户会话信息（用户 ID、权限）
- ✅ 不可变的元数据

不适合放在 Context 中的数据：
- ❌ 需要在节点间传递和修改的数据
- ❌ 工作流的中间结果
- ❌ 动态变化的状态

### Q3: 可以在节点中修改 Context 吗？

**A**: 不可以。Context 是只读的，节点函数无法修改 Context 的内容。如果需要修改数据，应该放在 State 中。

### Q4: Context 在哪里传入？

**A**: 在调用 `invoke()`, `stream()`, `ainvoke()` 等方法时，通过 `context` 参数传入：

```python
result = compiled.invoke(
    {"x": 0.5},           # State 初始值
    context={"r": 3.0}    # Context 数据
)
```

### Q5: 如果不定义 context_schema 会怎样？

**A**: 如果不定义 `context_schema`，节点函数就不能使用 `runtime: Runtime[ContextT]` 参数。但可以使用其他节点协议（如只接受 `state` 参数的基础节点）。

### Q6: Runtime 还包含什么？

**A**: Runtime 对象除了 `context` 外，还包含：
- `config`: RunnableConfig 配置
- `store`: BaseStore 存储接口
- `writer`: StreamWriter 流写入器

```python
def node(state: State, *, runtime: Runtime[Context]) -> dict:
    # 访问 context
    env = runtime.context.get("environment")

    # 访问 config
    run_id = runtime.config.get("run_id")

    # 访问 store（如果配置了）
    if runtime.store:
        data = runtime.store.get("key")

    # 访问 writer（如果配置了）
    if runtime.writer:
        runtime.writer.write({"event": "progress"})
```

---

## 最佳实践

### 1. Context 命名规范

```python
# ✅ 好的命名
class AppContext(TypedDict):
    environment: str
    api_endpoint: str

class LLMContext(TypedDict):
    model_name: str
    temperature: float

# ❌ 不好的命名
class Context(TypedDict):  # 太泛化
    data: dict  # 不明确

class Config(TypedDict):  # 容易与 RunnableConfig 混淆
    settings: dict
```

### 2. 使用类型约束

```python
from typing import Literal
from typing_extensions import NotRequired

class Context(TypedDict):
    # 使用 Literal 限制可选值
    environment: Literal["dev", "staging", "prod"]

    # 使用 NotRequired 标记可选字段
    api_key: NotRequired[str]

    # 使用具体类型而非 Any
    timeout: int  # ✅
    # timeout: Any  # ❌
```

### 3. 提供默认值

```python
def node(state: State, *, runtime: Runtime[Context]) -> dict:
    # ✅ 使用 get() 提供默认值
    max_count = runtime.context.get("max_count", 10)

    # ❌ 直接访问可能导致 KeyError
    max_count = runtime.context["max_count"]
```

### 4. 环境配置管理

```python
import os
from typing import Literal

def get_context(env: Literal["dev", "staging", "prod"]) -> dict:
    """根据环境返回对应的 Context 配置"""
    base_context = {
        "environment": env,
        "max_retries": 3,
    }

    if env == "dev":
        return {
            **base_context,
            "llm_model": "gpt-3.5-turbo",
            "timeout": 10,
            "vector_db_url": "http://localhost:19530"
        }
    elif env == "staging":
        return {
            **base_context,
            "llm_model": "gpt-4-turbo",
            "timeout": 30,
            "vector_db_url": "http://staging-db.example.com"
        }
    else:  # prod
        return {
            **base_context,
            "llm_model": "gpt-4",
            "timeout": 60,
            "vector_db_url": os.getenv("PROD_DB_URL"),
            "llm_api_key": os.getenv("OPENAI_API_KEY")
        }

# 使用
dev_result = graph.invoke(state, context=get_context("dev"))
prod_result = graph.invoke(state, context=get_context("prod"))
```

### 5. 敏感信息处理

```python
class SecureContext(TypedDict):
    api_key: NotRequired[str]
    db_password: NotRequired[str]

def node(state: State, *, runtime: Runtime[SecureContext]) -> dict:
    # ✅ 从环境变量读取敏感信息
    api_key = runtime.context.get("api_key") or os.getenv("API_KEY")

    # ✅ 不要在日志中打印敏感信息
    print(f"Using API key: {'*' * 8}")  # 脱敏

    # ❌ 不要直接打印敏感信息
    # print(f"API key: {api_key}")
```

### 6. Context 验证

```python
from typing import Literal

def validate_context(context: dict) -> None:
    """验证 Context 配置的有效性"""
    required_fields = ["environment", "llm_model", "vector_db_url"]

    for field in required_fields:
        if field not in context:
            raise ValueError(f"Missing required context field: {field}")

    # 验证环境值
    env = context["environment"]
    if env not in ["dev", "staging", "prod"]:
        raise ValueError(f"Invalid environment: {env}")

    # 验证数值范围
    if "timeout" in context and context["timeout"] <= 0:
        raise ValueError("Timeout must be positive")

# 使用
context = get_context("prod")
validate_context(context)
result = graph.invoke(state, context=context)
```

---

## 总结

**Context 注入机制的核心价值**：

1. **配置与数据分离**：State 专注于工作流数据，Context 专注于配置
2. **类型安全**：通过 TypedDict 和泛型提供完整的类型检查
3. **不可变性**：Context 只读，避免意外修改配置
4. **依赖注入**：通过 Runtime 对象优雅地注入依赖
5. **多环境支持**：轻松支持开发、测试、生产等多环境配置

**一句话总结**：Context Schema 定义不可变的运行时配置，通过 Runtime 对象注入到节点函数，实现配置与数据的分离，提供类型安全的依赖注入机制。

---

**相关文档**：
- 核心概念 9：Context Schema 与 Runtime
- 实战代码 - 场景1：最小 StateGraph 示例
- 实战代码 - 场景2：多节点状态流转
