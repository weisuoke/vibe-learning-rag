# 实战代码 - 场景3：Runtime Context 应用

## 场景概述

**Runtime Context** 是 LangGraph 中用于传递运行时配置和依赖的机制。与可变的 State 不同，Runtime Context 是**不可变的**，在整个图执行过程中保持不变。

### 核心特点

1. **不可变性**：Context 在运行时不可修改，确保配置的一致性
2. **类型安全**：通过 `context_schema` 定义类型，支持 TypedDict 或 dataclass
3. **依赖注入**：在节点中通过 `Runtime[Context]` 访问
4. **运行时传递**：在 `invoke()` 或 `stream()` 时通过 `context` 参数传递

### 典型应用场景

- **动态 LLM 选择**：根据用户偏好或任务类型选择不同的模型
- **数据库连接传递**：在节点间共享数据库连接或 API 客户端
- **用户 ID 管理**：实现多租户系统，隔离不同用户的数据
- **配置管理**：传递环境配置、特性开关等

## 代码示例 1：动态 LLM 选择

这个示例展示如何使用 Runtime Context 在运行时动态选择不同的 LLM 模型。

```python
"""
场景1：动态 LLM 选择
演示：使用 Runtime Context 在运行时选择不同的 LLM 模型
"""

import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.runtime import Runtime

# 加载环境变量
load_dotenv()

# ===== 1. 定义 Context Schema =====
@dataclass
class LLMContext:
    """运行时上下文：LLM 模型配置"""
    model_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

# ===== 2. 初始化模型字典 =====
# 预先初始化所有可能使用的模型
MODELS = {
    "openai": {
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0.7),
    },
    "anthropic": {
        # 注意：需要安装 langchain-anthropic 并配置 ANTHROPIC_API_KEY
        # "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    }
}

# ===== 3. 定义节点函数 =====
def call_model(state: MessagesState, runtime: Runtime[LLMContext]) -> MessagesState:
    """
    调用 LLM 模型

    通过 runtime.context 访问运行时配置，动态选择模型
    """
    # 从 runtime context 获取模型配置
    provider = runtime.context.model_provider
    model_name = runtime.context.model_name

    print(f"[call_model] 使用模型: {provider}/{model_name}")

    # 根据配置选择模型
    if provider in MODELS and model_name in MODELS[provider]:
        model = MODELS[provider][model_name]
    else:
        # 回退到默认模型
        print(f"[call_model] 模型 {provider}/{model_name} 不存在，使用默认模型")
        model = MODELS["openai"]["gpt-4o-mini"]

    # 调用模型
    response = model.invoke(state["messages"])

    return {"messages": [response]}

# ===== 4. 构建图 =====
def create_dynamic_llm_graph():
    """创建支持动态 LLM 选择的图"""
    builder = StateGraph(MessagesState, context_schema=LLMContext)

    # 添加节点
    builder.add_node("model", call_model)

    # 添加边
    builder.add_edge(START, "model")
    builder.add_edge("model", END)

    return builder.compile()

# ===== 5. 测试不同的模型配置 =====
if __name__ == "__main__":
    print("=== 场景1：动态 LLM 选择 ===\n")

    graph = create_dynamic_llm_graph()

    # 测试消息
    test_message = {"role": "user", "content": "用一句话介绍 LangGraph"}

    # 测试1：使用默认配置（OpenAI gpt-4o-mini）
    print("--- 测试1：默认配置 ---")
    result1 = graph.invoke(
        {"messages": [test_message]},
        context=LLMContext()  # 使用默认值
    )
    print(f"回复: {result1['messages'][-1].content}\n")

    # 测试2：使用 gpt-4o
    print("--- 测试2：使用 gpt-4o ---")
    result2 = graph.invoke(
        {"messages": [test_message]},
        context=LLMContext(model_name="gpt-4o")
    )
    print(f"回复: {result2['messages'][-1].content}\n")

    # 测试3：使用字典传递 context（更灵活）
    print("--- 测试3：字典方式传递 context ---")
    result3 = graph.invoke(
        {"messages": [test_message]},
        context={"model_provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.3}
    )
    print(f"回复: {result3['messages'][-1].content}\n")

    print("✅ 场景1 完成")
```

### 运行输出示例

```
=== 场景1：动态 LLM 选择 ===

--- 测试1：默认配置 ---
[call_model] 使用模型: openai/gpt-4o-mini
回复: LangGraph 是一个用于构建状态化、多步骤 AI 工作流的框架，支持复杂的决策流程和人机协作。

--- 测试2：使用 gpt-4o ---
[call_model] 使用模型: openai/gpt-4o
回复: LangGraph 是一个基于图结构的 AI 应用开发框架，让你能够构建可靠、可控的 Agent 系统。

--- 测试3：字典方式传递 context ---
[call_model] 使用模型: openai/gpt-4o-mini
回复: LangGraph 是一个用于构建状态化工作流的 Python 框架。

✅ 场景1 完成
```

## 代码示例 2：数据库连接传递

这个示例展示如何使用 Runtime Context 传递数据库连接，实现节点间的数据共享。

```python
"""
场景2：数据库连接传递
演示：使用 Runtime Context 传递数据库连接和 Store
"""

import uuid
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

# ===== 1. 定义 State 和 Context =====
class AgentState(TypedDict):
    """图的状态"""
    user_query: str
    search_results: list[str]
    memory_ids: list[str]

@dataclass
class DatabaseContext:
    """运行时上下文：数据库连接"""
    user_id: str
    store: InMemoryStore  # 长期记忆存储

# ===== 2. 定义节点函数 =====
async def search_node(state: AgentState, runtime: Runtime[DatabaseContext]) -> AgentState:
    """
    搜索节点：执行搜索并保存结果到 store
    """
    user_id = runtime.context.user_id
    store = runtime.context.store

    print(f"[search_node] 用户 {user_id} 执行搜索: {state['user_query']}")

    # 模拟搜索结果
    results = [
        f"搜索结果1: {state['user_query']} 相关内容",
        f"搜索结果2: {state['user_query']} 详细信息",
    ]

    # 保存搜索历史到 store（长期记忆）
    memory_id = str(uuid.uuid4())
    namespace = (user_id, "search_history")

    await store.aput(
        namespace,
        memory_id,
        {
            "query": state["user_query"],
            "results": results,
            "timestamp": "2026-02-26"
        }
    )

    print(f"[search_node] 保存搜索历史: {memory_id}")

    return {
        "search_results": results,
        "memory_ids": [memory_id]
    }

async def summarize_node(state: AgentState, runtime: Runtime[DatabaseContext]) -> AgentState:
    """
    总结节点：从 store 读取历史记录并生成总结
    """
    user_id = runtime.context.user_id
    store = runtime.context.store

    print(f"[summarize_node] 用户 {user_id} 生成总结")

    # 从 store 读取用户的搜索历史
    namespace = (user_id, "search_history")
    history = await store.asearch(namespace)

    print(f"[summarize_node] 找到 {len(history)} 条历史记录")

    # 生成总结（简化版）
    summary = f"基于 {len(state['search_results'])} 条搜索结果的总结"

    return {"search_results": state["search_results"] + [summary]}

# ===== 3. 构建图 =====
def create_database_graph():
    """创建使用数据库连接的图"""
    builder = StateGraph(AgentState, context_schema=DatabaseContext)

    # 添加节点
    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)

    # 添加边
    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()

# ===== 4. 测试 =====
async def test_database_context():
    print("=== 场景2：数据库连接传递 ===\n")

    graph = create_database_graph()

    # 创建共享的 store
    store = InMemoryStore()

    # 测试1：用户 A 的搜索
    print("--- 测试1：用户 A ---")
    result1 = await graph.ainvoke(
        {"user_query": "LangGraph 教程"},
        context=DatabaseContext(user_id="user_a", store=store)
    )
    print(f"结果: {result1['search_results']}\n")

    # 测试2：用户 B 的搜索
    print("--- 测试2：用户 B ---")
    result2 = await graph.ainvoke(
        {"user_query": "Python 异步编程"},
        context=DatabaseContext(user_id="user_b", store=store)
    )
    print(f"结果: {result2['search_results']}\n")

    # 测试3：用户 A 再次搜索（可以访问之前的历史）
    print("--- 测试3：用户 A 再次搜索 ---")
    result3 = await graph.ainvoke(
        {"user_query": "LangGraph 高级用法"},
        context=DatabaseContext(user_id="user_a", store=store)
    )
    print(f"结果: {result3['search_results']}\n")

    print("✅ 场景2 完成")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_database_context())
```

### 运行输出示例

```
=== 场景2：数据库连接传递 ===

--- 测试1：用户 A ---
[search_node] 用户 user_a 执行搜索: LangGraph 教程
[search_node] 保存搜索历史: 3f8a9b2c-...
[summarize_node] 用户 user_a 生成总结
[summarize_node] 找到 1 条历史记录
结果: ['搜索结果1: LangGraph 教程 相关内容', '搜索结果2: LangGraph 教程 详细信息', '基于 2 条搜索结果的总结']

--- 测试2：用户 B ---
[search_node] 用户 user_b 执行搜索: Python 异步编程
[search_node] 保存搜索历史: 7d2e4f1a-...
[summarize_node] 用户 user_b 生成总结
[summarize_node] 找到 1 条历史记录
结果: ['搜索结果1: Python 异步编程 相关内容', '搜索结果2: Python 异步编程 详细信息', '基于 2 条搜索结果的总结']

--- 测试3：用户 A 再次搜索 ---
[search_node] 用户 user_a 执行搜索: LangGraph 高级用法
[search_node] 保存搜索历史: 9c5b7e3d-...
[summarize_node] 用户 user_a 生成总结
[summarize_node] 找到 2 条历史记录
结果: ['搜索结果1: LangGraph 高级用法 相关内容', '搜索结果2: LangGraph 高级用法 详细信息', '基于 2 条搜索结果的总结']

✅ 场景2 完成
```

## 代码示例 3：多租户用户 ID 管理

这个示例展示如何使用 Runtime Context 实现多租户系统，隔离不同用户的数据。

```python
"""
场景3：多租户用户 ID 管理
演示：使用 Runtime Context 实现用户隔离和权限控制
"""

from dataclasses import dataclass
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

# ===== 1. 定义 State 和 Context =====
class MultiTenantState(TypedDict):
    """多租户状态"""
    action: str
    data: dict
    result: str

@dataclass
class UserContext:
    """用户上下文：包含用户信息和权限"""
    user_id: str
    role: Literal["admin", "user", "guest"]
    tenant_id: str

# ===== 2. 模拟数据库（按租户隔离）=====
TENANT_DATA = {
    "tenant_a": {
        "documents": ["文档A1", "文档A2", "文档A3"],
        "settings": {"theme": "dark", "language": "zh"}
    },
    "tenant_b": {
        "documents": ["文档B1", "文档B2"],
        "settings": {"theme": "light", "language": "en"}
    }
}

# ===== 3. 定义节点函数 =====
def check_permission(state: MultiTenantState, runtime: Runtime[UserContext]) -> MultiTenantState:
    """
    权限检查节点
    """
    user_id = runtime.context.user_id
    role = runtime.context.role
    action = state["action"]

    print(f"[check_permission] 用户 {user_id} (角色: {role}) 请求操作: {action}")

    # 权限规则
    if action == "read" and role in ["admin", "user", "guest"]:
        return {"result": "permission_granted"}
    elif action == "write" and role in ["admin", "user"]:
        return {"result": "permission_granted"}
    elif action == "delete" and role == "admin":
        return {"result": "permission_granted"}
    else:
        return {"result": "permission_denied"}

def execute_action(state: MultiTenantState, runtime: Runtime[UserContext]) -> MultiTenantState:
    """
    执行操作节点（仅在权限通过后执行）
    """
    if state["result"] == "permission_denied":
        print(f"[execute_action] 权限被拒绝，跳过执行")
        return {"result": "操作失败：权限不足"}

    tenant_id = runtime.context.tenant_id
    action = state["action"]

    print(f"[execute_action] 租户 {tenant_id} 执行操作: {action}")

    # 获取租户数据
    tenant_data = TENANT_DATA.get(tenant_id, {})

    if action == "read":
        docs = tenant_data.get("documents", [])
        return {"result": f"读取成功：找到 {len(docs)} 个文档"}
    elif action == "write":
        return {"result": "写入成功：文档已保存"}
    elif action == "delete":
        return {"result": "删除成功：文档已删除"}
    else:
        return {"result": "未知操作"}

# ===== 4. 构建图 =====
def create_multitenant_graph():
    """创建多租户图"""
    builder = StateGraph(MultiTenantState, context_schema=UserContext)

    # 添加节点
    builder.add_node("check_permission", check_permission)
    builder.add_node("execute_action", execute_action)

    # 添加边
    builder.add_edge(START, "check_permission")
    builder.add_edge("check_permission", "execute_action")
    builder.add_edge("execute_action", END)

    return builder.compile()

# ===== 5. 测试 =====
if __name__ == "__main__":
    print("=== 场景3：多租户用户 ID 管理 ===\n")

    graph = create_multitenant_graph()

    # 测试1：管理员读取
    print("--- 测试1：管理员读取 ---")
    result1 = graph.invoke(
        {"action": "read", "data": {}},
        context=UserContext(user_id="admin_001", role="admin", tenant_id="tenant_a")
    )
    print(f"结果: {result1['result']}\n")

    # 测试2：普通用户写入
    print("--- 测试2：普通用户写入 ---")
    result2 = graph.invoke(
        {"action": "write", "data": {"content": "新文档"}},
        context=UserContext(user_id="user_001", role="user", tenant_id="tenant_a")
    )
    print(f"结果: {result2['result']}\n")

    # 测试3：访客尝试删除（权限不足）
    print("--- 测试3：访客尝试删除 ---")
    result3 = graph.invoke(
        {"action": "delete", "data": {"doc_id": "123"}},
        context=UserContext(user_id="guest_001", role="guest", tenant_id="tenant_a")
    )
    print(f"结果: {result3['result']}\n")

    # 测试4：租户 B 的用户读取（数据隔离）
    print("--- 测试4：租户 B 的用户读取 ---")
    result4 = graph.invoke(
        {"action": "read", "data": {}},
        context=UserContext(user_id="user_002", role="user", tenant_id="tenant_b")
    )
    print(f"结果: {result4['result']}\n")

    print("✅ 场景3 完成")
```

### 运行输出示例

```
=== 场景3：多租户用户 ID 管理 ===

--- 测试1：管理员读取 ---
[check_permission] 用户 admin_001 (角色: admin) 请求操作: read
[execute_action] 租户 tenant_a 执行操作: read
结果: 读取成功：找到 3 个文档

--- 测试2：普通用户写入 ---
[check_permission] 用户 user_001 (角色: user) 请求操作: write
[execute_action] 租户 tenant_a 执行操作: write
结果: 写入成功：文档已保存

--- 测试3：访客尝试删除 ---
[check_permission] 用户 guest_001 (角色: guest) 请求操作: delete
[execute_action] 权限被拒绝，跳过执行
结果: 操作失败：权限不足

--- 测试4：租户 B 的用户读取 ---
[check_permission] 用户 user_002 (角色: user) 请求操作: read
[execute_action] 租户 tenant_b 执行操作: read
结果: 读取成功：找到 2 个文档

✅ 场景3 完成
```

## 核心要点总结

### 1. Context vs State 的区别

| 特性 | State | Runtime Context |
|------|-------|-----------------|
| 可变性 | 可变（节点可以更新） | 不可变（运行时固定） |
| 用途 | 存储工作流数据 | 传递配置和依赖 |
| 访问方式 | 直接作为参数 | 通过 `Runtime[Context]` |
| 生命周期 | 随图执行变化 | 整个执行期间不变 |

### 2. 最佳实践

1. **使用 dataclass 定义 Context**
   ```python
   @dataclass
   class Context:
       user_id: str
       config: dict
   ```

2. **在节点中访问 Context**
   ```python
   def node(state: State, runtime: Runtime[Context]):
       user_id = runtime.context.user_id
   ```

3. **运行时传递 Context**
   ```python
   graph.invoke(input, context=Context(user_id="123"))
   ```

4. **不要在 Context 中存储会变化的数据**
   - ❌ 错误：在 Context 中存储计数器、累积结果
   - ✅ 正确：在 State 中存储会变化的数据

### 3. 常见陷阱

1. **误区：尝试修改 Context**
   ```python
   # ❌ 错误：Context 是不可变的
   def node(state, runtime: Runtime[Context]):
       runtime.context.user_id = "new_id"  # 这不会生效
   ```

2. **误区：在 Context 中存储大对象**
   ```python
   # ❌ 不推荐：Context 会在每个节点中传递
   @dataclass
   class Context:
       large_model: LargeModel  # 太大了

   # ✅ 推荐：只存储标识符，在节点中按需加载
   @dataclass
   class Context:
       model_name: str
   ```

## 引用来源

### 官方文档
1. **LangGraph Context7 文档** - Runtime Context 机制
   - 来源：`reference/context7_langgraph_01.md`
   - 章节：Runtime Context（运行时上下文）、节点参数类型、动态 LLM 选择

2. **LangGraph Context7 文档** - Context Schema 定义与使用
   - 来源：`reference/context7_langgraph_01.md`
   - 章节：Context Schema 定义与使用、Functional API 中的 Injectable Parameters

### 技术博客
3. **LinkedIn: Context Engineering with LangGraph**
   - 作者：Sagar Mainkar
   - 来源：`reference/fetch_状态传递_04.md`
   - 主题：状态管理 vs 上下文窗口、外部存储、选择性检索

4. **LangChain 博客: Context Engineering**
   - 来源：`reference/fetch_状态传递_02.md`
   - 主题：Write、Select、Compress、Isolate 策略

### 源码分析
5. **LangGraph 源码分析**
   - 来源：`reference/source_状态传递_01.md`
   - 文件：`pregel/main.py` - CONFIG_KEY_RUNTIME 常量

## 完整代码文件

所有示例代码可以在以下位置找到：
- 场景1：动态 LLM 选择 - `examples/langgraph/runtime_context_llm.py`
- 场景2：数据库连接传递 - `examples/langgraph/runtime_context_database.py`
- 场景3：多租户用户 ID 管理 - `examples/langgraph/runtime_context_multitenant.py`

## 下一步学习

- **场景4：状态流转控制** - 学习如何使用 triggers 和 reducers 控制状态流转
- **L3_工作流编排** - 学习更复杂的工作流模式
- **L4_持久化与检查点** - 学习如何持久化 Context 和 State
