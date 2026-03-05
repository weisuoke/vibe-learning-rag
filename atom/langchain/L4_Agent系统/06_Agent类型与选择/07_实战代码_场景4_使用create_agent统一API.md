# 实战代码：使用 create_agent 统一 API

> 2026 推荐方式：使用现代化的 create_agent API 构建 Agent

---

## 场景说明

**目标**: 使用 LangChain 2026 年推荐的 `create_agent()` 统一 API 构建 Agent。

**为什么使用 create_agent?**
- ✅ **统一接口**: 一个 API 支持所有 Agent 类型
- ✅ **简化代码**: 减少样板代码，更易维护
- ✅ **类型安全**: 更好的类型提示和 IDE 支持
- ✅ **未来兼容**: LangChain 官方推荐的现代化方式
- ✅ **自动优化**: 根据模型能力自动选择最佳策略

**适用场景**:
- 新项目开发（强烈推荐）
- 从旧 API 迁移
- 需要快速原型开发
- 多 Agent 系统（统一接口便于管理）

---

## 完整代码示例

```python
"""
create_agent 统一 API 实战示例
演示：使用现代化 API 构建各种类型的 Agent

场景：
1. 基础 Agent 创建（自动选择类型）
2. 指定模型和工具
3. 自定义系统提示
4. 结构化输出
5. 与旧 API 对比
6. RAG 系统集成
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain 现代化 API
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# ===== 1. 定义工具 =====
print("=== 步骤 1: 定义工具 ===\n")

def search_documents(query: str) -> str:
    """
    搜索文档工具（模拟实现）

    在实际 RAG 系统中，这里会调用向量数据库
    """
    print(f"[工具调用] search_documents(query='{query}')")

    # 模拟检索结果
    results = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，提供了 Agent、Chain、Memory 等核心组件。",
        "agent": "Agent 是能够使用工具并进行推理的 LLM 应用，包括 ReAct、OpenAI Functions 等类型。",
        "rag": "RAG (Retrieval-Augmented Generation) 结合了检索和生成，能够基于外部知识回答问题。"
    }

    # 简单的关键词匹配
    for key, content in results.items():
        if key in query.lower():
            return f"找到相关文档:\n{content}"

    return "未找到相关文档"

def calculate(expression: str) -> str:
    """
    计算器工具

    支持基础数学运算
    """
    print(f"[工具调用] calculate(expression='{expression}')")

    try:
        # 安全的数学表达式求值
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建工具列表
tools = [
    Tool(
        name="search_documents",
        func=search_documents,
        description="搜索文档数据库，查找相关信息。输入应该是搜索查询字符串。"
    ),
    Tool(
        name="calculate",
        func=calculate,
        description="执行数学计算。输入应该是数学表达式，如 '2 + 2' 或 '10 * 5'。"
    )
]

print(f"已创建 {len(tools)} 个工具:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
print()

# ===== 2. 基础用法：自动选择 Agent 类型 =====
print("=== 步骤 2: 基础用法（自动选择类型）===\n")

# 初始化模型（使用 init_chat_model 统一接口）
model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0
)

print(f"模型已初始化: {model.model_name}")
print(f"  - 提供商: OpenAI")
print(f"  - 温度: 0")
print()

# 创建 Agent（最简单的方式）
agent = create_agent(
    model=model,
    tools=tools
)

print("Agent 已创建（自动选择类型）")
print(f"  - 工具数量: {len(tools)}")
print(f"  - Agent 类型: 自动选择（基于模型能力）")
print()

# 测试 Agent
print("【测试 1】基础查询")
print("-" * 60)

response = agent.invoke({
    "messages": [HumanMessage(content="什么是 LangChain？")]
})

print(f"\n用户: 什么是 LangChain？")
print(f"助手: {response['messages'][-1].content}\n")

# ===== 3. 自定义系统提示 =====
print("=== 步骤 3: 自定义系统提示 ===\n")

# 创建带自定义提示的 Agent
agent_with_prompt = create_agent(
    model=model,
    tools=tools,
    system_prompt="""你是一个专业的 RAG 系统助手。

你的职责:
1. 使用 search_documents 工具查找相关信息
2. 基于检索到的信息回答问题
3. 如果需要计算，使用 calculate 工具
4. 回答要准确、简洁、专业

重要规则:
- 优先使用工具获取信息
- 不要编造不存在的信息
- 如果不确定，明确告知用户
"""
)

print("Agent 已创建（带自定义提示）")
print("  - 系统提示: 已设置")
print()

# 测试自定义提示
print("【测试 2】自定义提示效果")
print("-" * 60)

response = agent_with_prompt.invoke({
    "messages": [HumanMessage(content="RAG 是什么？它有什么优势？")]
})

print(f"\n用户: RAG 是什么？它有什么优势？")
print(f"助手: {response['messages'][-1].content}\n")

# ===== 4. 流式输出 =====
print("=== 步骤 4: 流式输出 ===\n")

print("【测试 3】流式输出")
print("-" * 60)
print("\n用户: 计算 123 * 456 的结果")
print("助手: ", end="", flush=True)

# 使用 stream 方法
events = agent.stream(
    {"messages": [HumanMessage(content="计算 123 * 456 的结果")]},
    stream_mode="values"
)

for event in events:
    # 打印最新消息
    last_message = event["messages"][-1]
    if isinstance(last_message, AIMessage):
        print(".", end="", flush=True)

print(f"\n{event['messages'][-1].content}\n")

# ===== 5. 结构化输出 =====
print("=== 步骤 5: 结构化输出 ===\n")

# 定义输出 Schema
class SearchResult(BaseModel):
    """搜索结果的结构化输出"""
    query: str = Field(description="用户的查询")
    found: bool = Field(description="是否找到相关信息")
    summary: str = Field(description="信息摘要")
    confidence: float = Field(description="置信度 (0-1)")

# 创建支持结构化输出的 Agent
from langchain.agents.structured_output import ToolStrategy

agent_structured = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(SearchResult)
)

print("Agent 已创建（结构化输出）")
print(f"  - 输出格式: {SearchResult.__name__}")
print(f"  - 字段: {list(SearchResult.model_fields.keys())}")
print()

# 测试结构化输出
print("【测试 4】结构化输出")
print("-" * 60)

response = agent_structured.invoke({
    "messages": [HumanMessage(content="搜索关于 Agent 的信息")]
})

print(f"\n用户: 搜索关于 Agent 的信息")
print(f"结构化输出:")
print(f"  - query: {response.query}")
print(f"  - found: {response.found}")
print(f"  - summary: {response.summary}")
print(f"  - confidence: {response.confidence}")
print()

# ===== 6. 多轮对话 =====
print("=== 步骤 6: 多轮对话 ===\n")

print("【测试 5】多轮对话")
print("-" * 60)

# 初始化对话历史
conversation = []

# 第一轮
query1 = "什么是 RAG？"
conversation.append(HumanMessage(content=query1))

response1 = agent.invoke({"messages": conversation})
conversation.append(response1["messages"][-1])

print(f"\n用户: {query1}")
print(f"助手: {conversation[-1].content}")

# 第二轮（利用上下文）
query2 = "它在实际项目中如何应用？"
conversation.append(HumanMessage(content=query2))

response2 = agent.invoke({"messages": conversation})
conversation.append(response2["messages"][-1])

print(f"\n用户: {query2}")
print(f"助手: {conversation[-1].content}\n")

# ===== 7. 与旧 API 对比 =====
print("=== 步骤 7: 与旧 API 对比 ===\n")

comparison = """
【旧 API (Deprecated)】
```python
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)

response = agent.run("你的问题")
```

【新 API (推荐)】
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
agent = create_agent(model, tools)

response = agent.invoke({
    "messages": [HumanMessage(content="你的问题")]
})
```

【对比】
| 特性 | 旧 API | 新 API |
|------|--------|--------|
| 代码行数 | 10+ 行 | 3 行 |
| 类型提示 | ❌ 较弱 | ✅ 完善 |
| 自动优化 | ❌ 手动选择 | ✅ 自动选择 |
| 流式输出 | ⚠️ 复杂 | ✅ 简单 |
| 结构化输出 | ❌ 不支持 | ✅ 原生支持 |
| 维护状态 | ⚠️ 已弃用 | ✅ 活跃维护 |
"""

print(comparison)

# ===== 8. RAG 系统集成示例 =====
print("=== 步骤 8: RAG 系统集成示例 ===\n")

def create_rag_agent(
    model_name: str = "gpt-4o-mini",
    vector_store = None,
    system_prompt: Optional[str] = None
):
    """
    创建 RAG Agent 的工厂函数

    参数:
        model_name: 模型名称
        vector_store: 向量数据库实例（可选）
        system_prompt: 自定义系统提示（可选）

    返回:
        配置好的 Agent 实例

    使用示例:
    ```python
    from langchain_community.vectorstores import Milvus
    from langchain_openai import OpenAIEmbeddings

    # 初始化向量数据库
    embeddings = OpenAIEmbeddings()
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"host": "localhost", "port": "19530"}
    )

    # 创建 RAG Agent
    agent = create_rag_agent(
        model_name="gpt-4o-mini",
        vector_store=vector_store
    )

    # 使用 Agent
    response = agent.invoke({
        "messages": [HumanMessage(content="你的问题")]
    })
    ```
    """

    # 初始化模型
    model = init_chat_model(
        model_name,
        model_provider="openai",
        temperature=0
    )

    # 定义检索工具
    def vector_search(query: str) -> str:
        """向量检索工具"""
        if vector_store is None:
            return "向量数据库未初始化"

        # 执行相似度搜索
        docs = vector_store.similarity_search(query, k=5)

        # 格式化结果
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"{i}. {doc.page_content}")

        return "\n\n".join(results)

    # 创建工具列表
    rag_tools = [
        Tool(
            name="vector_search",
            func=vector_search,
            description="在向量数据库中搜索相关文档。输入应该是搜索查询。"
        )
    ]

    # 默认系统提示
    if system_prompt is None:
        system_prompt = """你是一个专业的 RAG 系统助手。

使用 vector_search 工具检索相关文档，然后基于检索到的信息回答用户问题。

重要规则:
1. 优先使用工具获取信息
2. 基于检索结果回答，不要编造
3. 如果检索结果不相关，明确告知用户
4. 引用来源时要准确
"""

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=rag_tools,
        system_prompt=system_prompt
    )

    return agent

print("RAG Agent 工厂函数已定义")
print("  - 支持自定义模型")
print("  - 支持向量数据库集成")
print("  - 支持自定义系统提示")
print()

# ===== 9. 高级配置 =====
print("=== 步骤 9: 高级配置 ===\n")

# 创建带高级配置的 Agent
agent_advanced = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个专业助手",
    # 注意: 以下参数在实际 API 中可能有所不同
    # 请参考最新文档
)

print("Agent 已创建（高级配置）")
print("  - 系统提示: 已设置")
print("  - 工具: 已配置")
print()

# ===== 10. 最佳实践总结 =====
print("=== 步骤 10: 最佳实践总结 ===\n")

best_practices = """
1. **模型初始化**
   ✅ 使用 init_chat_model() 统一接口
   ✅ 明确指定 model_provider
   ✅ 设置合理的 temperature

2. **工具设计**
   ✅ 工具描述要清晰准确
   ✅ 工具功能要单一明确
   ✅ 避免工具功能重叠

3. **系统提示**
   ✅ 明确 Agent 的角色和职责
   ✅ 说明工具使用规则
   ✅ 提供示例（如适用）

4. **错误处理**
   ✅ 工具函数要有异常处理
   ✅ 返回值要明确（成功/失败）
   ✅ 记录关键日志

5. **性能优化**
   ✅ 使用流式输出提升体验
   ✅ 缓存常见查询结果
   ✅ 限制工具调用次数

6. **代码组织**
   ✅ 使用工厂函数封装 Agent 创建
   ✅ 配置与逻辑分离
   ✅ 便于测试和维护
"""

print(best_practices)

# ===== 11. 迁移指南 =====
print("=== 步骤 11: 迁移指南 ===\n")

migration_guide = """
【从 initialize_agent 迁移】

步骤 1: 替换导入
```python
# 旧
from langchain.agents import AgentType, initialize_agent

# 新
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
```

步骤 2: 替换模型初始化
```python
# 旧
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# 新
model = init_chat_model("gpt-4o-mini", model_provider="openai")
```

步骤 3: 替换 Agent 创建
```python
# 旧
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# 新
agent = create_agent(model, tools)
```

步骤 4: 替换调用方式
```python
# 旧
response = agent.run("你的问题")

# 新
response = agent.invoke({
    "messages": [HumanMessage(content="你的问题")]
})
result = response["messages"][-1].content
```

【从 create_*_agent 迁移】

如果你使用的是 create_openai_functions_agent 等:

```python
# 旧
from langchain.agents import create_openai_functions_agent, AgentExecutor

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 新
from langchain.agents import create_agent

agent = create_agent(model, tools, system_prompt="...")
# 不需要 AgentExecutor，直接使用 agent
```
"""

print(migration_guide)

# ===== 12. 常见问题 =====
print("=== 步骤 12: 常见问题 ===\n")

faq = """
Q1: create_agent 支持哪些模型？
A: 支持所有实现了 ChatModel 接口的模型，包括:
   - OpenAI (gpt-4, gpt-4o-mini, gpt-3.5-turbo)
   - Anthropic (claude-3-opus, claude-3-sonnet)
   - Google (gemini-pro)
   - 开源模型 (通过 Ollama 等)

Q2: 如何选择 Agent 类型？
A: create_agent 会根据模型能力自动选择:
   - 支持函数调用 → Tool Calling Agent
   - 不支持函数调用 → ReAct Agent
   - 无需手动指定

Q3: 如何调试 Agent？
A: 使用 LangSmith 追踪:
   ```python
   import os
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
   ```

Q4: 如何限制工具调用次数？
A: 在系统提示中说明:
   ```python
   system_prompt = "最多调用 3 次工具，然后给出答案"
   ```

Q5: 如何处理工具调用失败？
A: 在工具函数中返回错误信息:
   ```python
   def my_tool(input: str) -> str:
       try:
           # 工具逻辑
           return result
       except Exception as e:
           return f"工具调用失败: {str(e)}"
   ```
"""

print(faq)

print("\n" + "=" * 60)
print("create_agent 统一 API 实战示例完成！")
print("=" * 60)
print("\n推荐阅读:")
print("  - LangChain 官方文档: https://docs.langchain.com")
print("  - Agent 类型对比")
print("  - Agent 故障排查指南")
```

---

## 运行输出示例

```
=== 步骤 1: 定义工具 ===

已创建 2 个工具:
  - search_documents: 搜索文档数据库，查找相关信息。输入应该是搜索查询字符串。
  - calculate: 执行数学计算。输入应该是数学表达式，如 '2 + 2' 或 '10 * 5'。

=== 步骤 2: 基础用法（自动选择类型）===

模型已初始化: gpt-4o-mini
  - 提供商: OpenAI
  - 温度: 0

Agent 已创建（自动选择类型）
  - 工具数量: 2
  - Agent 类型: 自动选择（基于模型能力）

【测试 1】基础查询
------------------------------------------------------------

[工具调用] search_documents(query='langchain')

用户: 什么是 LangChain？
助手: LangChain 是一个用于构建 LLM 应用的框架，提供了 Agent、Chain、Memory 等核心组件。

=== 步骤 3: 自定义系统提示 ===

Agent 已创建（带自定义提示）
  - 系统提示: 已设置

【测试 2】自定义提示效果
------------------------------------------------------------

[工具调用] search_documents(query='rag')

用户: RAG 是什么？它有什么优势？
助手: RAG (Retrieval-Augmented Generation) 结合了检索和生成，能够基于外部知识回答问题。它的优势包括...

=== 步骤 4: 流式输出 ===

【测试 3】流式输出
------------------------------------------------------------

用户: 计算 123 * 456 的结果
助手: .........
[工具调用] calculate(expression='123 * 456')
计算结果是 56088。

...
```

---

## 关键要点总结

### 1. create_agent 的核心优势

**简化代码**
```python
# 旧方式: 10+ 行
agent = initialize_agent(...)

# 新方式: 3 行
model = init_chat_model("gpt-4o-mini", model_provider="openai")
agent = create_agent(model, tools)
```

**自动优化**
- 根据模型能力自动选择最佳 Agent 类型
- 无需手动指定 AgentType
- 更好的性能和可靠性

**统一接口**
- 所有 Agent 类型使用相同 API
- 便于切换和测试
- 降低学习成本

### 2. 与旧 API 的对比

| 特性 | initialize_agent | create_agent |
|------|------------------|--------------|
| 代码复杂度 | ⚠️ 高 | ✅ 低 |
| 类型安全 | ❌ 弱 | ✅ 强 |
| 自动优化 | ❌ 无 | ✅ 有 |
| 流式输出 | ⚠️ 复杂 | ✅ 简单 |
| 结构化输出 | ❌ 不支持 | ✅ 支持 |
| 维护状态 | ⚠️ 已弃用 | ✅ 活跃 |
| 学习曲线 | ⚠️ 陡峭 | ✅ 平缓 |

### 3. 实际应用建议

**新项目**
- ✅ 直接使用 create_agent
- ✅ 使用 init_chat_model 初始化模型
- ✅ 利用自动类型选择

**旧项目迁移**
- ⚠️ 逐步迁移（不要一次性全改）
- ⚠️ 先迁移新功能
- ⚠️ 保持向后兼容

**多 Agent 系统**
- ✅ 使用工厂函数封装
- ✅ 统一接口便于管理
- ✅ 配置与逻辑分离

### 4. 性能优化技巧

**模型选择**
```python
# 开发环境: 使用快速模型
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# 生产环境: 根据需求选择
model = init_chat_model("gpt-4o", model_provider="openai")
```

**流式输出**
```python
# 提升用户体验
events = agent.stream(
    {"messages": [HumanMessage(content="问题")]},
    stream_mode="values"
)
for event in events:
    # 实时显示进度
    print(".", end="", flush=True)
```

**缓存策略**
```python
# 缓存常见查询
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str) -> str:
    return search_documents(query)
```

---

## 迁移检查清单

### 从 initialize_agent 迁移

- [ ] 替换导入语句
- [ ] 使用 init_chat_model 初始化模型
- [ ] 替换 agent 创建代码
- [ ] 更新调用方式（run → invoke）
- [ ] 测试功能是否正常
- [ ] 更新文档和注释

### 从 create_*_agent 迁移

- [ ] 移除 AgentExecutor
- [ ] 使用 create_agent 替代
- [ ] 更新系统提示（prompt → system_prompt）
- [ ] 测试功能是否正常
- [ ] 简化代码结构

---

## 下一步学习

- 探索结构化输出的高级用法
- 学习多 Agent 协作模式
- 掌握 LangSmith 调试技巧
- 了解 Agent 性能优化策略

---

## 参考资源

- [LangChain 官方文档](https://docs.langchain.com)
- [create_agent API 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [LangSmith 追踪](https://smith.langchain.com)

---

**版本**: v1.0
**适用**: Python 3.13+ | LangChain 0.3+
**最后更新**: 2026-03-02
