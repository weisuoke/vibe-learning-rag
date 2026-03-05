# 实战代码 - 场景1：从零构建 OpenAI Functions Agent

> 完整示例：使用 OpenAI Functions Agent 构建一个 RAG 知识库助手

---

## 场景描述

**目标**: 构建一个智能文档助手，能够：
- 搜索向量数据库中的相关文档
- 执行数学计算
- 获取当前时间
- 根据用户问题自动选择合适的工具

**为什么选择 OpenAI Functions Agent?**
- ✅ 工具参数简单（单一输入）
- ✅ 需要可靠的工具调用
- ✅ 使用支持函数调用的模型（OpenAI）
- ✅ 生产环境推荐

---

## 完整代码实现

```python
"""
OpenAI Functions Agent 实战示例
演示：从零构建一个 RAG 知识库助手

场景：用户可以询问文档内容、执行计算、获取时间
Agent 会自动选择合适的工具来回答问题
"""

import os
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 加载环境变量
load_dotenv()

# ===== 1. 定义工具函数 =====
print("=== 步骤1：定义工具函数 ===\n")

# 工具1：向量搜索（模拟 RAG 检索）
def search_documents(query: str) -> str:
    """
    在向量数据库中搜索相关文档

    Args:
        query: 搜索查询

    Returns:
        相关文档内容
    """
    # 模拟向量数据库（实际应用中会连接 Milvus/ChromaDB）
    knowledge_base = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，提供了 Agent、Chain、Memory 等核心组件。",
        "agent": "Agent 是能够使用工具并进行推理的智能体，可以根据用户输入自动选择和执行工具。",
        "rag": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，通过检索相关文档来增强 LLM 的回答质量。",
        "embedding": "Embedding 是将文本转换为向量表示的技术，用于计算文本之间的语义相似度。",
    }

    # 简单的关键词匹配（实际应用中会使用向量相似度）
    query_lower = query.lower()
    results = []
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in value.lower():
            results.append(f"[文档: {key}] {value}")

    if results:
        return "\n\n".join(results)
    else:
        return "未找到相关文档。"

# 工具2：计算器
def calculator(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式（如 "2 + 3 * 4"）

    Returns:
        计算结果
    """
    try:
        # 安全的数学计算（仅支持基本运算）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具3：获取当前时间
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前时间

    Args:
        format: 时间格式（默认: YYYY-MM-DD HH:MM:SS）

    Returns:
        格式化的当前时间
    """
    return f"当前时间: {datetime.now().strftime(format)}"

print("✅ 工具函数定义完成")
print(f"   - search_documents: 搜索知识库")
print(f"   - calculator: 执行数学计算")
print(f"   - get_current_time: 获取当前时间\n")

# ===== 2. 创建 LangChain 工具 =====
print("=== 步骤2：创建 LangChain 工具 ===\n")

tools = [
    Tool(
        name="search_documents",
        func=search_documents,
        description="在知识库中搜索相关文档。输入应该是搜索查询字符串。当用户询问关于 LangChain、Agent、RAG、Embedding 等技术问题时使用此工具。"
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="执行数学计算。输入应该是一个数学表达式字符串，如 '2 + 3 * 4'。当用户需要进行数学运算时使用此工具。"
    ),
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="获取当前时间。输入应该是时间格式字符串（可选，默认为 '%Y-%m-%d %H:%M:%S'）。当用户询问当前时间时使用此工具。"
    ),
]

print(f"✅ 创建了 {len(tools)} 个工具:")
for tool in tools:
    print(f"   - {tool.name}: {tool.description[:50]}...\n")

# ===== 3. 初始化 LLM =====
print("=== 步骤3：初始化 LLM ===\n")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # 降低随机性，提高可靠性
)

print(f"✅ LLM 初始化完成: {llm.model_name}\n")

# ===== 4. 创建 Prompt 模板 =====
print("=== 步骤4：创建 Prompt 模板 ===\n")

# OpenAI Functions Agent 需要特定的 prompt 结构
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能文档助手，可以帮助用户：
1. 搜索和查询知识库中的文档
2. 执行数学计算
3. 获取当前时间

请根据用户的问题，选择合适的工具来回答。
如果需要使用工具，请明确说明你在做什么。
如果不需要工具，可以直接回答。

回答要求：
- 简洁明了
- 基于事实
- 如果使用了工具，说明工具的输出"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

print("✅ Prompt 模板创建完成\n")

# ===== 5. 创建 OpenAI Functions Agent =====
print("=== 步骤5：创建 OpenAI Functions Agent ===\n")

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

print("✅ OpenAI Functions Agent 创建完成\n")

# ===== 6. 创建 AgentExecutor =====
print("=== 步骤6：创建 AgentExecutor ===\n")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细的执行过程
    max_iterations=5,  # 最大迭代次数
    handle_parsing_errors=True,  # 处理解析错误
)

print("✅ AgentExecutor 创建完成\n")
print("=" * 60)
print()

# ===== 7. 测试 Agent =====
print("=== 步骤7：测试 Agent ===\n")

# 测试用例
test_cases = [
    {
        "query": "什么是 RAG？",
        "expected_tool": "search_documents",
        "description": "测试知识库搜索"
    },
    {
        "query": "计算 15 * 8 + 32",
        "expected_tool": "calculator",
        "description": "测试数学计算"
    },
    {
        "query": "现在几点了？",
        "expected_tool": "get_current_time",
        "description": "测试时间查询"
    },
    {
        "query": "LangChain 的 Agent 是什么？它能做什么？",
        "expected_tool": "search_documents",
        "description": "测试复杂查询"
    },
]

# 执行测试
for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}")
    print(f"测试 {i}/{len(test_cases)}: {test_case['description']}")
    print(f"{'=' * 60}")
    print(f"问题: {test_case['query']}")
    print(f"预期工具: {test_case['expected_tool']}")
    print(f"\n执行过程:")
    print("-" * 60)

    try:
        # 调用 Agent
        result = agent_executor.invoke({
            "input": test_case['query'],
            "chat_history": []
        })

        print("-" * 60)
        print(f"\n最终回答:")
        print(result['output'])
        print()

    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}\n")

# ===== 8. 多轮对话示例 =====
print("\n" + "=" * 60)
print("=== 步骤8：多轮对话示例 ===")
print("=" * 60 + "\n")

# 模拟多轮对话
chat_history = []

conversations = [
    "什么是 Embedding？",
    "它在 RAG 中有什么作用？",
    "如果我有 1000 个文档，每个文档转换为 1536 维的向量，总共需要多少个数字？",
]

for i, user_input in enumerate(conversations, 1):
    print(f"\n{'=' * 60}")
    print(f"对话轮次 {i}")
    print(f"{'=' * 60}")
    print(f"用户: {user_input}")
    print(f"\n执行过程:")
    print("-" * 60)

    try:
        # 调用 Agent（带对话历史）
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        print("-" * 60)
        print(f"\nAssistant: {result['output']}")

        # 更新对话历史
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=result['output'])
        ])

    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")

# ===== 9. 总结 =====
print("\n" + "=" * 60)
print("=== 总结 ===")
print("=" * 60 + "\n")

print("✅ OpenAI Functions Agent 构建完成！")
print("\n核心特点:")
print("1. 自动工具选择 - Agent 根据问题自动选择合适的工具")
print("2. 结构化调用 - 使用 OpenAI 的函数调用能力，可靠性高")
print("3. 多轮对话 - 支持对话历史，可以进行上下文相关的对话")
print("4. 错误处理 - 内置错误处理机制，提高鲁棒性")
print("\n适用场景:")
print("- RAG 知识库助手")
print("- 文档问答系统")
print("- 智能客服")
print("- 数据分析助手")
print("\n下一步:")
print("- 连接真实的向量数据库（Milvus/ChromaDB）")
print("- 添加更多专业工具（API 调用、数据库查询等）")
print("- 部署为 API 服务（FastAPI）")
print("- 添加监控和日志（LangSmith）")
```

---

## 运行输出示例

```
=== 步骤1：定义工具函数 ===

✅ 工具函数定义完成
   - search_documents: 搜索知识库
   - calculator: 执行数学计算
   - get_current_time: 获取当前时间

=== 步骤2：创建 LangChain 工具 ===

✅ 创建了 3 个工具:
   - search_documents: 在知识库中搜索相关文档。输入应该是搜索查询字符串。当用户询问关于...
   - calculator: 执行数学计算。输入应该是一个数学表达式字符串，如 '2 + 3 * 4'...
   - get_current_time: 获取当前时间。输入应该是时间格式字符串（可选，默认为 '%Y-%m-%d...

=== 步骤3：初始化 LLM ===

✅ LLM 初始化完成: gpt-4o-mini

=== 步骤4：创建 Prompt 模板 ===

✅ Prompt 模板创建完成

=== 步骤5：创建 OpenAI Functions Agent ===

✅ OpenAI Functions Agent 创建完成

=== 步骤6：创建 AgentExecutor ===

✅ AgentExecutor 创建完成

============================================================

=== 步骤7：测试 Agent ===


============================================================
测试 1/4: 测试知识库搜索
============================================================
问题: 什么是 RAG？
预期工具: search_documents

执行过程:
------------------------------------------------------------


> Entering new AgentExecutor chain...

Invoking: `search_documents` with `{'query': 'RAG'}`


[文档: rag] RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，通过检索相关文档来增强 LLM 的回答质量。

RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，通过检索相关文档来增强 LLM 的回答质量。

> Finished chain.
------------------------------------------------------------

最终回答:
RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术，通过检索相关文档来增强 LLM 的回答质量。

============================================================
测试 2/4: 测试数学计算
============================================================
问题: 计算 15 * 8 + 32
预期工具: calculator

执行过程:
------------------------------------------------------------


> Entering new AgentExecutor chain...

Invoking: `calculator` with `{'expression': '15 * 8 + 32'}`


计算结果: 15 * 8 + 32 = 152

计算结果为 152。

> Finished chain.
------------------------------------------------------------

最终回答:
计算结果为 152。

============================================================
测试 3/4: 测试时间查询
============================================================
问题: 现在几点了？
预期工具: get_current_time

执行过程:
------------------------------------------------------------


> Entering new AgentExecutor chain...

Invoking: `get_current_time` with `{'format': '%Y-%m-%d %H:%M:%S'}`


当前时间: 2026-03-02 14:23:45

当前时间是 2026-03-02 14:23:45。

> Finished chain.
------------------------------------------------------------

最终回答:
当前时间是 2026-03-02 14:23:45。
```

---

## 代码详解

### 1. 工具定义

**关键点**:
- 每个工具都是一个简单的 Python 函数
- 参数简单（单一输入）- 这是选择 OpenAI Functions Agent 的原因
- 返回字符串结果

**在 RAG 开发中**:
- `search_documents` 对应向量数据库检索
- 实际应用中会使用 Milvus/ChromaDB 进行语义搜索

### 2. LangChain Tool 包装

```python
Tool(
    name="search_documents",
    func=search_documents,
    description="..."  # 重要！Agent 根据描述选择工具
)
```

**description 的重要性**:
- Agent 通过描述理解工具的用途
- 描述要清晰、具体、包含使用场景
- 好的描述 = 更准确的工具选择

### 3. Prompt 结构

OpenAI Functions Agent 需要特定的 prompt 结构:
- `system`: 系统指令
- `MessagesPlaceholder("chat_history")`: 对话历史
- `human`: 用户输入
- `MessagesPlaceholder("agent_scratchpad")`: Agent 的思考过程

### 4. AgentExecutor 配置

```python
AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 调试时开启
    max_iterations=5,  # 防止无限循环
    handle_parsing_errors=True,  # 提高鲁棒性
)
```

---

## 与 RAG 开发的联系

### 1. 知识库检索

```python
def search_documents(query: str) -> str:
    # 实际应用中的实现
    # 1. 将 query 转换为 embedding
    # 2. 在 Milvus 中进行向量检索
    # 3. 返回 top-k 相关文档
    pass
```

### 2. 多工具协作

在 RAG 系统中，Agent 可能需要：
- 搜索向量数据库
- 调用外部 API
- 执行数据处理
- 生成最终回答

OpenAI Functions Agent 能够自动协调这些工具。

### 3. 对话式 RAG

通过 `chat_history`，Agent 可以：
- 理解上下文
- 进行多轮对话
- 追问和澄清

---

## 最佳实践

### 1. 工具设计

✅ **好的工具**:
- 单一职责
- 参数简单
- 描述清晰
- 错误处理完善

❌ **避免**:
- 工具功能过于复杂
- 参数过多（考虑 Structured Chat Agent）
- 描述模糊

### 2. Prompt 优化

✅ **好的 Prompt**:
- 明确 Agent 的角色
- 说明工具的使用场景
- 提供输出格式要求

### 3. 错误处理

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 自动处理解析错误
    max_iterations=5,  # 防止无限循环
)
```

---

## 常见问题

### Q1: Agent 不调用工具怎么办？

**原因**:
- 工具描述不清晰
- Prompt 没有引导 Agent 使用工具
- 问题本身不需要工具

**解决**:
- 优化工具描述
- 在 system prompt 中明确说明工具的使用场景
- 使用 `verbose=True` 查看 Agent 的思考过程

### Q2: 如何添加更多工具？

```python
# 只需要添加到 tools 列表
tools.append(Tool(
    name="new_tool",
    func=new_tool_function,
    description="..."
))
```

### Q3: 如何连接真实的向量数据库？

```python
from langchain_milvus import Milvus

def search_documents(query: str) -> str:
    # 连接 Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="knowledge_base"
    )

    # 检索相关文档
    docs = vectorstore.similarity_search(query, k=3)

    # 格式化结果
    return "\n\n".join([doc.page_content for doc in docs])
```

---

## 下一步

1. **场景2**: 学习 ReAct Agent（适合开源模型）
2. **场景3**: 学习 Structured Chat Agent（复杂工具参数）
3. **场景4**: 使用 `create_agent()` 统一 API
4. **场景5**: Agent 类型迁移实战
5. **场景6**: Agent 故障排查与类型切换

---

**学习检查清单**:
- [ ] 理解 OpenAI Functions Agent 的工作原理
- [ ] 能够定义和包装工具
- [ ] 理解 Prompt 结构的重要性
- [ ] 能够处理多轮对话
- [ ] 知道如何调试 Agent（verbose=True）
- [ ] 理解与 RAG 开发的联系
- [ ] 能够连接真实的向量数据库

---

**参考资源**:
- [LangChain Agent 官方文档](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Functions 文档](https://platform.openai.com/docs/guides/function-calling)
- [Milvus Python SDK](https://milvus.io/docs/install-pymilvus.md)
