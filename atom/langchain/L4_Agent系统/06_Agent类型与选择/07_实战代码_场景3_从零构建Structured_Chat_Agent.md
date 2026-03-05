# 实战代码：从零构建 Structured Chat Agent

> 完整示例：使用 Structured Chat Agent 处理复杂多输入工具

---

## 场景说明

**目标**: 构建一个能够处理复杂工具调用的 Agent，支持多输入参数和结构化对话。

**适用场景**:
- 工具需要多个输入参数（如搜索工具需要 query + filters + limit）
- 需要结构化的参数传递（如 JSON 对象）
- 工具参数之间有依赖关系
- RAG 系统中的高级检索（混合检索、重排序等）

**为什么选择 Structured Chat Agent?**
- ✅ 支持多输入工具（OpenAI Functions 只支持简单参数）
- ✅ 参数结构化（自动验证参数格式）
- ✅ 更好的错误处理（参数错误时自动重试）
- ✅ 适合复杂业务逻辑

---

## 完整代码示例

```python
"""
Structured Chat Agent 完整实战示例
演示：构建支持复杂工具调用的 RAG 检索 Agent

场景：
1. 定义多输入工具（高级文档检索）
2. 创建 Structured Chat Agent
3. 处理复杂查询请求
4. 集成对话记忆
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain 核心组件
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain_classic.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# ===== 1. 定义复杂工具的输入 Schema =====
print("=== 步骤 1: 定义工具输入 Schema ===\n")

class AdvancedSearchInput(BaseModel):
    """高级文档检索工具的输入参数"""
    query: str = Field(description="用户的搜索查询")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="过滤条件，如 {'category': 'tech', 'date_range': '2024-01'}"
    )
    top_k: int = Field(
        default=5,
        description="返回结果数量，默认 5"
    )
    rerank: bool = Field(
        default=False,
        description="是否启用重排序"
    )

class DocumentSummaryInput(BaseModel):
    """文档摘要工具的输入参数"""
    doc_id: str = Field(description="文档 ID")
    max_length: int = Field(
        default=200,
        description="摘要最大长度（字符数）"
    )
    language: str = Field(
        default="zh",
        description="摘要语言，zh 或 en"
    )

# ===== 2. 实现工具函数 =====
print("=== 步骤 2: 实现工具函数 ===\n")

def advanced_search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    rerank: bool = False
) -> str:
    """
    高级文档检索工具（模拟实现）

    在实际 RAG 系统中，这里会调用:
    - 向量数据库检索
    - 标量过滤
    - 重排序模型
    """
    print(f"[工具调用] advanced_search")
    print(f"  - query: {query}")
    print(f"  - filters: {filters}")
    print(f"  - top_k: {top_k}")
    print(f"  - rerank: {rerank}")

    # 模拟检索结果
    results = [
        {
            "id": "doc_001",
            "title": "LangChain Agent 开发指南",
            "content": "详细介绍了 Agent 的类型和选择...",
            "score": 0.95
        },
        {
            "id": "doc_002",
            "title": "RAG 系统架构设计",
            "content": "RAG 系统的核心组件包括...",
            "score": 0.88
        },
        {
            "id": "doc_003",
            "title": "向量数据库选型",
            "content": "对比了 Milvus、Pinecone、Weaviate...",
            "score": 0.82
        }
    ]

    # 应用过滤器
    if filters:
        print(f"  - 应用过滤器: {filters}")
        # 实际实现中会根据 filters 过滤结果

    # 应用 top_k
    results = results[:top_k]

    # 应用重排序
    if rerank:
        print("  - 启用重排序")
        # 实际实现中会调用重排序模型
        results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 格式化输出
    output = f"找到 {len(results)} 个相关文档:\n\n"
    for i, doc in enumerate(results, 1):
        output += f"{i}. {doc['title']} (相关度: {doc['score']:.2f})\n"
        output += f"   内容: {doc['content']}\n"
        output += f"   文档ID: {doc['id']}\n\n"

    return output

def document_summary(
    doc_id: str,
    max_length: int = 200,
    language: str = "zh"
) -> str:
    """
    文档摘要工具（模拟实现）

    在实际 RAG 系统中，这里会:
    - 从向量数据库获取完整文档
    - 调用 LLM 生成摘要
    """
    print(f"[工具调用] document_summary")
    print(f"  - doc_id: {doc_id}")
    print(f"  - max_length: {max_length}")
    print(f"  - language: {language}")

    # 模拟文档内容
    doc_content = {
        "doc_001": "LangChain Agent 开发指南详细介绍了各种 Agent 类型，包括 OpenAI Functions、ReAct 和 Structured Chat。每种类型都有其适用场景和优缺点。",
        "doc_002": "RAG 系统架构设计涵盖了文档加载、文本分块、向量化、检索和生成等核心流程。合理的架构设计能显著提升系统性能。",
        "doc_003": "向量数据库选型需要考虑性能、成本、易用性等多个维度。Milvus 适合大规模部署，Pinecone 提供托管服务，Weaviate 支持混合检索。"
    }

    content = doc_content.get(doc_id, "文档未找到")

    # 截断到指定长度
    if len(content) > max_length:
        content = content[:max_length] + "..."

    # 根据语言返回
    if language == "en":
        content = f"[English Summary] {content}"

    return f"文档 {doc_id} 的摘要:\n{content}"

# ===== 3. 创建 StructuredTool =====
print("=== 步骤 3: 创建 StructuredTool ===\n")

# 使用 StructuredTool 包装工具函数
advanced_search_tool = StructuredTool.from_function(
    func=advanced_search,
    name="advanced_search",
    description="高级文档检索工具，支持过滤、top_k 和重排序。适用于需要精确控制检索参数的场景。",
    args_schema=AdvancedSearchInput
)

document_summary_tool = StructuredTool.from_function(
    func=document_summary,
    name="document_summary",
    description="文档摘要工具，根据文档 ID 生成指定长度和语言的摘要。",
    args_schema=DocumentSummaryInput
)

tools = [advanced_search_tool, document_summary_tool]

print(f"已创建 {len(tools)} 个工具:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
print()

# ===== 4. 创建 Structured Chat Agent =====
print("=== 步骤 4: 创建 Structured Chat Agent ===\n")

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 创建对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 定义 Prompt 模板
prefix = """你是一个专业的 RAG 系统助手，能够帮助用户检索和理解文档。

你可以使用以下工具:"""

suffix = """开始对话！记住要使用工具来回答用户的问题。

{chat_history}
问题: {input}
{agent_scratchpad}"""

# 创建 Prompt
prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

print("Prompt 模板已创建")
print(f"  - 输入变量: {prompt.input_variables}")
print()

# 创建 LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 创建 Agent
agent = StructuredChatAgent(
    llm_chain=llm_chain,
    tools=tools,
    verbose=True
)

# 创建 AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

print("Structured Chat Agent 已创建")
print(f"  - 工具数量: {len(tools)}")
print(f"  - 最大迭代次数: 5")
print(f"  - 错误处理: 启用")
print()

# ===== 5. 测试 Agent =====
print("=== 步骤 5: 测试 Agent ===\n")

# 测试 1: 简单检索
print("【测试 1】简单检索")
print("-" * 60)
response1 = agent_executor.invoke({
    "input": "搜索关于 LangChain Agent 的文档"
})
print(f"\n回答: {response1['output']}\n")

# 测试 2: 带过滤条件的检索
print("\n【测试 2】带过滤条件的检索")
print("-" * 60)
response2 = agent_executor.invoke({
    "input": "搜索技术类文档，只返回前 3 个结果，并启用重排序"
})
print(f"\n回答: {response2['output']}\n")

# 测试 3: 文档摘要
print("\n【测试 3】文档摘要")
print("-" * 60)
response3 = agent_executor.invoke({
    "input": "给我 doc_001 的摘要，最多 100 字"
})
print(f"\n回答: {response3['output']}\n")

# 测试 4: 多步骤任务（利用记忆）
print("\n【测试 4】多步骤任务（利用记忆）")
print("-" * 60)
response4 = agent_executor.invoke({
    "input": "刚才搜索到的第一个文档是什么？给我它的摘要"
})
print(f"\n回答: {response4['output']}\n")

# ===== 6. 查看对话历史 =====
print("=== 步骤 6: 查看对话历史 ===\n")
print("对话历史:")
for i, message in enumerate(memory.chat_memory.messages, 1):
    role = "用户" if message.type == "human" else "助手"
    print(f"{i}. {role}: {message.content[:100]}...")
print()

# ===== 7. 实际应用示例：RAG 系统集成 =====
print("=== 步骤 7: RAG 系统集成示例 ===\n")

def create_rag_agent_with_structured_chat(
    llm: ChatOpenAI,
    vector_store,  # 实际的向量数据库实例
    reranker=None  # 可选的重排序模型
) -> AgentExecutor:
    """
    创建集成了向量数据库的 RAG Agent

    在实际项目中的使用:
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
    agent = create_rag_agent_with_structured_chat(llm, vector_store)
    ```
    """

    # 定义实际的检索工具
    def real_advanced_search(
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        rerank: bool = False
    ) -> str:
        # 向量检索
        results = vector_store.similarity_search(
            query,
            k=top_k,
            filter=filters
        )

        # 重排序
        if rerank and reranker:
            results = reranker.rerank(query, results)

        # 格式化结果
        return "\n\n".join([doc.page_content for doc in results])

    # 创建工具
    search_tool = StructuredTool.from_function(
        func=real_advanced_search,
        name="vector_search",
        description="向量数据库检索工具",
        args_schema=AdvancedSearchInput
    )

    # 创建 Agent（与上面相同的流程）
    # ...

    return agent_executor

print("RAG 系统集成示例代码已定义")
print("  - 支持向量数据库集成")
print("  - 支持重排序模型")
print("  - 支持标量过滤")
print()

# ===== 8. 性能优化建议 =====
print("=== 步骤 8: 性能优化建议 ===\n")

optimization_tips = """
1. **工具设计优化**
   - 合并相似工具（减少 Agent 选择负担）
   - 工具描述要精确（帮助 Agent 正确选择）
   - 参数默认值要合理（减少用户输入）

2. **Prompt 优化**
   - 明确工具使用场景
   - 提供工具使用示例
   - 限制工具调用次数

3. **错误处理**
   - 启用 handle_parsing_errors
   - 设置合理的 max_iterations
   - 记录工具调用日志

4. **记忆管理**
   - 使用 ConversationBufferWindowMemory（限制历史长度）
   - 定期清理无关历史
   - 使用 ConversationSummaryMemory（压缩历史）

5. **成本控制**
   - 使用 gpt-4o-mini 而非 gpt-4
   - 缓存常见查询结果
   - 限制工具调用次数
"""

print(optimization_tips)

# ===== 9. 常见问题排查 =====
print("=== 步骤 9: 常见问题排查 ===\n")

troubleshooting = """
问题 1: Agent 不调用工具
解决:
  - 检查工具描述是否清晰
  - 检查 Prompt 是否包含工具使用指令
  - 增加 verbose=True 查看推理过程

问题 2: 工具参数解析失败
解决:
  - 检查 args_schema 定义是否正确
  - 启用 handle_parsing_errors=True
  - 在工具描述中说明参数格式

问题 3: Agent 陷入循环
解决:
  - 设置 max_iterations 限制
  - 检查工具返回值是否明确
  - 优化 Prompt 引导 Agent 结束

问题 4: 记忆占用过多 Token
解决:
  - 使用 ConversationBufferWindowMemory
  - 使用 ConversationSummaryMemory
  - 定期清理历史
"""

print(troubleshooting)

print("\n" + "=" * 60)
print("Structured Chat Agent 实战示例完成！")
print("=" * 60)
```

---

## 运行输出示例

```
=== 步骤 1: 定义工具输入 Schema ===

=== 步骤 2: 实现工具函数 ===

=== 步骤 3: 创建 StructuredTool ===

已创建 2 个工具:
  - advanced_search: 高级文档检索工具，支持过滤、top_k 和重排序。适用于需要精确控制检索参数的场景。
  - document_summary: 文档摘要工具，根据文档 ID 生成指定长度和语言的摘要。

=== 步骤 4: 创建 Structured Chat Agent ===

Prompt 模板已创建
  - 输入变量: ['input', 'chat_history', 'agent_scratchpad']

Structured Chat Agent 已创建
  - 工具数量: 2
  - 最大迭代次数: 5
  - 错误处理: 启用

=== 步骤 5: 测试 Agent ===

【测试 1】简单检索
------------------------------------------------------------

> Entering new AgentExecutor chain...
[工具调用] advanced_search
  - query: LangChain Agent
  - filters: None
  - top_k: 5
  - rerank: False

找到 3 个相关文档:

1. LangChain Agent 开发指南 (相关度: 0.95)
   内容: 详细介绍了 Agent 的类型和选择...
   文档ID: doc_001

2. RAG 系统架构设计 (相关度: 0.88)
   内容: RAG 系统的核心组件包括...
   文档ID: doc_002

3. 向量数据库选型 (相关度: 0.82)
   内容: 对比了 Milvus、Pinecone、Weaviate...
   文档ID: doc_003

> Finished chain.

回答: 找到了 3 个关于 LangChain Agent 的相关文档...

【测试 2】带过滤条件的检索
------------------------------------------------------------

> Entering new AgentExecutor chain...
[工具调用] advanced_search
  - query: 技术类文档
  - filters: {'category': 'tech'}
  - top_k: 3
  - rerank: True
  - 应用过滤器: {'category': 'tech'}
  - 启用重排序

...

【测试 4】多步骤任务（利用记忆）
------------------------------------------------------------

> Entering new AgentExecutor chain...
[工具调用] document_summary
  - doc_id: doc_001
  - max_length: 100
  - language: zh

文档 doc_001 的摘要:
LangChain Agent 开发指南详细介绍了各种 Agent 类型，包括 OpenAI Functions、ReAct 和 Structured Chat...

> Finished chain.

回答: 第一个文档是"LangChain Agent 开发指南"，摘要如下...
```

---

## 关键要点总结

### 1. Structured Chat Agent 的优势

- **多输入支持**: 工具可以接受复杂的参数结构
- **类型安全**: 使用 Pydantic 自动验证参数
- **错误处理**: 参数错误时自动重试
- **灵活性**: 适合复杂业务逻辑

### 2. 与 OpenAI Functions Agent 的对比

| 特性 | Structured Chat | OpenAI Functions |
|------|-----------------|------------------|
| 多输入工具 | ✅ 支持 | ❌ 仅简单参数 |
| 参数验证 | ✅ Pydantic | ⚠️ 基础验证 |
| 错误处理 | ✅ 自动重试 | ⚠️ 需手动处理 |
| 性能 | ⚠️ 稍慢 | ✅ 更快 |
| 成本 | ⚠️ 稍高 | ✅ 更低 |

### 3. 实际应用建议

**何时使用 Structured Chat Agent?**
- ✅ 工具需要 3+ 个参数
- ✅ 参数之间有依赖关系
- ✅ 需要复杂的过滤条件
- ✅ RAG 系统的高级检索

**何时不使用?**
- ❌ 工具参数简单（用 OpenAI Functions）
- ❌ 对性能要求极高（用 Tool Calling）
- ❌ 预算有限（用 ReAct）

---

## 下一步学习

- 学习 `create_agent()` 统一 API（2026 推荐方式）
- 了解 Agent 类型迁移策略
- 掌握 Agent 故障排查技巧
- 探索多 Agent 协作模式

---

**版本**: v1.0
**适用**: Python 3.13+ | LangChain 0.3+
**最后更新**: 2026-03-02
