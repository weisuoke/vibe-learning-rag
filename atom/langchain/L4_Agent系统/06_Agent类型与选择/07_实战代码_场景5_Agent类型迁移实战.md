# Agent类型迁移实战

> 从旧 API 迁移到新 API 的完整指南

---

## 场景说明

本文档演示如何将使用旧版 `AgentType` 枚举和 `initialize_agent()` 的代码迁移到 2026 年推荐的现代 API。

**为什么需要迁移？**
- `AgentType` 枚举已在 0.1.0 标记为 deprecated，将在 1.0 移除
- 新 API 提供更好的类型提示和文档
- 更灵活的配置和扩展能力
- 更好的错误处理和调试体验

**迁移路径：**
```
旧版 (Deprecated)          中间版 (稳定)              新版 (推荐)
AgentType 枚举      →    create_*_agent()    →    create_agent()
initialize_agent()        + AgentExecutor           统一 API
```

---

## 迁移场景 1：OpenAI Functions Agent

### Before: 旧版 API (Deprecated)

```python
"""
旧版 OpenAI Functions Agent - 使用 AgentType 枚举
⚠️ 此代码将在 LangChain 1.0 停止工作
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 加载环境变量
load_dotenv()

# ===== 1. 定义工具 =====
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关文档"""
    # 模拟知识库搜索
    knowledge = {
        "RAG": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术",
        "Embedding": "Embedding 是将文本转换为向量的技术",
        "LangChain": "LangChain 是一个用于构建 LLM 应用的框架"
    }

    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return "未找到相关信息"

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建工具列表
tools = [
    Tool(
        name="KnowledgeBase",
        func=search_knowledge_base,
        description="在知识库中搜索技术概念的定义和解释"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="计算数学表达式，输入格式如: 2+2 或 10*5"
    )
]

# ===== 2. 创建 LLM =====
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ===== 3. 使用旧版 API 创建 Agent (Deprecated) =====
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # ⚠️ Deprecated
    verbose=True,
    handle_parsing_errors=True
)

# ===== 4. 运行 Agent =====
print("=== 旧版 API 运行示例 ===\n")

# 测试 1: 知识库查询
response1 = agent.run("什么是 RAG？")
print(f"问题: 什么是 RAG？")
print(f"回答: {response1}\n")

# 测试 2: 计算
response2 = agent.run("计算 15 * 8")
print(f"问题: 计算 15 * 8")
print(f"回答: {response2}\n")
```

**旧版 API 的问题：**
- ❌ 使用已弃用的 `AgentType` 枚举
- ❌ `initialize_agent()` 返回的是 `AgentExecutor`，不够清晰
- ❌ 配置选项混在一起，不够灵活
- ❌ 缺少类型提示，IDE 支持差

---

### After: 新版 API (推荐)

```python
"""
新版 OpenAI Functions Agent - 使用 create_agent() 统一 API
✅ 2026 年推荐方式
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 加载环境变量
load_dotenv()

# ===== 1. 使用 @tool 装饰器定义工具 (现代方式) =====
@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关文档

    Args:
        query: 要搜索的关键词或问题

    Returns:
        相关文档的内容
    """
    # 模拟知识库搜索
    knowledge = {
        "RAG": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术",
        "Embedding": "Embedding 是将文本转换为向量的技术",
        "LangChain": "LangChain 是一个用于构建 LLM 应用的框架"
    }

    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return "未找到相关信息"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式，如 "2+2" 或 "10*5"

    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# ===== 2. 创建 LLM =====
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ===== 3. 使用新版 API 创建 Agent (推荐) =====
agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, calculate],
    system_prompt="""你是一个智能助手，可以使用以下工具：
    1. search_knowledge_base: 搜索技术知识库
    2. calculate: 进行数学计算

    请根据用户问题选择合适的工具，并给出准确的回答。"""
)

# ===== 4. 运行 Agent =====
print("=== 新版 API 运行示例 ===\n")

# 测试 1: 知识库查询
response1 = agent.invoke({"messages": [("user", "什么是 RAG？")]})
print(f"问题: 什么是 RAG？")
print(f"回答: {response1['messages'][-1].content}\n")

# 测试 2: 计算
response2 = agent.invoke({"messages": [("user", "计算 15 * 8")]})
print(f"问题: 计算 15 * 8")
print(f"回答: {response2['messages'][-1].content}\n")

# 测试 3: 多轮对话
print("=== 多轮对话示例 ===\n")
messages = []

# 第一轮
messages.append(("user", "什么是 Embedding？"))
response = agent.invoke({"messages": messages})
messages.append(("assistant", response['messages'][-1].content))
print(f"用户: 什么是 Embedding？")
print(f"助手: {response['messages'][-1].content}\n")

# 第二轮（带上下文）
messages.append(("user", "它在 RAG 中有什么作用？"))
response = agent.invoke({"messages": messages})
print(f"用户: 它在 RAG 中有什么作用？")
print(f"助手: {response['messages'][-1].content}\n")
```

**新版 API 的优势：**
- ✅ 使用 `create_agent()` 统一 API，不会被弃用
- ✅ 使用 `@tool` 装饰器，更简洁的工具定义
- ✅ 清晰的 `system_prompt` 配置
- ✅ 更好的类型提示和 IDE 支持
- ✅ 支持多轮对话（通过 messages 列表）

---

## 迁移场景 2：ReAct Agent

### Before: 旧版 API

```python
"""
旧版 ReAct Agent - 适用于开源模型
⚠️ 使用已弃用的 API
"""

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
def web_search(query: str) -> str:
    """模拟网络搜索"""
    # 实际应用中会调用真实的搜索 API
    return f"搜索结果: 关于 '{query}' 的最新信息..."

tools = [
    Tool(
        name="WebSearch",
        func=web_search,
        description="在互联网上搜索最新信息"
    )
]

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用旧版 API 创建 ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ⚠️ Deprecated
    verbose=True
)

# 运行
response = agent.run("2024年人工智能的最新进展是什么？")
print(response)
```

---

### After: 新版 API

```python
"""
新版 ReAct Agent - 使用 create_agent()
✅ 2026 年推荐方式
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 使用 @tool 装饰器定义工具
@tool
def web_search(query: str) -> str:
    """在互联网上搜索最新信息

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    # 实际应用中会调用真实的搜索 API
    return f"搜索结果: 关于 '{query}' 的最新信息..."

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用新版 API 创建 Agent
agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt="""你是一个研究助手，可以使用网络搜索工具获取最新信息。

    工作流程：
    1. 分析用户问题
    2. 使用 web_search 工具搜索相关信息
    3. 综合搜索结果给出回答

    请确保回答准确、全面。"""
)

# 运行
response = agent.invoke({
    "messages": [("user", "2024年人工智能的最新进展是什么？")]
})
print(response['messages'][-1].content)
```

---

## 迁移场景 3：Structured Chat Agent

### Before: 旧版 API

```python
"""
旧版 Structured Chat Agent - 支持复杂工具参数
⚠️ 使用已弃用的 API
"""

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 定义复杂工具参数
class DocumentSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    category: str = Field(description="文档分类: technical, business, general")
    max_results: int = Field(default=5, description="最大返回结果数")

def search_documents(query: str, category: str, max_results: int = 5) -> str:
    """在文档库中搜索"""
    return f"在 {category} 分类中搜索 '{query}'，返回 {max_results} 条结果"

# 创建结构化工具
tools = [
    StructuredTool.from_function(
        func=search_documents,
        name="DocumentSearch",
        description="在文档库中搜索，支持分类和结果数量控制",
        args_schema=DocumentSearchInput
    )
]

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用旧版 API
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ⚠️ Deprecated
    verbose=True
)

# 运行
response = agent.run("搜索技术文档中关于 RAG 的内容，返回3条结果")
print(response)
```

---

### After: 新版 API

```python
"""
新版 Structured Chat Agent - 使用 create_agent()
✅ 2026 年推荐方式
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field

# 定义工具参数模型
class DocumentSearchInput(BaseModel):
    """文档搜索参数"""
    query: str = Field(description="搜索关键词")
    category: str = Field(description="文档分类: technical, business, general")
    max_results: int = Field(default=5, description="最大返回结果数")

# 使用 @tool 装饰器 + args_schema
@tool(args_schema=DocumentSearchInput)
def search_documents(query: str, category: str, max_results: int = 5) -> str:
    """在文档库中搜索，支持分类和结果数量控制

    Args:
        query: 搜索关键词
        category: 文档分类 (technical, business, general)
        max_results: 最大返回结果数

    Returns:
        搜索结果摘要
    """
    return f"在 {category} 分类中搜索 '{query}'，返回 {max_results} 条结果"

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用新版 API
agent = create_agent(
    model=llm,
    tools=[search_documents],
    system_prompt="""你是一个文档搜索助手，可以使用 search_documents 工具。

    该工具支持以下参数：
    - query: 搜索关键词
    - category: 文档分类 (technical/business/general)
    - max_results: 返回结果数量

    请根据用户需求合理设置参数。"""
)

# 运行
response = agent.invoke({
    "messages": [("user", "搜索技术文档中关于 RAG 的内容，返回3条结果")]
})
print(response['messages'][-1].content)
```

---

## 迁移场景 4：带记忆的对话 Agent

### Before: 旧版 API

```python
"""
旧版对话 Agent - 使用 ConversationalReactDescription
⚠️ 使用已弃用的 API
"""

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# 定义工具
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    users = {
        "001": "张三，VIP用户，余额: 1000元",
        "002": "李四，普通用户，余额: 100元"
    }
    return users.get(user_id, "用户不存在")

tools = [
    Tool(
        name="GetUserInfo",
        func=get_user_info,
        description="获取用户信息，输入用户ID"
    )
]

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用旧版 API
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # ⚠️ Deprecated
    memory=memory,
    verbose=True
)

# 多轮对话
print("第一轮:")
response1 = agent.run("查询用户001的信息")
print(response1)

print("\n第二轮:")
response2 = agent.run("他的余额够买500元的商品吗？")
print(response2)
```

---

### After: 新版 API

```python
"""
新版对话 Agent - 使用 create_agent() + 手动管理对话历史
✅ 2026 年推荐方式
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 使用 @tool 装饰器
@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息

    Args:
        user_id: 用户ID

    Returns:
        用户详细信息
    """
    users = {
        "001": "张三，VIP用户，余额: 1000元",
        "002": "李四，普通用户，余额: 100元"
    }
    return users.get(user_id, "用户不存在")

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 使用新版 API
agent = create_agent(
    model=llm,
    tools=[get_user_info],
    system_prompt="""你是一个客服助手，可以查询用户信息。

    请记住对话历史，回答用户的后续问题。"""
)

# 手动管理对话历史（更灵活）
messages = []

# 第一轮对话
print("=== 第一轮对话 ===")
messages.append(("user", "查询用户001的信息"))
response1 = agent.invoke({"messages": messages})
assistant_msg1 = response1['messages'][-1].content
messages.append(("assistant", assistant_msg1))
print(f"用户: 查询用户001的信息")
print(f"助手: {assistant_msg1}\n")

# 第二轮对话（带上下文）
print("=== 第二轮对话 ===")
messages.append(("user", "他的余额够买500元的商品吗？"))
response2 = agent.invoke({"messages": messages})
assistant_msg2 = response2['messages'][-1].content
print(f"用户: 他的余额够买500元的商品吗？")
print(f"助手: {assistant_msg2}\n")
```

**新版对话管理的优势：**
- ✅ 更灵活的对话历史管理
- ✅ 可以自定义消息格式
- ✅ 更容易实现复杂的对话逻辑
- ✅ 更好的性能（按需加载历史）

---

## 完整迁移检查清单

### 代码层面

- [ ] 移除 `from langchain.agents import AgentType`
- [ ] 移除 `from langchain.agents import initialize_agent`
- [ ] 添加 `from langchain.agents import create_agent`
- [ ] 将 `Tool` 改为 `@tool` 装饰器
- [ ] 将 `agent.run()` 改为 `agent.invoke()`
- [ ] 更新工具定义（添加类型提示和文档字符串）
- [ ] 添加 `system_prompt` 配置
- [ ] 更新对话历史管理方式

### 配置层面

- [ ] 检查环境变量配置（`.env` 文件）
- [ ] 更新依赖版本（`langchain >= 0.1.0`）
- [ ] 测试所有工具是否正常工作
- [ ] 验证多轮对话功能
- [ ] 检查错误处理逻辑

### 测试层面

- [ ] 单元测试覆盖所有工具
- [ ] 集成测试覆盖完整流程
- [ ] 性能测试（对比新旧 API）
- [ ] 边界情况测试
- [ ] 错误恢复测试

---

## 迁移常见问题

### Q1: 迁移后性能有变化吗？

**A:** 新版 API 性能通常更好：
- ✅ 更高效的工具调用机制
- ✅ 更好的缓存策略
- ✅ 减少不必要的中间步骤

### Q2: 旧代码什么时候会完全不能用？

**A:** 根据官方计划：
- 0.1.0: 标记为 deprecated（当前）
- 1.0: 完全移除（预计 2026 年中）
- 建议尽快迁移，避免突然中断

### Q3: 可以混用新旧 API 吗？

**A:** 技术上可以，但不推荐：
- ❌ 代码风格不一致
- ❌ 维护成本高
- ❌ 可能出现兼容性问题
- ✅ 建议一次性完成迁移

### Q4: 迁移后如何调试？

**A:** 新版 API 提供更好的调试支持：
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用 LangSmith 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 查看中间步骤
response = agent.invoke(
    {"messages": [("user", "你的问题")]},
    config={"callbacks": [ConsoleCallbackHandler()]}
)
```

---

## 在 RAG 开发中的应用

### RAG Agent 迁移示例

**场景**: 构建一个 RAG 问答 Agent

```python
"""
RAG Agent 迁移示例
从旧版 API 迁移到新版 create_agent()
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ===== 1. 准备向量数据库 =====
# 加载文档
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# ===== 2. 定义 RAG 工具 =====
@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关文档

    Args:
        query: 用户问题或搜索关键词

    Returns:
        最相关的文档内容
    """
    # 向量检索
    docs = vectorstore.similarity_search(query, k=3)

    # 组合结果
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"找到以下相关内容:\n{context}"

# ===== 3. 创建 RAG Agent =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt="""你是一个 RAG 问答助手。

    工作流程：
    1. 使用 search_knowledge_base 工具检索相关文档
    2. 基于检索到的内容回答用户问题
    3. 如果知识库中没有相关信息，诚实告知用户

    请确保回答准确、有依据。"""
)

# ===== 4. 使用 Agent =====
response = agent.invoke({
    "messages": [("user", "什么是 RAG？")]
})
print(response['messages'][-1].content)
```

**迁移要点：**
- ✅ 使用 `@tool` 装饰器封装 RAG 检索逻辑
- ✅ 在 `system_prompt` 中明确 RAG 工作流程
- ✅ 使用 `invoke()` 方法调用 Agent
- ✅ 更容易集成到现有 RAG 系统

---

## 总结

### 迁移核心要点

1. **API 变化**
   - `AgentType` 枚举 → `create_agent()` 函数
   - `initialize_agent()` → `create_agent()`
   - `Tool` 类 → `@tool` 装饰器
   - `agent.run()` → `agent.invoke()`

2. **配置变化**
   - 分离的配置项 → 统一的 `system_prompt`
   - 内置记忆 → 手动管理对话历史
   - 隐式工具选择 → 显式工具定义

3. **优势**
   - ✅ 更清晰的代码结构
   - ✅ 更好的类型提示
   - ✅ 更灵活的配置
   - ✅ 更容易调试和维护

4. **建议**
   - 尽快完成迁移（1.0 版本前）
   - 一次性迁移所有代码
   - 充分测试迁移后的功能
   - 使用 LangSmith 监控 Agent 行为

---

## 下一步

- 学习 `07_实战代码_场景6_Agent故障排查与类型切换.md` - 诊断和解决迁移后的问题
- 学习 `07_实战代码_场景7_多Agent类型对比测试.md` - 对比不同 Agent 类型的性能
- 参考 `08_面试必问.md` - 掌握 Agent 迁移相关的面试问题

---

**最后更新**: 2026-03-02
**适用版本**: LangChain 0.1.0+
