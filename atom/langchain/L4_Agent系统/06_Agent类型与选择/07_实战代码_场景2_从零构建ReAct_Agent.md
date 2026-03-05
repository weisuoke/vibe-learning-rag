# 实战代码 - 场景2：从零构建 ReAct Agent

> 完整示例：使用 ReAct Agent 构建一个开源模型驱动的研究助手

---

## 场景描述

**目标**: 构建一个研究助手，能够：
- 搜索网络信息
- 查询本地知识库
- 进行逻辑推理
- 显式展示思考过程（Thought-Action-Observation）

**为什么选择 ReAct Agent?**
- ✅ 支持开源模型（不需要函数调用能力）
- ✅ 推理过程可见，便于调试
- ✅ 灵活性高，适合复杂推理任务
- ✅ 可解释性强

---

## ReAct 模式详解

**ReAct = Reasoning + Acting**

```
循环流程:
1. Thought (思考): Agent 分析当前状态，决定下一步
2. Action (行动): 选择并执行一个工具
3. Observation (观察): 获取工具执行结果
4. 重复 1-3，直到得出最终答案
```

**与 OpenAI Functions 的区别**:
- OpenAI Functions: 结构化函数调用（黑盒）
- ReAct: 显式推理过程（白盒）

---

## 完整代码实现

```python
"""
ReAct Agent 实战示例
演示：从零构建一个研究助手

场景：用户可以询问需要推理的复杂问题
Agent 会显式展示思考过程，逐步推理得出答案
"""

import os
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

# ===== 1. 定义工具函数 =====
print("=== 步骤1：定义工具函数 ===\n")

# 工具1：网络搜索（模拟）
def web_search(query: str) -> str:
    """
    搜索网络信息

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    # 模拟网络搜索结果
    search_results = {
        "langchain": "LangChain 是由 Harrison Chase 在 2022 年创建的开源框架，用于构建 LLM 应用。",
        "react": "ReAct (Reasoning and Acting) 是 2022 年提出的一种 Agent 架构，结合了推理和行动。",
        "openai": "OpenAI 在 2023 年推出了函数调用功能，使 Agent 能够更可靠地调用工具。",
        "milvus": "Milvus 是一个开源向量数据库，支持大规模向量检索，常用于 RAG 系统。",
        "python": "Python 3.13 是最新版本，发布于 2024 年，带来了性能提升和新特性。",
    }

    query_lower = query.lower()
    results = []
    for key, value in search_results.items():
        if key in query_lower:
            results.append(f"[搜索结果] {value}")

    if results:
        return "\n".join(results)
    else:
        return "[搜索结果] 未找到相关信息。"

# 工具2：知识库查询
def query_knowledge_base(topic: str) -> str:
    """
    查询本地知识库

    Args:
        topic: 查询主题

    Returns:
        知识库内容
    """
    knowledge_base = {
        "agent": """
        Agent 的三种主要类型:
        1. OpenAI Functions Agent - 使用函数调用，可靠性高
        2. ReAct Agent - 显式推理，适合开源模型
        3. Structured Chat Agent - 支持复杂工具参数
        """,
        "rag": """
        RAG 系统的核心流程:
        1. 文档加载与解析
        2. 文本分块 (Chunking)
        3. 向量化 (Embedding)
        4. 向量存储 (Vector Store)
        5. 检索 (Retrieval)
        6. 生成 (Generation)
        """,
        "embedding": """
        Embedding 模型选择:
        - OpenAI text-embedding-3-small: 1536 维，性价比高
        - OpenAI text-embedding-3-large: 3072 维，精度更高
        - sentence-transformers: 开源方案，可本地部署
        """,
    }

    topic_lower = topic.lower()
    for key, value in knowledge_base.items():
        if key in topic_lower:
            return f"[知识库] {value.strip()}"

    return "[知识库] 未找到相关主题。"

# 工具3：逻辑推理
def logical_reasoning(premise: str) -> str:
    """
    进行逻辑推理

    Args:
        premise: 前提条件

    Returns:
        推理结果
    """
    # 简单的逻辑推理示例
    reasoning_rules = {
        "如果": "基于条件推理",
        "因为": "因果关系推理",
        "所以": "结论推导",
        "比较": "对比分析",
    }

    for keyword, rule_type in reasoning_rules.items():
        if keyword in premise:
            return f"[推理] 这是一个{rule_type}问题。让我分析一下：{premise}"

    return f"[推理] 分析前提：{premise}"

# 工具4：计算器
def calculator(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"[计算] {expression} = {result}"
    except Exception as e:
        return f"[计算错误] {str(e)}"

print("✅ 工具函数定义完成")
print(f"   - web_search: 搜索网络信息")
print(f"   - query_knowledge_base: 查询本地知识库")
print(f"   - logical_reasoning: 进行逻辑推理")
print(f"   - calculator: 执行数学计算\n")

# ===== 2. 创建 LangChain 工具 =====
print("=== 步骤2：创建 LangChain 工具 ===\n")

tools = [
    Tool(
        name="WebSearch",
        func=web_search,
        description="搜索网络信息。当需要查找最新信息、事实、定义时使用。输入应该是搜索查询字符串。"
    ),
    Tool(
        name="KnowledgeBase",
        func=query_knowledge_base,
        description="查询本地知识库。当需要查找已知的技术文档、概念解释时使用。输入应该是主题关键词。"
    ),
    Tool(
        name="LogicalReasoning",
        func=logical_reasoning,
        description="进行逻辑推理。当需要分析条件、推导结论、进行对比时使用。输入应该是需要推理的前提条件。"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="执行数学计算。当需要进行数值运算时使用。输入应该是数学表达式。"
    ),
]

print(f"✅ 创建了 {len(tools)} 个工具\n")

# ===== 3. 初始化 LLM =====
print("=== 步骤3：初始化 LLM ===\n")

# ReAct Agent 可以使用任何 LLM（包括开源模型）
# 这里使用 OpenAI 作为示例，但可以替换为 Ollama、HuggingFace 等
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # ReAct 需要稳定的推理
)

print(f"✅ LLM 初始化完成: {llm.model_name}")
print("   注意: ReAct Agent 也支持开源模型（Llama, Mistral 等）\n")

# ===== 4. 创建 ReAct Prompt =====
print("=== 步骤4：创建 ReAct Prompt ===\n")

# ReAct Agent 使用特定的 prompt 模板
# 包含 Thought-Action-Observation 循环的指导
react_prompt = PromptTemplate.from_template("""
你是一个研究助手，能够使用工具来回答问题。

回答问题时，请遵循以下格式：

Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要执行的动作，必须是以下之一: [{tool_names}]
Action Input: 动作的输入
Observation: 动作的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

重要规则：
1. 每次只能执行一个 Action
2. Action 必须是提供的工具之一
3. 如果需要多个工具，分多次执行
4. 最后必须给出 Final Answer

可用工具：
{tools}

工具描述：
{tool_names}

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

print("✅ ReAct Prompt 创建完成")
print("   包含 Thought-Action-Observation 循环指导\n")

# ===== 5. 创建 ReAct Agent =====
print("=== 步骤5：创建 ReAct Agent ===\n")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

print("✅ ReAct Agent 创建完成\n")

# ===== 6. 创建 AgentExecutor =====
print("=== 步骤6：创建 AgentExecutor ===\n")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示完整的推理过程
    max_iterations=10,  # ReAct 可能需要更多迭代
    handle_parsing_errors=True,
    return_intermediate_steps=True,  # 返回中间步骤
)

print("✅ AgentExecutor 创建完成\n")
print("=" * 60)
print()

# ===== 7. 测试 ReAct Agent =====
print("=== 步骤7：测试 ReAct Agent ===\n")

# 测试用例 - 需要多步推理的问题
test_cases = [
    {
        "query": "LangChain 是什么时候创建的？它的主要用途是什么？",
        "description": "测试网络搜索 + 知识库查询",
        "expected_steps": ["WebSearch", "KnowledgeBase"]
    },
    {
        "query": "如果我有 100 个文档，每个文档平均 1000 个 token，使用 OpenAI text-embedding-3-small 进行向量化，总共需要多少维度的存储空间？",
        "description": "测试推理 + 计算",
        "expected_steps": ["KnowledgeBase", "Calculator"]
    },
    {
        "query": "比较 OpenAI Functions Agent 和 ReAct Agent 的优缺点",
        "description": "测试知识库查询 + 逻辑推理",
        "expected_steps": ["KnowledgeBase", "LogicalReasoning"]
    },
]

# 执行测试
for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}")
    print(f"测试 {i}/{len(test_cases)}: {test_case['description']}")
    print(f"{'=' * 60}")
    print(f"问题: {test_case['query']}")
    print(f"预期步骤: {', '.join(test_case['expected_steps'])}")
    print(f"\n推理过程:")
    print("-" * 60)

    try:
        # 调用 Agent
        result = agent_executor.invoke({
            "input": test_case['query']
        })

        print("-" * 60)
        print(f"\n最终回答:")
        print(result['output'])

        # 显示中间步骤
        if 'intermediate_steps' in result:
            print(f"\n执行的工具:")
            for step in result['intermediate_steps']:
                action, observation = step
                print(f"  - {action.tool}: {action.tool_input[:50]}...")

        print()

    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}\n")

# ===== 8. 复杂推理示例 =====
print("\n" + "=" * 60)
print("=== 步骤8：复杂推理示例 ===")
print("=" * 60 + "\n")

complex_query = """
我想构建一个 RAG 系统，需要处理 10000 个文档。
请帮我分析：
1. 应该选择哪种 Agent 类型？
2. 如果使用 OpenAI text-embedding-3-small (1536维)，存储需要多少空间？
3. 推荐使用哪个向量数据库？
"""

print(f"复杂问题:\n{complex_query}")
print(f"\n推理过程:")
print("-" * 60)

try:
    result = agent_executor.invoke({
        "input": complex_query
    })

    print("-" * 60)
    print(f"\n最终回答:")
    print(result['output'])

    # 分析推理路径
    if 'intermediate_steps' in result:
        print(f"\n推理路径分析:")
        print(f"总共执行了 {len(result['intermediate_steps'])} 个步骤:")
        for i, step in enumerate(result['intermediate_steps'], 1):
            action, observation = step
            print(f"\n步骤 {i}:")
            print(f"  工具: {action.tool}")
            print(f"  输入: {action.tool_input}")
            print(f"  输出: {observation[:100]}...")

except Exception as e:
    print(f"\n❌ 执行失败: {str(e)}")

# ===== 9. ReAct vs OpenAI Functions 对比 =====
print("\n" + "=" * 60)
print("=== 步骤9：ReAct vs OpenAI Functions 对比 ===")
print("=" * 60 + "\n")

comparison = """
| 特性 | ReAct Agent | OpenAI Functions Agent |
|------|-------------|------------------------|
| 推理可见性 | ✅ 显式 Thought-Action-Observation | ❌ 黑盒函数调用 |
| 模型要求 | ✅ 任何 LLM（包括开源） | ❌ 需要函数调用支持 |
| 可靠性 | ⚠️ 中等（依赖 prompt） | ✅ 高（结构化调用） |
| 调试难度 | ✅ 容易（过程可见） | ⚠️ 较难（过程隐藏） |
| Token 消耗 | ⚠️ 较高（显式推理） | ✅ 较低（结构化） |
| 适用场景 | 开源模型、复杂推理、调试 | 生产环境、简单工具 |
"""

print(comparison)

# ===== 10. 总结 =====
print("\n" + "=" * 60)
print("=== 总结 ===")
print("=" * 60 + "\n")

print("✅ ReAct Agent 构建完成！")
print("\n核心特点:")
print("1. 显式推理 - Thought-Action-Observation 循环可见")
print("2. 模型灵活 - 支持任何 LLM，包括开源模型")
print("3. 可解释性 - 推理过程清晰，便于调试")
print("4. 多步推理 - 适合需要复杂推理的任务")
print("\n适用场景:")
print("- 使用开源模型（Llama, Mistral, Qwen 等）")
print("- 需要可解释的推理过程")
print("- 复杂的多步推理任务")
print("- 调试和开发阶段")
print("\n何时选择 ReAct?")
print("✅ 使用开源模型")
print("✅ 需要显式推理过程")
print("✅ 调试和可解释性重要")
print("✅ 复杂的逻辑推理任务")
print("\n何时选择 OpenAI Functions?")
print("✅ 生产环境")
print("✅ 需要高可靠性")
print("✅ 工具调用简单")
print("✅ 成本敏感（Token 消耗）")
print("\n下一步:")
print("- 尝试使用开源模型（Ollama + Llama）")
print("- 优化 ReAct prompt 提高准确性")
print("- 添加更多推理工具")
print("- 对比不同 Agent 类型的性能")
```

---

## 运行输出示例

```
=== 步骤1：定义工具函数 ===

✅ 工具函数定义完成
   - web_search: 搜索网络信息
   - query_knowledge_base: 查询本地知识库
   - logical_reasoning: 进行逻辑推理
   - calculator: 执行数学计算

=== 步骤2：创建 LangChain 工具 ===

✅ 创建了 4 个工具

=== 步骤3：初始化 LLM ===

✅ LLM 初始化完成: gpt-4o-mini
   注意: ReAct Agent 也支持开源模型（Llama, Mistral 等）

=== 步骤4：创建 ReAct Prompt ===

✅ ReAct Prompt 创建完成
   包含 Thought-Action-Observation 循环指导

=== 步骤5：创建 ReAct Agent ===

✅ ReAct Agent 创建完成

=== 步骤6：创建 AgentExecutor ===

✅ AgentExecutor 创建完成

============================================================

=== 步骤7：测试 ReAct Agent ===


============================================================
测试 1/3: 测试网络搜索 + 知识库查询
============================================================
问题: LangChain 是什么时候创建的？它的主要用途是什么？
预期步骤: WebSearch, KnowledgeBase

推理过程:
------------------------------------------------------------


> Entering new AgentExecutor chain...

Thought: 我需要先搜索 LangChain 的创建时间
Action: WebSearch
Action Input: LangChain

[搜索结果] LangChain 是由 Harrison Chase 在 2022 年创建的开源框架，用于构建 LLM 应用。

Thought: 现在我知道创建时间了，接下来查询它的主要用途
Action: KnowledgeBase
Action Input: agent

[知识库]
        Agent 的三种主要类型:
        1. OpenAI Functions Agent - 使用函数调用，可靠性高
        2. ReAct Agent - 显式推理，适合开源模型
        3. Structured Chat Agent - 支持复杂工具参数

Thought: 我现在知道最终答案了
Final Answer: LangChain 是由 Harrison Chase 在 2022 年创建的开源框架。它的主要用途是构建 LLM 应用，提供了 Agent、Chain、Memory 等核心组件，帮助开发者快速构建基于大语言模型的应用。

> Finished chain.
------------------------------------------------------------

最终回答:
LangChain 是由 Harrison Chase 在 2022 年创建的开源框架。它的主要用途是构建 LLM 应用，提供了 Agent、Chain、Memory 等核心组件，帮助开发者快速构建基于大语言模型的应用。

执行的工具:
  - WebSearch: LangChain
  - KnowledgeBase: agent
```

---

## 代码详解

### 1. ReAct Prompt 结构

```python
react_prompt = PromptTemplate.from_template("""
Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要执行的动作
Action Input: 动作的输入
Observation: 动作的结果
... (重复)
Thought: 我现在知道最终答案了
Final Answer: 最终答案
""")
```

**关键点**:
- `Thought`: Agent 的思考过程
- `Action`: 选择的工具
- `Action Input`: 工具的输入
- `Observation`: 工具的输出
- `Final Answer`: 最终答案

### 2. 与 OpenAI Functions 的区别

**OpenAI Functions**:
```
用户问题 → [黑盒] → 工具调用 → 结果
```

**ReAct**:
```
用户问题
  ↓
Thought: 我需要搜索信息
  ↓
Action: WebSearch
  ↓
Observation: [搜索结果]
  ↓
Thought: 我需要进一步查询
  ↓
Action: KnowledgeBase
  ↓
Observation: [知识库结果]
  ↓
Final Answer: [综合答案]
```

### 3. 中间步骤追踪

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,  # 返回中间步骤
)

# 获取推理路径
result = agent_executor.invoke({"input": query})
for step in result['intermediate_steps']:
    action, observation = step
    print(f"工具: {action.tool}, 输入: {action.tool_input}")
```

---

## 与 RAG 开发的联系

### 1. 开源模型支持

ReAct Agent 是开源模型的最佳选择:

```python
# 使用 Ollama + Llama
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
agent = create_react_agent(llm, tools, react_prompt)
```

### 2. 复杂推理场景

在 RAG 系统中，可能需要：
- 先搜索向量数据库
- 再分析检索结果
- 最后综合生成答案

ReAct Agent 的显式推理非常适合这种场景。

### 3. 调试和优化

通过查看 Thought-Action-Observation，可以：
- 理解 Agent 的决策过程
- 发现工具选择错误
- 优化 prompt 和工具描述

---

## 使用开源模型示例

```python
# 使用 Ollama + Llama 3
from langchain_community.llms import Ollama

# 1. 安装 Ollama: https://ollama.ai
# 2. 下载模型: ollama pull llama3

llm = Ollama(
    model="llama3",
    temperature=0,
)

# 创建 ReAct Agent（代码相同）
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({"input": "什么是 RAG？"})
```

**优势**:
- ✅ 完全本地运行
- ✅ 无 API 成本
- ✅ 数据隐私保护
- ✅ 可定制化

---

## 最佳实践

### 1. Prompt 优化

✅ **好的 ReAct Prompt**:
- 明确 Thought-Action-Observation 格式
- 提供清晰的工具描述
- 包含示例（Few-shot）
- 强调最终答案格式

### 2. 工具设计

✅ **ReAct 友好的工具**:
- 返回清晰的文本结果
- 避免复杂的结构化输出
- 提供详细的描述

### 3. 错误处理

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # ReAct 可能需要更多迭代
    handle_parsing_errors=True,  # 处理格式错误
    early_stopping_method="generate",  # 提前停止策略
)
```

---

## 常见问题

### Q1: ReAct Agent 不遵循格式怎么办？

**原因**:
- Prompt 不够清晰
- 模型能力不足
- 工具描述模糊

**解决**:
- 优化 prompt，添加示例
- 使用更强的模型
- 简化工具描述

### Q2: 如何减少 Token 消耗？

**策略**:
- 减少 Thought 的冗长度
- 优化工具输出（只返回必要信息）
- 设置 `max_iterations` 限制

### Q3: ReAct vs OpenAI Functions 如何选择？

**选择 ReAct**:
- ✅ 使用开源模型
- ✅ 需要可解释性
- ✅ 复杂推理任务
- ✅ 调试阶段

**选择 OpenAI Functions**:
- ✅ 生产环境
- ✅ 需要高可靠性
- ✅ 成本敏感
- ✅ 工具调用简单

---

## 性能对比

| 指标 | ReAct Agent | OpenAI Functions Agent |
|------|-------------|------------------------|
| Token 消耗 | 高（显式推理） | 低（结构化） |
| 可靠性 | 中等（依赖 prompt） | 高（原生支持） |
| 调试难度 | 低（过程可见） | 高（黑盒） |
| 模型支持 | 任何 LLM | 需要函数调用 |
| 推理能力 | 强（显式推理） | 中等 |

---

## 下一步

1. **场景3**: 学习 Structured Chat Agent（复杂工具参数）
2. **场景4**: 使用 `create_agent()` 统一 API
3. **场景5**: Agent 类型迁移实战
4. **场景7**: 多 Agent 类型对比测试

---

**学习检查清单**:
- [ ] 理解 ReAct 的 Thought-Action-Observation 循环
- [ ] 能够创建 ReAct Prompt
- [ ] 理解与 OpenAI Functions 的区别
- [ ] 能够追踪中间步骤
- [ ] 知道如何使用开源模型
- [ ] 理解 ReAct 的适用场景
- [ ] 能够优化 ReAct Prompt

---

**参考资源**:
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct Agent 文档](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [Ollama 文档](https://ollama.ai)
- [开源 LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
