# Agent推理与决策 - 实战代码：MRKL 零样本推理

> 零样本（Zero-Shot）推理让 Agent 仅凭工具描述就能选择正确的工具，无需提供示例

---

## 场景说明

**目标：** 理解 MRKL/ZeroShot 推理模式，掌握如何通过工具描述引导 Agent 做出正确决策。

**你将学到：**
1. MRKL Prompt 的结构（prefix + tools + format + suffix）
2. 零样本 vs Few-shot 的区别与选择
3. 工具描述的最佳实践
4. 如何用 `create_react_agent` 实现零样本推理
5. 如何分析和优化 Prompt 模板

**核心概念：**
- **MRKL**（Modular Reasoning, Knowledge and Language）：模块化推理系统
- **Zero-Shot**：不提供示例，仅靠工具描述让 LLM 选择工具
- **ZeroShotAgent**：LangChain 中基于 MRKL 思想的 Agent 实现

---

## 前置准备

```bash
# 安装依赖
uv add langchain langchain-openai langchain-core python-dotenv
```

---

## 完整代码

```python
"""
MRKL 零样本推理实战
演示：使用零样本推理实现工具选择，深入理解 MRKL Prompt 结构

运行方式：python examples/mrkl_zero_shot_demo.py
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# ===== 0. 加载环境变量 =====
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===== 1. 理解 MRKL Prompt 结构 =====
print("=" * 60)
print("MRKL Prompt 结构解析")
print("=" * 60)

# MRKL Prompt 由 4 个部分组成：
# ┌──────────────────────────────────────┐
# │  PREFIX（角色设定 + 任务说明）         │
# ├──────────────────────────────────────┤
# │  TOOLS（可用工具列表 + 描述）          │
# ├──────────────────────────────────────┤
# │  FORMAT_INSTRUCTIONS（输出格式要求）   │
# ├──────────────────────────────────────┤
# │  SUFFIX（用户输入 + 推理开始）         │
# └──────────────────────────────────────┘

# --- Part 1: PREFIX ---
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""

# --- Part 2: TOOLS（自动生成） ---
# 这部分由 LangChain 根据注册的工具自动填充
# 格式：tool_name: tool_description

# --- Part 3: FORMAT_INSTRUCTIONS ---
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

# --- Part 4: SUFFIX ---
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

print("PREFIX:", PREFIX[:50], "...")
print("FORMAT_INSTRUCTIONS:", FORMAT_INSTRUCTIONS[:50], "...")
print("SUFFIX:", SUFFIX[:50], "...")


# ===== 2. 定义工具（重点：工具描述） =====
print("\n" + "=" * 60)
print("工具定义（零样本的关键：工具描述）")
print("=" * 60)

# 零样本推理的核心：LLM 完全依赖工具描述来决定用哪个工具
# 所以工具描述的质量直接决定了 Agent 的表现


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。
    适用场景：查询实时新闻、最新数据、当前事件。
    不适用：历史知识、数学计算、代码执行。
    输入：搜索查询字符串，如 '2025年AI最新进展' 或 '今天的天气'。"""

    # 模拟搜索结果
    mock_results = {
        "ai": "2025年AI领域最大突破：多模态大模型成为主流，Agent 系统在企业中广泛应用。",
        "python": "Python 3.13 于 2024 年发布，引入了自由线程模式和 JIT 编译器。",
        "langchain": "LangChain 在 2025 年推出了 LangGraph 2.0，支持更复杂的 Agent 编排。",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"搜索 '{query}' 的结果：未找到相关信息。"


@tool
def knowledge_base_query(topic: str) -> str:
    """查询内部知识库获取专业知识。
    适用场景：查询技术概念、定义、原理、最佳实践。
    不适用：实时信息、新闻、个人数据。
    输入：知识主题，如 'RAG原理' 或 'Transformer架构'。"""

    knowledge = {
        "rag": "RAG（检索增强生成）：先从知识库检索相关文档，再让 LLM 基于文档生成回答。"
               "核心流程：Query → Retrieve → Augment → Generate。优势：减少幻觉，知识可更新。",
        "transformer": "Transformer 是一种基于自注意力机制的神经网络架构。"
                       "核心组件：Multi-Head Attention + Feed Forward + Layer Norm。"
                       "是 GPT、BERT、T5 等模型的基础。",
        "embedding": "Embedding 是将离散数据（文本、图片）映射到连续向量空间的技术。"
                     "语义相似的内容在向量空间中距离更近。常用模型：text-embedding-3-small。",
    }
    for key, value in knowledge.items():
        if key in topic.lower():
            return value
    return f"知识库中未找到关于 '{topic}' 的信息。"


@tool
def code_executor(code: str) -> str:
    """执行 Python 代码并返回结果。
    适用场景：数学计算、数据处理、逻辑验证。
    不适用：网络请求、文件操作、系统命令。
    输入：有效的 Python 表达式或简单代码，如 '2**10' 或 'len([1,2,3])'。"""

    try:
        # 安全执行（仅允许基本运算）
        allowed_builtins = {
            "len": len, "sum": sum, "max": max, "min": min,
            "abs": abs, "round": round, "sorted": sorted,
            "range": range, "list": list, "int": int, "float": float,
            "str": str, "bool": bool, "pow": pow,
        }
        result = eval(code, {"__builtins__": allowed_builtins})
        return f"执行结果：{result}"
    except Exception as e:
        return f"执行错误：{e}"


tools = [web_search, knowledge_base_query, code_executor]

# 展示工具描述（这就是 LLM 在零样本推理时看到的全部信息）
print("\nLLM 看到的工具信息：")
for t in tools:
    print(f"\n  {t.name}: {t.description}")


# ===== 3. 组装完整的 MRKL Prompt =====
print("\n" + "=" * 60)
print("组装 MRKL Prompt")
print("=" * 60)

# 将 4 个部分组合成完整的 prompt
mrkl_template = f"""{PREFIX}

{{tools}}

{FORMAT_INSTRUCTIONS}

{SUFFIX}"""

mrkl_prompt = PromptTemplate.from_template(mrkl_template)

# 展示完整 prompt（填充工具信息后）
tool_strings = "\n".join([f"{t.name}: {t.description}" for t in tools])
tool_names = ", ".join([t.name for t in tools])

print("\n完整 Prompt 预览（前500字符）：")
preview = mrkl_prompt.format(
    tools=tool_strings,
    tool_names=tool_names,
    input="示例问题",
    agent_scratchpad="",
)
print(preview[:500])
print("...")


# ===== 4. 创建零样本 Agent =====
agent = create_react_agent(llm=llm, tools=tools, prompt=mrkl_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)


# ===== 5. 零样本推理演示 =====
# 注意：我们没有提供任何示例，LLM 完全靠工具描述来选择工具

print("\n" + "=" * 60)
print("零样本推理演示")
print("=" * 60)

# --- 测试1：应该选择 web_search ---
print("\n--- 测试1：实时信息查询（期望选择 web_search）---")
result1 = agent_executor.invoke({
    "input": "2025年 AI 领域有什么最新进展？"
})
print(f"答案：{result1['output']}")

# --- 测试2：应该选择 knowledge_base_query ---
print("\n--- 测试2：技术概念查询（期望选择 knowledge_base_query）---")
result2 = agent_executor.invoke({
    "input": "请解释一下 RAG 的原理是什么？"
})
print(f"答案：{result2['output']}")

# --- 测试3：应该选择 code_executor ---
print("\n--- 测试3：数学计算（期望选择 code_executor）---")
result3 = agent_executor.invoke({
    "input": "计算 2 的 20 次方是多少？"
})
print(f"答案：{result3['output']}")

# --- 测试4：多工具协作 ---
print("\n--- 测试4：多工具协作 ---")
result4 = agent_executor.invoke({
    "input": "什么是 Embedding？它的向量维度通常是多少？计算 1536 的平方根。"
})
print(f"答案：{result4['output']}")


# ===== 6. 零样本 vs Few-Shot 对比 =====
print("\n" + "=" * 60)
print("零样本 vs Few-Shot 对比")
print("=" * 60)

# Few-Shot：在 prompt 中提供推理示例
few_shot_template = f"""{PREFIX}

{{tools}}

{FORMAT_INSTRUCTIONS}

Here are some examples:

Question: What is the capital of France?
Thought: This is a factual question about geography. I should search for this information.
Action: knowledge_base_query
Action Input: capital of France
Observation: France's capital is Paris.
Thought: I now know the final answer
Final Answer: The capital of France is Paris.

Question: What is 15 * 23?
Thought: This is a math calculation. I should use the code executor.
Action: code_executor
Action Input: 15 * 23
Observation: 执行结果：345
Thought: I now know the final answer
Final Answer: 15 * 23 = 345

Now answer the following question:

Question: {{input}}
Thought:{{agent_scratchpad}}"""

few_shot_prompt = PromptTemplate.from_template(few_shot_template)

# 创建 Few-Shot Agent
few_shot_agent = create_react_agent(llm=llm, tools=tools, prompt=few_shot_prompt)
few_shot_executor = AgentExecutor(
    agent=few_shot_agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

print("\n--- Few-Shot Agent 测试 ---")
result_fs = few_shot_executor.invoke({
    "input": "Transformer 架构的核心组件是什么？"
})
print(f"答案：{result_fs['output']}")

# 对比总结
print("\n" + "=" * 60)
print("零样本 vs Few-Shot 对比总结")
print("=" * 60)
comparison = """
┌──────────────┬──────────────────────┬──────────────────────┐
│   维度        │  Zero-Shot           │  Few-Shot            │
├──────────────┼──────────────────────┼──────────────────────┤
│ Prompt 长度   │  短（省 Token）       │  长（包含示例）       │
│ 工具选择准确性 │  依赖工具描述质量     │  示例引导，更准确     │
│ 灵活性        │  高（不受示例限制）   │  可能被示例"锚定"     │
│ 适用场景      │  工具描述清晰时       │  工具容易混淆时       │
│ Token 成本    │  低                  │  高                  │
│ 维护成本      │  低（只需维护描述）   │  高（需要维护示例）   │
└──────────────┴──────────────────────┴──────────────────────┘
"""
print(comparison)
```

---

## 预期输出

```
============================================================
MRKL Prompt 结构解析
============================================================
PREFIX: Answer the following questions as best you can...
FORMAT_INSTRUCTIONS: Use the following format:

Question: the...
SUFFIX: Begin!

Question: {input}
Thought:{agent_scratchpad}...

============================================================
零样本推理演示
============================================================

--- 测试1：实时信息查询（期望选择 web_search）---

> Entering new AgentExecutor chain...
Thought: 这个问题需要最新的信息，我应该搜索互联网
Action: web_search
Action Input: 2025年AI最新进展
Observation: 2025年AI领域最大突破：多模态大模型成为主流...
Thought: I now know the final answer
Final Answer: 2025年AI领域的最新进展包括...

> Finished chain.

--- 测试2：技术概念查询（期望选择 knowledge_base_query）---

> Entering new AgentExecutor chain...
Thought: 这是一个技术概念问题，应该查询知识库
Action: knowledge_base_query
Action Input: RAG原理
Observation: RAG（检索增强生成）：先从知识库检索相关文档...
Thought: I now know the final answer
Final Answer: RAG 的原理是...

> Finished chain.
```

---

## 关键知识点解析

### 1. 工具描述的最佳实践

工具描述是零样本推理的核心。好的描述应该包含：

```python
@tool
def my_tool(input: str) -> str:
    """[一句话说明工具功能]。
    适用场景：[什么时候该用这个工具]。
    不适用：[什么时候不该用这个工具]。
    输入：[输入格式说明 + 示例]。"""
    ...

# ❌ 不好的描述
"""搜索工具"""
# 问题：LLM 不知道什么时候该用，输入什么格式

# ✅ 好的描述
"""搜索互联网获取最新信息。
适用场景：查询实时新闻、最新数据、当前事件。
不适用：历史知识、数学计算、代码执行。
输入：搜索查询字符串，如 '2025年AI最新进展'。"""
# 优势：LLM 能准确判断何时使用，输入什么
```

### 2. MRKL 的模块化思想

```
MRKL 的核心思想：
  不是让一个 LLM 做所有事情
  而是让 LLM 作为"路由器"，把任务分发给专业模块

  ┌─────────┐
  │   LLM   │ ← 路由器（决定用哪个工具）
  │ (大脑)   │
  └────┬────┘
       │
  ┌────┼────────────┬──────────────┐
  ↓    ↓            ↓              ↓
搜索  计算器      知识库        代码执行
(专家1) (专家2)   (专家3)       (专家4)
```

### 3. 零样本推理的局限性

```python
# 当工具功能有重叠时，零样本推理可能选错

# 例如：这两个工具的描述很相似
@tool
def search_web(query: str) -> str:
    """搜索网络获取信息"""  # 太模糊！
    ...

@tool
def search_docs(query: str) -> str:
    """搜索文档获取信息"""  # 和上面太像了！
    ...

# 解决方案：让描述更具体，明确区分
@tool
def search_web(query: str) -> str:
    """搜索互联网获取实时公开信息（新闻、百科、论坛）。
    不适用：公司内部文档。"""
    ...

@tool
def search_docs(query: str) -> str:
    """搜索公司内部文档库（技术文档、会议记录、规范）。
    不适用：互联网公开信息。"""
    ...
```

---

## 小结

MRKL 零样本推理的核心要点：

1. **Prompt 结构**：PREFIX + TOOLS + FORMAT + SUFFIX，四段式组合
2. **零样本关键**：工具描述的质量决定一切，要写清"适用/不适用/输入格式"
3. **vs Few-Shot**：零样本省 Token 更灵活，Few-Shot 更准确但成本高
4. **模块化思想**：LLM 是路由器，工具是专家，各司其职

---

## 下一步

- 想用思维链增强推理？→ 阅读 `07_实战代码_场景3_思维链增强Agent.md`
- 想自定义输出解析？→ 阅读 `07_实战代码_场景4_自定义输出解析器.md`
