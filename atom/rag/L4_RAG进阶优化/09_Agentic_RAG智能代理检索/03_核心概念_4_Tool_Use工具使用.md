# 核心概念 4: Tool Use 工具使用

## 一句话定义

**Tool Use 是让 AI 代理调用外部工具和 API 的能力,在 Agentic RAG 中实现检索器、重排序器、计算器等工具的动态集成和协作。**

---

## 详细解释

### 什么是 Tool Use?

Tool Use 是 Agentic RAG 的"工具箱",让 AI 代理能够:
- **调用检索器**: 向量检索、关键词检索、混合检索
- **使用重排序器**: ReRank 优化检索结果
- **执行计算**: 数学计算、数据处理
- **访问外部 API**: 实时数据、数据库查询

**核心价值**: 扩展 LLM 的能力边界,让 AI 能够"动手做事"而非只"动嘴说话"。

### 为什么需要 Tool Use?

LLM 的局限:
```python
# LLM 只能生成文本
query = "2023年营收增长率是多少?"
answer = llm.generate(query)
# 问题: LLM 无法访问数据库,只能猜测或拒绝回答
```

**Tool Use 解决方案**:
```python
# AI 代理可以调用工具
query = "2023年营收增长率是多少?"

# Step 1: 调用数据库工具
revenue_2023 = db_tool.query("SELECT revenue FROM reports WHERE year=2023")
revenue_2022 = db_tool.query("SELECT revenue FROM reports WHERE year=2022")

# Step 2: 调用计算工具
growth_rate = calculator.run(f"({revenue_2023} - {revenue_2022}) / {revenue_2022} * 100")

# Step 3: 生成答案
answer = llm.generate(f"2023年营收增长率是 {growth_rate}%")
```

### Tool Use 如何工作?

**工作流程**:
```
用户查询
    ↓
[AI 代理分析] 需要什么工具?
    ↓
[工具选择] 选择合适的工具
    ↓
[参数解析] 提取工具所需参数
    ↓
[工具执行] 调用工具获取结果
    ↓
[结果集成] 将结果整合到答案中
    ↓
最终答案
```

---

## 核心原理

### 原理图解

```
┌─────────────────────────────────────────┐
│         Tool Use 架构                   │
├─────────────────────────────────────────┤
│                                         │
│  查询: "检索 BERT 相关文档并重排序"     │
│       ↓                                 │
│  [工具注册表]                           │
│   - VectorSearch: 向量检索              │
│   - KeywordSearch: 关键词检索           │
│   - ReRank: 重排序                      │
│   - Calculator: 计算器                  │
│       ↓                                 │
│  [AI 代理决策]                          │
│   需要: VectorSearch + ReRank           │
│       ↓                                 │
│  [工具调用]                             │
│   1. VectorSearch("BERT") → 10 docs    │
│   2. ReRank(10 docs) → 5 best docs     │
│       ↓                                 │
│  [结果集成]                             │
│   基于 5 best docs 生成答案             │
│                                         │
└─────────────────────────────────────────┘
```

### 工作流程

**Step 1: 工具定义**
```python
from langchain.tools import Tool

def vector_search(query: str) -> str:
    """向量检索工具"""
    results = retriever.search(query)
    return str(results)

# 定义工具
search_tool = Tool(
    name="VectorSearch",
    func=vector_search,
    description="搜索相关文档,输入查询字符串"
)
```

**Step 2: 工具注册**
```python
tools = [
    search_tool,
    rerank_tool,
    calculator_tool
]

# 注册到代理
agent = create_agent(llm, tools)
```

**Step 3: 工具调用**
```python
# AI 代理自动选择和调用工具
result = agent.run("检索 BERT 文档并重排序")

# 内部流程:
# 1. 分析查询 → 需要 VectorSearch + ReRank
# 2. 调用 VectorSearch("BERT")
# 3. 调用 ReRank(results)
# 4. 生成最终答案
```

### 关键技术

**1. Function Calling (2023-2024)**
```python
# OpenAI Function Calling
functions = [
    {
        "name": "vector_search",
        "description": "搜索相关文档",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "查询字符串"}
            },
            "required": ["query"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "搜索 BERT 文档"}],
    functions=functions,
    function_call="auto"
)

# 解析函数调用
if response.choices[0].message.function_call:
    function_name = response.choices[0].message.function_call.name
    arguments = json.loads(response.choices[0].message.function_call.arguments)
    result = execute_function(function_name, arguments)
```

**2. Tool RAG (2025)**
```python
# 检索工具而非文档
tool_descriptions = [
    "VectorSearch: 语义检索工具",
    "KeywordSearch: 关键词检索工具",
    "ReRank: 重排序工具"
]

# 根据查询检索最相关的工具
relevant_tools = tool_retriever.search(query)

# 使用检索到的工具
for tool in relevant_tools:
    result = tool.run(query)
```

**3. MCP Protocol (2026)**
```python
# Model Context Protocol - 标准化工具接口
from mcp import MCPServer, Tool

server = MCPServer()

@server.tool("vector_search")
def vector_search(query: str) -> dict:
    """向量检索工具"""
    return {"results": retriever.search(query)}

# AI 代理通过 MCP 协议调用工具
agent = MCPAgent(server_url="http://localhost:8000")
result = agent.call_tool("vector_search", {"query": "BERT"})
```

---

## 手写实现

```python
"""
Tool Use 从零实现
演示: 工具定义、注册、调用
"""

from typing import List, Dict, Callable, Any
from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 工具定义 =====
class Tool:
    """工具基类"""
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def run(self, *args, **kwargs) -> Any:
        """执行工具"""
        return self.func(*args, **kwargs)

    def to_function_schema(self) -> Dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "工具输入"}
                },
                "required": ["input"]
            }
        }

# ===== 2. 具体工具实现 =====
def vector_search_func(query: str) -> str:
    """向量检索(模拟)"""
    knowledge_base = {
        "bert": "BERT 是双向编码器,使用 Masked LM 预训练",
        "gpt": "GPT 是单向解码器,使用自回归预训练",
        "transformer": "Transformer 使用 Self-Attention 机制"
    }

    for key, value in knowledge_base.items():
        if key in query.lower():
            return f"检索结果: {value}"

    return "未找到相关文档"

def rerank_func(docs: str) -> str:
    """重排序(模拟)"""
    return f"重排序后: {docs} (相关性提升)"

def calculator_func(expression: str) -> str:
    """计算器"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

# ===== 3. 工具注册 =====
tools = [
    Tool(
        name="VectorSearch",
        func=vector_search_func,
        description="搜索相关文档,输入查询字符串"
    ),
    Tool(
        name="ReRank",
        func=rerank_func,
        description="重排序文档,输入文档列表"
    ),
    Tool(
        name="Calculator",
        func=calculator_func,
        description="执行数学计算,输入表达式"
    )
]

# ===== 4. Tool Use Agent =====
class ToolUseAgent:
    """工具使用代理"""

    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.to_function_schema() for tool in tools]

    def run(self, query: str) -> str:
        """执行查询"""
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}\n")

        messages = [{"role": "user", "content": query}]
        max_iterations = 5

        for i in range(max_iterations):
            print(f"--- 迭代 {i + 1} ---\n")

            # 调用 LLM 决定是否使用工具
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=self.tool_schemas,
                function_call="auto"
            )

            message = response.choices[0].message

            # 如果没有函数调用,返回答案
            if not message.function_call:
                print(f"✅ 最终答案: {message.content}\n")
                return message.content

            # 解析函数调用
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)
            tool_input = arguments.get("input", "")

            print(f"🔧 调用工具: {function_name}({tool_input})")

            # 执行工具
            tool = self.tools[function_name]
            result = tool.run(tool_input)

            print(f"📊 工具结果: {result}\n")

            # 添加到消息历史
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            })
            messages.append({
                "role": "function",
                "name": function_name,
                "content": result
            })

        return "达到最大迭代次数"

# ===== 5. 测试 =====
if __name__ == "__main__":
    agent = ToolUseAgent(tools)

    test_queries = [
        "搜索 BERT 相关文档",
        "计算 (100 + 50) * 2",
        "搜索 Transformer 并重排序结果"
    ]

    for query in test_queries:
        answer = agent.run(query)
        print(f"\n{'='*50}\n")
```

---

## 在 RAG 中的应用

### 应用场景 1: 多检索器协作

**问题**: 不同查询需要不同检索器

**Tool Use 方案**:
```python
# 定义多个检索工具
tools = [
    Tool(name="VectorSearch", func=vector_search, description="语义检索"),
    Tool(name="KeywordSearch", func=keyword_search, description="关键词检索"),
    Tool(name="HybridSearch", func=hybrid_search, description="混合检索")
]

# AI 代理自动选择
agent = ToolUseAgent(tools)
result = agent.run("搜索 2023年营收数据")  # 自动选择 KeywordSearch
```

### 应用场景 2: 检索 + 重排序

**问题**: 初次检索结果需要优化

**Tool Use 方案**:
```python
# 定义检索和重排序工具
tools = [
    Tool(name="Search", func=search, description="检索文档"),
    Tool(name="ReRank", func=rerank, description="重排序文档")
]

# AI 代理自动组合
agent = ToolUseAgent(tools)
result = agent.run("搜索 BERT 并优化结果")
# 内部: Search("BERT") → ReRank(results)
```

### 应用场景 3: RAG + 计算

**问题**: 需要结合检索和计算

**Tool Use 方案**:
```python
# 定义检索和计算工具
tools = [
    Tool(name="Search", func=search, description="检索数据"),
    Tool(name="Calculate", func=calculate, description="执行计算")
]

# AI 代理自动组合
agent = ToolUseAgent(tools)
result = agent.run("检索2022和2023年营收,计算增长率")
# 内部: Search("2022营收") → Search("2023营收") → Calculate(增长率)
```

---

## 主流框架实现

### LangChain 实现 (推荐)

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="VectorSearch",
        func=vector_search,
        description="搜索相关文档"
    ),
    Tool(
        name="ReRank",
        func=rerank,
        description="重排序文档"
    )
]

# 创建代理
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# 执行
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "搜索 BERT 并重排序"})
```

### LangGraph 实现

```python
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# 定义工具执行器
tool_executor = ToolExecutor(tools)

def agent_node(state):
    """代理节点"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state):
    """工具节点"""
    messages = state["messages"]
    last_message = messages[-1]

    # 执行工具
    tool_invocation = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    )

    response = tool_executor.invoke(tool_invocation)
    return {"messages": [FunctionMessage(content=str(response), name=tool_invocation.tool)]}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

### LlamaIndex 实现

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# 定义工具
def vector_search(query: str) -> str:
    """向量检索工具"""
    return retriever.search(query)

search_tool = FunctionTool.from_defaults(fn=vector_search)

# 创建代理
agent = ReActAgent.from_tools(
    [search_tool],
    llm=llm,
    verbose=True
)

# 执行
response = agent.chat("搜索 BERT 文档")
```

---

## 最佳实践 (2025-2026)

### 性能优化

**1. 工具缓存**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    """缓存检索结果"""
    return retriever.search(query)
```

**2. 并行工具调用**
```python
import asyncio

async def parallel_tools(tool_calls: List[Dict]):
    """并行执行多个工具"""
    tasks = [execute_tool_async(call) for call in tool_calls]
    return await asyncio.gather(*tasks)
```

**3. 工具选择优化**
```python
# 使用小模型选择工具
tool_selector = ChatOpenAI(model="gpt-4o-mini")

# 使用大模型执行任务
task_executor = ChatOpenAI(model="gpt-4o")
```

### 成本控制

**1. 限制工具调用次数**
```python
agent = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5  # 限制最大迭代次数
)
```

**2. 工具调用日志**
```python
def logged_tool(func):
    """记录工具调用"""
    def wrapper(*args, **kwargs):
        print(f"调用工具: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"工具结果: {result}")
        return result
    return wrapper
```

### 错误处理

**1. 工具调用失败**
```python
def safe_tool_call(tool: Tool, input: str):
    """安全的工具调用"""
    try:
        return tool.run(input)
    except Exception as e:
        return f"工具调用失败: {e}"
```

**2. 参数验证**
```python
def validate_tool_input(tool: Tool, input: Dict):
    """验证工具输入"""
    required_params = tool.get_required_params()

    for param in required_params:
        if param not in input:
            raise ValueError(f"缺少必需参数: {param}")

    return True
```

---

## 常见问题

### 问题 1: 工具调用失败怎么办?

**解决方案**:
```python
# 1. 添加重试机制
def retry_tool_call(tool: Tool, input: str, max_retries: int = 3):
    """重试工具调用"""
    for attempt in range(max_retries):
        try:
            return tool.run(input)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"重试 {attempt + 1}/{max_retries}")

# 2. 提供回退工具
def fallback_tool_call(primary_tool: Tool, fallback_tool: Tool, input: str):
    """回退工具调用"""
    try:
        return primary_tool.run(input)
    except Exception:
        return fallback_tool.run(input)
```

### 问题 2: 如何选择合适的工具?

**评估标准**:
```python
def evaluate_tool_selection(query: str, selected_tool: str, expected_tool: str):
    """评估工具选择"""
    return {
        "query": query,
        "selected": selected_tool,
        "expected": expected_tool,
        "correct": selected_tool == expected_tool
    }

# 测试用例
test_cases = [
    {"query": "搜索 BERT", "expected": "VectorSearch"},
    {"query": "计算 1+1", "expected": "Calculator"}
]

for case in test_cases:
    result = agent.run(case["query"])
    evaluate_tool_selection(case["query"], result["tool"], case["expected"])
```

### 问题 3: Tool Use vs ReAct 如何选择?

**对比**:
```python
# Tool Use: 专注工具调用
tool_agent = ToolUseAgent(tools)
result = tool_agent.run("搜索 BERT")  # 直接调用工具

# ReAct: 推理 + 行动循环
react_agent = ReActAgent(tools)
result = react_agent.run("搜索 BERT")  # 思考 → 行动 → 观察 → 反思

# 选择建议:
# - 简单工具调用 → Tool Use
# - 需要推理决策 → ReAct
# - 复杂任务 → ReAct + Tool Use 结合
```

---

## 参考资源

### 论文
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (arXiv 2302.04761, 2023)
- "Tool RAG: The Next Breakthrough in Scalable AI Agents" (Red Hat, 2025)

### 博客
- IBM: "Agentic RAG Tutorial" (2026) - Tool Use 实践
  https://www.ibm.com/think/tutorials/agentic-rag
- "Beyond RAG: Why 2026 is the Year of Agentic AI" (Medium, 2026)
  https://medium.com/@isuruig/beyond-rag-why-2026-is-the-year-of-agentic-ai
- LangChain: "Build a RAG agent" (2026)
  https://docs.langchain.com/oss/python/langchain/rag

### 框架文档
- LangChain Tools: https://python.langchain.com/docs/modules/agents/tools/
- LangGraph Tool Executor: https://langchain-ai.github.io/langgraph/
- LlamaIndex Function Tools: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/

### 协议标准
- MCP (Model Context Protocol): https://modelcontextprotocol.io/
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling

---

**版本**: v1.0
**最后更新**: 2026-02-17
**字数**: ~450 行
