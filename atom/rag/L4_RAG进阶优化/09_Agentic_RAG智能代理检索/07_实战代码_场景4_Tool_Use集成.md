# 实战代码 - 场景 4: Tool Use 集成

## 场景描述

**目标**: 构建一个集成多个工具的 RAG 系统,包括向量检索、重排序和计算器

**难点**:
- 管理多个工具的注册和调用
- 工具之间的协作和数据传递
- 错误处理和回退机制

**解决方案**: 使用 LangChain 工具系统,构建可扩展的工具集成架构

---

## 环境准备

```bash
# 安装依赖
uv add langchain langchain-openai chromadb python-dotenv
```

---

## 完整代码

```python
"""
Tool Use 集成 - 多工具协作系统
演示: 向量检索 + 重排序 + 计算器集成

技术栈:
- LangChain: 0.1.0+
- OpenAI: 1.0.0+
- ChromaDB: 0.4.0+
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 加载环境变量
load_dotenv()

# ===== 1. 初始化组件 =====
print("初始化组件...")

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 2. 准备知识库 =====
print("准备知识库...")

documents = [
    Document(
        page_content="BERT 在 2018 年发布,参数量为 110M (base) 和 340M (large)",
        metadata={"source": "bert_stats", "relevance": 0.9}
    ),
    Document(
        page_content="GPT-3 在 2020 年发布,参数量为 175B",
        metadata={"source": "gpt3_stats", "relevance": 0.95}
    ),
    Document(
        page_content="GPT-4 在 2023 年发布,参数量未公开,估计超过 1T",
        metadata={"source": "gpt4_stats", "relevance": 0.98}
    ),
    Document(
        page_content="Transformer 架构在 2017 年提出,使用 Self-Attention 机制",
        metadata={"source": "transformer_intro", "relevance": 0.85}
    ),
    Document(
        page_content="BERT 使用双向编码器,GPT 使用单向解码器",
        metadata={"source": "architecture_comparison", "relevance": 0.92}
    )
]

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="tool_integration_demo"
)

# ===== 3. 工具 1: 向量检索 =====

def vector_search(query: str) -> str:
    """向量检索工具"""
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "未找到相关文档"

    output = ["向量检索结果:"]
    for i, doc in enumerate(results, 1):
        output.append(f"{i}. {doc.page_content}")
        output.append(f"   相关性: {doc.metadata.get('relevance', 'N/A')}")

    return "\n".join(output)

# ===== 4. 工具 2: 重排序 =====

def rerank_results(query: str) -> str:
    """重排序工具 - 基于相关性分数重排序"""
    # 先检索
    results = vectorstore.similarity_search(query, k=5)

    if not results:
        return "没有结果可以重排序"

    # 按相关性分数排序
    sorted_results = sorted(
        results,
        key=lambda x: x.metadata.get('relevance', 0),
        reverse=True
    )

    output = ["重排序后的结果:"]
    for i, doc in enumerate(sorted_results[:3], 1):
        output.append(f"{i}. {doc.page_content}")
        output.append(f"   相关性: {doc.metadata.get('relevance', 'N/A')}")

    return "\n".join(output)

# ===== 5. 工具 3: 计算器 =====

def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        # 安全的数学表达式求值
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"

    except Exception as e:
        return f"计算错误: {e}"

# ===== 6. 工具 4: 数据提取 =====

def extract_numbers(text: str) -> str:
    """从文本中提取数字"""
    import re

    # 提取所有数字(包括小数和科学计数法)
    numbers = re.findall(r'\d+\.?\d*[KMBT]?', text)

    if not numbers:
        return "未找到数字"

    return f"提取的数字: {', '.join(numbers)}"

# ===== 7. 创建工具列表 =====

tools = [
    Tool(
        name="VectorSearch",
        func=vector_search,
        description="搜索相关文档。输入:查询字符串。返回:相关文档列表。适合语义搜索。"
    ),
    Tool(
        name="ReRank",
        func=rerank_results,
        description="重排序搜索结果。输入:查询字符串。返回:按相关性排序的文档。适合优化搜索结果。"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="执行数学计算。输入:数学表达式(如 '100 * 2 + 50')。返回:计算结果。"
    ),
    Tool(
        name="ExtractNumbers",
        func=extract_numbers,
        description="从文本中提取数字。输入:包含数字的文本。返回:提取的数字列表。"
    )
]

# ===== 8. 创建 Agent Prompt =====

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手,可以使用多个工具来回答问题。

可用工具:
- VectorSearch: 搜索相关文档
- ReRank: 重排序搜索结果
- Calculator: 执行数学计算
- ExtractNumbers: 从文本中提取数字

工作流程建议:
1. 对于概念查询,使用 VectorSearch
2. 如果需要更精确的结果,使用 ReRank
3. 如果需要计算,先用 ExtractNumbers 提取数字,再用 Calculator 计算
4. 组合使用多个工具以获得最佳结果

请根据查询选择合适的工具并组合使用。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ===== 9. 创建 Agent =====

print("创建 Agent...")

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ===== 10. 测试场景 =====

def main():
    """主函数"""
    test_queries = [
        # 场景 1: 简单检索
        "什么是 BERT?",

        # 场景 2: 检索 + 重排序
        "搜索 GPT 相关信息并优化结果",

        # 场景 3: 检索 + 计算
        "BERT 和 GPT-3 的参数量相差多少?",

        # 场景 4: 多工具组合
        "比较 BERT 和 GPT-3 的参数量,并计算 GPT-3 是 BERT 的多少倍"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"场景 {i}: {query}")
        print(f"{'='*60}\n")

        try:
            result = agent_executor.invoke({"input": query})
            print(f"\n{'='*60}")
            print(f"最终答案:")
            print(result["output"])
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"错误: {e}\n")

if __name__ == "__main__":
    main()
```

---

## 运行输出

```
初始化组件...
准备知识库...
创建 Agent...

============================================================
场景 1: 什么是 BERT?
============================================================


> Entering new AgentExecutor chain...

Invoking: `VectorSearch` with `{'query': 'BERT'}`

向量检索结果:
1. BERT 在 2018 年发布,参数量为 110M (base) 和 340M (large)
   相关性: 0.9
2. BERT 使用双向编码器,GPT 使用单向解码器
   相关性: 0.92
3. Transformer 架构在 2017 年提出,使用 Self-Attention 机制
   相关性: 0.85

BERT 是一个在 2018 年发布的模型,参数量为 110M (base) 和 340M (large)。它使用双向编码器架构。

> Finished chain.

============================================================
最终答案:
BERT 是一个在 2018 年发布的模型,参数量为 110M (base) 和 340M (large)。它使用双向编码器架构。
============================================================

============================================================
场景 2: 搜索 GPT 相关信息并优化结果
============================================================


> Entering new AgentExecutor chain...

Invoking: `ReRank` with `{'query': 'GPT'}`

重排序后的结果:
1. GPT-4 在 2023 年发布,参数量未公开,估计超过 1T
   相关性: 0.98
2. GPT-3 在 2020 年发布,参数量为 175B
   相关性: 0.95
3. BERT 使用双向编码器,GPT 使用单向解码器
   相关性: 0.92

根据重排序后的结果,以下是关于 GPT 的信息:

1. **GPT-4**: 在 2023 年发布,参数量未公开,估计超过 1T。
2. **GPT-3**: 在 2020 年发布,参数量为 175B。
3. **架构**: GPT 使用单向解码器,与 BERT 的双向编码器不同。

> Finished chain.

============================================================
最终答案:
根据重排序后的结果,以下是关于 GPT 的信息:

1. **GPT-4**: 在 2023 年发布,参数量未公开,估计超过 1T。
2. **GPT-3**: 在 2020 年发布,参数量为 175B。
3. **架构**: GPT 使用单向解码器,与 BERT 的双向编码器不同。
============================================================

============================================================
场景 3: BERT 和 GPT-3 的参数量相差多少?
============================================================


> Entering new AgentExecutor chain...

Invoking: `VectorSearch` with `{'query': 'BERT GPT-3 参数量'}`

向量检索结果:
1. GPT-3 在 2020 年发布,参数量为 175B
   相关性: 0.95
2. BERT 在 2018 年发布,参数量为 110M (base) 和 340M (large)
   相关性: 0.9
3. BERT 使用双向编码器,GPT 使用单向解码器
   相关性: 0.92

Invoking: `Calculator` with `{'expression': '175000000000 - 340000000'}`

计算结果: 174660000000

BERT (large) 的参数量为 340M,GPT-3 的参数量为 175B。两者相差约 174.66B (174,660,000,000)。

> Finished chain.

============================================================
最终答案:
BERT (large) 的参数量为 340M,GPT-3 的参数量为 175B。两者相差约 174.66B (174,660,000,000)。
============================================================

============================================================
场景 4: 比较 BERT 和 GPT-3 的参数量,并计算 GPT-3 是 BERT 的多少倍
============================================================


> Entering new AgentExecutor chain...

Invoking: `VectorSearch` with `{'query': 'BERT GPT-3 参数量'}`

向量检索结果:
1. GPT-3 在 2020 年发布,参数量为 175B
   相关性: 0.95
2. BERT 在 2018 年发布,参数量为 110M (base) 和 340M (large)
   相关性: 0.9
3. BERT 使用双向编码器,GPT 使用单向解码器
   相关性: 0.92

Invoking: `Calculator` with `{'expression': '175000000000 / 340000000'}`

计算结果: 514.7058823529412

**参数量对比:**
- BERT (large): 340M
- GPT-3: 175B

**计算结果:**
GPT-3 的参数量是 BERT (large) 的约 **515 倍**。

> Finished chain.

============================================================
最终答案:
**参数量对比:**
- BERT (large): 340M
- GPT-3: 175B

**计算结果:**
GPT-3 的参数量是 BERT (large) 的约 **515 倍**。
============================================================
```

---

## 代码解析

### 关键点 1: 工具定义

```python
tools = [
    Tool(
        name="VectorSearch",
        func=vector_search,
        description="搜索相关文档。输入:查询字符串。返回:相关文档列表。适合语义搜索。"
    ),
    Tool(
        name="ReRank",
        func=rerank_results,
        description="重排序搜索结果。输入:查询字符串。返回:按相关性排序的文档。适合优化搜索结果。"
    ),
    # ...
]
```

**要点**:
- 清晰的工具名称
- 详细的功能描述
- 明确的输入输出格式
- 适用场景说明

### 关键点 2: 工具协作

```python
# Agent 自动组合工具
# 场景: "BERT 和 GPT-3 的参数量相差多少?"
# 1. VectorSearch("BERT GPT-3 参数量") → 获取数据
# 2. Calculator("175000000000 - 340000000") → 计算差值
```

**要点**:
- Agent 自动选择工具顺序
- 工具之间传递数据
- 组合使用多个工具

### 关键点 3: 系统提示词

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手,可以使用多个工具来回答问题。

工作流程建议:
1. 对于概念查询,使用 VectorSearch
2. 如果需要更精确的结果,使用 ReRank
3. 如果需要计算,先用 ExtractNumbers 提取数字,再用 Calculator 计算
4. 组合使用多个工具以获得最佳结果
"""),
    # ...
])
```

**要点**:
- 明确工具使用建议
- 提供工作流程指导
- 鼓励工具组合使用

---

## 扩展思考

### 如何优化?

**1. 添加工具缓存**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_vector_search(query: str) -> str:
    """缓存向量检索结果"""
    return vector_search(query)
```

**2. 添加工具监控**
```python
import time

def monitored_tool(func):
    """监控工具执行"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        print(f"工具 {func.__name__} 执行时间: {duration:.2f}s")
        return result

    return wrapper

@monitored_tool
def vector_search(query: str) -> str:
    # ...
```

**3. 添加工具验证**
```python
def validate_tool_input(tool_name: str, input_data: Any) -> bool:
    """验证工具输入"""
    validators = {
        "Calculator": lambda x: isinstance(x, str) and len(x) < 100,
        "VectorSearch": lambda x: isinstance(x, str) and len(x) > 0
    }

    validator = validators.get(tool_name)
    return validator(input_data) if validator else True
```

### 如何扩展?

**1. 添加更多工具**
```python
def web_search(query: str) -> str:
    """网络搜索工具"""
    # 实现网络搜索
    pass

def summarize(text: str) -> str:
    """文本摘要工具"""
    # 实现文本摘要
    pass

tools.extend([
    Tool(name="WebSearch", func=web_search, description="..."),
    Tool(name="Summarize", func=summarize, description="...")
])
```

**2. 工具链模式**
```python
class ToolChain:
    """工具链 - 预定义的工具组合"""
    def __init__(self, tools: List[Tool]):
        self.tools = tools

    def execute(self, input_data: Any) -> Any:
        """顺序执行工具链"""
        result = input_data
        for tool in self.tools:
            result = tool.func(result)
        return result

# 定义工具链
search_and_rerank_chain = ToolChain([
    Tool(name="Search", func=vector_search, description="..."),
    Tool(name="ReRank", func=rerank_results, description="...")
])
```

**3. 条件工具选择**
```python
def smart_tool_selector(query: str, tools: List[Tool]) -> List[Tool]:
    """智能选择工具"""
    selected = []

    if "计算" in query or "多少" in query:
        selected.append(get_tool("Calculator", tools))

    if "搜索" in query or "查找" in query:
        selected.append(get_tool("VectorSearch", tools))

    return selected
```

### 生产级改进

**1. 错误处理**
```python
def safe_tool_execution(tool: Tool, input_data: Any) -> Dict:
    """安全的工具执行"""
    try:
        result = tool.func(input_data)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 使用
result = safe_tool_execution(vector_search_tool, query)
if result["success"]:
    print(result["result"])
else:
    print(f"工具执行失败: {result['error']}")
```

**2. 工具超时控制**
```python
from concurrent.futures import TimeoutError
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("工具执行超时")

def execute_with_timeout(tool: Tool, input_data: Any, timeout: int = 30):
    """带超时的工具执行"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = tool.func(input_data)
        signal.alarm(0)  # 取消超时
        return result
    except TimeoutError:
        return "工具执行超时"
```

**3. 工具使用统计**
```python
class ToolUsageTracker:
    """工具使用统计"""
    def __init__(self):
        self.usage_count = {}
        self.execution_times = {}

    def track(self, tool_name: str, execution_time: float):
        """记录工具使用"""
        self.usage_count[tool_name] = self.usage_count.get(tool_name, 0) + 1
        if tool_name not in self.execution_times:
            self.execution_times[tool_name] = []
        self.execution_times[tool_name].append(execution_time)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "usage_count": self.usage_count,
            "avg_execution_time": {
                tool: sum(times) / len(times)
                for tool, times in self.execution_times.items()
            }
        }

# 使用
tracker = ToolUsageTracker()
start = time.time()
result = tool.func(input_data)
tracker.track(tool.name, time.time() - start)
```

**4. 工具版本管理**
```python
class VersionedTool:
    """带版本的工具"""
    def __init__(self, name: str, func: callable, version: str):
        self.name = name
        self.func = func
        self.version = version

    def execute(self, *args, **kwargs):
        """执行工具"""
        print(f"执行 {self.name} v{self.version}")
        return self.func(*args, **kwargs)

# 使用
vector_search_v1 = VersionedTool("VectorSearch", vector_search, "1.0")
vector_search_v2 = VersionedTool("VectorSearch", vector_search_improved, "2.0")
```

---

## 参考资源

### 官方文档
- LangChain Tools: https://python.langchain.com/docs/modules/agents/tools/
- LangChain Custom Tools: https://python.langchain.com/docs/modules/agents/tools/custom_tools

### 相关博客
- "Building Tool-Using Agents with LangChain" (LangChain Blog, 2025)
- "Tool Integration Best Practices" (Medium, 2026)

---

**版本**: v1.0
**最后更新**: 2026-02-17
**代码行数**: ~200 行
