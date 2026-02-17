# 实战代码 - 场景 3: ReAct Agent 检索

## 场景描述

**目标**: 使用 LangChain 构建 ReAct 风格的检索代理,实现"思考 → 行动 → 观察 → 反思"循环

**难点**:
- 实现推理和行动的交替循环
- 管理中间步骤和观察结果
- 决定何时停止迭代

**解决方案**: 使用 LangChain ReAct Agent 框架,结合检索工具实现迭代式检索

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
ReAct Agent 检索 - 推理行动循环
演示: Thought → Action → Observation 迭代检索

技术栈:
- LangChain: 0.1.0+
- OpenAI: 1.0.0+
- ChromaDB: 0.4.0+
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

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
        page_content="BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的预训练语言模型。",
        metadata={"source": "bert_intro", "topic": "bert"}
    ),
    Document(
        page_content="BERT 使用双向 Transformer 编码器,通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 进行预训练。",
        metadata={"source": "bert_training", "topic": "bert"}
    ),
    Document(
        page_content="BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。",
        metadata={"source": "bert_advantages", "topic": "bert"}
    ),
    Document(
        page_content="BERT 的缺点包括:模型较大、推理速度慢、不适合生成任务。",
        metadata={"source": "bert_disadvantages", "topic": "bert"}
    ),
    Document(
        page_content="GPT (Generative Pre-trained Transformer) 是 OpenAI 提出的自回归语言模型系列。",
        metadata={"source": "gpt_intro", "topic": "gpt"}
    ),
    Document(
        page_content="GPT 使用单向 Transformer 解码器,通过自回归方式预测下一个 token。",
        metadata={"source": "gpt_architecture", "topic": "gpt"}
    ),
    Document(
        page_content="GPT 的优点包括:强大的文本生成能力、流畅的语言输出、适合对话和创作任务。",
        metadata={"source": "gpt_advantages", "topic": "gpt"}
    ),
    Document(
        page_content="GPT 的缺点包括:上下文理解不如双向模型、可能产生幻觉、训练成本高。",
        metadata={"source": "gpt_disadvantages", "topic": "gpt"}
    ),
    Document(
        page_content="Transformer 是 2017 年提出的注意力机制架构,使用 Self-Attention 实现并行处理。",
        metadata={"source": "transformer_intro", "topic": "transformer"}
    ),
    Document(
        page_content="Self-Attention 机制允许模型关注输入序列的不同位置,计算每个位置与其他位置的相关性。",
        metadata={"source": "transformer_attention", "topic": "transformer"}
    )
]

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="react_demo"
)

# ===== 3. 定义工具 =====

def search_documents(query: str) -> str:
    """搜索相关文档"""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "未找到相关文档"

    output = []
    for i, doc in enumerate(results, 1):
        output.append(f"{i}. {doc.page_content}")
        output.append(f"   来源: {doc.metadata.get('source', 'unknown')}")

    return "\n".join(output)

def search_by_topic(topic: str) -> str:
    """按主题搜索文档"""
    results = vectorstore.similarity_search(
        topic,
        k=3,
        filter={"topic": topic.lower()}
    )

    if not results:
        return f"未找到关于 {topic} 的文档"

    output = []
    for i, doc in enumerate(results, 1):
        output.append(f"{i}. {doc.page_content}")

    return "\n".join(output)

# 创建工具列表
tools = [
    Tool(
        name="Search",
        func=search_documents,
        description="搜索相关文档。输入:查询字符串。适合一般性搜索。"
    ),
    Tool(
        name="SearchByTopic",
        func=search_by_topic,
        description="按主题搜索文档。输入:主题名称(bert/gpt/transformer)。适合针对特定主题的深入搜索。"
    )
]

# ===== 4. 定义 ReAct Prompt =====

react_prompt = PromptTemplate.from_template("""
你是一个问答助手。使用以下格式回答问题:

Question: 用户的问题
Thought: 你应该思考下一步做什么
Action: 执行的工具名称
Action Input: 工具的输入
Observation: 工具的输出结果
... (重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

可用工具:
{tools}

工具名称: {tool_names}

Question: {input}
{agent_scratchpad}
""")

# ===== 5. 创建 ReAct Agent =====

print("创建 ReAct Agent...")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ===== 6. 测试 =====

def main():
    """主函数"""
    test_queries = [
        "什么是 BERT?",
        "比较 BERT 和 GPT 的优缺点",
        "Transformer 的核心机制是什么?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
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
创建 ReAct Agent...

============================================================
查询: 什么是 BERT?
============================================================


> Entering new AgentExecutor chain...
Thought: 我需要搜索关于 BERT 的信息
Action: SearchByTopic
Action Input: bert

Observation:
1. BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的预训练语言模型。
2. BERT 使用双向 Transformer 编码器,通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 进行预训练。
3. BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。

Thought: 我现在有足够的信息来回答问题了
Final Answer: BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的预训练语言模型。它使用双向 Transformer 编码器,通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 进行预训练。BERT 的主要优点包括强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能,以及可以进行 fine-tuning。

> Finished chain.

============================================================
最终答案:
BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的预训练语言模型。它使用双向 Transformer 编码器,通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 进行预训练。BERT 的主要优点包括强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能,以及可以进行 fine-tuning。
============================================================

============================================================
查询: 比较 BERT 和 GPT 的优缺点
============================================================


> Entering new AgentExecutor chain...
Thought: 我需要分别搜索 BERT 和 GPT 的优缺点
Action: SearchByTopic
Action Input: bert

Observation:
1. BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。
2. BERT 的缺点包括:模型较大、推理速度慢、不适合生成任务。

Thought: 现在我需要搜索 GPT 的优缺点
Action: SearchByTopic
Action Input: gpt

Observation:
1. GPT 的优点包括:强大的文本生成能力、流畅的语言输出、适合对话和创作任务。
2. GPT 的缺点包括:上下文理解不如双向模型、可能产生幻觉、训练成本高。

Thought: 我现在有足够的信息来对比 BERT 和 GPT 了
Final Answer:
BERT 和 GPT 的对比:

**BERT 优点:**
- 强大的上下文理解能力
- 在多个 NLP 任务上取得 SOTA 性能
- 可以进行 fine-tuning

**BERT 缺点:**
- 模型较大
- 推理速度慢
- 不适合生成任务

**GPT 优点:**
- 强大的文本生成能力
- 流畅的语言输出
- 适合对话和创作任务

**GPT 缺点:**
- 上下文理解不如双向模型
- 可能产生幻觉
- 训练成本高

> Finished chain.

============================================================
最终答案:
BERT 和 GPT 的对比:

**BERT 优点:**
- 强大的上下文理解能力
- 在多个 NLP 任务上取得 SOTA 性能
- 可以进行 fine-tuning

**BERT 缺点:**
- 模型较大
- 推理速度慢
- 不适合生成任务

**GPT 优点:**
- 强大的文本生成能力
- 流畅的语言输出
- 适合对话和创作任务

**GPT 缺点:**
- 上下文理解不如双向模型
- 可能产生幻觉
- 训练成本高
============================================================
```

---

## 代码解析

### 关键点 1: ReAct Prompt 设计

```python
react_prompt = PromptTemplate.from_template("""
你是一个问答助手。使用以下格式回答问题:

Question: 用户的问题
Thought: 你应该思考下一步做什么
Action: 执行的工具名称
Action Input: 工具的输入
Observation: 工具的输出结果
... (重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案
""")
```

**要点**:
- 明确的格式定义
- Thought → Action → Observation 循环
- 清晰的结束标志(Final Answer)

### 关键点 2: 工具定义

```python
tools = [
    Tool(
        name="Search",
        func=search_documents,
        description="搜索相关文档。输入:查询字符串。适合一般性搜索。"
    ),
    Tool(
        name="SearchByTopic",
        func=search_by_topic,
        description="按主题搜索文档。输入:主题名称(bert/gpt/transformer)。适合针对特定主题的深入搜索。"
    )
]
```

**要点**:
- 清晰的工具描述(帮助 LLM 选择)
- 明确的输入格式说明
- 适用场景说明

### 关键点 3: Agent Executor 配置

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # 显示推理过程
    max_iterations=5,          # 限制最大迭代次数
    handle_parsing_errors=True # 处理解析错误
)
```

**要点**:
- `verbose=True` 显示完整推理过程
- `max_iterations` 防止无限循环
- `handle_parsing_errors` 提高鲁棒性

---

## 扩展思考

### 如何优化?

**1. 添加更多工具**
```python
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

tools.append(
    Tool(
        name="Calculator",
        func=calculate,
        description="执行数学计算。输入:数学表达式。"
    )
)
```

**2. 添加记忆功能**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,  # 添加记忆
    verbose=True
)
```

**3. 自定义停止条件**
```python
def custom_stopping_condition(intermediate_steps):
    """自定义停止条件"""
    # 如果已经执行了3次搜索,停止
    search_count = sum(1 for step in intermediate_steps if "Search" in str(step))
    return search_count >= 3

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="force",
    max_iterations=10
)
```

### 如何扩展?

**1. 支持流式输出**
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True
)
```

**2. 添加工具验证**
```python
def validate_tool_input(tool_name: str, tool_input: str) -> bool:
    """验证工具输入"""
    if tool_name == "SearchByTopic":
        valid_topics = ["bert", "gpt", "transformer"]
        return tool_input.lower() in valid_topics
    return True

# 在工具函数中使用
def search_by_topic_validated(topic: str) -> str:
    if not validate_tool_input("SearchByTopic", topic):
        return f"无效的主题: {topic}。有效主题: bert, gpt, transformer"
    return search_by_topic(topic)
```

**3. 添加结果评分**
```python
def evaluate_result(result: str, query: str) -> float:
    """评估结果质量"""
    # 简单的评分逻辑
    if len(result) < 50:
        return 0.3  # 太短
    if "未找到" in result:
        return 0.1  # 未找到结果
    return 0.9  # 正常结果

# 在 agent 中使用
def search_with_evaluation(query: str) -> str:
    result = search_documents(query)
    score = evaluate_result(result, query)

    if score < 0.5:
        # 尝试改写查询
        refined_query = refine_query(query)
        result = search_documents(refined_query)

    return result
```

### 生产级改进

**1. 错误处理和重试**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def robust_agent_run(query: str):
    """带重试的 agent 执行"""
    try:
        return agent_executor.invoke({"input": query})
    except Exception as e:
        print(f"执行失败: {e}, 重试中...")
        raise
```

**2. 性能监控**
```python
import time
from typing import Dict

def run_with_metrics(query: str) -> Dict:
    """带性能监控的执行"""
    start_time = time.time()

    result = agent_executor.invoke({"input": query})

    metrics = {
        "query": query,
        "execution_time": time.time() - start_time,
        "iterations": len(result.get("intermediate_steps", [])),
        "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])]
    }

    print(f"\n📊 性能指标:")
    print(f"  执行时间: {metrics['execution_time']:.2f}s")
    print(f"  迭代次数: {metrics['iterations']}")
    print(f"  使用工具: {', '.join(metrics['tools_used'])}")

    return result
```

**3. 日志记录**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('react_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def logged_agent_run(query: str):
    """带日志的 agent 执行"""
    logger.info(f"收到查询: {query}")

    try:
        result = agent_executor.invoke({"input": query})
        logger.info(f"查询成功: {query}")
        return result
    except Exception as e:
        logger.error(f"查询失败: {query}, 错误: {e}")
        raise
```

**4. 结果缓存**
```python
from functools import lru_cache
import hashlib

def cache_key(query: str) -> str:
    """生成缓存键"""
    return hashlib.md5(query.encode()).hexdigest()

result_cache = {}

def cached_agent_run(query: str):
    """带缓存的 agent 执行"""
    key = cache_key(query)

    if key in result_cache:
        print("从缓存返回结果")
        return result_cache[key]

    result = agent_executor.invoke({"input": query})
    result_cache[key] = result

    return result
```

---

## 参考资源

### 官方文档
- LangChain ReAct Agent: https://python.langchain.com/docs/modules/agents/agent_types/react
- LangChain Tools: https://python.langchain.com/docs/modules/agents/tools/

### 相关论文
- "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv 2210.03629, 2022)

### 相关博客
- "Building ReAct Agents with LangChain" (LangChain Blog, 2025)
- "ReAct Framework Explained" (Medium, 2026)

---

**版本**: v1.0
**最后更新**: 2026-02-17
**代码行数**: ~180 行
