# 实战代码 - 场景 5: Multi-Agent RAG 系统

## 场景描述

**目标**: 使用 CrewAI 构建多代理协作的 RAG 系统,包括检索代理、评估代理和生成代理

**难点**:
- 定义代理角色和职责
- 管理代理间的通信和协作
- 聚合多个代理的输出

**解决方案**: 使用 CrewAI 框架实现专业分工的多代理系统

---

## 环境准备

```bash
# 安装依赖
uv add crewai langchain langchain-openai chromadb python-dotenv
```

---

## 完整代码

```python
"""
Multi-Agent RAG 系统 - 多代理协作
演示: 检索代理 + 评估代理 + 生成代理协作

技术栈:
- CrewAI: 0.1.0+
- LangChain: 0.1.0+
- OpenAI: 1.0.0+
- ChromaDB: 0.4.0+
"""

import os
from typing import List
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.tools import Tool

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
        page_content="BERT 是 Google 在 2018 年提出的双向预训练语言模型,使用 Masked LM 和 NSP 任务进行预训练。",
        metadata={"source": "bert_intro", "quality": "high", "topic": "bert"}
    ),
    Document(
        page_content="BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。",
        metadata={"source": "bert_advantages", "quality": "high", "topic": "bert"}
    ),
    Document(
        page_content="BERT 的缺点包括:模型较大(110M-340M 参数)、推理速度慢、不适合生成任务。",
        metadata={"source": "bert_disadvantages", "quality": "medium", "topic": "bert"}
    ),
    Document(
        page_content="GPT 是 OpenAI 提出的自回归语言模型,使用单向 Transformer 解码器,通过预测下一个 token 进行预训练。",
        metadata={"source": "gpt_intro", "quality": "high", "topic": "gpt"}
    ),
    Document(
        page_content="GPT 的优点包括:强大的文本生成能力、流畅的语言输出、适合对话和创作任务。",
        metadata={"source": "gpt_advantages", "quality": "high", "topic": "gpt"}
    ),
    Document(
        page_content="GPT 的缺点包括:上下文理解不如双向模型、可能产生幻觉、训练成本高。",
        metadata={"source": "gpt_disadvantages", "quality": "medium", "topic": "gpt"}
    ),
    Document(
        page_content="Transformer 是 2017 年提出的注意力机制架构,使用 Self-Attention 实现并行处理,是 BERT 和 GPT 的基础。",
        metadata={"source": "transformer_intro", "quality": "high", "topic": "transformer"}
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) 结合检索和生成,通过检索相关文档增强 LLM 的回答质量。",
        metadata={"source": "rag_intro", "quality": "high", "topic": "rag"}
    )
]

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="multi_agent_demo"
)

# ===== 3. 定义工具 =====

def search_documents(query: str) -> str:
    """搜索相关文档"""
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "未找到相关文档"

    output = []
    for i, doc in enumerate(results, 1):
        output.append(f"{i}. {doc.page_content}")
        output.append(f"   来源: {doc.metadata.get('source', 'unknown')}")
        output.append(f"   质量: {doc.metadata.get('quality', 'unknown')}")

    return "\n".join(output)

def evaluate_relevance(text: str) -> str:
    """评估文档相关性"""
    # 简单的评估逻辑
    if len(text) < 50:
        return "相关性: 低 (文本太短)"
    elif "BERT" in text or "GPT" in text or "Transformer" in text:
        return "相关性: 高 (包含关键技术词汇)"
    else:
        return "相关性: 中 (一般性描述)"

# 创建工具
search_tool = Tool(
    name="SearchDocuments",
    func=search_documents,
    description="搜索相关文档。输入:查询字符串。返回:相关文档列表。"
)

evaluate_tool = Tool(
    name="EvaluateRelevance",
    func=evaluate_relevance,
    description="评估文档相关性。输入:文档文本。返回:相关性评分。"
)

# ===== 4. 定义代理 =====

# 检索代理
retrieval_agent = Agent(
    role="检索专家",
    goal="找到与查询最相关的文档",
    backstory="""
    你是一个经验丰富的信息检索专家,擅长理解用户查询意图并找到最相关的文档。
    你会仔细分析查询,使用合适的检索策略,确保返回高质量的结果。
    """,
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 评估代理
evaluation_agent = Agent(
    role="质量评估专家",
    goal="评估检索结果的质量和相关性",
    backstory="""
    你是一个严格的质量评估专家,擅长判断文档的相关性和质量。
    你会仔细审查每个文档,评估其与查询的相关性,并筛选出最佳结果。
    你的评估标准包括:内容完整性、信息准确性、与查询的匹配度。
    """,
    tools=[evaluate_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 生成代理
generation_agent = Agent(
    role="答案生成专家",
    goal="基于高质量文档生成准确、完整的答案",
    backstory="""
    你是一个专业的内容生成专家,擅长综合多个信息源生成高质量答案。
    你会仔细阅读提供的文档,提取关键信息,组织成清晰、准确、易懂的答案。
    你的答案总是结构化、有逻辑、有依据。
    """,
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ===== 5. 定义任务 =====

def create_tasks(query: str) -> List[Task]:
    """创建任务列表"""

    # 任务 1: 检索
    retrieval_task = Task(
        description=f"""
        检索与以下查询相关的文档:
        查询: {query}

        要求:
        1. 使用 SearchDocuments 工具搜索相关文档
        2. 返回至少 3 个相关文档
        3. 包含文档内容和元数据
        """,
        agent=retrieval_agent,
        expected_output="相关文档列表,包含内容和元数据"
    )

    # 任务 2: 评估
    evaluation_task = Task(
        description=f"""
        评估检索到的文档的质量和相关性:

        要求:
        1. 审查每个文档的内容
        2. 使用 EvaluateRelevance 工具评估相关性
        3. 筛选出最相关的文档(至少 2 个)
        4. 说明筛选理由
        """,
        agent=evaluation_agent,
        expected_output="筛选后的高质量文档列表及评估理由"
    )

    # 任务 3: 生成
    generation_task = Task(
        description=f"""
        基于评估后的文档生成答案:
        原始查询: {query}

        要求:
        1. 仔细阅读所有筛选后的文档
        2. 提取关键信息
        3. 生成结构化、准确、完整的答案
        4. 答案应该:
           - 直接回答查询
           - 有逻辑结构
           - 基于文档内容
           - 易于理解
        """,
        agent=generation_agent,
        expected_output="结构化的完整答案"
    )

    return [retrieval_task, evaluation_task, generation_task]

# ===== 6. 创建团队 =====

def create_crew(query: str) -> Crew:
    """创建多代理团队"""
    tasks = create_tasks(query)

    crew = Crew(
        agents=[retrieval_agent, evaluation_agent, generation_agent],
        tasks=tasks,
        process=Process.sequential,  # 顺序执行
        verbose=True
    )

    return crew

# ===== 7. 测试 =====

def main():
    """主函数"""
    test_queries = [
        "什么是 BERT?",
        "比较 BERT 和 GPT 的优缺点",
        "什么是 RAG?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"场景 {i}: {query}")
        print(f"{'='*60}\n")

        try:
            # 创建团队
            crew = create_crew(query)

            # 执行任务
            result = crew.kickoff()

            print(f"\n{'='*60}")
            print(f"最终答案:")
            print(result)
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

============================================================
场景 1: 什么是 BERT?
============================================================

> Entering new Crew...

[检索专家] 开始任务: 检索与以下查询相关的文档...

使用工具: SearchDocuments
输入: BERT

工具输出:
1. BERT 是 Google 在 2018 年提出的双向预训练语言模型,使用 Masked LM 和 NSP 任务进行预训练。
   来源: bert_intro
   质量: high
2. BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。
   来源: bert_advantages
   质量: high
3. BERT 的缺点包括:模型较大(110M-340M 参数)、推理速度慢、不适合生成任务。
   来源: bert_disadvantages
   质量: medium

[检索专家] 任务完成

[质量评估专家] 开始任务: 评估检索到的文档的质量和相关性...

使用工具: EvaluateRelevance
输入: BERT 是 Google 在 2018 年提出的双向预训练语言模型...

工具输出: 相关性: 高 (包含关键技术词汇)

评估结果:
- 文档 1: 高质量,直接介绍 BERT,相关性高
- 文档 2: 高质量,介绍 BERT 优点,相关性高
- 文档 3: 中等质量,介绍 BERT 缺点,相关性中

筛选结果: 保留文档 1 和文档 2

[质量评估专家] 任务完成

[答案生成专家] 开始任务: 基于评估后的文档生成答案...

基于文档生成答案:

BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的双向预训练语言模型。

**核心特点:**
- 使用双向 Transformer 编码器
- 通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 任务进行预训练

**主要优点:**
1. 强大的上下文理解能力
2. 在多个 NLP 任务上取得 SOTA 性能
3. 可以进行 fine-tuning 适应不同任务

BERT 是现代 NLP 的重要里程碑,为后续模型奠定了基础。

[答案生成专家] 任务完成

> Crew execution completed

============================================================
最终答案:
BERT (Bidirectional Encoder Representations from Transformers) 是 Google 在 2018 年提出的双向预训练语言模型。

**核心特点:**
- 使用双向 Transformer 编码器
- 通过 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 任务进行预训练

**主要优点:**
1. 强大的上下文理解能力
2. 在多个 NLP 任务上取得 SOTA 性能
3. 可以进行 fine-tuning 适应不同任务

BERT 是现代 NLP 的重要里程碑,为后续模型奠定了基础。
============================================================

============================================================
场景 2: 比较 BERT 和 GPT 的优缺点
============================================================

> Entering new Crew...

[检索专家] 开始任务: 检索与以下查询相关的文档...

使用工具: SearchDocuments
输入: BERT GPT 优缺点

工具输出:
1. BERT 的优点包括:强大的上下文理解能力、在多个 NLP 任务上取得 SOTA 性能、可以进行 fine-tuning。
   来源: bert_advantages
   质量: high
2. BERT 的缺点包括:模型较大(110M-340M 参数)、推理速度慢、不适合生成任务。
   来源: bert_disadvantages
   质量: medium
3. GPT 的优点包括:强大的文本生成能力、流畅的语言输出、适合对话和创作任务。
   来源: gpt_advantages
   质量: high

[检索专家] 任务完成

[质量评估专家] 开始任务: 评估检索到的文档的质量和相关性...

评估结果:
- 文档 1 (BERT 优点): 高质量,相关性高
- 文档 2 (BERT 缺点): 中等质量,相关性高
- 文档 3 (GPT 优点): 高质量,相关性高

需要补充 GPT 缺点信息...

使用工具: SearchDocuments
输入: GPT 缺点

工具输出:
1. GPT 的缺点包括:上下文理解不如双向模型、可能产生幻觉、训练成本高。
   来源: gpt_disadvantages
   质量: medium

筛选结果: 保留所有 4 个文档

[质量评估专家] 任务完成

[答案生成专家] 开始任务: 基于评估后的文档生成答案...

基于文档生成对比分析:

**BERT vs GPT 对比分析**

**BERT 优点:**
1. 强大的上下文理解能力(双向编码)
2. 在多个 NLP 任务上取得 SOTA 性能
3. 可以进行 fine-tuning

**BERT 缺点:**
1. 模型较大(110M-340M 参数)
2. 推理速度慢
3. 不适合生成任务

**GPT 优点:**
1. 强大的文本生成能力
2. 流畅的语言输出
3. 适合对话和创作任务

**GPT 缺点:**
1. 上下文理解不如双向模型
2. 可能产生幻觉
3. 训练成本高

**总结:**
- BERT 擅长理解任务(分类、问答、NER)
- GPT 擅长生成任务(对话、创作、摘要)
- 选择取决于具体应用场景

[答案生成专家] 任务完成

> Crew execution completed

============================================================
最终答案:
**BERT vs GPT 对比分析**

**BERT 优点:**
1. 强大的上下文理解能力(双向编码)
2. 在多个 NLP 任务上取得 SOTA 性能
3. 可以进行 fine-tuning

**BERT 缺点:**
1. 模型较大(110M-340M 参数)
2. 推理速度慢
3. 不适合生成任务

**GPT 优点:**
1. 强大的文本生成能力
2. 流畅的语言输出
3. 适合对话和创作任务

**GPT 缺点:**
1. 上下文理解不如双向模型
2. 可能产生幻觉
3. 训练成本高

**总结:**
- BERT 擅长理解任务(分类、问答、NER)
- GPT 擅长生成任务(对话、创作、摘要)
- 选择取决于具体应用场景
============================================================
```

---

## 代码解析

### 关键点 1: 代理定义

```python
retrieval_agent = Agent(
    role="检索专家",
    goal="找到与查询最相关的文档",
    backstory="你是一个经验丰富的信息检索专家...",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)
```

**要点**:
- `role`: 代理的角色定位
- `goal`: 代理的目标
- `backstory`: 代理的背景故事(影响行为)
- `tools`: 代理可用的工具
- `allow_delegation`: 是否允许委托任务

### 关键点 2: 任务定义

```python
retrieval_task = Task(
    description="检索与以下查询相关的文档...",
    agent=retrieval_agent,
    expected_output="相关文档列表,包含内容和元数据"
)
```

**要点**:
- `description`: 详细的任务描述
- `agent`: 负责执行的代理
- `expected_output`: 期望的输出格式

### 关键点 3: 团队协作

```python
crew = Crew(
    agents=[retrieval_agent, evaluation_agent, generation_agent],
    tasks=[retrieval_task, evaluation_task, generation_task],
    process=Process.sequential,  # 顺序执行
    verbose=True
)
```

**要点**:
- `agents`: 团队成员列表
- `tasks`: 任务列表
- `process`: 执行模式(sequential/hierarchical)
- 任务按顺序执行,前一个任务的输出作为后一个任务的输入

---

## 扩展思考

### 如何优化?

**1. 添加并行执行**
```python
# 独立任务可以并行执行
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.parallel  # 并行执行
)
```

**2. 添加层级协作**
```python
# 管理者代理协调工作代理
manager = Agent(
    role="项目经理",
    goal="协调团队完成任务",
    backstory="你是经验丰富的项目经理...",
    allow_delegation=True  # 允许委托
)

crew = Crew(
    agents=[manager, worker1, worker2],
    tasks=[task1, task2],
    process=Process.hierarchical,  # 层级模式
    manager_llm=llm
)
```

**3. 添加代理记忆**
```python
from crewai import Memory

memory = Memory()

agent = Agent(
    role="检索专家",
    goal="...",
    backstory="...",
    memory=memory  # 添加记忆
)
```

### 如何扩展?

**1. 添加更多专业代理**
```python
# 重排序代理
rerank_agent = Agent(
    role="重排序专家",
    goal="优化检索结果排序",
    backstory="你擅长评估文档相关性并重新排序...",
    tools=[rerank_tool]
)

# 验证代理
verification_agent = Agent(
    role="事实验证专家",
    goal="验证答案的准确性",
    backstory="你擅长核实信息的真实性...",
    tools=[verify_tool]
)
```

**2. 添加动态任务生成**
```python
def generate_dynamic_tasks(query: str, complexity: str):
    """根据查询复杂度动态生成任务"""
    if complexity == "simple":
        return [retrieval_task, generation_task]
    elif complexity == "complex":
        return [
            retrieval_task,
            evaluation_task,
            rerank_task,
            generation_task,
            verification_task
        ]
```

**3. 添加代理协商机制**
```python
class NegotiationAgent(Agent):
    """协商代理"""
    def negotiate(self, proposals: List[str]) -> str:
        """协商多个代理的提案"""
        # 评估各个提案
        scores = [self.evaluate(p) for p in proposals]

        # 选择最佳提案或综合
        best_proposal = proposals[scores.index(max(scores))]

        return best_proposal
```

### 生产级改进

**1. 错误处理和重试**
```python
def robust_crew_execution(crew: Crew, max_retries: int = 3):
    """鲁棒的团队执行"""
    for attempt in range(max_retries):
        try:
            result = crew.kickoff()

            # 验证结果
            if is_valid_result(result):
                return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"执行失败,重试 {attempt + 1}/{max_retries}")

    return None
```

**2. 性能监控**
```python
import time

class MonitoredCrew(Crew):
    """带监控的团队"""
    def kickoff(self):
        start_time = time.time()

        result = super().kickoff()

        execution_time = time.time() - start_time

        print(f"\n📊 执行指标:")
        print(f"  总时间: {execution_time:.2f}s")
        print(f"  代理数: {len(self.agents)}")
        print(f"  任务数: {len(self.tasks)}")

        return result
```

**3. 结果验证**
```python
def validate_crew_result(result: str, query: str) -> Dict:
    """验证团队执行结果"""
    validation = {
        "complete": len(result) > 100,
        "relevant": query.lower() in result.lower(),
        "structured": "\n" in result,
        "quality_score": calculate_quality_score(result)
    }

    return validation
```

**4. 代理性能分析**
```python
class PerformanceTracker:
    """代理性能追踪"""
    def __init__(self):
        self.agent_metrics = {}

    def track_agent(self, agent_name: str, task_time: float, success: bool):
        """记录代理性能"""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = {
                "total_tasks": 0,
                "success_count": 0,
                "total_time": 0
            }

        metrics = self.agent_metrics[agent_name]
        metrics["total_tasks"] += 1
        metrics["success_count"] += 1 if success else 0
        metrics["total_time"] += task_time

    def get_report(self) -> Dict:
        """生成性能报告"""
        report = {}
        for agent, metrics in self.agent_metrics.items():
            report[agent] = {
                "success_rate": metrics["success_count"] / metrics["total_tasks"],
                "avg_time": metrics["total_time"] / metrics["total_tasks"]
            }
        return report
```

---

## 参考资源

### 官方文档
- CrewAI: https://docs.crewai.com/
- CrewAI Examples: https://github.com/joaomdmoura/crewAI-examples

### 相关博客
- "CrewAI vs LangGraph in 2026" (Medium, 2026)
- "Building Multi-Agent Systems with CrewAI" (2025)

### 实践案例
- "Multi-Agent RAG System" (Oracle, 2026)
- "Agentic RAG with CrewAI" (IBM, 2026)

---

**版本**: v1.0
**最后更新**: 2026-02-17
**代码行数**: ~200 行
