# 核心概念 5: Multi-Agent 多代理协作

## 一句话定义

**Multi-Agent 是多个专业 AI 代理协同工作的架构,在 Agentic RAG 中通过任务分工、通信协议和结果聚合实现复杂问题的高效求解。**

---

## 详细解释

### 什么是 Multi-Agent?

Multi-Agent 是 Agentic RAG 的"团队协作"模式,包含:
- **专业分工**: 不同代理负责不同任务(检索、评估、生成)
- **通信协议**: 代理之间如何传递信息
- **结果聚合**: 如何整合多个代理的输出

**核心价值**: 像人类团队一样,通过专业分工和协作提升整体效率和质量。

### 为什么需要 Multi-Agent?

单代理的局限:
```python
# 单代理: 一个代理做所有事情
single_agent = Agent(tools=[search, rerank, generate])
result = single_agent.run(query)
# 问题: 代理需要在多个任务间切换,效率低,质量参差不齐
```

**Multi-Agent 优势**:
```python
# 多代理: 专业分工
retrieval_agent = Agent(tools=[search], expertise="检索")
evaluation_agent = Agent(tools=[rerank], expertise="评估")
generation_agent = Agent(tools=[generate], expertise="生成")

# 协作流程
docs = retrieval_agent.run(query)
best_docs = evaluation_agent.run(docs)
answer = generation_agent.run(best_docs)
# 优势: 每个代理专注自己擅长的任务,整体质量更高
```

### Multi-Agent 如何工作?

**协作模式**:
```
用户查询
    ↓
[协调器 Coordinator]
    ↓
┌─────────┬─────────┬─────────┐
│检索代理 │评估代理 │生成代理 │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
[结果聚合 Aggregator]
    ↓
最终答案
```

---

## 核心原理

### 原理图解

```
┌─────────────────────────────────────────┐
│       Multi-Agent RAG 架构              │
├─────────────────────────────────────────┤
│                                         │
│  查询: "比较 BERT 和 GPT 的优缺点"      │
│       ↓                                 │
│  [协调器 Coordinator]                   │
│   分析任务 → 分配给专业代理             │
│       ↓                                 │
│  ┌─────────────────────────────────┐   │
│  │ 检索代理 (Retrieval Agent)     │   │
│  │ - 搜索 BERT 相关文档            │   │
│  │ - 搜索 GPT 相关文档             │   │
│  │ 结果: 10 个文档                 │   │
│  └─────────────────────────────────┘   │
│       ↓                                 │
│  ┌─────────────────────────────────┐   │
│  │ 评估代理 (Evaluation Agent)    │   │
│  │ - 评估文档相关性                │   │
│  │ - 重排序文档                    │   │
│  │ 结果: 5 个最佳文档              │   │
│  └─────────────────────────────────┘   │
│       ↓                                 │
│  ┌─────────────────────────────────┐   │
│  │ 生成代理 (Generation Agent)    │   │
│  │ - 基于最佳文档生成对比分析      │   │
│  │ 结果: 完整答案                  │   │
│  └─────────────────────────────────┘   │
│       ↓                                 │
│  [聚合器 Aggregator]                    │
│   整合结果 → 返回最终答案               │
│                                         │
└─────────────────────────────────────────┘
```

### 工作流程

**Step 1: 任务分解**
```python
def decompose_task(query: str) -> List[Task]:
    """将查询分解为子任务"""
    tasks = [
        Task(agent="retrieval", action="搜索 BERT 文档"),
        Task(agent="retrieval", action="搜索 GPT 文档"),
        Task(agent="evaluation", action="评估文档质量"),
        Task(agent="generation", action="生成对比分析")
    ]
    return tasks
```

**Step 2: 任务分配**
```python
def assign_tasks(tasks: List[Task], agents: Dict[str, Agent]):
    """分配任务给合适的代理"""
    for task in tasks:
        agent = agents[task.agent]
        agent.add_task(task)
```

**Step 3: 并行执行**
```python
import asyncio

async def execute_parallel(agents: List[Agent]):
    """并行执行独立任务"""
    tasks = [agent.run_async() for agent in agents]
    results = await asyncio.gather(*tasks)
    return results
```

**Step 4: 结果聚合**
```python
def aggregate_results(results: List[Result]) -> FinalAnswer:
    """聚合多个代理的结果"""
    # 整合检索结果
    all_docs = [r.docs for r in results if r.type == "retrieval"]

    # 整合评估结果
    best_docs = [r.docs for r in results if r.type == "evaluation"]

    # 生成最终答案
    answer = generate_final_answer(best_docs)
    return answer
```

### 关键技术

**1. 协作模式 (2025-2026)**

**Router 模式** (路由分发):
```python
# 根据查询类型路由到不同代理
def router_pattern(query: str):
    if "技术" in query:
        return tech_agent.run(query)
    elif "业务" in query:
        return business_agent.run(query)
```

**Supervisor 模式** (监督协调):
```python
# 监督者协调多个工作代理
def supervisor_pattern(query: str):
    supervisor = SupervisorAgent()
    workers = [retrieval_agent, evaluation_agent, generation_agent]

    # 监督者分配任务
    tasks = supervisor.plan(query)
    for task, worker in zip(tasks, workers):
        worker.execute(task)

    # 监督者聚合结果
    return supervisor.aggregate(workers)
```

**Hierarchical 模式** (层级协作):
```python
# 层级代理结构
def hierarchical_pattern(query: str):
    # 顶层代理
    manager = ManagerAgent()
    plan = manager.create_plan(query)

    # 中层代理
    team_leads = [retrieval_lead, evaluation_lead]
    for lead in team_leads:
        lead.assign_tasks(plan)

    # 底层代理
    workers = [worker1, worker2, worker3]
    results = [worker.execute() for worker in workers]

    return manager.finalize(results)
```

**2. 通信协议 (2026)**

**Message Passing**:
```python
class Message:
    def __init__(self, sender: str, receiver: str, content: Any):
        self.sender = sender
        self.receiver = receiver
        self.content = content

# 代理间通信
retrieval_agent.send_message(
    Message("retrieval", "evaluation", docs)
)
```

**Shared Memory**:
```python
# 共享内存
shared_memory = {}

# 检索代理写入
shared_memory["docs"] = retrieval_agent.run(query)

# 评估代理读取
docs = shared_memory["docs"]
shared_memory["best_docs"] = evaluation_agent.run(docs)
```

**3. 框架实现 (2026)**

**CrewAI** (角色协作):
```python
from crewai import Agent, Task, Crew

# 定义代理
retriever = Agent(
    role="检索专家",
    goal="找到最相关的文档",
    backstory="我擅长语义检索"
)

evaluator = Agent(
    role="评估专家",
    goal="评估文档质量",
    backstory="我擅长判断相关性"
)

# 定义任务
task1 = Task(description="检索 BERT 文档", agent=retriever)
task2 = Task(description="评估文档质量", agent=evaluator)

# 创建团队
crew = Crew(agents=[retriever, evaluator], tasks=[task1, task2])
result = crew.kickoff()
```

**LangGraph** (状态图):
```python
from langgraph.graph import StateGraph

# 定义状态
class AgentState(TypedDict):
    query: str
    docs: List[str]
    best_docs: List[str]
    answer: str

# 定义节点
def retrieval_node(state):
    docs = retrieval_agent.run(state["query"])
    return {"docs": docs}

def evaluation_node(state):
    best_docs = evaluation_agent.run(state["docs"])
    return {"best_docs": best_docs}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_edge("retrieval", "evaluation")

app = workflow.compile()
```

---

## 手写实现

```python
"""
Multi-Agent 从零实现
演示: 检索代理 + 评估代理 + 生成代理协作
"""

from typing import List, Dict, Any
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 代理基类 =====
class Agent:
    """代理基类"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def run(self, input: Any) -> Any:
        """执行任务"""
        raise NotImplementedError

# ===== 2. 专业代理 =====
class RetrievalAgent(Agent):
    """检索代理"""
    def __init__(self):
        super().__init__("Retriever", "检索专家")

    def run(self, query: str) -> List[str]:
        """检索文档"""
        print(f"[{self.name}] 检索: {query}")

        # 模拟检索
        knowledge_base = {
            "bert": "BERT 是双向编码器,使用 Masked LM 预训练,擅长理解任务",
            "gpt": "GPT 是单向解码器,使用自回归预训练,擅长生成任务",
            "transformer": "Transformer 使用 Self-Attention 机制"
        }

        docs = []
        for key, value in knowledge_base.items():
            if key in query.lower():
                docs.append(value)

        print(f"[{self.name}] 找到 {len(docs)} 个文档\n")
        return docs

class EvaluationAgent(Agent):
    """评估代理"""
    def __init__(self):
        super().__init__("Evaluator", "评估专家")

    def run(self, docs: List[str]) -> List[str]:
        """评估文档质量"""
        print(f"[{self.name}] 评估 {len(docs)} 个文档")

        # 模拟评估(这里简单返回所有文档)
        best_docs = docs[:3]  # 取前3个

        print(f"[{self.name}] 选出 {len(best_docs)} 个最佳文档\n")
        return best_docs

class GenerationAgent(Agent):
    """生成代理"""
    def __init__(self):
        super().__init__("Generator", "生成专家")

    def run(self, docs: List[str]) -> str:
        """生成答案"""
        print(f"[{self.name}] 基于 {len(docs)} 个文档生成答案")

        context = "\n".join(docs)
        prompt = f"""
        基于以下信息生成答案:
        {context}

        答案:
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        print(f"[{self.name}] 答案已生成\n")
        return answer

# ===== 3. 协调器 =====
class Coordinator:
    """协调器"""
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    def run(self, query: str) -> str:
        """协调多个代理"""
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}\n")

        # Step 1: 检索
        docs = self.agents["retrieval"].run(query)

        # Step 2: 评估
        best_docs = self.agents["evaluation"].run(docs)

        # Step 3: 生成
        answer = self.agents["generation"].run(best_docs)

        return answer

# ===== 4. Multi-Agent 系统 =====
class MultiAgentRAG:
    """多代理 RAG 系统"""
    def __init__(self):
        # 创建专业代理
        self.agents = {
            "retrieval": RetrievalAgent(),
            "evaluation": EvaluationAgent(),
            "generation": GenerationAgent()
        }

        # 创建协调器
        self.coordinator = Coordinator(self.agents)

    def query(self, question: str) -> str:
        """查询接口"""
        return self.coordinator.run(question)

# ===== 5. 测试 =====
if __name__ == "__main__":
    system = MultiAgentRAG()

    test_queries = [
        "什么是 BERT?",
        "比较 BERT 和 GPT",
        "Transformer 的核心机制"
    ]

    for query in test_queries:
        answer = system.query(query)
        print(f"{'='*50}")
        print(f"最终答案:\n{answer}")
        print(f"{'='*50}\n\n")
```

---

## 在 RAG 中的应用

### 应用场景 1: 复杂查询处理

**问题**: "比较 2022 和 2023 年的营收,并分析增长原因"

**Multi-Agent 方案**:
```python
# 数据检索代理
data_agent = Agent(role="数据检索", tools=[db_search])
revenue_2022 = data_agent.run("2022年营收")
revenue_2023 = data_agent.run("2023年营收")

# 计算代理
calc_agent = Agent(role="数据计算", tools=[calculator])
growth = calc_agent.run(f"({revenue_2023} - {revenue_2022}) / {revenue_2022}")

# 分析代理
analysis_agent = Agent(role="业务分析", tools=[doc_search])
reasons = analysis_agent.run("2023年营收增长原因")

# 生成代理
gen_agent = Agent(role="报告生成", tools=[llm])
report = gen_agent.run(f"营收对比: {revenue_2022} vs {revenue_2023}, 增长: {growth}, 原因: {reasons}")
```

### 应用场景 2: 多源信息整合

**问题**: "整合内部文档和外部新闻,分析市场趋势"

**Multi-Agent 方案**:
```python
# 内部文档代理
internal_agent = Agent(role="内部检索", tools=[vector_search])
internal_docs = internal_agent.run("市场趋势")

# 外部新闻代理
external_agent = Agent(role="外部检索", tools=[web_search])
external_news = external_agent.run("市场趋势 2026")

# 整合代理
integration_agent = Agent(role="信息整合", tools=[llm])
analysis = integration_agent.run(f"内部: {internal_docs}, 外部: {external_news}")
```

### 应用场景 3: 质量保证

**问题**: 确保生成答案的准确性和完整性

**Multi-Agent 方案**:
```python
# 生成代理
gen_agent = Agent(role="答案生成")
answer = gen_agent.run(query)

# 验证代理
verify_agent = Agent(role="答案验证")
is_valid = verify_agent.run(f"验证答案: {answer}")

# 改进代理
if not is_valid:
    improve_agent = Agent(role="答案改进")
    answer = improve_agent.run(f"改进答案: {answer}")
```

---

## 主流框架实现

### CrewAI 实现 (推荐)

```python
from crewai import Agent, Task, Crew, Process

# 定义代理
retriever = Agent(
    role="检索专家",
    goal="找到最相关的文档",
    backstory="我是检索专家,擅长语义搜索",
    tools=[search_tool],
    verbose=True
)

evaluator = Agent(
    role="评估专家",
    goal="评估文档质量并重排序",
    backstory="我是评估专家,擅长判断相关性",
    tools=[rerank_tool],
    verbose=True
)

generator = Agent(
    role="生成专家",
    goal="生成高质量答案",
    backstory="我是生成专家,擅长综合信息",
    tools=[llm_tool],
    verbose=True
)

# 定义任务
task1 = Task(
    description="检索关于 {topic} 的文档",
    agent=retriever,
    expected_output="相关文档列表"
)

task2 = Task(
    description="评估文档质量并重排序",
    agent=evaluator,
    expected_output="最佳文档列表"
)

task3 = Task(
    description="基于最佳文档生成答案",
    agent=generator,
    expected_output="完整答案"
)

# 创建团队
crew = Crew(
    agents=[retriever, evaluator, generator],
    tasks=[task1, task2, task3],
    process=Process.sequential,  # 顺序执行
    verbose=True
)

# 执行
result = crew.kickoff(inputs={"topic": "BERT"})
```

### LangGraph 实现

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    query: str
    docs: List[str]
    best_docs: List[str]
    answer: str

def retrieval_node(state):
    """检索节点"""
    docs = retrieval_agent.run(state["query"])
    return {"docs": docs}

def evaluation_node(state):
    """评估节点"""
    best_docs = evaluation_agent.run(state["docs"])
    return {"best_docs": best_docs}

def generation_node(state):
    """生成节点"""
    answer = generation_agent.run(state["best_docs"])
    return {"answer": answer}

# 构建图
workflow = StateGraph(MultiAgentState)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_node("generation", generation_node)

workflow.set_entry_point("retrieval")
workflow.add_edge("retrieval", "evaluation")
workflow.add_edge("evaluation", "generation")
workflow.add_edge("generation", END)

app = workflow.compile()

# 执行
result = app.invoke({"query": "什么是 BERT?"})
```

### AutoGen 实现

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 定义代理
retriever = AssistantAgent(
    name="Retriever",
    system_message="你是检索专家,负责搜索相关文档",
    llm_config={"model": "gpt-4o"}
)

evaluator = AssistantAgent(
    name="Evaluator",
    system_message="你是评估专家,负责评估文档质量",
    llm_config={"model": "gpt-4o"}
)

generator = AssistantAgent(
    name="Generator",
    system_message="你是生成专家,负责生成最终答案",
    llm_config={"model": "gpt-4o"}
)

# 创建群聊
groupchat = GroupChat(
    agents=[retriever, evaluator, generator],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)

# 执行
user_proxy = UserProxyAgent(name="User")
user_proxy.initiate_chat(manager, message="搜索 BERT 相关文档")
```

---

## 最佳实践 (2025-2026)

### 性能优化

**1. 并行执行**
```python
import asyncio

async def parallel_agents(agents: List[Agent], query: str):
    """并行执行独立代理"""
    tasks = [agent.run_async(query) for agent in agents]
    results = await asyncio.gather(*tasks)
    return results
```

**2. 代理缓存**
```python
# 缓存代理结果
agent_cache = {}

def cached_agent_run(agent: Agent, input: str):
    cache_key = f"{agent.name}:{input}"
    if cache_key in agent_cache:
        return agent_cache[cache_key]

    result = agent.run(input)
    agent_cache[cache_key] = result
    return result
```

### 成本控制

**1. 代理选择**
```python
# 简单任务用小模型代理
simple_agent = Agent(llm="gpt-4o-mini")

# 复杂任务用大模型代理
complex_agent = Agent(llm="gpt-4o")
```

**2. 任务优先级**
```python
# 高优先级任务优先执行
def prioritize_tasks(tasks: List[Task]):
    return sorted(tasks, key=lambda t: t.priority, reverse=True)
```

### 错误处理

**1. 代理失败回退**
```python
def robust_multi_agent(agents: List[Agent], query: str):
    """鲁棒的多代理执行"""
    results = []
    for agent in agents:
        try:
            result = agent.run(query)
            results.append(result)
        except Exception as e:
            print(f"代理 {agent.name} 失败: {e}")
            # 使用备用代理
            backup_agent = get_backup_agent(agent)
            result = backup_agent.run(query)
            results.append(result)

    return results
```

**2. 结果验证**
```python
def validate_results(results: List[Result]):
    """验证代理结果"""
    for result in results:
        if not is_valid(result):
            raise ValueError(f"无效结果: {result}")

    return True
```

---

## 常见问题

### 问题 1: 多代理通信开销大怎么办?

**解决方案**:
```python
# 1. 使用共享内存
shared_state = {}

# 2. 减少通信次数
def batch_communication(messages: List[Message]):
    """批量通信"""
    # 一次发送多条消息
    pass

# 3. 异步通信
async def async_communication(agent1, agent2, message):
    """异步通信"""
    await agent1.send_async(message)
```

### 问题 2: 如何协调代理冲突?

**解决方案**:
```python
# 1. 投票机制
def vote_on_result(agents: List[Agent], results: List[Result]):
    """代理投票"""
    votes = {}
    for agent in agents:
        preferred = agent.vote(results)
        votes[preferred] = votes.get(preferred, 0) + 1

    return max(votes, key=votes.get)

# 2. 权重聚合
def weighted_aggregation(results: List[Result], weights: List[float]):
    """加权聚合"""
    final_result = sum(r * w for r, w in zip(results, weights))
    return final_result
```

### 问题 3: Multi-Agent vs Single Agent 如何选择?

**对比**:
```python
# Single Agent: 简单快速
single_agent = Agent(tools=[search, rerank, generate])
result = single_agent.run(query)  # 一个代理做所有事

# Multi-Agent: 专业高质
multi_agent = MultiAgentSystem([
    retrieval_agent,
    evaluation_agent,
    generation_agent
])
result = multi_agent.run(query)  # 专业分工

# 选择建议:
# - 简单任务 → Single Agent
# - 复杂任务 → Multi-Agent
# - 需要高质量 → Multi-Agent
# - 需要快速响应 → Single Agent
```

---

## 参考资源

### 论文
- "Communicative Agents for Software Development" (arXiv 2307.07924, 2023)
- "AutoGen: Enabling Next-Gen LLM Applications" (Microsoft, 2023)

### 博客
- "CrewAI vs LangGraph in 2026" (Medium, 2026)
  https://medium.com/@vikrantdheer/crewai-vs-langgraph-in-2026
- "Multi-Agent AI Orchestration Guide for 2026" (Dev.to, 2026)
  https://dev.to/pockit_tools/langgraph-vs-crewai-vs-autogen
- IBM: "The 2026 Guide to AI Agents" (2026)
  https://www.ibm.com/think/ai-agents

### 框架文档
- CrewAI: https://docs.crewai.com/
- LangGraph Multi-Agent: https://langchain-ai.github.io/langgraph/
- AutoGen: https://microsoft.github.io/autogen/

### 实践案例
- Oracle: "Build a Scalable Multi Agent RAG system" (2026)
  https://blogs.oracle.com/developers/build-a-scalable-multi-agent-rag-system

---

**版本**: v1.0
**最后更新**: 2026-02-17
**字数**: ~450 行
