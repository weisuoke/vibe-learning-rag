# 实战代码 - Agentic RAG

演示如何实现具有自主决策能力的RAG系统，支持多跳推理、工具使用和反思机制。

---

## 代码说明

**演示场景：** 构建一个能够自主规划检索步骤的智能RAG系统

**核心能力：**
1. 自主规划：根据查询复杂度决定检索策略
2. 多跳推理：逐步深入检索相关信息
3. 工具使用：调用计算器、搜索等工具
4. 反思机制：评估结果质量并调整策略

**技术栈：**
- OpenAI API（规划和生成）
- ChromaDB（向量存储）
- LangChain（Agent框架）

---

## 完整代码

```python
"""
Agentic RAG实现
演示：自主规划、多跳推理、工具使用
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import OpenAI
import chromadb

load_dotenv()

# ===== 1. 准备测试数据 =====
print("=== 准备测试数据 ===")

documents = [
    "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
    "Python 3.11于2022年10月发布，引入了性能优化和新特性。",
    "FastAPI是一个现代Python Web框架，基于Starlette和Pydantic构建。",
    "RAG系统结合了检索和生成技术，可以访问外部知识。",
    "向量数据库用于存储和检索高维向量，支持语义搜索。",
    "LangChain是一个用于构建LLM应用的框架，支持Agent开发。",
    "Agent可以自主决策、使用工具、进行多步推理。",
    "多跳推理是指Agent根据中间结果决定下一步行动。",
    "工具使用允许Agent调用外部API、计算器等功能。",
    "反思机制让Agent能够评估自己的输出质量并改进。"
]

print(f"✓ 准备了 {len(documents)} 个测试文档")

# ===== 2. 初始化组件 =====
print("\n=== 初始化组件 ===")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 向量存储
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("agentic_rag")

# 嵌入并存储文档
def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

embeddings = embed_texts(documents)
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("✓ 组件初始化完成")

# ===== 3. 定义工具 =====
print("\n=== 定义工具 ===")

class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError

class SearchTool(Tool):
    """搜索工具"""
    def __init__(self, collection, embed_fn):
        super().__init__(
            name="search",
            description="在知识库中搜索相关信息。输入：查询字符串。输出：相关文档列表。"
        )
        self.collection = collection
        self.embed_fn = embed_fn

    def execute(self, query: str, k: int = 3) -> List[str]:
        """执行搜索"""
        query_embedding = self.embed_fn([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"][0]

class CalculatorTool(Tool):
    """计算器工具"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算。输入：数学表达式字符串。输出：计算结果。"
        )

    def execute(self, expression: str) -> float:
        """执行计算"""
        try:
            # 安全的eval（仅用于演示）
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return f"计算错误: {str(e)}"

# 初始化工具
search_tool = SearchTool(collection, embed_texts)
calculator_tool = CalculatorTool()

tools = {
    "search": search_tool,
    "calculator": calculator_tool
}

print(f"✓ 定义了 {len(tools)} 个工具")

# ===== 4. 实现Planner（规划器） =====
print("\n=== 实现Planner（规划器） ===")

class Planner:
    """规划器：分析查询并生成执行计划"""

    def __init__(self, client, tools):
        self.client = client
        self.tools = tools

    def create_plan(self, query: str) -> List[Dict]:
        """创建执行计划"""
        # 构建工具描述
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        prompt = f"""你是一个智能规划器。分析用户查询，生成执行计划。

可用工具：
{tools_desc}

用户查询：{query}

请生成一个执行计划，包含以下步骤：
1. 分析查询的复杂度（简单/中等/复杂）
2. 决定需要使用哪些工具
3. 规划执行顺序

以JSON格式返回计划：
{{
  "complexity": "simple|medium|complex",
  "steps": [
    {{"action": "search", "query": "...", "reason": "..."}},
    {{"action": "calculator", "expression": "...", "reason": "..."}}
  ]
}}

只返回JSON，不要其他内容。"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # 解析JSON
        try:
            plan = json.loads(response.choices[0].message.content)
            return plan
        except json.JSONDecodeError:
            # 如果解析失败，返回默认计划
            return {
                "complexity": "simple",
                "steps": [{"action": "search", "query": query, "reason": "直接搜索"}]
            }

planner = Planner(client, tools)

# 测试规划器
test_query = "Python 3.11是什么时候发布的？"
plan = planner.create_plan(test_query)
print(f"\n查询: {test_query}")
print(f"复杂度: {plan['complexity']}")
print(f"计划步骤:")
for i, step in enumerate(plan['steps']):
    print(f"  {i+1}. {step['action']}: {step.get('query', step.get('expression', ''))}")
    print(f"     原因: {step['reason']}")

# ===== 5. 实现Executor（执行器） =====
print("\n=== 实现Executor（执行器） ===")

class Executor:
    """执行器：执行计划中的每个步骤"""

    def __init__(self, tools):
        self.tools = tools

    def execute_step(self, step: Dict) -> Any:
        """执行单个步骤"""
        action = step["action"]
        tool = self.tools.get(action)

        if not tool:
            return f"错误：工具 {action} 不存在"

        # 执行工具
        if action == "search":
            return tool.execute(step["query"])
        elif action == "calculator":
            return tool.execute(step["expression"])
        else:
            return "未知操作"

    def execute_plan(self, plan: Dict) -> List[Dict]:
        """执行完整计划"""
        results = []

        for step in plan["steps"]:
            result = self.execute_step(step)
            results.append({
                "step": step,
                "result": result
            })

        return results

executor = Executor(tools)

# 测试执行器
execution_results = executor.execute_plan(plan)
print(f"\n执行结果:")
for i, result in enumerate(execution_results):
    print(f"\n步骤 {i+1}:")
    print(f"  操作: {result['step']['action']}")
    print(f"  结果: {result['result']}")

# ===== 6. 实现Reflector（反思器） =====
print("\n=== 实现Reflector（反思器） ===")

class Reflector:
    """反思器：评估结果质量并决定是否需要额外步骤"""

    def __init__(self, client):
        self.client = client

    def reflect(self, query: str, results: List[Dict]) -> Dict:
        """反思执行结果"""
        # 构建结果摘要
        results_summary = "\n".join([
            f"步骤{i+1}: {r['step']['action']} -> {r['result']}"
            for i, r in enumerate(results)
        ])

        prompt = f"""你是一个反思器。评估执行结果是否足以回答用户查询。

用户查询：{query}

执行结果：
{results_summary}

请评估：
1. 结果是否充分回答了查询？
2. 是否需要额外的搜索或计算？
3. 如果需要，应该执行什么操作？

以JSON格式返回：
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "reason": "...",
  "next_action": {{"action": "...", "query": "..."}} // 如果需要
}}

只返回JSON。"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        try:
            reflection = json.loads(response.choices[0].message.content)
            return reflection
        except json.JSONDecodeError:
            return {
                "sufficient": True,
                "confidence": 0.5,
                "reason": "无法解析反思结果"
            }

reflector = Reflector(client)

# 测试反思器
reflection = reflector.reflect(test_query, execution_results)
print(f"\n反思结果:")
print(f"  是否充分: {reflection['sufficient']}")
print(f"  置信度: {reflection['confidence']}")
print(f"  原因: {reflection['reason']}")

# ===== 7. 实现Generator（生成器） =====
print("\n=== 实现Generator（生成器） ===")

class Generator:
    """生成器：基于执行结果生成最终答案"""

    def __init__(self, client):
        self.client = client

    def generate(self, query: str, results: List[Dict]) -> str:
        """生成最终答案"""
        # 构建上下文
        context = "\n\n".join([
            f"信息{i+1}（来自{r['step']['action']}）:\n{r['result']}"
            for i, r in enumerate(results)
        ])

        prompt = f"""基于以下信息回答用户查询。

用户查询：{query}

可用信息：
{context}

要求：
1. 仅基于提供的信息回答
2. 如果信息不足，明确说明
3. 保持简洁准确

答案："""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        return response.choices[0].message.content

generator = Generator(client)

# 测试生成器
answer = generator.generate(test_query, execution_results)
print(f"\n最终答案:")
print(answer)

# ===== 8. 完整的Agentic RAG系统 =====
print("\n=== 完整的Agentic RAG系统 ===")

class AgenticRAG:
    """完整的Agentic RAG系统"""

    def __init__(self, client, tools):
        self.planner = Planner(client, tools)
        self.executor = Executor(tools)
        self.reflector = Reflector(client)
        self.generator = Generator(client)
        self.max_iterations = 3

    def query(self, question: str, verbose: bool = True) -> Dict:
        """查询系统"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"查询: {question}")
            print(f"{'='*60}")

        all_results = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"\n--- 迭代 {iteration} ---")

            # 1. 规划
            if iteration == 1:
                plan = self.planner.create_plan(question)
            else:
                # 后续迭代使用反思结果
                plan = {
                    "complexity": "medium",
                    "steps": [reflection["next_action"]]
                }

            if verbose:
                print(f"计划: {len(plan['steps'])} 个步骤")

            # 2. 执行
            results = self.executor.execute_plan(plan)
            all_results.extend(results)

            if verbose:
                for i, r in enumerate(results):
                    print(f"  步骤{i+1}: {r['step']['action']}")

            # 3. 反思
            reflection = self.reflector.reflect(question, all_results)

            if verbose:
                print(f"反思: 充分={reflection['sufficient']}, 置信度={reflection['confidence']:.2f}")

            # 如果结果充分，退出循环
            if reflection["sufficient"]:
                break

            # 如果没有下一步行动，退出循环
            if "next_action" not in reflection:
                break

        # 4. 生成最终答案
        answer = self.generator.generate(question, all_results)

        return {
            "question": question,
            "answer": answer,
            "iterations": iteration,
            "steps": len(all_results),
            "confidence": reflection.get("confidence", 0.5)
        }

# 创建Agentic RAG系统
agentic_rag = AgenticRAG(client, tools)

# ===== 9. 测试不同复杂度的查询 =====
print("\n=== 测试不同复杂度的查询 ===")

test_queries = [
    "Python是什么？",  # 简单查询
    "Python 3.11是什么时候发布的？",  # 中等查询
    "FastAPI基于哪些技术构建？它与Python有什么关系？"  # 复杂查询（需要多跳）
]

for query in test_queries:
    result = agentic_rag.query(query, verbose=True)
    print(f"\n最终答案: {result['answer']}")
    print(f"迭代次数: {result['iterations']}, 步骤数: {result['steps']}, 置信度: {result['confidence']:.2f}")

# ===== 10. 对比普通RAG vs Agentic RAG =====
print("\n=== 对比普通RAG vs Agentic RAG ===")

def simple_rag(query: str) -> str:
    """简单的RAG系统（无Agent能力）"""
    # 直接搜索
    results = search_tool.execute(query, k=3)
    context = "\n".join(results)

    # 直接生成
    prompt = f"基于以下上下文回答问题。\n\n上下文：\n{context}\n\n问题：{query}\n\n答案："
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content

complex_query = "FastAPI基于哪些技术构建？它与Python有什么关系？"

print(f"\n查询: {complex_query}\n")

print("【普通RAG】")
simple_answer = simple_rag(complex_query)
print(f"答案: {simple_answer}\n")

print("【Agentic RAG】")
agentic_result = agentic_rag.query(complex_query, verbose=False)
print(f"答案: {agentic_result['answer']}")
print(f"迭代: {agentic_result['iterations']}, 步骤: {agentic_result['steps']}")

print("\n=== Agentic RAG演示完成 ===")
```

---

## 运行输出示例

```
=== 准备测试数据 ===
✓ 准备了 10 个测试文档

=== 初始化组件 ===
✓ 组件初始化完成

=== 定义工具 ===
✓ 定义了 2 个工具

=== 实现Planner（规划器） ===

查询: Python 3.11是什么时候发布的？
复杂度: simple
计划步骤:
  1. search: Python 3.11发布时间
     原因: 直接搜索相关信息

=== 实现Executor（执行器） ===

执行结果:

步骤 1:
  操作: search
  结果: ['Python 3.11于2022年10月发布，引入了性能优化和新特性。', 'Python是一种高级编程语言，由Guido van Rossum于1991年创建。', 'FastAPI是一个现代Python Web框架，基于Starlette和Pydantic构建。']

=== 实现Reflector（反思器） ===

反思结果:
  是否充分: True
  置信度: 0.95
  原因: 第一个搜索结果直接回答了问题

=== 实现Generator（生成器） ===

最终答案:
Python 3.11于2022年10月发布。

=== 完整的Agentic RAG系统 ===

=== 测试不同复杂度的查询 ===

============================================================
查询: Python是什么？
============================================================

--- 迭代 1 ---
计划: 1 个步骤
  步骤1: search
反思: 充分=True, 置信度=0.90

最终答案: Python是一种高级编程语言，由Guido van Rossum于1991年创建。
迭代次数: 1, 步骤数: 1, 置信度: 0.90

============================================================
查询: Python 3.11是什么时候发布的？
============================================================

--- 迭代 1 ---
计划: 1 个步骤
  步骤1: search
反思: 充分=True, 置信度=0.95

最终答案: Python 3.11于2022年10月发布。
迭代次数: 1, 步骤数: 1, 置信度: 0.95

============================================================
查询: FastAPI基于哪些技术构建？它与Python有什么关系？
============================================================

--- 迭代 1 ---
计划: 2 个步骤
  步骤1: search
  步骤2: search
反思: 充分=False, 置信度=0.70

--- 迭代 2 ---
计划: 1 个步骤
  步骤1: search
反思: 充分=True, 置信度=0.85

最终答案: FastAPI是一个现代Python Web框架，基于Starlette和Pydantic构建。它是用Python编写的，专门用于构建API。
迭代次数: 2, 步骤数: 3, 置信度: 0.85

=== 对比普通RAG vs Agentic RAG ===

查询: FastAPI基于哪些技术构建？它与Python有什么关系？

【普通RAG】
答案: FastAPI是一个Python Web框架，基于Starlette和Pydantic构建。

【Agentic RAG】
答案: FastAPI是一个现代Python Web框架，基于Starlette和Pydantic构建。它是用Python编写的，专门用于构建API。
迭代: 2, 步骤: 3

=== Agentic RAG演示完成 ===
```

---

## 关键要点

**1. Agentic RAG的核心组件**
- Planner：分析查询，生成执行计划
- Executor：执行计划中的每个步骤
- Reflector：评估结果质量，决定是否继续
- Generator：生成最终答案

**2. 与普通RAG的区别**

| 维度 | 普通RAG | Agentic RAG |
|------|---------|-------------|
| 检索策略 | 固定（一次检索） | 动态（多跳推理） |
| 工具使用 | 无 | 支持多种工具 |
| 反思能力 | 无 | 评估并调整 |
| 复杂查询 | 效果差 | 效果好 |
| LLM调用 | 1-2次 | 3-5次 |

**3. 适用场景**
- 需要多步推理的复杂查询
- 需要调用外部工具（计算、搜索）
- 需要动态调整检索策略
- 对答案质量要求高

**4. 成本权衡**
- LLM调用次数增加3-5倍
- 延迟增加2-3倍
- 准确率提升40-60%（复杂查询）

---

## 扩展建议

**1. 添加更多工具**
```python
class WebSearchTool(Tool):
    """网络搜索工具"""
    def execute(self, query: str):
        # 调用搜索API
        pass

class DatabaseTool(Tool):
    """数据库查询工具"""
    def execute(self, sql: str):
        # 执行SQL查询
        pass
```

**2. 改进规划策略**
- 使用Few-shot示例提升规划质量
- 添加查询分解（Query Decomposition）
- 支持并行执行多个步骤

**3. 增强反思机制**
- 添加置信度阈值
- 支持回溯（撤销错误步骤）
- 记录历史决策

---

## 下一步

学习 **07_实战代码_04_GraphRAG.md**，实现知识图谱增强的RAG系统。
