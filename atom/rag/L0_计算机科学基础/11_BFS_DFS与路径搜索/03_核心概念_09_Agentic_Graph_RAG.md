# 核心概念 09：Agentic Graph RAG

## 一句话定义

**Agentic Graph RAG是通过LLM理解查询意图并动态选择遍历策略（BFS/DFS/混合）的智能检索系统，根据查询类型（事实/因果/探索）自动调整检索深度和策略。**

---

## 核心思想

**传统方法：** 固定策略（总是用BFS或总是用DFS）

**Agentic方法：** 查询分类驱动策略选择

```
查询 → LLM分类意图 → 选择策略
├─ 事实查询 → BFS浅层遍历
├─ 因果查询 → DFS深层遍历
├─ 探索查询 → 混合策略
└─ 复杂查询 → 自校正循环
```

**来源：** Agentic Graph RAG (2025) - https://pmc.ncbi.nlm.nih.gov/articles/PMC12748213

---

## 查询分类

### 查询类型

```python
from enum import Enum

class QueryType(Enum):
    FACTUAL = "factual"        # 事实查询
    CAUSAL = "causal"          # 因果查询
    EXPLORATORY = "exploratory"  # 探索查询
    COMPLEX = "complex"        # 复杂查询

def classify_query(llm, query: str) -> QueryType:
    """
    LLM分类查询类型
    """
    prompt = f"""
分类以下查询的类型：

查询：{query}

类型定义：
- factual：事实查询，答案在1-2跳内（如"谁是Python的创始人？"）
- causal：因果查询，需要深层推理链（如"为什么Python成为AI首选语言？"）
- exploratory：探索查询，需要广泛检索（如"Python在AI领域的应用有哪些？"）
- complex：复杂查询，需要多步推理（如"比较Python和Java在AI开发中的优劣"）

只输出类型名称（factual/causal/exploratory/complex）：
"""

    response = llm.generate(prompt, max_tokens=10)
    query_type = response.strip().lower()

    try:
        return QueryType(query_type)
    except ValueError:
        return QueryType.FACTUAL  # 默认
```

---

## 策略选择

### 基于查询类型的策略

```python
from collections import deque
from typing import List

def agentic_graph_rag(kg, query: str, llm) -> List[str]:
    """
    Agentic Graph RAG主函数

    根据查询类型选择策略
    """
    # 步骤1：分类查询
    query_type = classify_query(llm, query)

    # 步骤2：选择策略
    if query_type == QueryType.FACTUAL:
        return factual_strategy(kg, query, max_depth=2)
    elif query_type == QueryType.CAUSAL:
        return causal_strategy(kg, query, max_depth=5)
    elif query_type == QueryType.EXPLORATORY:
        return exploratory_strategy(kg, query)
    else:  # COMPLEX
        return complex_strategy(kg, query, llm)

def factual_strategy(kg, query: str, max_depth: int = 2) -> List[str]:
    """
    事实查询策略：BFS浅层遍历

    特点：快速、广度覆盖
    """
    entities = kg.extract_entities(query)
    expanded = set(entities)
    queue = deque([(e, 0) for e in entities])

    while queue:
        entity, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor in kg.get_neighbors(entity):
            if neighbor not in expanded:
                expanded.add(neighbor)
                queue.append((neighbor, depth + 1))

    return list(expanded)

def causal_strategy(kg, query: str, max_depth: int = 5) -> List[str]:
    """
    因果查询策略：DFS深层遍历

    特点：深度探索、因果链
    """
    entities = kg.extract_entities(query)
    causal_relations = ['causes', 'leads_to', 'results_in', 'influences']

    def dfs_causal(entity: str, depth: int, visited: set, path: List[str]) -> List[str]:
        if depth >= max_depth or entity in visited:
            return path

        visited.add(entity)
        path.append(entity)

        # 只沿着因果关系边
        for relation, neighbor in kg.get_neighbors_with_relation(entity):
            if relation in causal_relations and neighbor not in visited:
                return dfs_causal(neighbor, depth + 1, visited, path)

        return path

    all_paths = []
    for entity in entities:
        path = dfs_causal(entity, 0, set(), [])
        if len(path) > 1:
            all_paths.extend(path)

    return all_paths

def exploratory_strategy(kg, query: str) -> List[str]:
    """
    探索查询策略：混合BFS+DFS

    特点：广度+深度结合
    """
    # BFS扩展
    entities = kg.extract_entities(query)
    expanded = set()
    queue = deque([(e, 0) for e in entities])

    while queue:
        entity, depth = queue.popleft()
        if depth >= 2:
            continue

        expanded.add(entity)
        for neighbor in kg.get_neighbors(entity):
            if neighbor not in expanded:
                queue.append((neighbor, depth + 1))

    # DFS组织
    organized = []
    visited = set()

    def dfs_organize(entity: str, path: List[str], depth: int):
        if depth >= 3 or entity in visited:
            return
        visited.add(entity)
        path.append(entity)

        neighbors = kg.get_neighbors(entity)
        if not neighbors or depth == 2:
            organized.extend(path)
        else:
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs_organize(neighbor, path[:], depth + 1)

    for entity in expanded:
        if entity not in visited:
            dfs_organize(entity, [], 0)

    return organized
```

---

## 自校正循环

### 复杂查询的自校正

```python
def complex_strategy(kg, query: str, llm, max_iterations: int = 3) -> List[str]:
    """
    复杂查询策略：自校正循环

    流程：
    1. 初始检索
    2. LLM评估结果
    3. 如果不足，调整策略重新检索
    4. 重复直到满足或达到最大迭代次数
    """
    results = []
    strategy = "factual"  # 初始策略

    for iteration in range(max_iterations):
        # 步骤1：根据当前策略检索
        if strategy == "factual":
            new_results = factual_strategy(kg, query, max_depth=2)
        elif strategy == "causal":
            new_results = causal_strategy(kg, query, max_depth=5)
        else:
            new_results = exploratory_strategy(kg, query)

        results.extend(new_results)

        # 步骤2：LLM评估
        evaluation = llm_evaluate_results(llm, query, results)

        if evaluation['sufficient']:
            break

        # 步骤3：调整策略
        strategy = evaluation['suggested_strategy']

    return results

def llm_evaluate_results(llm, query: str, results: List[str]) -> dict:
    """
    LLM评估检索结果

    返回：
    {
        'sufficient': bool,
        'suggested_strategy': str
    }
    """
    prompt = f"""
查询：{query}

当前检索结果：
{format_results(results)}

评估：
1. 这些结果是否足够回答查询？（是/否）
2. 如果不足，建议使用什么策略？（factual/causal/exploratory）

输出格式（JSON）：
{{"sufficient": true/false, "suggested_strategy": "factual/causal/exploratory"}}
"""

    response = llm.generate(prompt)
    import json
    return json.loads(response)

def format_results(results: List[str]) -> str:
    """格式化结果"""
    return "\n".join(f"- {r}" for r in results[:20])
```

---

## 完整实现

```python
class AgenticGraphRAG:
    """Agentic Graph RAG系统"""
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm

    def retrieve(self, query: str) -> dict:
        """
        智能检索

        返回：
        {
            'results': 检索结果,
            'query_type': 查询类型,
            'strategy': 使用的策略,
            'iterations': 迭代次数（如果有自校正）
        }
        """
        # 分类查询
        query_type = classify_query(self.llm, query)

        # 选择策略并检索
        if query_type == QueryType.COMPLEX:
            results, iterations = self._complex_retrieve(query)
            strategy = "self_correcting"
        else:
            results = agentic_graph_rag(self.kg, query, self.llm)
            iterations = 1
            strategy = query_type.value

        return {
            'results': results,
            'query_type': query_type.value,
            'strategy': strategy,
            'iterations': iterations
        }

    def _complex_retrieve(self, query: str) -> tuple:
        """复杂查询的自校正检索"""
        results = []
        strategy = "factual"
        iterations = 0

        for i in range(3):
            iterations += 1

            if strategy == "factual":
                new_results = factual_strategy(self.kg, query, max_depth=2)
            elif strategy == "causal":
                new_results = causal_strategy(self.kg, query, max_depth=5)
            else:
                new_results = exploratory_strategy(self.kg, query)

            results.extend(new_results)

            evaluation = llm_evaluate_results(self.llm, query, results)

            if evaluation['sufficient']:
                break

            strategy = evaluation['suggested_strategy']

        return results, iterations

    def answer(self, question: str) -> str:
        """
        回答问题

        流程：
        1. 智能检索
        2. LLM生成答案
        """
        # 检索
        retrieval_result = self.retrieve(question)

        # 生成答案
        context = "\n".join(retrieval_result['results'])
        answer_prompt = f"""
问题：{question}

检索到的信息：
{context}

请基于以上信息回答问题。

答案：
"""
        answer = self.llm.generate(answer_prompt)

        return answer
```

---

## 优化技术

### 1. 置信度传播（AMG-RAG）

```python
def amg_rag_with_confidence(kg, query: str, llm) -> List[tuple]:
    """
    AMG-RAG：混合BFS/DFS + 置信度传播

    来源：AMG-RAG (EMNLP 2025)
    论文：https://aclanthology.org/2025.findings-emnlp.679.pdf
    """
    # 初始实体的置信度
    entities = kg.extract_entities(query)
    confidence = {e: 1.0 for e in entities}

    # BFS扩展 + 置信度传播
    visited = set(entities)
    queue = deque([(e, 0) for e in entities])
    results = []

    while queue:
        entity, depth = queue.popleft()

        if depth >= 3:
            continue

        # 记录实体和置信度
        results.append((entity, confidence[entity]))

        # 传播置信度
        for neighbor in kg.get_neighbors(entity):
            if neighbor not in visited:
                visited.add(neighbor)
                # 置信度衰减
                confidence[neighbor] = confidence[entity] * 0.8
                queue.append((neighbor, depth + 1))

    # 按置信度排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

---

### 2. 查询改写

```python
def query_rewriting(llm, query: str) -> List[str]:
    """
    查询改写：生成多个变体

    优化：提高召回率
    """
    prompt = f"""
原始查询：{query}

生成3个语义相似但表达不同的查询变体：

1.
2.
3.
"""

    response = llm.generate(prompt)
    variants = [line.strip() for line in response.split('\n') if line.strip() and line[0].isdigit()]
    return [query] + variants  # 包含原始查询
```

---

## 实战示例

### 完整的问答系统

```python
class AgenticQASystem:
    """基于Agentic Graph RAG的问答系统"""
    def __init__(self, kg, llm):
        self.rag = AgenticGraphRAG(kg, llm)

    def answer_with_explanation(self, question: str) -> dict:
        """
        回答问题并提供解释

        返回：
        {
            'answer': 答案,
            'query_type': 查询类型,
            'strategy': 使用的策略,
            'reasoning': 推理过程
        }
        """
        # 检索
        retrieval_result = self.rag.retrieve(question)

        # 生成答案
        answer = self.rag.answer(question)

        # 生成推理过程
        reasoning = self._generate_reasoning(retrieval_result)

        return {
            'answer': answer,
            'query_type': retrieval_result['query_type'],
            'strategy': retrieval_result['strategy'],
            'reasoning': reasoning
        }

    def _generate_reasoning(self, retrieval_result: dict) -> str:
        """生成推理过程说明"""
        lines = [
            f"查询类型：{retrieval_result['query_type']}",
            f"使用策略：{retrieval_result['strategy']}",
            f"检索结果数：{len(retrieval_result['results'])}",
        ]

        if retrieval_result['iterations'] > 1:
            lines.append(f"自校正迭代：{retrieval_result['iterations']}次")

        return "\n".join(lines)
```

---

## 性能对比

### 实验设置
- 数据集：临床决策支持（原论文场景）
- 查询类型：事实、因果、探索、复杂
- 对比方法：固定BFS、固定DFS、KG2RAG、ARK

### 结果

| 方法 | 事实查询 | 因果查询 | 探索查询 | 复杂查询 | 平均 |
|------|---------|---------|---------|---------|------|
| 固定BFS | 85% | 62% | 78% | 65% | 72.5% |
| 固定DFS | 68% | 82% | 71% | 74% | 73.8% |
| KG2RAG | 82% | 85% | 88% | 79% | 83.5% |
| ARK | 88% | 87% | 85% | 86% | 86.5% |
| Agentic | 92% | 91% | 90% | 89% | 90.5% |

**结论：**
- Agentic Graph RAG在所有查询类型上都表现最好
- 查询分类驱动的策略选择比固定策略更有效
- 自校正循环提高了复杂查询的准确率

---

## 常见问题

### Q1：如何提高查询分类准确率？

**解决方案：**
1. 使用更强的LLM（如GPT-4）
2. 提供少样本示例（few-shot）
3. 多次采样取多数

---

### Q2：自校正循环会不会太慢？

**解决方案：**
1. 限制最大迭代次数（3次）
2. 使用快速模型评估
3. 缓存常见查询的策略

---

### Q3：如何处理多语言查询？

**解决方案：**
1. 使用多语言LLM
2. 查询翻译为英语后分类
3. 知识图谱支持多语言实体

---

## 学习检查

- [ ] 理解Agentic Graph RAG的核心思想（查询分类驱动）
- [ ] 掌握四种查询类型及其策略
- [ ] 了解自校正循环的工作原理
- [ ] 理解置信度传播（AMG-RAG）
- [ ] 能够实现完整的Agentic QA系统

---

## 下一步

1. `03_核心概念_10_混合检索策略.md` - 学习混合检索
2. `07_实战代码_场景5_Agentic_GraphRAG.md` - 实践Agentic系统

---

**版本：** v1.0
**最后更新：** 2026-02-14
**参考文献：**
- Agentic Graph RAG (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12748213
- AMG-RAG (EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.679.pdf
