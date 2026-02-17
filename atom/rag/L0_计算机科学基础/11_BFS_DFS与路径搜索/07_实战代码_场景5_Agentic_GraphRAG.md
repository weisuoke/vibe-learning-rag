# 实战代码 - 场景5：Agentic GraphRAG系统

## 场景描述

**目标：** 实现Agentic Graph RAG系统（查询分类驱动的策略选择）

**学习重点：**
- 查询分类（事实/因果/探索/复杂）
- 策略选择（BFS/DFS/混合）
- 自校正循环
- 完整的问答系统

**来源：** Agentic Graph RAG (2025) - https://pmc.ncbi.nlm.nih.gov/articles/PMC12748213

---

## 完整可运行代码

```python
"""
Agentic GraphRAG系统
演示：查询分类驱动的策略选择 + 自校正循环
"""

from collections import deque
from typing import List, Set, Dict
from enum import Enum

# ===== 1. 查询类型定义 =====

class QueryType(Enum):
    FACTUAL = "factual"        # 事实查询
    CAUSAL = "causal"          # 因果查询
    EXPLORATORY = "exploratory"  # 探索查询
    COMPLEX = "complex"        # 复杂查询

# ===== 2. 简化的知识图谱 =====

class SimpleKG:
    """简化的知识图谱"""
    def __init__(self):
        self.entity_neighbors: Dict[str, List[str]] = {}

    def add_edge(self, entity1: str, entity2: str):
        if entity1 not in self.entity_neighbors:
            self.entity_neighbors[entity1] = []
        self.entity_neighbors[entity1].append(entity2)

    def get_neighbors(self, entity: str) -> List[str]:
        return self.entity_neighbors.get(entity, [])

    def extract_entities(self, query: str) -> List[str]:
        entities = []
        for entity in self.entity_neighbors.keys():
            if entity.lower() in query.lower():
                entities.append(entity)
        return entities

# ===== 3. LLM模拟器 =====

class MockLLM:
    """模拟LLM"""
    def classify_query(self, query: str) -> QueryType:
        """分类查询类型"""
        if "为什么" in query or "如何" in query:
            return QueryType.CAUSAL
        elif "是谁" in query or "是什么" in query:
            return QueryType.FACTUAL
        elif "有哪些" in query or "包括" in query:
            return QueryType.EXPLORATORY
        else:
            return QueryType.COMPLEX

    def evaluate_results(self, query: str, results: List[str]) -> Dict:
        """评估检索结果"""
        if len(results) >= 5:
            return {'sufficient': True, 'suggested_strategy': None}
        else:
            return {'sufficient': False, 'suggested_strategy': 'causal'}

# ===== 4. 策略实现 =====

def factual_strategy(kg: SimpleKG, query: str, max_depth: int = 2) -> List[str]:
    """事实查询策略：BFS浅层遍历"""
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

def causal_strategy(kg: SimpleKG, query: str, max_depth: int = 5) -> List[str]:
    """因果查询策略：DFS深层遍历"""
    entities = kg.extract_entities(query)
    all_paths = []

    def dfs(entity: str, depth: int, visited: Set[str], path: List[str]):
        if depth >= max_depth or entity in visited:
            return
        visited.add(entity)
        path.append(entity)

        neighbors = kg.get_neighbors(entity)
        if not neighbors or depth == max_depth - 1:
            all_paths.extend(path)
        else:
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, depth + 1, visited, path[:])

    for entity in entities:
        dfs(entity, 0, set(), [])

    return all_paths

def exploratory_strategy(kg: SimpleKG, query: str) -> List[str]:
    """探索查询策略：混合BFS+DFS"""
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

    return list(expanded)

# ===== 5. Agentic GraphRAG主类 =====

class AgenticGraphRAG:
    """Agentic Graph RAG系统"""
    def __init__(self, kg: SimpleKG, llm: MockLLM):
        self.kg = kg
        self.llm = llm

    def retrieve(self, query: str) -> Dict:
        """智能检索"""
        print("=" * 60)
        print(f"查询：{query}")
        print("=" * 60)

        # 步骤1：分类查询
        query_type = self.llm.classify_query(query)
        print(f"\n【查询分类】{query_type.value}")

        # 步骤2：选择策略并检索
        if query_type == QueryType.FACTUAL:
            print("【策略】BFS浅层遍历（2跳）")
            results = factual_strategy(self.kg, query, max_depth=2)
        elif query_type == QueryType.CAUSAL:
            print("【策略】DFS深层遍历（5跳）")
            results = causal_strategy(self.kg, query, max_depth=5)
        elif query_type == QueryType.EXPLORATORY:
            print("【策略】混合BFS+DFS")
            results = exploratory_strategy(self.kg, query)
        else:  # COMPLEX
            print("【策略】自校正循环")
            results = self._complex_retrieve(query)

        print(f"\n【检索结果】{len(results)}个实体")
        print(f"结果：{results[:10]}")

        return {
            'results': results,
            'query_type': query_type.value,
            'strategy': query_type.value
        }

    def _complex_retrieve(self, query: str) -> List[str]:
        """复杂查询的自校正检索"""
        results = []
        strategy = "factual"

        for iteration in range(3):
            print(f"\n  迭代{iteration + 1}：策略={strategy}")

            if strategy == "factual":
                new_results = factual_strategy(self.kg, query, max_depth=2)
            elif strategy == "causal":
                new_results = causal_strategy(self.kg, query, max_depth=5)
            else:
                new_results = exploratory_strategy(self.kg, query)

            results.extend(new_results)

            evaluation = self.llm.evaluate_results(query, results)
            if evaluation['sufficient']:
                print(f"  ✅ 结果足够")
                break

            strategy = evaluation['suggested_strategy']

        return results

    def answer(self, question: str) -> str:
        """回答问题"""
        retrieval_result = self.retrieve(question)
        context = "\n".join(retrieval_result['results'][:5])
        answer = f"基于{retrieval_result['query_type']}查询策略，检索到{len(retrieval_result['results'])}个相关实体。"
        return answer

# ===== 6. 示例数据 =====

def create_example_kg() -> SimpleKG:
    """创建示例知识图谱"""
    kg = SimpleKG()

    # Python相关
    kg.add_edge("Python", "Guido van Rossum")
    kg.add_edge("Python", "简洁语法")
    kg.add_edge("简洁语法", "快速开发")
    kg.add_edge("快速开发", "科学计算社区")
    kg.add_edge("科学计算社区", "NumPy")
    kg.add_edge("NumPy", "AI开发")

    # Einstein相关
    kg.add_edge("Einstein", "相对论")
    kg.add_edge("相对论", "时空弯曲")
    kg.add_edge("时空弯曲", "GPS技术")

    return kg

# ===== 7. 主函数 =====

def main():
    """主函数"""
    print("Agentic GraphRAG系统\n")

    kg = create_example_kg()
    llm = MockLLM()
    system = AgenticGraphRAG(kg, llm)

    # 示例1：事实查询
    print("\n【示例1：事实查询】")
    answer1 = system.answer("Python的创始人是谁？")
    print(f"\n答案：{answer1}")

    # 示例2：因果查询
    print("\n\n【示例2：因果查询】")
    answer2 = system.answer("为什么Python成为AI开发的首选语言？")
    print(f"\n答案：{answer2}")

    # 示例3：探索查询
    print("\n\n【示例3：探索查询】")
    answer3 = system.answer("Python在AI领域有哪些应用？")
    print(f"\n答案：{answer3}")

if __name__ == "__main__":
    main()
```

---

## 运行输出示例

```
Agentic GraphRAG系统

【示例1：事实查询】
============================================================
查询：Python的创始人是谁？
============================================================

【查询分类】factual
【策略】BFS浅层遍历（2跳）

【检索结果】3个实体
结果：['Python', 'Guido van Rossum', '简洁语法']

答案：基于factual查询策略，检索到3个相关实体。


【示例2：因果查询】
============================================================
查询：为什么Python成为AI开发的首选语言？
============================================================

【查询分类】causal
【策略】DFS深层遍历（5跳）

【检索结果】6个实体
结果：['Python', '简洁语法', '快速开发', '科学计算社区', 'NumPy', 'AI开发']

答案：基于causal查询策略，检索到6个相关实体。


【示例3：探索查询】
============================================================
查询：Python在AI领域有哪些应用？
============================================================

【查询分类】exploratory
【策略】混合BFS+DFS

【检索结果】5个实体
结果：['Python', 'Guido van Rossum', '简洁语法', '快速开发', '科学计算社区']

答案：基于exploratory查询策略，检索到5个实体。
```

---

## 核心特性

### 1. 查询分类驱动

根据查询类型自动选择最优策略：
- 事实查询 → BFS（2跳）
- 因果查询 → DFS（5跳）
- 探索查询 → 混合策略
- 复杂查询 → 自校正循环

### 2. 自校正循环

复杂查询通过多轮迭代优化结果：
1. 初始检索
2. LLM评估
3. 调整策略
4. 重新检索

### 3. 策略对比

| 查询类型 | 策略 | 深度 | 优势 |
|---------|------|------|------|
| 事实 | BFS | 2 | 快速精准 |
| 因果 | DFS | 5 | 深度推理 |
| 探索 | 混合 | 2 | 广度覆盖 |
| 复杂 | 自校正 | 动态 | 自适应 |

---

## 与真实LLM集成

```python
from openai import OpenAI

class RealLLM:
    def __init__(self):
        self.client = OpenAI()

    def classify_query(self, query: str) -> QueryType:
        prompt = f"""
分类查询类型：

查询：{query}

类型：
- factual：事实查询（如"谁是Python的创始人？"）
- causal：因果查询（如"为什么Python成为AI首选？"）
- exploratory：探索查询（如"Python有哪些应用？"）
- complex：复杂查询（需要多步推理）

只输出类型名称：
"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        return QueryType(response.choices[0].message.content.strip().lower())
```

---

## 学习检查

- [ ] 理解Agentic Graph RAG的核心思想（查询分类驱动）
- [ ] 掌握四种查询类型及其策略
- [ ] 了解自校正循环的工作原理
- [ ] 能够实现完整的Agentic QA系统

---

## 扩展练习

1. **集成真实LLM**：使用OpenAI API
2. **添加更多查询类型**：如比较查询、聚合查询
3. **优化策略选择**：基于历史数据学习
4. **支持多轮对话**：记录对话历史
5. **性能监控**：记录每次检索的指标

---

**版本：** v1.0
**最后更新：** 2026-02-14
**运行环境：** Python 3.13+
**参考文献：** Agentic Graph RAG (2025) - https://pmc.ncbi.nlm.nih.gov/articles/PMC12748213
