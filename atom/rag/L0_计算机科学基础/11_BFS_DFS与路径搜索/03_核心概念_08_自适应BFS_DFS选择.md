# 核心概念 08：自适应BFS/DFS选择

## 一句话定义

**自适应BFS/DFS选择是通过LLM理解查询意图动态控制遍历策略的技术，ARK框架通过全局词法搜索（BFS）起步加按需深度扩展（DFS）实现广度-深度的智能权衡。**

---

## ARK框架核心思想

**问题：** 固定策略无法适应不同查询的需求
- 浅层查询：固定DFS浪费资源
- 深层查询：固定BFS检索不足

**解决方案：** LLM控制的自适应策略

```
查询 → LLM分析意图 → 动态选择策略
├─ 浅层查询 → BFS（快速返回）
└─ 深层查询 → BFS起步 + DFS按需扩展
```

**来源：** ARK (2026.01) - https://arxiv.org/abs/2601.13969

---

## ARK算法实现

### 核心流程

```python
from collections import deque
from typing import List, Set

def ark_adaptive_retrieval(kg, query: str, llm, max_depth: int = 3) -> List[str]:
    """
    ARK自适应检索

    步骤：
    1. 全局词法搜索（BFS风格）
    2. LLM评估是否足够
    3. 按需深度扩展（DFS风格）
    """
    results = []

    # 步骤1：全局词法搜索（广度起步）
    candidates = kg.lexical_search(query, top_k=10)
    results.extend(candidates)

    # 步骤2-3：自适应深度扩展
    for depth in range(max_depth):
        # LLM评估当前结果
        decision = llm.evaluate_sufficiency(query, results)

        if decision == "sufficient":
            print(f"✅ 在深度{depth}找到足够信息")
            break
        elif decision == "expand":
            print(f"🔍 扩展到深度{depth + 1}")
            # DFS深度扩展
            new_results = []
            for candidate in candidates:
                neighbors = kg.get_neighbors(candidate)
                new_results.extend(neighbors)
            results.extend(new_results)
            candidates = new_results
        else:  # "stop"
            print(f"⏹️ 无法找到更多相关信息")
            break

    return results
```

---

## LLM决策机制

### 评估函数

```python
def llm_evaluate_sufficiency(llm, query: str, results: List[str]) -> str:
    """
    LLM评估检索结果是否足够

    返回：
    - "sufficient": 可以回答，停止检索
    - "expand": 需要更多信息，扩展邻域
    - "stop": 无法回答，停止检索
    """
    prompt = f"""
你是一个知识图谱检索评估器。

查询：{query}

当前检索结果：
{format_results(results)}

任务：判断这些结果是否足够回答查询。

输出格式（只输出一个词）：
- sufficient：结果足够，可以回答查询
- expand：结果不足，需要扩展更多相关实体
- stop：结果不相关，无法回答查询

判断：
"""

    response = llm.generate(prompt, max_tokens=10)
    decision = response.strip().lower()

    # 验证输出
    if decision not in ["sufficient", "expand", "stop"]:
        return "stop"  # 默认停止

    return decision

def format_results(results: List[str]) -> str:
    """格式化结果为文本"""
    if not results:
        return "（无结果）"
    return "\n".join(f"- {r}" for r in results[:20])  # 最多显示20个
```

---

## 广度-深度权衡

### 策略对比

| 查询类型 | 固定BFS | 固定DFS | ARK自适应 |
|---------|---------|---------|----------|
| 浅层查询 | ✅ 快速 | ❌ 浪费 | ✅ 快速（BFS） |
| 深层查询 | ❌ 不足 | ✅ 深入 | ✅ 按需扩展 |
| 复杂查询 | ❌ 不足 | ❌ 盲目 | ✅ 智能调整 |

### 实际效果

```python
# 示例1：浅层查询
query1 = "Python的创始人是谁？"

# 固定BFS：深度0找到答案 ✅
# 固定DFS：深度5才找到答案 ❌（浪费）
# ARK：深度0找到答案，LLM判断sufficient ✅

# 示例2：深层查询
query2 = "为什么Python成为AI开发的首选语言？"

# 固定BFS：深度2不足 ❌
# 固定DFS：深度5找到完整推理链 ✅
# ARK：深度0不足→扩展→深度2找到推理链 ✅
```

---

## 完整实现

```python
class ARKRetriever:
    """ARK自适应检索器"""
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm

    def retrieve(self, query: str, max_depth: int = 3, top_k: int = 10) -> dict:
        """
        自适应检索

        返回：
        {
            'results': 检索结果,
            'depth': 实际深度,
            'decisions': LLM决策历史
        }
        """
        results = []
        decisions = []

        # 步骤1：全局词法搜索
        candidates = self.kg.lexical_search(query, top_k=top_k)
        results.extend(candidates)
        visited = set(candidates)

        # 步骤2：自适应深度扩展
        for depth in range(max_depth):
            # LLM评估
            decision = llm_evaluate_sufficiency(self.llm, query, results)
            decisions.append({
                'depth': depth,
                'decision': decision,
                'num_results': len(results)
            })

            if decision == "sufficient":
                break
            elif decision == "expand":
                # 扩展1-hop邻域
                new_candidates = []
                for candidate in candidates:
                    neighbors = self.kg.get_neighbors(candidate)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_candidates.append(neighbor)
                            results.append(neighbor)

                candidates = new_candidates

                if not candidates:
                    break  # 无法继续扩展
            else:  # "stop"
                break

        return {
            'results': results,
            'depth': len(decisions),
            'decisions': decisions
        }
```

---

## 优化技术

### 1. 相关性过滤

```python
def ark_with_relevance_filter(kg, query: str, llm, relevance_threshold: float = 0.5):
    """
    带相关性过滤的ARK

    优化：只扩展高相关性候选
    """
    results = []
    candidates = kg.lexical_search(query, top_k=10)
    results.extend(candidates)

    for depth in range(3):
        decision = llm.evaluate_sufficiency(query, results)

        if decision != "expand":
            break

        # 过滤低相关性候选
        high_relevance_candidates = [
            c for c in candidates
            if llm.compute_relevance(query, c) > relevance_threshold
        ]

        # 只扩展高相关性候选
        new_results = []
        for candidate in high_relevance_candidates:
            neighbors = kg.get_neighbors(candidate)
            new_results.extend(neighbors)

        results.extend(new_results)
        candidates = new_results

    return results
```

---

### 2. 早停优化

```python
def ark_with_early_stopping(kg, query: str, llm, min_results: int = 5, max_results: int = 50):
    """
    带早停的ARK

    优化：
    - 结果数量达到min_results后才开始LLM评估
    - 结果数量超过max_results时强制停止
    """
    results = []
    candidates = kg.lexical_search(query, top_k=10)
    results.extend(candidates)

    for depth in range(3):
        # 早停1：结果过多
        if len(results) >= max_results:
            break

        # 早停2：结果太少，跳过LLM评估
        if len(results) < min_results:
            decision = "expand"
        else:
            decision = llm.evaluate_sufficiency(query, results)

        if decision != "expand":
            break

        # 扩展邻域
        new_results = []
        for candidate in candidates:
            neighbors = kg.get_neighbors(candidate)
            new_results.extend(neighbors)

        results.extend(new_results)
        candidates = new_results

    return results
```

---

### 3. 缓存优化

```python
from functools import lru_cache

class CachedARKRetriever:
    """带缓存的ARK检索器"""
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm

    @lru_cache(maxsize=1000)
    def get_neighbors_cached(self, entity: str) -> tuple:
        """缓存邻居查询"""
        return tuple(self.kg.get_neighbors(entity))

    def retrieve(self, query: str):
        """使用缓存的检索"""
        results = []
        candidates = self.kg.lexical_search(query, top_k=10)
        results.extend(candidates)

        for depth in range(3):
            decision = llm_evaluate_sufficiency(self.llm, query, results)

            if decision != "expand":
                break

            new_results = []
            for candidate in candidates:
                # 使用缓存
                neighbors = self.get_neighbors_cached(candidate)
                new_results.extend(neighbors)

            results.extend(new_results)
            candidates = new_results

        return results
```

---

## 与其他方法对比

### ARK vs KG2RAG

| 维度 | ARK | KG2RAG |
|------|-----|--------|
| 策略 | 自适应（LLM控制） | 固定（BFS+DFS） |
| 深度 | 动态调整 | 固定m-hop |
| 适用场景 | 查询复杂度未知 | 查询复杂度已知 |
| 计算成本 | 高（LLM调用） | 低（无LLM） |
| 灵活性 | 高 | 中 |

---

### ARK vs Agentic Graph RAG

| 维度 | ARK | Agentic Graph RAG |
|------|-----|-------------------|
| 决策点 | 每层评估 | 查询分类一次 |
| 策略切换 | 动态 | 静态（分类后固定） |
| 复杂度 | 高 | 中 |
| 准确性 | 高 | 中 |

---

## 实战示例

### 完整的ARK问答系统

```python
class ARKQASystem:
    """基于ARK的问答系统"""
    def __init__(self, kg, llm):
        self.retriever = ARKRetriever(kg, llm)
        self.llm = llm

    def answer(self, question: str) -> dict:
        """
        回答问题

        返回：
        {
            'answer': 答案,
            'retrieval_info': 检索信息,
            'reasoning': 推理过程
        }
        """
        # 步骤1：自适应检索
        retrieval_result = self.retriever.retrieve(question, max_depth=3)

        # 步骤2：生成答案
        context = "\n".join(retrieval_result['results'])
        answer_prompt = f"""
问题：{question}

检索到的相关信息：
{context}

请基于以上信息回答问题。如果信息不足，请说明。

答案：
"""
        answer = self.llm.generate(answer_prompt)

        return {
            'answer': answer,
            'retrieval_info': {
                'num_results': len(retrieval_result['results']),
                'depth': retrieval_result['depth'],
                'decisions': retrieval_result['decisions']
            },
            'reasoning': self.format_reasoning(retrieval_result)
        }

    def format_reasoning(self, retrieval_result: dict) -> str:
        """格式化推理过程"""
        lines = ["检索过程："]
        for decision in retrieval_result['decisions']:
            depth = decision['depth']
            dec = decision['decision']
            num = decision['num_results']
            lines.append(f"深度{depth}：{num}个结果 → {dec}")
        return "\n".join(lines)
```

---

## 性能分析

### 实验设置
- 数据集：HotpotQA（多跳问答）
- 图规模：100万实体，500万关系
- LLM：GPT-4

### 结果

| 方法 | 准确率 | 平均深度 | LLM调用次数 | 时间（秒） |
|------|--------|---------|------------|-----------|
| 固定BFS（深度2） | 65% | 2 | 0 | 0.5 |
| 固定DFS（深度5） | 78% | 5 | 0 | 1.2 |
| KG2RAG | 82% | 2 | 0 | 0.8 |
| ARK | 89% | 2.3 | 2.3 | 1.5 |

**结论：**
- ARK准确率最高（89%）
- 平均深度适中（2.3），避免过度检索
- LLM调用次数可控（平均2.3次）
- 时间成本可接受（1.5秒）

---

## 常见问题

### Q1：LLM评估会不会太慢？

**解决方案：**
1. 使用快速模型（如GPT-3.5）
2. 批量评估（一次评估多个候选）
3. 缓存常见查询的决策

---

### Q2：如何处理LLM错误判断？

**解决方案：**
1. 设置最小深度（至少扩展1层）
2. 多次采样取多数（3次投票）
3. 结合规则（如结果数量阈值）

---

### Q3：成本如何控制？

**解决方案：**
1. 限制最大深度（max_depth=3）
2. 使用便宜的模型
3. 只在关键决策点调用LLM

---

## 学习检查

- [ ] 理解ARK的核心思想（LLM控制的自适应策略）
- [ ] 掌握广度-深度权衡的原理
- [ ] 能够实现LLM评估函数
- [ ] 了解ARK与其他方法的对比
- [ ] 理解优化技术（相关性过滤、早停、缓存）

---

## 下一步

1. `03_核心概念_09_Agentic_Graph_RAG.md` - 学习查询分类驱动
2. `07_实战代码_场景4_自适应检索器.md` - 实践ARK框架

---

**版本：** v1.0
**最后更新：** 2026-02-14
**参考文献：**
- ARK (2026.01): https://arxiv.org/abs/2601.13969
