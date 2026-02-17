# 核心概念 4: Tree of Thoughts (ToT)

## 一句话定义

**通过探索多条推理分支并评估每条路径的质量,像搜索树一样系统化地寻找最优解,适合需要战略规划和回溯的复杂任务。**

**RAG应用:** 在RAG系统中,ToT用于探索多种检索策略(语义检索、关键词检索、混合检索等),评估每种策略的效果,选择最优方案。

---

## 为什么重要?

### 问题场景

```python
# 场景:复杂的多步推理任务
from openai import OpenAI

client = OpenAI()

# ❌ 线性推理(CoT):一条路走到黑
prompt = """
问题:用4个4和基本运算符(+,-,*,/)得到24

让我们一步步思考:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# 可能输出:
# 4 + 4 + 4 + 4 = 16 (错误,无法得到24)
# 问题:如果第一步错了,整个推理失败
```

### 解决方案

```python
# ✅ Tree of Thoughts:探索多条路径
prompt = """
问题:用4个4和基本运算符得到24

让我们探索多种可能:

方案1: 先加后乘
- 4 + 4 = 8
- 8 + 4 = 12
- 12 * 4 = 48 (不对)

方案2: 先乘后加
- 4 * 4 = 16
- 16 + 4 = 20
- 20 + 4 = 24 ✓ (正确!)

方案3: 混合运算
- 4 * 4 = 16
- 4 + 4 = 8
- 16 + 8 = 24 ✓ (也正确!)

最优解:4 * 4 + 4 + 4 = 24
"""
```

**性能提升:**

| 任务类型 | CoT | ToT | 提升 |
|---------|-----|-----|------|
| 创意写作 | 12% | 56% | +367% |
| 24点游戏 | 4% | 74% | +1750% |
| 迷你填字游戏 | 16% | 78% | +388% |

**来源:** [Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)

---

## 核心原理

### 原理1:搜索树结构

**定义:** 将推理过程建模为树形搜索空间,每个节点是一个中间思考状态。

**树结构:**

```
                    根问题
                      |
        +-------------+-------------+
        |             |             |
      思路A         思路B         思路C
        |             |             |
    +---+---+     +---+---+     +---+---+
    |       |     |       |     |       |
  A1      A2    B1      B2    C1      C2
    |       |     |       |     |       |
  结果1   结果2  结果3   结果4  结果5   结果6
```

**与CoT的区别:**

```
CoT (线性):
问题 → 步骤1 → 步骤2 → 步骤3 → 答案
       (如果步骤1错了,全盘皆输)

ToT (树形):
问题 → 思路1 → 结果1
    → 思路2 → 结果2 ✓ (最优)
    → 思路3 → 结果3
       (可以比较选择最优)
```

**来源:** [Tree of Thoughts Paper (2023)](https://arxiv.org/abs/2305.10601)

---

### 原理2:广度优先vs深度优先

**两种搜索策略:**

```python
# 广度优先(BFS):先探索所有第一层,再探索第二层
def bfs_tot(problem, depth=3, breadth=3):
    """
    depth: 搜索深度
    breadth: 每层探索的分支数
    """
    current_level = [problem]
    
    for d in range(depth):
        next_level = []
        for node in current_level:
            # 为每个节点生成breadth个子思路
            thoughts = generate_thoughts(node, n=breadth)
            # 评估并保留最好的
            scored = [(evaluate(t), t) for t in thoughts]
            best = sorted(scored, reverse=True)[:breadth]
            next_level.extend([t for _, t in best])
        current_level = next_level
    
    return max(current_level, key=evaluate)

# 深度优先(DFS):先探索一条路径到底,再回溯
def dfs_tot(problem, max_depth=5):
    """递归深度优先搜索"""
    def dfs(node, depth):
        if depth == max_depth or is_solution(node):
            return node
        
        thoughts = generate_thoughts(node, n=3)
        for thought in thoughts:
            result = dfs(thought, depth + 1)
            if is_solution(result):
                return result
        
        return None
    
    return dfs(problem, 0)
```

**选择策略:**

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 解空间大 | BFS | 避免陷入局部最优 |
| 解空间深 | DFS | 节省内存 |
| 需要多个解 | BFS | 可以并行探索 |
| 需要快速解 | DFS | 找到一个就停止 |

---

### 原理3:评估函数

**定义:** 评估每个思路节点的质量,决定是否继续探索。

**评估方法:**

```python
# 方法1:LLM评分
def llm_evaluate(thought, problem):
    """使用LLM评估思路质量"""
    prompt = f"""
    问题:{problem}
    当前思路:{thought}
    
    请评估这个思路的质量(1-10分):
    - 是否接近解决方案?
    - 是否有逻辑错误?
    - 是否值得继续探索?
    
    评分:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    score = extract_score(response.choices[0].message.content)
    return score

# 方法2:启发式评估
def heuristic_evaluate(thought, problem):
    """基于规则的评估"""
    score = 0
    
    # 规则1:包含关键词
    if any(keyword in thought for keyword in problem_keywords):
        score += 3
    
    # 规则2:逻辑连贯
    if is_coherent(thought):
        score += 3
    
    # 规则3:接近目标
    if is_close_to_goal(thought, problem):
        score += 4
    
    return score

# 方法3:投票评估
def vote_evaluate(thought, problem, n=3):
    """多次LLM调用投票"""
    scores = []
    for _ in range(n):
        score = llm_evaluate(thought, problem)
        scores.append(score)
    return sum(scores) / len(scores)
```

---

## 手写实现

### 从零实现 Tree of Thoughts

```python
"""
Tree of Thoughts Implementation
功能:树形搜索推理
"""

from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import heapq

@dataclass
class ThoughtNode:
    """思路节点"""
    content: str
    score: float
    depth: int
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __lt__(self, other):
        return self.score > other.score  # 分数高的优先

class TreeOfThoughts:
    """Tree of Thoughts实现"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_thoughts(
        self,
        problem: str,
        current_thought: str,
        n: int = 3,
        model: str = "gpt-4o-mini"
    ) -> List[str]:
        """生成n个子思路"""
        prompt = f"""
问题:{problem}

当前思路:{current_thought}

请生成{n}个不同的下一步思路:
"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=0.8
        )
        
        thoughts = [choice.message.content for choice in response.choices]
        return thoughts
    
    def evaluate_thought(
        self,
        thought: str,
        problem: str,
        model: str = "gpt-4o-mini"
    ) -> float:
        """评估思路质量(0-10分)"""
        prompt = f"""
问题:{problem}
思路:{thought}

请评估这个思路的质量(0-10分):
- 是否接近解决方案?
- 是否有逻辑错误?
- 是否值得继续探索?

只返回数字分数:
"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0), 10)  # 限制在0-10
        except:
            return 5.0  # 默认中等分数
    
    def bfs_search(
        self,
        problem: str,
        max_depth: int = 3,
        breadth: int = 3,
        model: str = "gpt-4o-mini"
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """广度优先搜索"""
        # 初始化根节点
        root = ThoughtNode(
            content=problem,
            score=5.0,
            depth=0
        )
        
        current_level = [root]
        all_nodes = [root]
        
        for depth in range(max_depth):
            next_level = []
            
            for node in current_level:
                # 生成子思路
                thoughts = self.generate_thoughts(
                    problem, node.content, n=breadth, model=model
                )
                
                # 评估并创建子节点
                for thought in thoughts:
                    score = self.evaluate_thought(thought, problem, model)
                    child = ThoughtNode(
                        content=thought,
                        score=score,
                        depth=depth + 1,
                        parent=node
                    )
                    node.children.append(child)
                    next_level.append(child)
                    all_nodes.append(child)
            
            # 保留最好的breadth个节点进入下一层
            next_level.sort(key=lambda x: x.score, reverse=True)
            current_level = next_level[:breadth]
        
        # 返回最优节点和所有节点
        best_node = max(all_nodes, key=lambda x: x.score)
        return best_node, all_nodes
    
    def dfs_search(
        self,
        problem: str,
        max_depth: int = 5,
        breadth: int = 3,
        threshold: float = 8.0,
        model: str = "gpt-4o-mini"
    ) -> Optional[ThoughtNode]:
        """深度优先搜索"""
        def dfs(node: ThoughtNode, depth: int) -> Optional[ThoughtNode]:
            # 达到最大深度或找到高分解
            if depth >= max_depth or node.score >= threshold:
                return node
            
            # 生成子思路
            thoughts = self.generate_thoughts(
                problem, node.content, n=breadth, model=model
            )
            
            # 评估并按分数排序
            children = []
            for thought in thoughts:
                score = self.evaluate_thought(thought, problem, model)
                child = ThoughtNode(
                    content=thought,
                    score=score,
                    depth=depth + 1,
                    parent=node
                )
                children.append(child)
            
            children.sort(key=lambda x: x.score, reverse=True)
            node.children = children
            
            # 深度优先探索最好的分支
            for child in children:
                result = dfs(child, depth + 1)
                if result and result.score >= threshold:
                    return result
            
            # 返回当前最好的子节点
            return children[0] if children else node
        
        root = ThoughtNode(content=problem, score=5.0, depth=0)
        return dfs(root, 0)
    
    def get_path(self, node: ThoughtNode) -> List[str]:
        """获取从根到节点的路径"""
        path = []
        current = node
        while current:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))
    
    def solve(
        self,
        problem: str,
        strategy: str = "bfs",
        max_depth: int = 3,
        breadth: int = 3,
        model: str = "gpt-4o-mini"
    ) -> Dict:
        """解决问题"""
        if strategy == "bfs":
            best_node, all_nodes = self.bfs_search(
                problem, max_depth, breadth, model
            )
            path = self.get_path(best_node)
            
            return {
                "solution": best_node.content,
                "score": best_node.score,
                "path": path,
                "total_nodes": len(all_nodes),
                "strategy": "BFS"
            }
        
        elif strategy == "dfs":
            best_node = self.dfs_search(
                problem, max_depth, breadth, model=model
            )
            path = self.get_path(best_node)
            
            return {
                "solution": best_node.content,
                "score": best_node.score,
                "path": path,
                "strategy": "DFS"
            }
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    client = OpenAI()
    tot = TreeOfThoughts(client)
    
    # 测试:24点游戏
    problem = "用4个4和基本运算符(+,-,*,/)得到24"
    
    print("=== BFS搜索 ===")
    result_bfs = tot.solve(problem, strategy="bfs", max_depth=2, breadth=3)
    print(f"解决方案:{result_bfs['solution']}")
    print(f"分数:{result_bfs['score']}")
    print(f"路径:{' -> '.join(result_bfs['path'])}")
    print(f"探索节点数:{result_bfs['total_nodes']}")
    print()
    
    print("=== DFS搜索 ===")
    result_dfs = tot.solve(problem, strategy="dfs", max_depth=3, breadth=3)
    print(f"解决方案:{result_dfs['solution']}")
    print(f"分数:{result_dfs['score']}")
    print(f"路径:{' -> '.join(result_dfs['path'])}")
```

---

## RAG 应用场景

### 场景1:多策略检索

**问题:** 不确定哪种检索策略最好

**解决方案:** 使用ToT探索多种策略

```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("docs")

# 添加文档
docs = [
    "Python是解释型语言,由Guido创建于1991年",
    "JavaScript是脚本语言,主要用于Web开发",
    "FastAPI是Python Web框架,基于Starlette"
]
collection.add(documents=docs, ids=[f"doc{i}" for i in range(len(docs))])

def multi_strategy_retrieval(query: str) -> Dict:
    """多策略检索"""
    
    strategies = {
        "直接语义检索": lambda q: collection.query(query_texts=[q], n_results=2),
        "关键词提取": lambda q: collection.query(
            query_texts=[extract_keywords(q)], n_results=2
        ),
        "问题改写": lambda q: collection.query(
            query_texts=[rewrite_query(q)], n_results=2
        )
    }
    
    results = {}
    for name, strategy in strategies.items():
        docs = strategy(query)['documents'][0]
        answer = generate_answer(query, docs)
        score = evaluate_answer(answer, query)
        results[name] = {
            "docs": docs,
            "answer": answer,
            "score": score
        }
    
    # 选择最优策略
    best_strategy = max(results.items(), key=lambda x: x[1]['score'])
    
    return {
        "query": query,
        "best_strategy": best_strategy[0],
        "answer": best_strategy[1]['answer'],
        "all_strategies": results
    }

def extract_keywords(query: str) -> str:
    """提取关键词"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"提取关键词:{query}"}]
    )
    return response.choices[0].message.content

def rewrite_query(query: str) -> str:
    """改写查询"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"改写查询:{query}"}]
    )
    return response.choices[0].message.content

def generate_answer(query: str, docs: List[str]) -> str:
    """生成答案"""
    prompt = f"文档:{' | '.join(docs)}\n问题:{query}\n答案:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def evaluate_answer(answer: str, query: str) -> float:
    """评估答案质量"""
    prompt = f"问题:{query}\n答案:{answer}\n评分(0-10):"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 5.0

# 测试
result = multi_strategy_retrieval("Python是什么时候创建的?")
print(f"最优策略:{result['best_strategy']}")
print(f"答案:{result['answer']}")
```

---

## 最佳实践

### 1. 搜索深度和广度

```python
# 根据任务复杂度选择
simple_task = {"depth": 2, "breadth": 2}  # 4个节点
medium_task = {"depth": 3, "breadth": 3}  # 27个节点
complex_task = {"depth": 4, "breadth": 3}  # 81个节点

# 成本控制
# depth * breadth^depth = 总LLM调用次数
```

### 2. 评估函数设计

```python
# 快速评估:启发式规则
def fast_evaluate(thought):
    return heuristic_score(thought)

# 准确评估:LLM打分
def accurate_evaluate(thought):
    return llm_score(thought)

# 混合评估:先快后准
def hybrid_evaluate(thought):
    fast_score = fast_evaluate(thought)
    if fast_score > 7:  # 只对高分候选用LLM
        return accurate_evaluate(thought)
    return fast_score
```

### 3. 剪枝策略

```python
# 提前终止低分分支
def should_prune(node, threshold=3.0):
    return node.score < threshold

# 限制总节点数
def limit_nodes(nodes, max_nodes=100):
    return nodes[:max_nodes]
```

---

## 参考资源

- [Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)
- [Prompt Engineering Guide - ToT](https://www.promptingguide.ai/techniques/tot)
