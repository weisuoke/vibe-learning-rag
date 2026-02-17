# 实战代码7：Context Pruning

> 上下文修剪技术实现

---

## 代码示例

```python
"""
Context Pruning实战
演示：相关性过滤、多样性去重、Token预算控制
"""

from openai import OpenAI
import os
from typing import List, Dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 相关性过滤 =====

def relevance_based_pruning(docs: List[Dict], min_score: float = 0.7) -> List[Dict]:
    """
    基于相关性的修剪
    """
    return [doc for doc in docs if doc.get('score', 0) >= min_score]

def example_relevance_pruning():
    """相关性过滤示例"""
    print("=== 相关性过滤 ===\n")
    
    docs = [
        {"content": "Python是解释型语言", "score": 0.92},
        {"content": "Python支持OOP", "score": 0.88},
        {"content": "Java是编译型语言", "score": 0.45},
        {"content": "C++性能很高", "score": 0.38}
    ]
    
    print(f"原始文档数：{len(docs)}")
    
    pruned = relevance_based_pruning(docs, min_score=0.7)
    
    print(f"过滤后文档数：{len(pruned)}")
    print("\n保留的文档：")
    for doc in pruned:
        print(f"  - {doc['content']} (score: {doc['score']})")

# ===== 2. 多样性去重 =====

def diversity_based_pruning(docs: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    基于多样性的修剪
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode([doc['content'] for doc in docs])
    
    selected = []
    selected_embeddings = []
    
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        is_diverse = True
        for sel_emb in selected_embeddings:
            similarity = np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
            if similarity > similarity_threshold:
                is_diverse = False
                break
        
        if is_diverse:
            selected.append(doc)
            selected_embeddings.append(emb)
    
    return selected

def example_diversity_pruning():
    """多样性去重示例"""
    print("\n=== 多样性去重 ===\n")
    
    docs = [
        {"content": "Python是解释型语言"},
        {"content": "Python是一种解释型编程语言"},
        {"content": "Python支持面向对象编程"}
    ]
    
    print(f"原始文档数：{len(docs)}")
    
    pruned = diversity_based_pruning(docs, similarity_threshold=0.8)
    
    print(f"去重后文档数：{len(pruned)}")
    print("\n保留的文档：")
    for doc in pruned:
        print(f"  - {doc['content']}")

# ===== 3. Token预算控制 =====

def token_budget_pruning(docs: List[Dict], max_tokens: int = 2000) -> List[Dict]:
    """
    基于Token预算的修剪
    """
    pruned = []
    total_tokens = 0
    
    for doc in docs:
        doc_tokens = len(doc['content'].split()) * 1.3
        
        if total_tokens + doc_tokens <= max_tokens:
            pruned.append(doc)
            total_tokens += doc_tokens
        else:
            break
    
    return pruned

def example_token_pruning():
    """Token预算控制示例"""
    print("\n=== Token预算控制 ===\n")
    
    docs = [
        {"content": "Python是一种解释型、面向对象的编程语言..." * 50},
        {"content": "Python支持多种编程范式"},
        {"content": "Python有丰富的标准库"}
    ]
    
    print(f"原始文档数：{len(docs)}")
    
    pruned = token_budget_pruning(docs, max_tokens=2000)
    
    print(f"修剪后文档数：{len(pruned)}")

# ===== 4. 综合修剪 =====

class ComprehensivePruner:
    """综合修剪器"""
    
    def __init__(
        self,
        min_score: float = 0.7,
        similarity_threshold: float = 0.8,
        max_tokens: int = 2000
    ):
        self.min_score = min_score
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
    
    def prune(self, query: str, docs: List[Dict]) -> Dict:
        """多阶段修剪"""
        # 阶段1：相关性过滤
        stage1 = relevance_based_pruning(docs, self.min_score)
        
        # 阶段2：多样性过滤
        stage2 = diversity_based_pruning(stage1, self.similarity_threshold)
        
        # 阶段3：Token预算控制
        stage3 = token_budget_pruning(stage2, self.max_tokens)
        
        return {
            "original_count": len(docs),
            "after_relevance": len(stage1),
            "after_diversity": len(stage2),
            "final_count": len(stage3),
            "pruned_docs": stage3,
            "pruning_rate": 1 - len(stage3) / len(docs) if docs else 0
        }

def example_comprehensive_pruning():
    """综合修剪示例"""
    print("\n=== 综合修剪 ===\n")
    
    pruner = ComprehensivePruner(
        min_score=0.7,
        similarity_threshold=0.8,
        max_tokens=2000
    )
    
    docs = [
        {"content": f"文档{i}内容", "score": 0.9 - i*0.1}
        for i in range(20)
    ]
    
    result = pruner.prune("查询问题", docs)
    
    print(f"原始文档数：{result['original_count']}")
    print(f"相关性过滤后：{result['after_relevance']}")
    print(f"多样性过滤后：{result['after_diversity']}")
    print(f"最终文档数：{result['final_count']}")
    print(f"修剪率：{result['pruning_rate']:.2%}")

# ===== 5. 上下文利用率追踪 =====

def track_context_utilization(context: str, answer: str) -> Dict:
    """追踪上下文利用率"""
    prompt = f"""
分析答案中使用了哪些上下文内容。

上下文：
{context}

答案：
{answer}

返回JSON格式：
{{
  "utilization_rate": 0.0-1.0
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是利用率分析专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    import json
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"utilization_rate": 0.5}

def example_utilization_tracking():
    """利用率追踪示例"""
    print("\n=== 上下文利用率追踪 ===\n")
    
    context = """
Python是解释型语言。
Python支持面向对象编程。
Python有丰富的标准库。
Java是编译型语言。
"""
    
    answer = "Python是解释型语言，支持面向对象编程。"
    
    utilization = track_context_utilization(context, answer)
    
    print(f"利用率：{utilization.get('utilization_rate', 0):.2%}")

# ===== 6. 完整Context Pruning系统 =====

class ProductionContextPruner:
    """生产级Context Pruning系统"""
    
    def __init__(self):
        self.pruner = ComprehensivePruner(
            min_score=0.7,
            similarity_threshold=0.8,
            max_tokens=2000
        )
    
    def process_query(
        self,
        query: str,
        docs: List[Dict]
    ) -> Dict:
        """处理查询（完整流程）"""
        # 1. 修剪上下文
        pruning_result = self.pruner.prune(query, docs)
        
        # 2. 构建Prompt
        context = "\n\n".join([
            doc['content'] for doc in pruning_result['pruned_docs']
        ])
        
        prompt = f"""
参考资料：
{context}

问题：{query}

回答：
"""
        
        # 3. 生成答案
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是知识助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # 4. 追踪利用率
        utilization = track_context_utilization(context, answer)
        
        return {
            "answer": answer,
            "pruning_stats": pruning_result,
            "utilization": utilization
        }

def example_production_system():
    """生产系统示例"""
    print("\n=== 生产级Context Pruning系统 ===\n")
    
    system = ProductionContextPruner()
    
    docs = [
        {"content": "RAG是检索增强生成技术", "score": 0.95},
        {"content": "RAG结合检索和生成", "score": 0.92},
        {"content": "RAG可以减少幻觉", "score": 0.88}
    ]
    
    result = system.process_query("什么是RAG？", docs)
    
    print(f"答案：{result['answer']}")
    print(f"\n修剪统计：")
    print(f"  原始文档数：{result['pruning_stats']['original_count']}")
    print(f"  最终文档数：{result['pruning_stats']['final_count']}")
    print(f"  修剪率：{result['pruning_stats']['pruning_rate']:.2%}")
    print(f"\n利用率：{result['utilization'].get('utilization_rate', 0):.2%}")

# ===== 运行所有示例 =====

if __name__ == "__main__":
    example_relevance_pruning()
    example_diversity_pruning()
    example_token_pruning()
    example_comprehensive_pruning()
    example_utilization_tracking()
    example_production_system()
```

---

## 关键要点

1. **相关性过滤**：只保留score >0.7的文档
2. **多样性去重**：移除相似度>0.8的重复内容
3. **Token预算**：控制在1500-2500 tokens
4. **利用率追踪**：监控上下文使用效率
5. **完整流程**：修剪→生成→追踪一体化

---

**版本：** v1.0
**最后更新：** 2026-02-16
