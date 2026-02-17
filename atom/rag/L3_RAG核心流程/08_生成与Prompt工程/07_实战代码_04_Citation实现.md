# 实战代码4：Citation实现

> Citation-aware RAG with Spatial Metadata实现

---

## 代码示例

```python
"""
Citation实现实战
演示：内联引用锚点、空间元数据、引用验证
"""

from openai import OpenAI
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 1. 基础引用实现 =====

def basic_citation_example():
    """
    基础引用标记示例
    """
    print("=== 基础引用实现 ===\n")
    
    docs = [
        {"id": 1, "content": "Python是一种解释型语言"},
        {"id": 2, "content": "Python支持面向对象编程"}
    ]
    
    # 构建带编号的上下文
    context = "\n\n".join([
        f"[文档{doc['id']}] {doc['content']}"
        for doc in docs
    ])
    
    prompt = f"""
参考资料：
{context}

问题：Python有什么特点？

要求：为关键信息添加引用标记 [文档X]

回答：
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是知识助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    print(f"答案：{answer}")
    
    # 提取引用
    import re
    citations = re.findall(r'\[文档(\d+)\]', answer)
    print(f"\n引用的文档：{set(citations)}")

# ===== 2. 空间元数据Citation =====

@dataclass
class SpatialMetadata:
    """空间元数据"""
    doc_id: str
    page: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str
    confidence: float

@dataclass
class Citation:
    """引用对象"""
    id: str
    metadata: SpatialMetadata

class CitationManager:
    """引用管理器"""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self.next_id = 1
    
    def add_citation(
        self,
        doc_id: str,
        page: int,
        bbox: Tuple[float, float, float, float],
        text: str,
        confidence: float = 1.0
    ) -> str:
        """添加引用"""
        citation_id = f"{self.next_id}.{len(self.citations) + 1}"
        
        metadata = SpatialMetadata(
            doc_id=doc_id,
            page=page,
            bbox=bbox,
            text=text,
            confidence=confidence
        )
        
        citation = Citation(id=citation_id, metadata=metadata)
        self.citations[citation_id] = citation
        
        return citation_id
    
    def get_citation(self, citation_id: str) -> Citation:
        """获取引用"""
        return self.citations.get(citation_id)
    
    def format_citation(self, citation_id: str) -> str:
        """格式化引用信息"""
        citation = self.get_citation(citation_id)
        if not citation:
            return "引用未找到"
        
        meta = citation.metadata
        return f"""
引用 {citation.id}:
- 文档: {meta.doc_id}
- 页码: {meta.page}
- 位置: {meta.bbox}
- 原文: {meta.text}
- 置信度: {meta.confidence:.2f}
"""

def spatial_citation_example():
    """
    空间元数据Citation示例
    """
    print("\n=== 空间元数据Citation ===\n")
    
    manager = CitationManager()
    
    # 添加引用
    cite_id1 = manager.add_citation(
        doc_id="python_intro.pdf",
        page=15,
        bbox=(100, 200, 400, 250),
        text="Python is an interpreted language",
        confidence=0.95
    )
    
    cite_id2 = manager.add_citation(
        doc_id="python_oop.pdf",
        page=23,
        bbox=(150, 300, 450, 350),
        text="Python supports object-oriented programming",
        confidence=0.92
    )
    
    print(f"创建的引用ID：{cite_id1}, {cite_id2}")
    print(manager.format_citation(cite_id1))
    print(manager.format_citation(cite_id2))

# ===== 3. 生成带Citation的答案 =====

def generate_with_citations(
    query: str,
    docs: List[Dict],
    citation_manager: CitationManager
) -> Dict:
    """
    生成带Citation的答案
    """
    # 为每个文档创建引用
    context_with_citations = []
    
    for i, doc in enumerate(docs):
        cite_id = citation_manager.add_citation(
            doc_id=doc.get('doc_id', f'doc{i+1}'),
            page=doc.get('page', 1),
            bbox=doc.get('bbox', (0, 0, 0, 0)),
            text=doc['content'],
            confidence=doc.get('score', 1.0)
        )
        
        context_with_citations.append(
            f"[引用{cite_id}] {doc['content']}"
        )
    
    context = "\n\n".join(context_with_citations)
    
    prompt = f"""
参考资料：
{context}

问题：{query}

要求：为关键信息添加引用标记 <c>X.Y</c>

回答：
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是知识助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    
    # 提取引用ID
    import re
    citation_ids = re.findall(r'<c>([\d.]+)</c>', answer)
    
    citations = [
        citation_manager.get_citation(cid)
        for cid in set(citation_ids)
        if citation_manager.get_citation(cid)
    ]
    
    return {
        "answer": answer,
        "citations": citations,
        "citation_count": len(citations)
    }

def citation_generation_example():
    """
    生成带Citation的答案示例
    """
    print("\n=== 生成带Citation的答案 ===\n")
    
    manager = CitationManager()
    
    docs = [
        {
            "content": "RAG是检索增强生成技术",
            "doc_id": "rag_intro.pdf",
            "page": 1,
            "bbox": (100, 100, 500, 150),
            "score": 0.95
        },
        {
            "content": "RAG结合检索和生成两个步骤",
            "doc_id": "rag_arch.pdf",
            "page": 3,
            "bbox": (100, 200, 500, 250),
            "score": 0.92
        }
    ]
    
    result = generate_with_citations("什么是RAG？", docs, manager)
    
    print(f"答案：{result['answer']}")
    print(f"\n引用数量：{result['citation_count']}")
    print("\n引用详情：")
    for citation in result['citations']:
        print(manager.format_citation(citation.id))

# ===== 4. 引用验证 =====

def verify_citation_accuracy(
    answer: str,
    citations: List[Citation]
) -> Dict:
    """
    验证引用准确性
    """
    results = {
        "total_citations": len(citations),
        "verified": 0,
        "unverified": 0,
        "issues": []
    }
    
    for citation in citations:
        # 检查引用文本是否在答案中
        if citation.metadata.text.lower() in answer.lower():
            results["verified"] += 1
        else:
            results["unverified"] += 1
            results["issues"].append({
                "citation_id": citation.id,
                "issue": "引用文本未在答案中出现"
            })
        
        # 检查置信度
        if citation.metadata.confidence < 0.7:
            results["issues"].append({
                "citation_id": citation.id,
                "issue": f"置信度过低: {citation.metadata.confidence:.2f}"
            })
    
    results["accuracy"] = results["verified"] / results["total_citations"] if results["total_citations"] > 0 else 0
    
    return results

def citation_verification_example():
    """
    引用验证示例
    """
    print("\n=== 引用验证 ===\n")
    
    manager = CitationManager()
    
    # 创建引用
    cite_id = manager.add_citation(
        doc_id="test.pdf",
        page=1,
        bbox=(0, 0, 100, 100),
        text="Python是解释型语言",
        confidence=0.95
    )
    
    answer = "Python是解释型语言<c>1.1</c>"
    citations = [manager.get_citation(cite_id)]
    
    verification = verify_citation_accuracy(answer, citations)
    
    print(f"总引用数：{verification['total_citations']}")
    print(f"已验证：{verification['verified']}")
    print(f"未验证：{verification['unverified']}")
    print(f"准确率：{verification['accuracy']:.2%}")
    
    if verification["issues"]:
        print("\n问题：")
        for issue in verification["issues"]:
            print(f"  - {issue}")

# ===== 5. 引用覆盖率检查 =====

def check_citation_coverage(answer: str) -> Dict:
    """
    检查引用覆盖率
    """
    # 简化实现：检查句子中是否有引用
    sentences = answer.split('。')
    
    import re
    citation_pattern = r'<c>[\d.]+</c>'
    
    results = {
        "total_sentences": len([s for s in sentences if s.strip()]),
        "cited_sentences": 0,
        "uncited_sentences": []
    }
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        if re.search(citation_pattern, sentence):
            results["cited_sentences"] += 1
        else:
            results["uncited_sentences"].append(sentence.strip())
    
    results["coverage"] = results["cited_sentences"] / results["total_sentences"] if results["total_sentences"] > 0 else 0
    
    return results

def coverage_check_example():
    """
    引用覆盖率检查示例
    """
    print("\n=== 引用覆盖率检查 ===\n")
    
    answer = "RAG是检索增强生成技术<c>1.1</c>。它结合检索和生成。"
    
    coverage = check_citation_coverage(answer)
    
    print(f"总句子数：{coverage['total_sentences']}")
    print(f"已引用句子：{coverage['cited_sentences']}")
    print(f"覆盖率：{coverage['coverage']:.2%}")
    
    if coverage["uncited_sentences"]:
        print("\n未引用的句子：")
        for sentence in coverage["uncited_sentences"]:
            print(f"  - {sentence}")

# ===== 6. 引用持久化 =====

class CitationStore:
    """引用持久化存储"""
    
    def __init__(self, storage_path: str = "./citations"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_citations(
        self,
        session_id: str,
        answer: str,
        citations: List[Citation]
    ):
        """保存引用数据"""
        data = {
            "session_id": session_id,
            "answer": answer,
            "citations": [
                {
                    "id": c.id,
                    "doc_id": c.metadata.doc_id,
                    "page": c.metadata.page,
                    "bbox": c.metadata.bbox,
                    "text": c.metadata.text,
                    "confidence": c.metadata.confidence
                }
                for c in citations
            ]
        }
        
        file_path = os.path.join(self.storage_path, f"{session_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_citations(self, session_id: str) -> Dict:
        """加载引用数据"""
        file_path = os.path.join(self.storage_path, f"{session_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def citation_storage_example():
    """
    引用持久化示例
    """
    print("\n=== 引用持久化 ===\n")
    
    store = CitationStore()
    manager = CitationManager()
    
    # 创建引用
    cite_id = manager.add_citation(
        doc_id="test.pdf",
        page=1,
        bbox=(0, 0, 100, 100),
        text="测试内容",
        confidence=0.95
    )
    
    answer = "这是测试答案<c>1.1</c>"
    citations = [manager.get_citation(cite_id)]
    
    # 保存
    store.save_citations("session_123", answer, citations)
    print("引用已保存")
    
    # 加载
    loaded = store.load_citations("session_123")
    print(f"\n加载的数据：")
    print(json.dumps(loaded, ensure_ascii=False, indent=2))

# ===== 7. 完整Citation系统 =====

class CompleteCitationSystem:
    """完整Citation系统"""
    
    def __init__(self):
        self.manager = CitationManager()
        self.store = CitationStore()
    
    def process_query(
        self,
        session_id: str,
        query: str,
        docs: List[Dict]
    ) -> Dict:
        """处理查询（完整流程）"""
        # 1. 生成带Citation的答案
        result = generate_with_citations(query, docs, self.manager)
        
        # 2. 验证引用
        verification = verify_citation_accuracy(
            result["answer"],
            result["citations"]
        )
        
        # 3. 检查覆盖率
        coverage = check_citation_coverage(result["answer"])
        
        # 4. 保存引用
        self.store.save_citations(
            session_id,
            result["answer"],
            result["citations"]
        )
        
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "verification": verification,
            "coverage": coverage,
            "session_id": session_id
        }

def complete_system_example():
    """
    完整Citation系统示例
    """
    print("\n=== 完整Citation系统 ===\n")
    
    system = CompleteCitationSystem()
    
    docs = [
        {
            "content": "RAG是检索增强生成技术",
            "doc_id": "rag.pdf",
            "page": 1,
            "bbox": (0, 0, 100, 100),
            "score": 0.95
        }
    ]
    
    result = system.process_query(
        "session_456",
        "什么是RAG？",
        docs
    )
    
    print(f"答案：{result['answer']}")
    print(f"\n验证准确率：{result['verification']['accuracy']:.2%}")
    print(f"引用覆盖率：{result['coverage']['coverage']:.2%}")
    print(f"会话ID：{result['session_id']}")

# ===== 运行所有示例 =====

if __name__ == "__main__":
    basic_citation_example()
    spatial_citation_example()
    citation_generation_example()
    citation_verification_example()
    coverage_check_example()
    citation_storage_example()
    complete_system_example()
```

---

## 关键要点

1. **基础引用**：使用`[文档X]`标记
2. **空间元数据**：包含doc_id、page、bbox、confidence
3. **引用验证**：检查准确性和覆盖率
4. **持久化**：保存引用数据供后续追溯
5. **完整系统**：生成→验证→存储一体化

---

**版本：** v1.0
**最后更新：** 2026-02-16
