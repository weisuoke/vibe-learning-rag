# 实战代码：端到端RAG系统

> 完整可运行的端到端RAG系统，集成幻觉检测、引用溯源、多策略缓解

---

## 场景说明

**目标：** 构建一个生产级的端到端RAG系统，集成所有幻觉防护技术

**核心功能：**
- 文档加载和索引
- 向量检索
- 多层防护（检索过滤、约束生成、一致性检测、置信度决策）
- 引用溯源
- 完整的错误处理和日志

**技术栈：**
- `chromadb`：向量存储
- `openai`：LLM 和 Embedding
- `sentence-transformers`：NLI 检测
- `pypdf`：PDF 文档解析

---

## 完整代码

```python
"""
端到端 RAG 系统
集成幻觉检测、引用溯源、多策略缓解
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import CrossEncoder
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# ===== 配置 =====
@dataclass
class RAGConfig:
    """RAG 系统配置"""
    # API 配置
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"

    # 模型配置
    llm_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # 检索配置
    top_k: int = 5
    min_retrieval_score: float = 0.6

    # 生成配置
    temperature: float = 0.3
    max_tokens: int = 1000

    # 防护配置
    consistency_threshold: float = 0.7
    require_citations: bool = True

    # 向量数据库配置
    collection_name: str = "rag_documents"
    persist_directory: str = "./chroma_db"


# ===== 文档管理器 =====
class DocumentManager:
    """
    文档管理器
    负责文档的加载、分块、索引
    """

    def __init__(self, config: RAGConfig):
        """初始化文档管理器"""
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.persist_directory
        )

        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            metadata={"description": "RAG documents with hallucination protection"}
        )

        print(f"[文档管理器] 初始化完成，集合: {config.collection_name}")

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 100
    ):
        """
        添加文档到向量数据库

        Args:
            documents: 文档列表，每个文档包含 'content', 'title', 'url' 等
            batch_size: 批处理大小
        """
        print(f"[文档管理器] 开始添加 {len(documents)} 个文档")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 生成 embeddings
            texts = [doc['content'] for doc in batch]
            embeddings = self._get_embeddings(texts)

            # 准备元数据
            metadatas = []
            ids = []
            for j, doc in enumerate(batch):
                metadatas.append({
                    'title': doc.get('title', f'Document {i+j+1}'),
                    'url': doc.get('url', ''),
                    'source': doc.get('source', 'unknown')
                })
                ids.append(f"doc_{i+j+1}")

            # 添加到集合
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

        print(f"[文档管理器] ✓ 文档添加完成")

    def search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = self.config.top_k

        # 生成查询 embedding
        query_embedding = self._get_embeddings([query])[0]

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 格式化结果
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'title': results['metadatas'][0][i].get('title', ''),
                'url': results['metadatas'][0][i].get('url', ''),
                'score': 1 - results['distances'][0][i]  # 转换为相似度分数
            })

        return documents

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的 embeddings

        Args:
            texts: 文本列表

        Returns:
            embeddings 列表
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=texts
        )

        return [item.embedding for item in response.data]


# ===== 幻觉防护系统 =====
class HallucinationProtection:
    """
    幻觉防护系统
    集成一致性检测、引用溯源、多策略缓解
    """

    def __init__(self, config: RAGConfig):
        """初始化防护系统"""
        self.config = config
        self.nli_model = CrossEncoder(config.nli_model)
        print(f"[防护系统] 初始化完成")

    def filter_retrieval(self, docs: List[Dict]) -> List[Dict]:
        """第1层：检索质量过滤"""
        filtered = []
        for doc in docs:
            if doc['score'] >= self.config.min_retrieval_score:
                if len(doc['content']) >= 50:
                    filtered.append(doc)

        return filtered

    def check_consistency(self, answer: str, docs: List[Dict]) -> float:
        """第3层：一致性检测"""
        if not docs:
            return 0.0

        # NLI 检测
        nli_scores = []
        for doc in docs:
            scores = self.nli_model.predict([(doc['content'], answer)])
            nli_scores.append(scores[0][2])

        return max(nli_scores) if nli_scores else 0.0

    def verify_citations(self, answer: str, num_docs: int) -> Dict:
        """验证引用有效性"""
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        citations = [int(c) for c in citations]

        invalid = [c for c in citations if c > num_docs or c < 1]

        return {
            'valid': len(invalid) == 0,
            'invalid_citations': invalid,
            'citation_count': len(set(citations))
        }


# ===== 端到端 RAG 系统 =====
class EndToEndRAGSystem:
    """
    端到端 RAG 系统
    集成所有组件
    """

    def __init__(self, config: RAGConfig):
        """初始化 RAG 系统"""
        print("\n" + "=" * 60)
        print("初始化端到端 RAG 系统")
        print("=" * 60 + "\n")

        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

        # 初始化组件
        self.doc_manager = DocumentManager(config)
        self.protection = HallucinationProtection(config)

        print("\n[系统] ✓ 所有组件初始化完成\n")

    def query(
        self,
        query: str,
        verbose: bool = True
    ) -> Dict:
        """
        处理用户查询

        Args:
            query: 用户查询
            verbose: 是否输出详细信息

        Returns:
            查询结果
        """
        if verbose:
            print("=" * 60)
            print(f"查询: {query}")
            print("=" * 60 + "\n")

        # 步骤1：检索
        if verbose:
            print("[步骤1] 检索相关文档")

        docs = self.doc_manager.search(query, top_k=self.config.top_k)

        if verbose:
            print(f"  检索到 {len(docs)} 个文档\n")

        # 步骤2：过滤
        if verbose:
            print("[步骤2] 过滤低质量文档")

        filtered_docs = self.protection.filter_retrieval(docs)

        if verbose:
            print(f"  保留 {len(filtered_docs)} 个文档\n")

        if len(filtered_docs) == 0:
            return {
                'status': 'no_docs',
                'answer': '抱歉，没有找到相关信息',
                'confidence': 0.0
            }

        # 步骤3：生成答案
        if verbose:
            print("[步骤3] 生成带引用的答案")

        answer = self._generate_answer(query, filtered_docs)

        if verbose:
            print(f"  ✓ 答案生成完成\n")

        # 步骤4：一致性检测
        if verbose:
            print("[步骤4] 一致性检测")

        consistency = self.protection.check_consistency(answer, filtered_docs)

        if verbose:
            print(f"  一致性分数: {consistency:.2f}\n")

        # 步骤5：验证引用
        if verbose:
            print("[步骤5] 验证引用")

        citation_verification = self.protection.verify_citations(
            answer,
            len(filtered_docs)
        )

        if verbose:
            print(f"  引用有效: {citation_verification['valid']}")
            print(f"  引用数量: {citation_verification['citation_count']}\n")

        # 步骤6：决策
        if verbose:
            print("[步骤6] 置信度决策")

        result = self._make_decision(
            answer,
            consistency,
            citation_verification,
            filtered_docs
        )

        if verbose:
            print(f"  状态: {result['status']}")
            print(f"  置信度: {result['confidence']:.2f}\n")

        return result

    def _generate_answer(
        self,
        query: str,
        docs: List[Dict]
    ) -> str:
        """生成带引用的答案"""
        # 构建 Prompt
        context = "\n\n".join([
            f"文档{i+1}: {doc['content']}"
            for i, doc in enumerate(docs)
        ])

        citation_req = ""
        if self.config.require_citations:
            citation_req = "\n2. 每个事实都要标注来源 [1], [2]"

        prompt = f"""基于以下文档回答问题：

{context}

问题：{query}

要求：
1. 只使用文档中的信息{citation_req}
3. 不要添加文档外的信息

答案："""

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        answer = response.choices[0].message.content

        # 添加来源详情
        if self.config.require_citations:
            answer += "\n\n来源：\n"
            for i, doc in enumerate(docs):
                title = doc.get('title', f'文档{i+1}')
                url = doc.get('url', '')
                answer += f"[{i+1}] {title}"
                if url:
                    answer += f"\n     {url}"
                answer += "\n"

        return answer

    def _make_decision(
        self,
        answer: str,
        consistency: float,
        citation_verification: Dict,
        docs: List[Dict]
    ) -> Dict:
        """置信度决策"""
        threshold = self.config.consistency_threshold

        # 计算综合置信度
        confidence = consistency

        if consistency >= threshold:
            if citation_verification['valid']:
                status = "approved"
                final_answer = answer
                message = "通过所有检测"
            else:
                status = "warning"
                final_answer = answer
                message = f"引用存在问题: {citation_verification['invalid_citations']}"
        elif consistency >= threshold - 0.2:
            status = "warning"
            final_answer = f"（不太确定）{answer}"
            message = "一致性分数较低"
        else:
            status = "rejected"
            final_answer = "抱歉，我对这个答案不够确定"
            message = "未通过一致性检测"

        return {
            'status': status,
            'answer': final_answer,
            'original_answer': answer,
            'confidence': confidence,
            'message': message,
            'consistency': consistency,
            'citation_verification': citation_verification,
            'sources': [
                {'title': doc['title'], 'url': doc['url']}
                for doc in docs
            ]
        }

    def save_result(self, result: Dict, filename: str = None):
        """保存查询结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_result_{timestamp}.json"

        result['timestamp'] = datetime.now().isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[保存] ✓ 结果已保存到 {filename}")


# ===== 示例使用 =====
def example_usage():
    """示例：如何使用端到端 RAG 系统"""

    # 1. 配置
    config = RAGConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        consistency_threshold=0.7,
        require_citations=True
    )

    # 2. 初始化系统
    rag = EndToEndRAGSystem(config)

    # 3. 添加文档
    print("=" * 60)
    print("添加文档到知识库")
    print("=" * 60 + "\n")

    documents = [
        {
            "content": "Python 3.9 于 2020 年 10 月 5 日正式发布，这是一个重要的版本更新。",
            "title": "Python 3.9 发布公告",
            "url": "https://www.python.org/downloads/release/python-390/",
            "source": "official"
        },
        {
            "content": "Python 3.9 新增了字典合并运算符 |，可以方便地合并两个字典。例如：d1 | d2 会返回一个新字典。",
            "title": "PEP 584 - 字典合并运算符",
            "url": "https://peps.python.org/pep-0584/",
            "source": "pep"
        },
        {
            "content": "Python 3.9 改进了类型提示功能，支持使用内置集合类型（如 list、dict）作为泛型，不再需要从 typing 模块导入。",
            "title": "PEP 585 - 类型提示泛型",
            "url": "https://peps.python.org/pep-0585/",
            "source": "pep"
        },
        {
            "content": "Python 3.9 引入了新的字符串方法 removeprefix() 和 removesuffix()，用于移除字符串的前缀和后缀。",
            "title": "Python 3.9 新特性",
            "url": "https://docs.python.org/3.9/whatsnew/3.9.html",
            "source": "docs"
        },
        {
            "content": "Python 3.9 的性能得到了显著提升，特别是在字典操作和函数调用方面。",
            "title": "Python 3.9 性能改进",
            "url": "https://docs.python.org/3.9/whatsnew/3.9.html#optimizations",
            "source": "docs"
        }
    ]

    rag.doc_manager.add_documents(documents)

    # 4. 查询
    print("\n" + "=" * 60)
    print("开始查询")
    print("=" * 60 + "\n")

    queries = [
        "Python 3.9 有什么新特性？",
        "Python 3.9 什么时候发布的？",
        "如何合并两个字典？"
    ]

    for query in queries:
        result = rag.query(query, verbose=True)

        print("\n" + "=" * 60)
        print("查询结果")
        print("=" * 60)
        print(f"状态: {result['status']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"消息: {result['message']}")
        print(f"\n答案:\n{result['answer']}")
        print("=" * 60 + "\n")

        # 保存结果
        # rag.save_result(result)


# ===== 主函数 =====
def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("端到端 RAG 系统演示")
    print("=" * 60 + "\n")

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：未设置 OPENAI_API_KEY 环境变量")
        print("请运行：export OPENAI_API_KEY='your-api-key'")
        return

    try:
        example_usage()

    except Exception as e:
        print(f"\n[错误] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

## 运行说明

### 1. 安装依赖

```bash
pip install chromadb openai sentence-transformers pypdf
```

### 2. 设置环境变量

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. 运行代码

```bash
python 12_实战代码_端到端RAG系统.py
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户查询                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              文档管理器 (DocumentManager)                │
│  - 向量检索 (ChromaDB)                                   │
│  - Embedding 生成 (OpenAI)                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          幻觉防护系统 (HallucinationProtection)          │
│  第1层：检索质量过滤                                      │
│  第2层：约束生成 (Prompt 工程)                            │
│  第3层：一致性检测 (NLI)                                  │
│  第4层：置信度决策                                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   最终答案                               │
│  - 带引用的答案                                          │
│  - 置信度分数                                            │
│  - 来源列表                                              │
└─────────────────────────────────────────────────────────┘
```

---

## 核心组件说明

### 1. RAGConfig

**功能：** 系统配置管理

**关键参数：**
- `llm_model`: LLM 模型
- `embedding_model`: Embedding 模型
- `consistency_threshold`: 一致性阈值
- `require_citations`: 是否要求引用

### 2. DocumentManager

**功能：** 文档管理和检索

**关键方法：**
- `add_documents()`: 添加文档到向量数据库
- `search()`: 检索相关文档
- `_get_embeddings()`: 生成 embeddings

### 3. HallucinationProtection

**功能：** 幻觉防护

**关键方法：**
- `filter_retrieval()`: 检索质量过滤
- `check_consistency()`: 一致性检测
- `verify_citations()`: 验证引用

### 4. EndToEndRAGSystem

**功能：** 端到端系统集成

**关键方法：**
- `query()`: 处理用户查询
- `_generate_answer()`: 生成答案
- `_make_decision()`: 置信度决策
- `save_result()`: 保存结果

---

## 使用示例

### 基础使用

```python
# 1. 配置
config = RAGConfig(
    openai_api_key="your-key",
    consistency_threshold=0.7
)

# 2. 初始化
rag = EndToEndRAGSystem(config)

# 3. 添加文档
documents = [
    {"content": "...", "title": "...", "url": "..."}
]
rag.doc_manager.add_documents(documents)

# 4. 查询
result = rag.query("你的问题")
print(result['answer'])
```

### 高级配置

```python
# 医疗场景：高阈值
config = RAGConfig(
    openai_api_key="your-key",
    consistency_threshold=0.9,
    min_retrieval_score=0.8,
    temperature=0.0,
    require_citations=True
)

# 客服场景：中等阈值
config = RAGConfig(
    openai_api_key="your-key",
    consistency_threshold=0.7,
    min_retrieval_score=0.6,
    temperature=0.3,
    require_citations=False
)
```

---

## 扩展建议

### 1. 添加文档解析

```python
from pypdf import PdfReader

def load_pdf(file_path: str) -> List[Dict]:
    """加载 PDF 文档"""
    reader = PdfReader(file_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append({
            "content": text,
            "title": f"{file_path} - Page {i+1}",
            "source": "pdf"
        })

    return documents
```

### 2. 添加 API 服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
rag_system = None

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = rag_system.query(request.query, verbose=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. 添加流式输出

```python
def query_stream(self, query: str):
    """流式输出答案"""
    # 检索和过滤
    docs = self.doc_manager.search(query)
    filtered_docs = self.protection.filter_retrieval(docs)

    # 流式生成
    stream = self.client.chat.completions.create(
        model=self.config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

---

## 总结

**本示例展示了：**

1. ✅ 完整的端到端 RAG 系统
2. ✅ 文档管理和向量检索
3. ✅ 多层幻觉防护
4. ✅ 引用溯源和验证
5. ✅ 生产级的架构设计

**系统特点：**

- **模块化设计**：各组件独立，易于扩展
- **完整防护**：四层防护体系
- **可配置**：灵活的配置管理
- **生产就绪**：包含错误处理和日志

**适用场景：**

- 企业知识库问答
- 文档智能检索
- 客服机器人
- 学术研究助手

**下一步：**

- 部署到生产环境
- 添加监控和日志
- 优化性能
- 扩展功能（多模态、多语言等）
