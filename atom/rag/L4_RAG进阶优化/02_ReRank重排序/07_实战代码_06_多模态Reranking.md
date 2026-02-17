# 07_实战代码_06_多模态Reranking

## 场景说明

多模态Reranking支持对包含图像、文本、表格等多种内容的文档进行重排序,特别适合PDF文档问答、图文混合检索等场景。Jina reranker-m0是首个支持多模态的开源reranker。

**核心价值:**
- 统一处理文本和图像
- 支持PDF、PPT等视觉文档
- 跨模态相关性评分
- 适合复杂文档场景

**适用场景:**
- PDF文档问答
- 图文混合检索
- 技术文档搜索
- 学术论文检索

---

## 完整实现代码

### 1. Jina Reranker-m0基础使用

```python
"""
Jina reranker-m0多模态重排序
支持文本和图像混合输入
"""

import os
from typing import List, Dict, Union
from dotenv import load_dotenv
import requests
from PIL import Image
import base64
from io import BytesIO

load_dotenv()


class JinaMultimodalReranker:
    """Jina多模态重排序器"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "jina-reranker-m0-v1"
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = model
        self.api_url = "https://api.jina.ai/v1/rerank"

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def rerank_text(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5
    ) -> List[Dict]:
        """
        纯文本重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的文档数量

        Returns:
            排序后的文档列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            results = response.json()
            return results.get("results", [])

        except Exception as e:
            print(f"重排序失败: {e}")
            return []

    def rerank_multimodal(
        self,
        query: str,
        documents: List[Dict[str, Union[str, List[str]]]],
        top_n: int = 5
    ) -> List[Dict]:
        """
        多模态重排序

        Args:
            query: 查询文本
            documents: 文档列表,每个文档可包含text和images
                例如: [
                    {"text": "文本内容", "images": ["path1.jpg"]},
                    {"text": "另一个文本"}
                ]
            top_n: 返回的文档数量

        Returns:
            排序后的文档列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 处理文档,编码图像
        processed_docs = []
        for doc in documents:
            processed_doc = {}

            if "text" in doc:
                processed_doc["text"] = doc["text"]

            if "images" in doc and doc["images"]:
                processed_doc["images"] = [
                    self._encode_image(img_path)
                    for img_path in doc["images"]
                ]

            processed_docs.append(processed_doc)

        payload = {
            "model": self.model,
            "query": query,
            "documents": processed_docs,
            "top_n": top_n
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            results = response.json()
            return results.get("results", [])

        except Exception as e:
            print(f"多模态重排序失败: {e}")
            return []


# 使用示例
def main():
    reranker = JinaMultimodalReranker()

    # 1. 纯文本重排序
    print("=== 纯文本重排序 ===")
    query = "How does RAG work?"
    documents = [
        "RAG combines retrieval with generation.",
        "Python is a programming language.",
        "Vector databases enable semantic search."
    ]

    results = reranker.rerank_text(query, documents, top_n=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result['relevance_score']:.4f}]")
        print(f"   {result['document']['text']}\n")

    # 2. 多模态重排序
    print("\n=== 多模态重排序 ===")
    query = "Show me diagrams about RAG architecture"
    multimodal_docs = [
        {
            "text": "RAG architecture diagram",
            "images": ["diagrams/rag_architecture.png"]
        },
        {
            "text": "Simple text explanation of RAG"
        },
        {
            "text": "Vector search flowchart",
            "images": ["diagrams/vector_search.png"]
        }
    ]

    results = reranker.rerank_multimodal(query, multimodal_docs, top_n=2)
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result['relevance_score']:.4f}]")
        doc = result['document']
        print(f"   文本: {doc.get('text', 'N/A')}")
        print(f"   图像数: {len(doc.get('images', []))}\n")


if __name__ == "__main__":
    main()
```

---

### 2. PDF文档多模态检索

```python
"""
PDF文档多模态检索与重排序
提取文本和图像,进行混合检索
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document


class PDFMultimodalRetriever:
    """PDF多模态检索器"""

    def __init__(self, reranker: JinaMultimodalReranker):
        self.reranker = reranker

    def extract_pdf_content(
        self,
        pdf_path: str,
        output_dir: str = "extracted_images"
    ) -> List[Dict]:
        """
        提取PDF的文本和图像

        Args:
            pdf_path: PDF文件路径
            output_dir: 图像输出目录

        Returns:
            包含文本和图像路径的文档列表
        """
        Path(output_dir).mkdir(exist_ok=True)
        documents = []

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 提取文本
            text = page.get_text()

            # 提取图像
            image_list = page.get_images()
            image_paths = []

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # 保存图像
                image_path = f"{output_dir}/page{page_num+1}_img{img_index+1}.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_paths.append(image_path)

            # 创建文档
            doc_data = {
                "text": text.strip(),
                "images": image_paths,
                "metadata": {
                    "page": page_num + 1,
                    "source": pdf_path
                }
            }

            if text.strip() or image_paths:
                documents.append(doc_data)

        doc.close()
        return documents

    def search_pdf(
        self,
        query: str,
        pdf_path: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        在PDF中搜索相关内容

        Args:
            query: 查询文本
            pdf_path: PDF文件路径
            top_k: 返回的结果数量

        Returns:
            重排序后的文档列表
        """
        # 1. 提取PDF内容
        print(f"提取PDF内容: {pdf_path}")
        documents = self.extract_pdf_content(pdf_path)
        print(f"提取了 {len(documents)} 页内容")

        # 2. 多模态重排序
        print(f"执行多模态重排序...")
        results = self.reranker.rerank_multimodal(
            query=query,
            documents=documents,
            top_n=top_k
        )

        # 3. 添加元数据
        for result in results:
            doc_index = result['index']
            result['metadata'] = documents[doc_index]['metadata']

        return results


# 使用示例
def pdf_search_example():
    reranker = JinaMultimodalReranker()
    retriever = PDFMultimodalRetriever(reranker)

    # 搜索PDF
    query = "Show me the RAG architecture diagram"
    pdf_path = "documents/rag_paper.pdf"

    results = retriever.search_pdf(query, pdf_path, top_k=3)

    print(f"\n查询: {query}\n")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"相关性分数: {result['relevance_score']:.4f}")
        print(f"页码: {result['metadata']['page']}")
        print(f"文本预览: {result['document']['text'][:200]}...")
        print(f"包含图像: {len(result['document'].get('images', []))} 张")
        print("-" * 80)


if __name__ == "__main__":
    pdf_search_example()
```

---

### 3. 图文混合RAG管道

```python
"""
图文混合RAG管道
结合向量检索和多模态重排序
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class MultimodalRAGPipeline:
    """图文混合RAG管道"""

    def __init__(
        self,
        reranker: JinaMultimodalReranker,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.reranker = reranker
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None

    def index_documents(
        self,
        documents: List[Dict]
    ):
        """
        索引文档

        Args:
            documents: 包含text和images的文档列表
        """
        # 只对文本部分建立向量索引
        text_docs = []
        for i, doc in enumerate(documents):
            text_docs.append(
                Document(
                    page_content=doc.get("text", ""),
                    metadata={
                        "doc_index": i,
                        "has_images": bool(doc.get("images"))
                    }
                )
            )

        self.vectorstore = Chroma.from_documents(
            documents=text_docs,
            embedding=self.embeddings,
            collection_name="multimodal_rag"
        )

        # 保存原始文档(包含图像)
        self.documents = documents

    def search(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[Dict]:
        """
        执行图文混合检索

        流程:
        1. 向量检索获取候选
        2. 多模态重排序精排

        Args:
            query: 查询文本
            initial_k: 初检数量
            final_k: 最终返回数量

        Returns:
            重排序后的文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先调用index_documents索引文档")

        # 1. 向量检索
        print(f"向量检索 top {initial_k}...")
        vector_results = self.vectorstore.similarity_search(
            query,
            k=initial_k
        )

        # 2. 准备多模态文档
        multimodal_docs = []
        for doc in vector_results:
            doc_index = doc.metadata['doc_index']
            original_doc = self.documents[doc_index]
            multimodal_docs.append(original_doc)

        # 3. 多模态重排序
        print(f"多模态重排序 top {final_k}...")
        reranked = self.reranker.rerank_multimodal(
            query=query,
            documents=multimodal_docs,
            top_n=final_k
        )

        return reranked


# 使用示例
def multimodal_rag_example():
    # 准备文档
    documents = [
        {
            "text": "RAG architecture consists of retrieval and generation components.",
            "images": ["diagrams/rag_arch.png"]
        },
        {
            "text": "Vector databases store embeddings for semantic search.",
            "images": []
        },
        {
            "text": "This diagram shows the complete RAG pipeline.",
            "images": ["diagrams/rag_pipeline.png", "diagrams/rag_flow.png"]
        },
        {
            "text": "Python code example for implementing RAG.",
            "images": []
        },
        {
            "text": "Performance comparison chart of different rerankers.",
            "images": ["charts/reranker_comparison.png"]
        }
    ]

    # 创建管道
    reranker = JinaMultimodalReranker()
    pipeline = MultimodalRAGPipeline(reranker)

    # 索引文档
    print("索引文档...")
    pipeline.index_documents(documents)

    # 搜索
    query = "Show me diagrams explaining RAG architecture"
    results = pipeline.search(query, initial_k=10, final_k=3)

    print(f"\n查询: {query}\n")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"相关性分数: {result['relevance_score']:.4f}")
        doc = result['document']
        print(f"文本: {doc.get('text', 'N/A')[:100]}...")
        print(f"图像数: {len(doc.get('images', []))}")
        if doc.get('images'):
            print(f"图像: {', '.join(doc['images'])}")
        print("-" * 80)


if __name__ == "__main__":
    multimodal_rag_example()
```

---

### 4. 本地部署Jina Reranker-m0

```python
"""
本地部署Jina reranker-m0
使用Transformers库
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from PIL import Image
from typing import List, Dict, Union


class LocalJinaReranker:
    """本地Jina重排序器"""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-m0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"加载模型: {model_name} 到 {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        self.device = device

    def rerank_text(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        纯文本重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量

        Returns:
            排序后的文档列表
        """
        # 准备输入
        pairs = [[query, doc] for doc in documents]

        # 编码
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # 推理
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # 排序
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "index": int(idx),
                "relevance_score": float(scores[idx]),
                "document": {"text": documents[idx]}
            })

        return results


# 使用示例
def local_deployment_example():
    # 加载本地模型
    reranker = LocalJinaReranker()

    # 测试
    query = "How does RAG improve LLM accuracy?"
    documents = [
        "RAG combines retrieval with generation for factual answers.",
        "Python is a popular programming language.",
        "Vector databases enable semantic search.",
        "Reranking improves the quality of retrieved documents."
    ]

    results = reranker.rerank_text(query, documents, top_k=3)

    print(f"查询: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result['relevance_score']:.4f}]")
        print(f"   {result['document']['text']}\n")


if __name__ == "__main__":
    local_deployment_example()
```

---

## 代码说明

### 核心组件

1. **JinaMultimodalReranker**: API调用版本
   - 支持纯文本和多模态输入
   - 自动处理图像编码
   - 简单易用的接口

2. **PDFMultimodalRetriever**: PDF处理
   - 提取文本和图像
   - 页面级别索引
   - 保留元数据

3. **MultimodalRAGPipeline**: 完整管道
   - 向量检索初排
   - 多模态重排序精排
   - 统一的搜索接口

4. **LocalJinaReranker**: 本地部署
   - 使用Transformers库
   - GPU加速支持
   - 无需API调用

### 多模态文档格式

```python
{
    "text": "文本内容",
    "images": ["path1.jpg", "path2.png"],
    "metadata": {
        "page": 1,
        "source": "document.pdf"
    }
}
```

---

## 运行示例

### 环境准备

```bash
# 安装依赖
pip install jina-ai pymupdf pillow transformers torch

# 配置API Key
export JINA_API_KEY="your_key"
```

### 执行代码

```bash
python jina_multimodal_basic.py
python pdf_multimodal_search.py
python multimodal_rag_pipeline.py
python local_jina_deployment.py
```

### 预期输出

```
=== 纯文本重排序 ===
1. [分数: 0.9234]
   RAG combines retrieval with generation.

2. [分数: 0.7123]
   Vector databases enable semantic search.

=== 多模态重排序 ===
提取PDF内容: documents/rag_paper.pdf
提取了 15 页内容
执行多模态重排序...

查询: Show me the RAG architecture diagram

结果 1:
相关性分数: 0.9567
页码: 3
文本预览: Figure 1 shows the complete RAG architecture...
包含图像: 2 张
```

---

## 性能优化

### 1. API调用优化

```python
# 批处理
documents_batch = [documents[i:i+100] for i in range(0, len(documents), 100)]

# 并发调用
import asyncio
results = await asyncio.gather(*[
    rerank_async(query, batch)
    for batch in documents_batch
])
```

### 2. 图像处理优化

```python
# 压缩图像
from PIL import Image

def compress_image(image_path: str, max_size: int = 1024) -> str:
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    # 保存压缩后的图像
    compressed_path = f"compressed_{image_path}"
    img.save(compressed_path, optimize=True, quality=85)
    return compressed_path
```

### 3. 本地部署优化

```python
# 使用量化模型
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

---

## 常见问题

### Q1: Jina reranker-m0 vs 纯文本reranker?

**A:** 选择依据:

| 场景 | 推荐方案 |
|------|----------|
| 纯文本检索 | Cross-Encoder (更快) |
| PDF文档问答 | Jina reranker-m0 |
| 图文混合检索 | Jina reranker-m0 |
| 技术文档搜索 | Jina reranker-m0 |

### Q2: 如何处理大量图像?

**A:** 三种策略:
1. 压缩图像(降低分辨率)
2. 只对关键页面提取图像
3. 使用缩略图

### Q3: API vs 本地部署?

**A:**
- **API**: 简单易用,无需GPU,按需付费
- **本地**: 成本可控,数据隐私,需要GPU

### Q4: 如何提高多模态检索质量?

**A:**
```python
# 1. 提取高质量图像
min_image_size = (200, 200)  # 过滤小图

# 2. 保留图像上下文
context_window = 2  # 前后2段文本

# 3. 使用OCR增强
from pytesseract import image_to_string
image_text = image_to_string(image)
```

---

## 参考资料

### 官方文档
- [Jina AI Reranker](https://jina.ai/reranker) - 官方API文档
- [jina-reranker-m0 Model Card](https://huggingface.co/jinaai/jina-reranker-m0) - Hugging Face模型页

### 技术文章
- [Jina Reranker-m0 Release](https://jina.ai/news/jina-reranker-m0-multilingual-multimodal-document-reranker) - 发布公告
- [Multimodal RAG 2025](https://arxiv.org/html/2501.04695v1) - 多模态RAG论文
- [ColPali for RAG](https://medium.com/@intuitivedl/rag-with-colpali-everything-you-need-to-know-46b7bd50901b) - ColPali实践

### 代码示例
- [Jina vLLM Deployment](https://docs.vllm.ai/projects/recipes/en/latest/Jina/Jina-reranker-m0.html) - vLLM部署指南
- [Elastic Jina Integration](https://www.elastic.co/search-labs/blog/jina-rerankers-elastic-inference-service) - Elasticsearch集成

---

**版本:** v1.0 (2026年标准)
**最后更新:** 2026-02-16
**代码测试:** Python 3.13 + jina-ai 1.x + pymupdf 1.x
**模型:** jina-reranker-m0-v1
