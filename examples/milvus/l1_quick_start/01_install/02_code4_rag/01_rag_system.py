"""
端到端 RAG 系统实现
"""

from typing import List, Dict, Optional

from langchain_core.documents import Document
from dotenv import load_dotenv
from pymilvus import MilvusClient
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 加载环境变量
load_dotenv()


def _docs_have_text(docs: List[Document], min_total_chars: int = 20) -> bool:
    total = sum(len((d.page_content or "").strip()) for d in docs)
    return total >= min_total_chars


class RAGSystem:
    """端到端 RAG 系统"""

    def delete_collection(self):
        """删除当前 Collection（用于示例脚本每次运行后清理数据）。"""
        if not self.milvus_client.has_collection(self.collection_name):
            return
        self.milvus_client.drop_collection(self.collection_name)
        print(f"✅ Collection '{self.collection_name}' 已删除")

    def _ocr_pdf_with_tesseract(
        self,
        file_path: str,
        *,
        dpi: int = 300,
        psm: int = 6,
    ) -> List[Document]:
        """使用 Tesseract 对扫描版 PDF 做 OCR（pdf -> images -> text）。"""
        from pdf2image import convert_from_path
        import pytesseract

        lang = "+".join(self.ocr_languages)
        images = convert_from_path(file_path, dpi=dpi)

        docs: List[Document] = []
        for page_idx, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_idx},
                    )
                )
        return docs

    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "rag_knowledge_base",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        ocr_languages: Optional[List[str]] = None,
    ):
        # 初始化 Milvus
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

        # 初始化 OpenAI
        self.openai_client = OpenAI()
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # 初始化文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # Unstructured/Tesseract OCR languages (e.g. ["chi_sim", "eng"])
        self.ocr_languages = ocr_languages or ["eng"]

        # 创建 Collection
        self._create_collection()

    def _create_collection(self):
        """创建 Milvus Collection"""
        if self.milvus_client.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' 已存在")
            return
        
        # 创建 Collection (1536 维 - text-embedding-3-small)
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=1536,
            metric_type="COSINE",
            auto_id=True
        )
        print(f"✅ Collection '{self.collection_name}' 创建成功")

    def load_documents(self, file_path: str) -> List[Document]:
        """加载文件"""
        print(f"\n[1/5] 加载文档：{file_path}")

        # 根据文件类型选择加载器
        lower_path = file_path.lower()
        if lower_path.endswith(".pdf"):
            # NOTE:
            # - “扫描版/图片版PDF”或“文字转路径(outline)的PDF”往往提取不到文本；
            #   需要 OCR 或换更强的解析器。
            loaders_to_try = []

            # 用户期望优先走 PyMuPDFLoader（pymupdf 已安装时生效）
            try:
                from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
                loaders_to_try.append(("PyMuPDFLoader", PyMuPDFLoader))
            except Exception:
                pass

            loaders_to_try.append(("PyPDFLoader", PyPDFLoader))

            try:
                from langchain_community.document_loaders import PDFPlumberLoader  # type: ignore
                loaders_to_try.append(("PDFPlumberLoader", PDFPlumberLoader))
            except Exception:
                pass

            for name, cls in loaders_to_try:
                try:
                    docs = cls(file_path).load()
                except Exception:
                    continue
                if _docs_have_text(docs):
                    if name != "PyPDFLoader":
                        print(f"✅ 使用 {name} 成功提取文本")
                    print(f"加载 {len(docs)} 个文档")
                    return docs

            try:
                docs = self._ocr_pdf_with_tesseract(file_path, dpi=300, psm=6)
                if _docs_have_text(docs):
                    print("✅ 使用 Tesseract OCR 成功提取文本")
                    print(f"加载 {len(docs)} 个文档")
                    return docs
            except Exception:
                pass

            hint = [
                "PDF 文本提取结果为空（0 字符）。这通常意味着：",
                "1) PDF 是扫描版/图片版（需要 OCR）；或",
                "2) PDF 的文字被转换成矢量路径（pypdf 无法还原为文本）。",
                "",
                "可选解决方案：",
                "- 先把 PDF 做 OCR/转成可复制文本（例如 ocrmypdf / Adobe / 在线工具），再摄入；",
                "- 或安装更强的解析器：",
                "  - `pip install pdfplumber` 然后自动回退到 PDFPlumberLoader",
                "  - `pip install pymupdf` 然后自动回退到 PyMuPDFLoader",
            ]
            raise RuntimeError("\n".join(hint))

        if lower_path.endswith(".txt"):
            documents = TextLoader(file_path).load()
            print(f"加载 {len(documents)} 个文档")
            return documents

        raise ValueError(f"不支持的文件类型: {file_path}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """文本分块"""
        print(f"\n[2/5] 文本分块...")

        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ 分块完成,共 {len(chunks)} 个 chunks")
        return chunks
    
    def embed_and_store(self, chunks: List[Document]):
        """向量化并存储到 Milvus"""
        print(f"\n[3/5] 向量化并存储...")

        if not chunks:
            print("⚠️  没有可用 chunks（可能文档未提取到文本），跳过向量化/存储。")
            return

        data = []
        for i, chunk in enumerate(chunks):
            content = (chunk.page_content or "").strip()
            if not content:
                continue

            # 向量化
            embedding = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=content
            ).data[0].embedding

            # 准备数据
            data.append({
                "vector": embedding,
                "text": content,
                "source": chunk.metadata.get("source", ""),
                # PyPDFLoader 默认元数据里包含 page
                "page": chunk.metadata.get("page", None),
            })

            if (i + 1) % 10 == 0:
                print(f"   已处理 {i + 1}/{len(chunks)}")

        if not data:
            print("⚠️  chunks 全部为空文本，未写入 Milvus。")
            return

        # 批量插入
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data
        )

        print(f"✅ 存储 {len(data)} 条数据到 Milvus")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量检索"""
        print(f"\n[4/5] 向量检索: {query}")

        # 查询向量化
        query_embedding = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        ).data[0].embedding

        # Milvus 检索
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source", "page"]
        )

        hits = results[0] if results else []
        print(f"✅ 检索到 {len(hits)} 条相关文档")
        return hits
    
    def generate_answer(self, query: str, context: str) -> str:
        """LLM 生成答案"""
        print(f"\n[5/5] LLM 生成答案...")

        # 构建 Prompt
        prompt = (
            "请根据【上下文】回答【问题】。如果上下文中没有相关信息，请只回答“我不知道”。\n\n"
            f"【上下文】\n{context}\n\n"
            f"【问题】\n{query}\n\n"
            "【回答要求】\n"
            "- 用中文回答\n"
            "- 用要点列出具体可自动化的事情（若有步骤/循环，请描述）\n"
        )

        # LLM 生成
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手,根据提供的上下文准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        print(f"✅ 答案生成完成")
        return answer
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """完整的 RAG 查询流程"""
        # 1. 向量检索
        search_results = self.search(question, top_k=top_k)

        # 2. 构建上下文
        context_parts = []
        for i, hit in enumerate(search_results):
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            source = entity.get("source", "")
            page = entity.get("page", None)
            text = entity.get("text", "")
            page_display = page if page is not None else "N/A"
            context_parts.append(f"[文档 {i+1}] (来源: {source}, 页码: {page_display})\n{text}")
        context = "\n\n".join(context_parts)

        # 3. LLM 生成答案
        answer = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "text": (hit.get("entity", {}).get("text", "")[:200] + "..."),
                    "source": hit.get("entity", {}).get("source", ""),
                    "page": hit.get("entity", {}).get("page", None),
                    "similarity": hit.get("distance", 0.0),
                }
                for hit in search_results
            ]
        }
    
    def ingest_documents(self, file_path: str):
        """完整的文档摄入流程"""
        # 1. 加载文档
        documents = self.load_documents(file_path)

        print(f"🆚 documents: {documents}")

        # 2. 文本分块
        chunks = self.chunk_documents(documents)

        print(f"🆚 chunks: {chunks}")

        # 3. 向量化并存储
        self.embed_and_store(chunks)

        print(f"\n✅ 文档摄入完成!")

# 使用示例
def main():
    """主函数"""
    print("=" * 60)
    print("端到端 RAG 系统演示")
    print("=" * 60)

    # 1. 初始化 RAG 系统
    rag = RAGSystem(
        milvus_uri="http://localhost:19530",
        collection_name="rag_demo",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o",
        ocr_languages=["chi_sim", "eng"],
    )

    try:
        # 2. 摄入文档（示例）
        rag.ingest_documents("/Users/wuxiao/Downloads/grok-ralph-loop.pdf")

        # 3. 查询
        question = "ralph-loop在英语学习中可以做哪些自动化的事情？"
        result = rag.query(question, top_k=3)

        # 4. 输出结果
        print("\n" + "=" * 60)
        print("查询结果")
        print("=" * 60)
        print(f"\n问题: {result['question']}")
        print(f"\n答案:\n{result['answer']}")
        print(f"\n参考来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] 相似度: {source['similarity']:.4f}")
            print(f"    来源: {source['source']}")
            print(f"    页码: {source['page']}")
            print(f"    内容: {source['text']}")
    finally:
        # 示例脚本：每次运行结束后都删除 collection，避免数据累积影响演示结果
        try:
            rag.delete_collection()
        except Exception as e:
            print(f"WARN: 删除 Collection 失败: {e}")

if __name__ == "__main__":
    main()
