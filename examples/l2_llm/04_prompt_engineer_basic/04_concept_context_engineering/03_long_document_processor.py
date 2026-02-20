from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

class LongDocumentProcessor:
    """长文档处理器"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.client = OpenAI()

    def chunk_document(self, document: str) -> list:
        """将文档分块"""
        chunks = []
        start = 0

        while start < len(document):
            end = start + self.chunk_size
            chunk = document[start:end]
            chunks.append(chunk)
            start = end - self.overlap # 重叠部分

        return chunks
    
    def process_long_document(self, document: str, query: str) -> dict:
        """处理长文档"""

        # 步骤 1: 分块
        chunks = self.chunk_document(document)
        print(f"文档长度: {len(document)} 字符")
        print(f"分块数量: {len(chunks)}")
        print(f"每块大小: {self.chunk_size} 字符")
        print(f"重叠大小: {self.overlap} 字符")

        # 步骤 2： 对每个块进行相关性评分
        print(f"\n=== 相关性评分 ===")

        scored_chunks = []

        for i, chunk in enumerate(chunks):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "你是文档分析专家, 只返回合法 JSON（不要 Markdown 代码块，不要额外解释）。"},
                    {"role": "user", "content": f"""
                        评估这段文档与问题的相关性(0.0-1.0)。

                        问题：{query}

                        文档片段：{chunk}

                        返回 JSON:
                        {{
                            "relevance_score": 0.0-1.0,
                            "reason": "评分理由"
                        }}
                    """}
                ],
                temperature=0.0
            )

            result = json.loads(response.choices[0].message.content)
            scored_chunks.append({
                "chunk": chunk,
                "score": result['relevance_score'],
                "reason": result['reason']
            })

            print(f" 块 {i+1}: {result['relevance_score']:.2f} - {result['reason']}")

        # 步骤 3：选择最相关的块
        top_chunks = sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:3]

        # 步骤 4：构建上下文
        context = "\n\n".join([
            f"## 相关片段 {i+1} (相关度：{chunk['score']:.2f}) \n{chunk['chunk']}"
            for i, chunk in enumerate(top_chunks)
        ])

        # 步骤 5：生成答案
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是专业的文档分析助手"},
                {"role": "user", "content": f"""
                    基于以下文档片段回答问题.
                 
                    {context}

                    问题：{query}
                """}
            ],
            temperature=0.1
        )

        return {
            "answer": response.choices[0].message.content,
            "chunks_used": len(top_chunks),
            "total_chunks": len(chunks),
            "top_chunks": top_chunks
        }
    
# 测试 
processor = LongDocumentProcessor(chunk_size=500, overlap=100)

# 模拟长文档
long_doc = """
RAG（检索增强生成，Retrieval-Augmented Generation）是一种创新的技术架构，它通过结合信息检索和大语言模型生成能力，有效解决了传统大模型的知识局限性问题。RAG 的核心思想是在生成答案之前，先从外部知识库中检索相关信息，然后将检索到的内容作为上下文注入到大模型的提示词中，从而让模型基于最新、最准确的信息进行回答。

为什么需要 RAG 技术？传统的大语言模型虽然在自然语言理解和生成方面表现出色，但它们存在几个关键限制。首先是知识截止日期问题，模型的训练数据只到某个时间点，无法获取最新信息。其次是幻觉问题，当模型不确定答案时，可能会编造看似合理但实际错误的内容。第三是领域知识局限，通用模型对特定领域的深度知识掌握不足。第四是无法访问私有数据，企业内部文档、个人笔记等私有信息无法被模型直接使用。RAG 技术通过引入外部知识检索机制，优雅地解决了这些问题。

RAG 的基本架构包含三个核心阶段：检索阶段、增强阶段和生成阶段。在检索阶段，系统首先将用户的查询转换为向量表示（Embedding），然后在向量数据库中进行相似度搜索，找出与查询最相关的文档片段。这个过程类似于在图书馆中根据关键词找到相关书籍。在增强阶段，系统将检索到的文档片段与原始查询组合，构建一个包含丰富上下文信息的提示词。这就像是给大模型提供了参考资料，让它可以基于这些资料来回答问题。在生成阶段，大语言模型接收增强后的提示词，基于检索到的上下文信息生成准确、相关的答案。

RAG 系统的关键组件包括文档加载器、文本分块器、嵌入模型、向量存储、检索器和生成模型。文档加载器负责从各种数据源（PDF、Word、网页等）中提取文本内容。文本分块器将长文档切分成适当大小的片段，这是因为大模型的上下文窗口有限，无法一次处理整个文档。嵌入模型将文本转换为高维向量，使得语义相似的文本在向量空间中距离较近。向量存储（向量数据库）高效地存储和检索这些向量。检索器根据查询向量找到最相关的文档片段。生成模型则基于检索到的上下文生成最终答案。

RAG 技术的应用场景非常广泛。在企业知识库问答系统中，员工可以快速查询公司文档、政策和流程，系统会自动检索相关内容并生成准确答案。在客户服务领域，RAG 可以基于产品手册、FAQ 和历史工单提供智能客服支持。在内容创作方面，RAG 可以帮助作者检索相关资料，生成有据可查的文章。在代码辅助开发中，RAG 可以检索代码库和文档，提供上下文相关的代码建议。在医疗健康领域，RAG 可以检索医学文献和病例，辅助医生诊断。在法律咨询中，RAG 可以检索法律条文和判例，提供专业法律意见。

RAG 技术的优势显而易见。首先是知识时效性，可以随时更新知识库，无需重新训练模型。其次是可解释性强，生成的答案可以追溯到具体的文档来源，增强了可信度。第三是成本效益高，相比于训练专用大模型，RAG 只需维护知识库和检索系统。第四是灵活性好，可以轻松切换不同的知识库，适应不同场景。第五是减少幻觉，基于真实文档生成答案，降低了模型编造内容的风险。

然而，RAG 技术也面临一些挑战。检索质量直接影响生成效果，如果检索不到相关文档或检索到错误文档，生成的答案质量会下降。文本分块策略需要仔细设计，分块太小可能丢失上下文，分块太大可能包含无关信息。向量检索可能存在语义理解偏差，相似的向量不一定表示语义相关。上下文窗口限制意味着只能使用有限数量的检索结果。系统延迟问题，检索和生成都需要时间，影响用户体验。知识库维护成本，需要持续更新和管理文档。

为了优化 RAG 系统性能，可以采用多种策略。混合检索结合向量检索和关键词检索，提高召回率。查询改写将用户的自然语言查询转换为更适合检索的形式。重排序（ReRank）对初步检索结果进行二次排序，提高精确度。多跳检索针对复杂问题进行多轮检索，逐步收集信息。文档摘要对长文档先生成摘要，减少上下文长度。缓存机制缓存常见查询的检索结果，提高响应速度。

向量数据库是 RAG 系统的核心基础设施，它专门用于存储和检索高维向量数据。与传统的关系型数据库不同，向量数据库针对向量相似度搜索进行了优化，能够在毫秒级别内从数百万甚至数十亿个向量中找到最相似的结果。向量数据库的出现是因为传统数据库无法高效处理向量相似度计算，而机器学习和深度学习的发展使得向量表示成为处理非结构化数据（文本、图像、音频）的标准方法。

向量数据库的工作原理基于向量索引技术。最简单的方法是暴力搜索，计算查询向量与所有存储向量的距离，但这在大规模数据集上效率极低。为了提高效率，向量数据库使用各种索引算法。HNSW（Hierarchical Navigable Small World）算法构建多层图结构，通过图遍历快速找到近似最近邻。IVF（Inverted File Index）算法先将向量空间划分为多个区域，查询时只搜索最相关的几个区域。PQ（Product Quantization）算法通过向量量化压缩存储空间，牺牲少量精度换取更快的搜索速度。这些算法在精确度、速度和内存使用之间做出不同的权衡。

向量数据库的核心操作包括插入、搜索和删除。插入操作将文本通过嵌入模型转换为向量，然后存储到数据库中，同时更新索引结构。搜索操作接收查询向量，使用索引算法快速找到 Top-K 个最相似的向量，返回对应的文档内容。删除操作从数据库中移除指定向量，并更新索引。许多向量数据库还支持元数据过滤，可以在向量搜索的同时根据元数据条件（如时间范围、文档类型）进行过滤，这在实际应用中非常有用。

市场上有多种向量数据库可供选择，各有特点。Chroma 是一个轻量级的开源向量数据库，易于上手，适合原型开发和小规模应用。Faiss 是 Facebook 开发的向量检索库，性能极高，但需要自己管理持久化。Milvus 是一个云原生的分布式向量数据库，支持大规模数据和高并发查询，适合生产环境。Pinecone 是一个完全托管的向量数据库服务，无需运维，但是商业产品。Weaviate 是一个开源的向量搜索引擎，支持多种数据类型和复杂查询。Qdrant 是一个高性能的向量数据库，提供丰富的过滤功能。

选择向量数据库时需要考虑多个因素。数据规模决定了是否需要分布式架构，小规模应用可以使用 Chroma 或 Faiss，大规模应用需要 Milvus 或 Pinecone。性能要求包括查询延迟和吞吐量，实时应用需要低延迟数据库。功能需求如是否需要元数据过滤、混合搜索、多租户支持等。部署方式可以选择自托管或云服务，自托管更灵活但需要运维能力。成本考虑包括开源免费方案和商业付费方案。社区支持和文档质量也很重要，活跃的社区可以提供更好的技术支持。

向量数据库的性能优化涉及多个方面。索引参数调优可以在精确度和速度之间找到平衡点，例如 HNSW 的 ef_construction 和 M 参数。批量操作比单条操作更高效，应该尽量批量插入和查询。缓存策略可以缓存热点查询结果，减少重复计算。硬件优化如使用 GPU 加速向量计算，使用 SSD 提高 I/O 性能。分片和副本可以提高并发能力和可用性。定期维护如重建索引、清理无效数据可以保持系统健康。

LangChain 是一个强大的开源框架，专门用于开发基于大语言模型的应用程序。它提供了一套完整的工具链，简化了 RAG 系统的构建过程。LangChain 的设计哲学是将复杂的 LLM 应用开发流程模块化，通过组合不同的组件快速构建功能丰富的应用。它支持多种大语言模型（OpenAI、Anthropic、HuggingFace 等），多种向量数据库，以及丰富的文档加载器和文本处理工具。

LangChain 的核心抽象包括几个关键概念。Models 是对各种大语言模型的统一封装，包括聊天模型和文本生成模型。Prompts 提供了提示词模板管理功能，支持变量替换和少样本学习。Indexes 处理文档的加载、分块、嵌入和存储。Chains 将多个组件串联起来，形成完整的处理流程。Agents 是能够使用工具并根据观察结果做出决策的智能体。Memory 提供对话历史管理，支持多种记忆策略。

在 RAG 应用中，LangChain 提供了专门的组件。Document Loaders 支持从 PDF、Word、HTML、Markdown 等多种格式加载文档。Text Splitters 提供多种文本分块策略，如按字符数分块、按 Token 数分块、递归分块等。Embeddings 封装了多种嵌入模型，如 OpenAI Embeddings、HuggingFace Embeddings。Vector Stores 集成了主流向量数据库，提供统一的接口。Retrievers 实现了多种检索策略，如相似度检索、MMR（最大边际相关性）检索、自查询检索等。

使用 LangChain 构建 RAG 应用的典型流程如下。首先，使用 Document Loader 加载文档，例如 PyPDFLoader 加载 PDF 文件。然后，使用 Text Splitter 将文档分块，例如 RecursiveCharacterTextSplitter 按照段落和句子边界智能分块。接着，选择嵌入模型，例如 OpenAIEmbeddings，将文本块转换为向量。之后，选择向量数据库，例如 Chroma，存储向量和文档。创建检索器，配置检索参数如 Top-K 值。最后，构建 QA Chain，将检索器和大语言模型组合起来，形成完整的问答系统。

LangChain 的优势在于它大大降低了 RAG 应用的开发门槛。开发者不需要深入了解每个组件的底层实现，只需要通过简单的 API 调用就能构建功能完整的应用。它的模块化设计使得组件可以灵活替换，例如可以轻松切换不同的向量数据库或嵌入模型。丰富的预置组件覆盖了大部分常见需求，减少了重复开发。活跃的社区持续贡献新的功能和集成。详细的文档和示例帮助开发者快速上手。

LangChain 也在不断演进，引入了更多高级特性。LCEL（LangChain Expression Language）提供了一种声明式的方式来构建复杂的处理链。LangSmith 是配套的调试和监控工具，帮助开发者追踪和优化应用性能。LangServe 简化了将 LangChain 应用部署为 API 服务的过程。这些工具共同构成了一个完整的 LLM 应用开发生态系统，使得从原型到生产的过渡更加顺畅。
"""

result = processor.process_long_document(long_doc, "什么是 RAG?")

print(f"\n=== 处理结果 ===")
print(f"使用块数：{result['chunks_used']}/{result['total_chunks']}")
print(f"答案：{result['answer']}")