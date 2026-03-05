---
type: context7_documentation
library: langchain
version: latest (2026)
fetched_at: 2026-02-27
knowledge_point: 13_DocumentTransformer文档转换
context7_query: DocumentTransformer BaseDocumentTransformer document transformation
---

# Context7 文档：LangChain DocumentTransformer

## 文档来源
- 库名称：LangChain
- Context7 ID：/websites/langchain
- 官方文档链接：https://docs.langchain.com

## 关键信息提取

### 1. Html2TextTransformer 使用
```python
from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
```
- 将 HTML 文档转换为纯文本
- 移除 HTML 标签和格式
- 用于后续文本处理（分块、嵌入）

### 2. DoctranQATransformer 使用
```python
documents = [Document(page_content=sample_text)]
qa_transformer = DoctranQATransformer()
transformed_document = qa_transformer.transform_documents(documents)
```
- 从文档中提取问答对
- 结果存储在 metadata 属性中
- 用于自动化文档信息提取

### 3. EmbeddingsRedundantFilter + LongContextReorder 组合
```python
from langchain_community.document_transformers import LongContextReorder

filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)
```
- 先过滤冗余文档，再重排序
- 通过 DocumentCompressorPipeline 组合多个转换器
- 集成到 ContextualCompressionRetriever 中

### 4. EmbeddingsClusteringFilter 使用
```python
filter_ordered_cluster = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=10,
    num_closest=1,
)
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=10,
    num_closest=1,
    sorted=True,  # 按原始检索分数排序
)
```
- 将文档向量聚类为语义簇
- 从每个簇中选择代表性文档
- 支持按簇分组或按检索分数排序

### 5. RecursiveCharacterTextSplitter（作为 DocumentTransformer）
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
```
- TextSplitter 继承自 BaseDocumentTransformer
- `transform_documents()` 等价于 `split_documents()`
