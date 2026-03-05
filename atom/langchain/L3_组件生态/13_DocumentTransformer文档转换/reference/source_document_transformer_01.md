---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/documents/transformers.py
  - libs/langchain/langchain_classic/document_transformers/__init__.py
  - libs/text-splitters/langchain_text_splitters/base.py
analyzed_at: 2026-02-27
knowledge_point: 13_DocumentTransformer文档转换
---

# 源码分析：DocumentTransformer 核心架构

## 分析的文件

### 1. BaseDocumentTransformer 抽象基类
- 文件：`libs/core/langchain_core/documents/transformers.py`
- 位置：`langchain_core.documents.transformers`

**关键代码：**
```python
class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation."""

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents."""

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents."""
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )
```

**关键发现：**
- 接口极简：只有两个方法 `transform_documents` 和 `atransform_documents`
- 输入输出一致：`Sequence[Document]` → `Sequence[Document]`
- 异步默认实现：使用 `run_in_executor` 包装同步方法
- 纯函数式设计：不修改原始文档，返回新的文档序列

### 2. 内置转换器列表（langchain_community）
- 文件：`libs/langchain/langchain_classic/document_transformers/__init__.py`

**所有内置转换器（已迁移到 langchain_community）：**
1. `BeautifulSoupTransformer` - HTML 解析和提取
2. `DoctranQATransformer` - 文档问答提取
3. `DoctranTextTranslator` - 文档翻译
4. `DoctranPropertyExtractor` - 属性提取
5. `EmbeddingsClusteringFilter` - 嵌入聚类过滤
6. `EmbeddingsRedundantFilter` - 冗余文档过滤
7. `GoogleTranslateTransformer` - Google 翻译
8. `Html2TextTransformer` - HTML 转纯文本
9. `LongContextReorder` - 长上下文重排序
10. `NucliaTextTransformer` - Nuclia 文本转换
11. `OpenAIMetadataTagger` - OpenAI 元数据标注

### 3. TextSplitter 作为 DocumentTransformer
- 文件：`libs/text-splitters/langchain_text_splitters/base.py`

**关键代码：**
```python
class TextSplitter(BaseDocumentTransformer, ABC):
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return self.split_documents(list(documents))
```

**关键发现：**
- TextSplitter 继承自 BaseDocumentTransformer
- `transform_documents()` 委托给 `split_documents()`
- 这意味着所有 TextSplitter 都是 DocumentTransformer
- 支持 chunk_size、chunk_overlap、length_function 等参数
- 支持 tiktoken 和 HuggingFace tokenizer

## 架构总结

```
BaseDocumentTransformer (ABC)
├── TextSplitter (ABC) → 文本分块
│   ├── CharacterTextSplitter
│   ├── RecursiveCharacterTextSplitter
│   ├── TokenTextSplitter
│   └── ... (17+ 种分块器)
├── BeautifulSoupTransformer → HTML 清洗
├── Html2TextTransformer → HTML 转文本
├── EmbeddingsRedundantFilter → 冗余过滤
├── EmbeddingsClusteringFilter → 聚类过滤
├── LongContextReorder → 上下文重排序
├── OpenAIMetadataTagger → 元数据标注
├── DoctranQATransformer → QA 提取
├── DoctranTextTranslator → 文本翻译
├── DoctranPropertyExtractor → 属性提取
├── GoogleTranslateTransformer → Google 翻译
└── NucliaTextTransformer → Nuclia 转换
```
