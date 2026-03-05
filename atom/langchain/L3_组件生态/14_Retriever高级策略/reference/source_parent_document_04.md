---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/langchain/langchain_classic/retrievers/multi_vector.py
  - libs/langchain/langchain_classic/retrievers/parent_document_retriever.py
analyzed_at: 2026-02-27
knowledge_point: 14_Retriever高级策略
---

# 源码分析：MultiVectorRetriever 与 ParentDocumentRetriever

## 分析的文件
- `multi_vector.py` - 多向量检索器基类
- `parent_document_retriever.py` - 父文档检索器

## 关键发现

### MultiVectorRetriever(BaseRetriever)
"小到大"模式：搜索小块，返回父文档。

**核心属性：**
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vectorstore` | `VectorStore` | (必需) | 存储子块嵌入 |
| `docstore` | `BaseStore[str, Document]` | (必需) | 存储父文档 |
| `id_key` | `str` | `"doc_id"` | 链接子块到父 ID 的元数据键 |
| `search_type` | `SearchType` | `similarity` | 搜索策略（similarity/mmr/threshold） |

**检索流程：**
1. 根据 search_type 在 vectorstore 中搜索子块
2. 收集唯一父 ID（保持首次出现顺序）
3. 从 docstore 批量获取父文档
4. 过滤 None 结果

### ParentDocumentRetriever(MultiVectorRetriever)
扩展 MultiVectorRetriever，添加文档分割逻辑。

**额外属性：**
- `child_splitter`: TextSplitter（必需）- 将父文档分割成小块用于嵌入
- `parent_splitter`: TextSplitter | None - 可选，创建中间"父"块
- `child_metadata_fields`: Sequence[str] | None - 子块保留的元数据字段白名单

**三层层次结构（当 parent_splitter 存在时）：**
原始文档 → 父块 → 子块

**两层层次结构（无 parent_splitter）：**
原始文档 → 子块

### 设计模式
- **两层存储**：VectorStore（嵌入）+ DocStore（完整文档），解耦搜索索引和文档存储
- **策略模式**：SearchType 枚举实现可插拔搜索行为
- **模板方法**：`_split_docs_for_adding` 封装分割逻辑
- **元数据过滤**：防止大元数据膨胀向量索引
