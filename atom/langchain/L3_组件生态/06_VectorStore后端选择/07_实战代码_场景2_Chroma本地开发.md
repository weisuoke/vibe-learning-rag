# 实战代码 - 场景2：Chroma 本地开发

> **来源**：基于 `reference/source_chroma_03.md` 源码分析 + `reference/context7_chroma_03.md` 官方文档

## 场景说明

Chroma 是一个开源的向量数据库，专为 AI 应用设计，适合：
- 本地开发和测试
- 原型验证
- 中小规模应用（< 100K 文档）
- 需要持久化的场景

**核心特点**：
- 本地优先：支持本地文件系统存储
- 易于使用：零配置启动
- 持久化：支持数据持久化
- 完整功能：支持元数据过滤、MMR 检索等

## 环境准备

```bash
# 安装依赖
pip install langchain-chroma chromadb langchain-openai

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 完整代码示例

### 示例1：基础本地开发

```python
"""
Chroma 本地开发 - 基础示例
演示如何使用 Chroma 进行本地开发和持久化存储
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import shutil

# 加载环境变量
load_dotenv()


def basic_local_development():
    """基础本地开发示例"""
    print("=" * 60)
    print("场景1：Chroma 本地开发 - 基础示例")
    print("=" * 60)

    # 1. 设置持久化目录
    persist_directory = "./chroma_db"
    print(f"
[步骤1] 设置持久化目录: {persist_directory}")
    
    # 清理旧数据（仅用于演示）
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("  清理旧数据完成")

    # 2. 初始化 Chroma（带持久化）
    print("
[步骤2] 初始化 Chroma 向量存储...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("✓ 初始化完成（数据将持久化到本地）")

    # 3. 添加文档
    print("
[步骤3] 添加文档...")
    documents = [
        Document(
            page_content="Chroma 是一个开源的向量数据库。",
            metadata={"source": "intro.txt", "category": "database"}
        ),
        Document(
            page_content="LangChain 支持多种向量存储后端。",
            metadata={"source": "langchain.txt", "category": "framework"}
        ),
        Document(
            page_content="向量检索是 RAG 系统的核心。",
            metadata={"source": "rag.txt", "category": "technique"}
        ),
    ]
    
    ids = vector_store.add_documents(documents)
    print(f"✓ 添加了 {len(ids)} 个文档")
    print(f"  文档 IDs: {ids}")

    # 4. 检索测试
    print("
[步骤4] 检索测试...")
    query = "向量数据库"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"  查询: {query}")
    print(f"  返回 {len(results)} 个结果:")
    for i, doc in enumerate(results, 1):
        print(f"
  结果 {i}:")
        print(f"    内容: {doc.page_content}")
        print(f"    元数据: {doc.metadata}")

    # 5. 验证持久化
    print("
[步骤5] 验证持久化...")
    print(f"  持久化目录: {persist_directory}")
    print(f"  目录存在: {os.path.exists(persist_directory)}")
    
    if os.path.exists(persist_directory):
        files = os.listdir(persist_directory)
        print(f"  目录内容: {files}")

    return vector_store, persist_directory


def load_from_disk():
    """从磁盘加载已有数据"""
    print("
" + "=" * 60)
    print("场景2：从磁盘加载数据")
    print("=" * 60)

    persist_directory = "./chroma_db"
    
    # 检查目录是否存在
    if not os.path.exists(persist_directory):
        print("  错误: 持久化目录不存在，请先运行 basic_local_development()")
        return None

    # 1. 加载已有数据
    print("
[步骤1] 从磁盘加载 Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("✓ 加载完成")

    # 2. 验证数据
    print("
[步骤2] 验证数据...")
    query = "LangChain"
    results = vector_store.similarity_search(query, k=3)
    
    print(f"  查询: {query}")
    print(f"  返回 {len(results)} 个结果")
    for i, doc in enumerate(results, 1):
        print(f"    {i}. {doc.page_content}")

    return vector_store


def metadata_filtering():
    """元数据过滤示例"""
    print("
" + "=" * 60)
    print("场景3：元数据过滤")
    print("=" * 60)

    # 初始化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="filtered_collection",
        embedding_function=embeddings
    )

    # 添加带元数据的文档
    documents = [
        Document(
            page_content="Python 3.13 发布了新特性。",
            metadata={"language": "python", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="JavaScript ES2024 标准发布。",
            metadata={"language": "javascript", "year": 2024, "type": "news"}
        ),
        Document(
            page_content="Python 教程：入门指南。",
            metadata={"language": "python", "year": 2023, "type": "tutorial"}
        ),
        Document(
            page_content="Go 语言并发编程。",
            metadata={"language": "go", "year": 2024, "type": "tutorial"}
        ),
    ]
    
    vector_store.add_documents(documents)

    # 1. 单条件过滤
    print("
[示例1] 单条件过滤 (language=python):")
    results = vector_store.similarity_search(
        "编程语言",
        k=5,
        filter={"language": "python"}
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")

    # 2. 多条件过滤（AND）
    print("
[示例2] 多条件过滤 (language=python AND year=2024):")
    results = vector_store.similarity_search(
        "编程",
        k=5,
        filter={
            "": [
                {"language": "python"},
                {"year": {"": 2024}}
            ]
        }
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")

    # 3. 范围过滤
    print("
[示例3] 范围过滤 (year >= 2024):")
    results = vector_store.similarity_search(
        "新特性",
        k=5,
        filter={"year": {"": 2024}}
    )
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")


def main():
    """主函数"""
    print("
" + "=" * 60)
    print("Chroma 本地开发 - 完整示例")
    print("=" * 60)

    # 运行所有示例
    vector_store, persist_dir = basic_local_development()
    load_from_disk()
    metadata_filtering()

    print("
" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 运行说明

```bash
# 1. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 2. 运行示例
python 07_实战代码_场景2_Chroma本地开发.py
```

## 最佳实践

### 1. 持久化策略

```python
# 开发环境：使用本地目录
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 测试环境：使用临时目录
import tempfile
temp_dir = tempfile.mkdtemp()
vector_store = Chroma(
    persist_directory=temp_dir,
    embedding_function=embeddings
)
```

### 2. Collection 管理

```python
# 使用不同的 collection 隔离数据
dev_store = Chroma(collection_name="dev", ...)
test_store = Chroma(collection_name="test", ...)
prod_store = Chroma(collection_name="prod", ...)
```

### 3. 元数据设计

```python
# 好的元数据设计
metadata = {
    "source": "file.pdf",
    "page": 1,
    "category": "technical",
    "created_at": "2024-01-01",
    "author": "John Doe"
}

# 避免过于复杂的嵌套
# 不推荐
metadata = {
    "nested": {
        "deep": {
            "structure": "value"
        }
    }
}
```

## 常见问题

### Q1: Chroma 和 InMemory 如何选择？

**Chroma**：
- 支持持久化
- 更好的性能
- 适合本地开发

**InMemory**：
- 零配置
- 不持久化
- 适合测试

### Q2: 如何清理 Chroma 数据？

```python
# 方法1：删除持久化目录
import shutil
shutil.rmtree("./chroma_db")

# 方法2：删除 collection
vector_store.delete_collection()
```

### Q3: Chroma 支持多大规模的数据？

- < 10K 文档：性能很好
- 10K - 100K 文档：性能可接受
- > 100K 文档：建议使用 Qdrant 或 Milvus

### Q4: 如何备份 Chroma 数据？

```bash
# 直接复制持久化目录
cp -r ./chroma_db ./chroma_db_backup
```

## 总结

Chroma 是本地开发的首选向量存储，适合：
- **本地开发**：零配置，易于使用
- **持久化**：数据保存到本地文件
- **中小规模**：< 100K 文档

**核心优势**：
- 零配置启动
- 持久化支持
- 完整功能（元数据过滤、MMR 等）
- 开发友好

**主要限制**：
- 性能限制（大规模数据）
- 单机限制（不支持分布式）
- 并发限制（高并发场景）

**下一步**：
- 需要高性能 → 使用 FAISS
- 需要生产部署 → 使用 Qdrant 或 Pinecone
- 需要企业级 → 使用 Milvus
