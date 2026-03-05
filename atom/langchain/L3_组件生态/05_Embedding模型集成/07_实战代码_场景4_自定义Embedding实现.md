# 实战代码：场景4 - 自定义Embedding实现

## 场景概述

本文档展示如何实现自定义Embedding类，包括简单实现、缓存优化、异步支持和LangChain集成测试。

## 代码示例1：简单自定义Embedding

### 实现目标
- 继承LangChain的Embeddings基类
- 实现必需的抽象方法
- 使用简单的向量化逻辑（示例用途）

### 完整代码

```python
from langchain_core.embeddings import Embeddings
from typing import List
import hashlib

class SimpleCustomEmbeddings(Embeddings):
    """
    简单的自定义Embedding实现
    使用哈希函数模拟向量化（仅用于演示）
    """
    
    def __init__(self, dimension: int = 384):
        """
        初始化
        
        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        """
        单个文本的嵌入逻辑
        使用哈希函数生成固定维度的向量（仅用于演示）
        
        Args:
            text: 输入文本
            
        Returns:
            向量
        """
        # 使用SHA256哈希
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            # 循环使用哈希字节
            byte_val = hash_bytes[i % len(hash_bytes)]
            # 归一化到[-1, 1]
            normalized_val = (byte_val / 255.0) * 2 - 1
            vector.append(normalized_val)
        
        return vector

# 使用示例
if __name__ == "__main__":
    # 创建实例
    embeddings = SimpleCustomEmbeddings(dimension=384)
    
    # 测试单个查询
    query = "什么是RAG？"
    query_vector = embeddings.embed_query(query)
    print(f"查询向量维度: {len(query_vector)}")
    print(f"查询向量前5个值: {query_vector[:5]}")
    
    # 测试批量文档
    documents = [
        "RAG是检索增强生成",
        "Embedding是文本向量化",
        "向量数据库用于存储向量"
    ]
    doc_vectors = embeddings.embed_documents(documents)
    print(f"\n文档数量: {len(doc_vectors)}")
    print(f"每个文档向量维度: {len(doc_vectors[0])}")
```

### 关键点说明

1. **继承Embeddings基类**：必须实现`embed_documents`和`embed_query`两个抽象方法
2. **向量维度一致性**：所有向量必须具有相同的维度
3. **批量处理**：`embed_documents`支持批量处理以提高效率
4. **查询与文档分离**：允许对查询和文档使用不同的处理策略

## 代码示例2：带缓存的自定义Embedding

### 实现目标
- 添加内存缓存层
- 避免重复计算相同文本的向量
- 提供缓存统计信息

### 完整代码

```python
from langchain_core.embeddings import Embeddings
from typing import List, Dict
import hashlib
from functools import lru_cache

class CachedCustomEmbeddings(Embeddings):
    """
    带缓存的自定义Embedding实现
    使用LRU缓存避免重复计算
    """
    
    def __init__(self, dimension: int = 384, cache_size: int = 1000):
        """
        初始化
        
        Args:
            dimension: 向量维度
            cache_size: 缓存大小
        """
        self.dimension = dimension
        self.cache_size = cache_size
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档（带缓存）
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        results = []
        for text in texts:
            # 检查缓存
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                results.append(self._cache[cache_key])
            else:
                self._cache_misses += 1
                vector = self._embed_single(text)
                # 添加到缓存
                if len(self._cache) >= self.cache_size:
                    # 简单的FIFO策略
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = vector
                results.append(vector)
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询（带缓存）
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        vector = self._embed_single(text)
        
        # 添加到缓存
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = vector
        
        return vector
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _embed_single(self, text: str) -> List[float]:
        """
        单个文本的嵌入逻辑
        
        Args:
            text: 输入文本
            
        Returns:
            向量
        """
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            normalized_val = (byte_val / 255.0) * 2 - 1
            vector.append(normalized_val)
        
        return vector
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

# 使用示例
if __name__ == "__main__":
    # 创建实例
    embeddings = CachedCustomEmbeddings(dimension=384, cache_size=100)
    
    # 测试缓存效果
    documents = [
        "RAG是检索增强生成",
        "Embedding是文本向量化",
        "RAG是检索增强生成",  # 重复
        "向量数据库用于存储向量",
        "Embedding是文本向量化"  # 重复
    ]
    
    # 第一次嵌入
    vectors1 = embeddings.embed_documents(documents)
    stats1 = embeddings.get_cache_stats()
    print("第一次嵌入后的缓存统计:")
    print(f"  缓存大小: {stats1['cache_size']}")
    print(f"  缓存命中: {stats1['cache_hits']}")
    print(f"  缓存未命中: {stats1['cache_misses']}")
    print(f"  命中率: {stats1['hit_rate']:.2%}")
    
    # 第二次嵌入相同文档
    vectors2 = embeddings.embed_documents(documents)
    stats2 = embeddings.get_cache_stats()
    print("\n第二次嵌入后的缓存统计:")
    print(f"  缓存大小: {stats2['cache_size']}")
    print(f"  缓存命中: {stats2['cache_hits']}")
    print(f"  缓存未命中: {stats2['cache_misses']}")
    print(f"  命中率: {stats2['hit_rate']:.2%}")
```

### 关键点说明

1. **缓存策略**：使用字典实现简单的LRU缓存
2. **缓存键**：使用MD5哈希作为缓存键
3. **缓存统计**：跟踪命中率以评估缓存效果
4. **缓存大小限制**：防止内存无限增长

## 代码示例3：异步自定义Embedding

### 实现目标
- 覆盖异步方法以提供原生异步支持
- 使用asyncio进行并发处理
- 提高高并发场景下的性能

### 完整代码

```python
from langchain_core.embeddings import Embeddings
from typing import List
import hashlib
import asyncio

class AsyncCustomEmbeddings(Embeddings):
    """
    异步自定义Embedding实现
    提供原生异步支持以提高并发性能
    """
    
    def __init__(self, dimension: int = 384, batch_size: int = 10):
        """
        初始化
        
        Args:
            dimension: 向量维度
            batch_size: 异步批处理大小
        """
        self.dimension = dimension
        self.batch_size = batch_size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        同步批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        同步嵌入单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        return self._embed_single(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量嵌入文档
        使用并发处理提高性能
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        # 分批处理
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # 并发处理批次
            batch_results = await asyncio.gather(
                *[self._aembed_single(text) for text in batch]
            )
            results.extend(batch_results)
        
        return results
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        异步嵌入单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        return await self._aembed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        """
        同步单个文本的嵌入逻辑
        
        Args:
            text: 输入文本
            
        Returns:
            向量
        """
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            normalized_val = (byte_val / 255.0) * 2 - 1
            vector.append(normalized_val)
        
        return vector
    
    async def _aembed_single(self, text: str) -> List[float]:
        """
        异步单个文本的嵌入逻辑
        模拟异步API调用
        
        Args:
            text: 输入文本
            
        Returns:
            向量
        """
        # 模拟异步IO操作
        await asyncio.sleep(0.01)
        
        # 实际的嵌入逻辑
        return self._embed_single(text)

# 使用示例
async def main():
    # 创建实例
    embeddings = AsyncCustomEmbeddings(dimension=384, batch_size=5)
    
    # 测试异步嵌入
    documents = [
        f"文档{i}: RAG开发中的Embedding技术"
        for i in range(20)
    ]
    
    # 异步批量嵌入
    import time
    start = time.time()
    vectors = await embeddings.aembed_documents(documents)
    async_time = time.time() - start
    
    print(f"异步嵌入完成:")
    print(f"  文档数量: {len(vectors)}")
    print(f"  向量维度: {len(vectors[0])}")
    print(f"  耗时: {async_time:.2f}秒")
    
    # 对比同步嵌入
    start = time.time()
    vectors_sync = embeddings.embed_documents(documents)
    sync_time = time.time() - start
    
    print(f"\n同步嵌入完成:")
    print(f"  文档数量: {len(vectors_sync)}")
    print(f"  向量维度: {len(vectors_sync[0])}")
    print(f"  耗时: {sync_time:.2f}秒")
    
    print(f"\n性能提升: {sync_time / async_time:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
```

### 关键点说明

1. **原生异步支持**：覆盖`aembed_documents`和`aembed_query`方法
2. **并发处理**：使用`asyncio.gather`并发处理多个文本
3. **批处理**：分批处理以控制并发数量
4. **性能优势**：在IO密集型场景下显著提升性能

## 代码示例4：与LangChain集成测试

### 实现目标
- 测试自定义Embedding与LangChain组件的集成
- 使用向量存储进行实际检索
- 验证端到端RAG流程

### 完整代码

```python
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from typing import List
import hashlib
import tempfile
import shutil

class TestCustomEmbeddings(Embeddings):
    """
    用于测试的自定义Embedding实现
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            normalized_val = (byte_val / 255.0) * 2 - 1
            vector.append(normalized_val)
        
        return vector

# 集成测试
def test_with_vectorstore():
    """测试与向量存储的集成"""
    print("=== 测试1: 与Chroma向量存储集成 ===\n")
    
    # 创建自定义Embedding
    embeddings = TestCustomEmbeddings(dimension=384)
    
    # 准备测试文档
    documents = [
        "RAG是检索增强生成技术，结合了检索和生成两个步骤",
        "Embedding是将文本转换为向量的过程",
        "向量数据库用于高效存储和检索向量",
        "LangChain是一个用于构建LLM应用的框架",
        "Prompt Engineering是设计有效提示词的技术"
    ]
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建向量存储
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory=temp_dir
        )
        
        print(f"向量存储创建成功，文档数量: {len(documents)}\n")
        
        # 测试相似度检索
        query = "什么是RAG？"
        results = vectorstore.similarity_search(query, k=3)
        
        print(f"查询: {query}")
        print(f"检索到 {len(results)} 个相关文档:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)

def test_with_text_splitter():
    """测试与文本分割器的集成"""
    print("\n=== 测试2: 与文本分割器集成 ===\n")
    
    # 创建自定义Embedding
    embeddings = TestCustomEmbeddings(dimension=384)
    
    # 准备长文本
    long_text = """
    RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
    它首先从知识库中检索相关文档，然后将这些文档作为上下文输入到大语言模型中。
    这种方法可以有效减少模型幻觉，提高生成内容的准确性。
    
    Embedding是RAG系统的核心组件之一。
    它负责将文本转换为向量表示，使得我们可以在向量空间中进行相似度计算。
    常用的Embedding模型包括OpenAI的text-embedding-ada-002和开源的sentence-transformers。
    
    向量数据库是存储和检索Embedding向量的专用数据库。
    常见的向量数据库包括Chroma、Milvus、Pinecone等。
    它们提供了高效的向量相似度搜索功能。
    """
    
    # 文本分割
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separator="\n"
    )
    chunks = text_splitter.split_text(long_text)
    
    print(f"文本分割完成，共 {len(chunks)} 个块\n")
    
    # 嵌入所有块
    chunk_vectors = embeddings.embed_documents(chunks)
    
    print(f"嵌入完成:")
    print(f"  块数量: {len(chunk_vectors)}")
    print(f"  向量维度: {len(chunk_vectors[0])}")
    
    # 显示前3个块
    print(f"\n前3个文本块:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"{i}. {chunk.strip()[:50]}...")

def test_end_to_end_rag():
    """测试端到端RAG流程"""
    print("\n=== 测试3: 端到端RAG流程 ===\n")
    
    # 创建自定义Embedding
    embeddings = TestCustomEmbeddings(dimension=384)
    
    # 准备知识库文档
    knowledge_base = [
        "Python是一种高级编程语言，广泛用于数据科学和AI开发",
        "LangChain是一个用于构建LLM应用的Python框架",
        "RAG系统需要三个核心组件：文档加载器、Embedding模型和向量存储",
        "Prompt Engineering是设计有效提示词以获得更好LLM输出的技术",
        "向量数据库使用近似最近邻算法进行高效检索"
    ]
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. 构建向量存储
        print("步骤1: 构建向量存储")
        vectorstore = Chroma.from_texts(
            texts=knowledge_base,
            embedding=embeddings,
            persist_directory=temp_dir
        )
        print(f"  ✓ 已索引 {len(knowledge_base)} 个文档\n")
        
        # 2. 执行检索
        print("步骤2: 执行检索")
        queries = [
            "LangChain是什么？",
            "RAG需要哪些组件？",
            "如何提高LLM输出质量？"
        ]
        
        for query in queries:
            results = vectorstore.similarity_search(query, k=2)
            print(f"\n查询: {query}")
            print(f"检索结果:")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. {doc.page_content}")
        
        print("\n✓ RAG流程测试完成")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)

# 运行所有测试
if __name__ == "__main__":
    test_with_vectorstore()
    test_with_text_splitter()
    test_end_to_end_rag()
    
    print("\n" + "="*50)
    print("所有集成测试完成！")
    print("="*50)
```

### 关键点说明

1. **向量存储集成**：自定义Embedding可以直接用于Chroma等向量存储
2. **文本分割集成**：与LangChain的文本分割器无缝配合
3. **端到端测试**：验证完整的RAG流程
4. **临时目录管理**：使用tempfile确保测试环境清洁

## 参考来源

本文档基于以下资料编写：

1. **源码分析**：`atom/langchain/L3_组件生态/05_Embedding模型集成/reference/source_embeddings_base_01.md`
   - LangChain Embeddings基类设计
   - 抽象方法定义
   - 异步支持机制

2. **搜索结果**：`atom/langchain/L3_组件生态/05_Embedding模型集成/reference/search_custom_embeddings_02.md`
   - 自定义Embeddings实现模式
   - 多提供商切换策略
   - 生产级实践经验

## 总结

本文档展示了4个完整的自定义Embedding实现示例：

1. **简单实现**：基础的Embeddings子类，实现必需的抽象方法
2. **缓存优化**：添加内存缓存层，提高重复文本的处理效率
3. **异步支持**：覆盖异步方法，提供原生异步支持以提高并发性能
4. **集成测试**：验证与LangChain组件的集成，包括向量存储和文本分割器

所有代码示例都是完整可运行的，可以直接用于学习和实践。
