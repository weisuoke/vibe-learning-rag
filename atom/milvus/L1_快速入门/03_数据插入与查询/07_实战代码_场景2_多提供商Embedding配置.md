# 实战代码 - 场景2: 多提供商Embedding配置

> **对比不同嵌入模型提供商的配置和性能**

---

## 场景概述

本场景演示如何配置和使用不同的嵌入模型提供商，包括：
- **OpenAI**: API 服务，高质量
- **Cohere**: API 服务，多语言支持
- **Sentence Transformers**: 本地模型，免费

**核心价值**:
- ✅ 灵活选择：根据需求选择合适的提供商
- ✅ 成本优化：本地模型 vs API 服务
- ✅ 性能对比：不同模型的效果差异
- ✅ 统一接口：Milvus 2.6 统一管理

---

## 环境准备

```bash
# 安装依赖
pip install pymilvus openai cohere sentence-transformers python-dotenv

# 配置 .env
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
```

---

## 完整代码示例

### 1. 导入和配置

```python
"""
场景2: 多提供商Embedding配置
对比 OpenAI, Cohere, Sentence Transformers
"""

import os
import time
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, Function, FunctionType

load_dotenv()

# 配置
MILVUS_URI = "http://localhost:19530"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

milvus_client = MilvusClient(uri=MILVUS_URI)
```

### 2. 提供商配置对比

```python
# 提供商配置字典
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "collection": "openai_collection",
        "function_params": {
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
            "dim": 1536
        },
        "index_params": {
            "metric_type": "COSINE"
        }
    },
    "cohere": {
        "name": "Cohere",
        "collection": "cohere_collection",
        "function_params": {
            "provider": "cohere",
            "model_name": "embed-english-v3.0",
            "api_key": COHERE_API_KEY,
            "dim": 1024
        },
        "index_params": {
            "metric_type": "COSINE"
        }
    },
    "sentence_transformers": {
        "name": "Sentence Transformers",
        "collection": "st_collection",
        "function_params": {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dim": 384
        },
        "index_params": {
            "metric_type": "COSINE"
        }
    }
}
```

### 3. 创建 Collection（通用函数）

```python
def create_collection_with_provider(provider_key):
    """
    为指定提供商创建 Collection

    参数:
        provider_key: 提供商键名 (openai, cohere, sentence_transformers)
    """
    config = PROVIDERS[provider_key]
    collection_name = config["collection"]

    # 删除旧 Collection
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # 创建 Schema
    schema = milvus_client.create_schema()

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("document", DataType.VARCHAR, max_length=5000)
    schema.add_field("dense", DataType.FLOAT_VECTOR,
                     dim=config["function_params"]["dim"])

    # 配置 Embedding Function
    schema.add_function(
        Function(
            name=f"{provider_key}_embedding",
            function_type=FunctionType.TEXTEMBEDDING,
            input_field_names=["document"],
            output_field_names=["dense"],
            params=config["function_params"]
        )
    )

    # 创建索引
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="AUTOINDEX",
        metric_type=config["index_params"]["metric_type"]
    )

    # 创建 Collection
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"✅ {config['name']} Collection 创建成功")
    print(f"   - 模型: {config['function_params']['model_name']}")
    print(f"   - 维度: {config['function_params']['dim']}")
```

### 4. 插入测试数据

```python
def insert_test_documents(provider_key):
    """
    插入测试文档并测量性能

    返回:
        插入耗时（秒）
    """
    config = PROVIDERS[provider_key]
    collection_name = config["collection"]

    # 测试文档
    documents = [
        {"document": "Milvus is an open-source vector database for AI applications."},
        {"document": "Embedding Functions simplify the RAG workflow in Milvus 2.6."},
        {"document": "Vector search enables semantic similarity matching."},
        {"document": "Hybrid search combines dense and sparse vectors."},
        {"document": "COSINE metric is recommended for text embeddings."}
    ]

    # 测量插入时间
    start_time = time.time()
    result = milvus_client.insert(collection_name, documents)
    elapsed = time.time() - start_time

    print(f"✅ {config['name']}: 插入 {result['insert_count']} 条文档")
    print(f"   - 耗时: {elapsed:.2f}s")

    return elapsed
```

### 5. 搜索测试

```python
def search_test(provider_key, query_text):
    """
    执行搜索测试并测量性能

    返回:
        (搜索耗时, 结果列表)
    """
    config = PROVIDERS[provider_key]
    collection_name = config["collection"]

    # 测量搜索时间
    start_time = time.time()
    results = milvus_client.search(
        collection_name=collection_name,
        data=[query_text],
        anns_field="dense",
        limit=3,
        output_fields=["document"]
    )
    elapsed = time.time() - start_time

    print(f"\n🔍 {config['name']} 搜索结果:")
    print(f"   - 耗时: {elapsed:.2f}s")

    for i, hit in enumerate(results[0], 1):
        print(f"   {i}. 相似度: {hit['distance']:.4f}")
        print(f"      文档: {hit['entity']['document'][:60]}...")

    return elapsed, results[0]
```

### 6. 性能对比测试

```python
def performance_comparison():
    """
    对比不同提供商的性能
    """
    print("=" * 60)
    print("多提供商性能对比测试")
    print("=" * 60)

    query_text = "What is vector search?"

    results = {}

    for provider_key in PROVIDERS.keys():
        print(f"\n【{PROVIDERS[provider_key]['name']}】")

        # 创建 Collection
        create_collection_with_provider(provider_key)

        # 插入数据
        insert_time = insert_test_documents(provider_key)

        # 搜索测试
        search_time, search_results = search_test(provider_key, query_text)

        # 记录结果
        results[provider_key] = {
            "name": PROVIDERS[provider_key]['name'],
            "model": PROVIDERS[provider_key]['function_params']['model_name'],
            "dim": PROVIDERS[provider_key]['function_params']['dim'],
            "insert_time": insert_time,
            "search_time": search_time,
            "top1_score": search_results[0]['distance'] if search_results else 0
        }

        print("-" * 60)

    return results
```

### 7. 结果分析

```python
def analyze_results(results):
    """
    分析和展示对比结果
    """
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)

    print("\n| 提供商 | 模型 | 维度 | 插入耗时 | 搜索耗时 | Top-1 相似度 |")
    print("|--------|------|------|----------|----------|--------------|")

    for provider_key, data in results.items():
        print(f"| {data['name']:<15} | {data['model']:<25} | "
              f"{data['dim']:<4} | {data['insert_time']:.2f}s | "
              f"{data['search_time']:.2f}s | {data['top1_score']:.4f} |")

    # 推荐建议
    print("\n📊 推荐建议:")

    # 找出最快的
    fastest_insert = min(results.items(), key=lambda x: x[1]['insert_time'])
    fastest_search = min(results.items(), key=lambda x: x[1]['search_time'])

    print(f"- 插入最快: {fastest_insert[1]['name']} ({fastest_insert[1]['insert_time']:.2f}s)")
    print(f"- 搜索最快: {fastest_search[1]['name']} ({fastest_search[1]['search_time']:.2f}s)")

    print("\n💡 选择建议:")
    print("- OpenAI: 高质量，适合生产环境，需要 API 费用")
    print("- Cohere: 多语言支持，适合国际化应用")
    print("- Sentence Transformers: 免费本地模型，适合开发测试")
```

### 8. 主函数

```python
def main():
    """
    主函数: 执行完整的对比测试
    """
    # 执行性能对比
    results = performance_comparison()

    # 分析结果
    analyze_results(results)

    print("\n✅ 对比测试完成！")

if __name__ == "__main__":
    main()
```

---

## 运行结果示例

```
==============================================================
多提供商性能对比测试
==============================================================

【OpenAI】
✅ OpenAI Collection 创建成功
   - 模型: text-embedding-3-small
   - 维度: 1536
✅ OpenAI: 插入 5 条文档
   - 耗时: 1.23s

🔍 OpenAI 搜索结果:
   - 耗时: 0.18s
   1. 相似度: 0.8923
      文档: Vector search enables semantic similarity matching...
   2. 相似度: 0.8456
      文档: Milvus is an open-source vector database for AI applic...
   3. 相似度: 0.8234
      文档: Embedding Functions simplify the RAG workflow in Milvu...
------------------------------------------------------------

【Cohere】
✅ Cohere Collection 创建成功
   - 模型: embed-english-v3.0
   - 维度: 1024
✅ Cohere: 插入 5 条文档
   - 耗时: 1.45s

🔍 Cohere 搜索结果:
   - 耗时: 0.21s
   1. 相似度: 0.8712
      文档: Vector search enables semantic similarity matching...
   2. 相似度: 0.8345
      文档: Milvus is an open-source vector database for AI applic...
   3. 相似度: 0.8123
      文档: Hybrid search combines dense and sparse vectors...
------------------------------------------------------------

【Sentence Transformers】
✅ Sentence Transformers Collection 创建成功
   - 模型: all-MiniLM-L6-v2
   - 维度: 384
✅ Sentence Transformers: 插入 5 条文档
   - 耗时: 0.45s

🔍 Sentence Transformers 搜索结果:
   - 耗时: 0.08s
   1. 相似度: 0.8534
      文档: Vector search enables semantic similarity matching...
   2. 相似度: 0.8123
      文档: Milvus is an open-source vector database for AI applic...
   3. 相似度: 0.7956
      文档: Embedding Functions simplify the RAG workflow in Milvu...
------------------------------------------------------------

==============================================================
性能对比总结
==============================================================

| 提供商 | 模型 | 维度 | 插入耗时 | 搜索耗时 | Top-1 相似度 |
|--------|------|------|----------|----------|--------------|
| OpenAI          | text-embedding-3-small    | 1536 | 1.23s | 0.18s | 0.8923 |
| Cohere          | embed-english-v3.0        | 1024 | 1.45s | 0.21s | 0.8712 |
| Sentence Transformers | all-MiniLM-L6-v2    | 384  | 0.45s | 0.08s | 0.8534 |

📊 推荐建议:
- 插入最快: Sentence Transformers (0.45s)
- 搜索最快: Sentence Transformers (0.08s)

💡 选择建议:
- OpenAI: 高质量，适合生产环境，需要 API 费用
- Cohere: 多语言支持，适合国际化应用
- Sentence Transformers: 免费本地模型，适合开发测试

✅ 对比测试完成！
```

---

## 提供商详细对比

### OpenAI

**优势**:
- 高质量嵌入
- 稳定的 API 服务
- 持续更新的模型

**劣势**:
- 需要 API 费用
- 依赖网络连接
- 有速率限制

**适用场景**:
- 生产环境
- 高质量要求
- 英文为主

### Cohere

**优势**:
- 多语言支持
- 良好的跨语言性能
- 灵活的模型选择

**劣势**:
- 需要 API 费用
- 相对较慢

**适用场景**:
- 国际化应用
- 多语言文档
- 跨语言检索

### Sentence Transformers

**优势**:
- 完全免费
- 本地运行，无网络依赖
- 速度快

**劣势**:
- 质量略低于 API 服务
- 需要本地计算资源
- 模型更新需要手动

**适用场景**:
- 开发测试
- 成本敏感
- 离线环境

---

## 切换提供商

### 方法1: 修改配置

```python
# 只需修改 provider_key
provider_key = "openai"  # 或 "cohere", "sentence_transformers"
create_collection_with_provider(provider_key)
```

### 方法2: 环境变量

```python
# .env 文件
EMBEDDING_PROVIDER=openai  # 或 cohere, sentence_transformers

# 代码中读取
provider_key = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
```

---

## 常见问题

### Q1: 如何选择合适的提供商？

**决策树**:
```
是否有预算？
├─ 是 → 是否需要多语言？
│        ├─ 是 → Cohere
│        └─ 否 → OpenAI
└─ 否 → Sentence Transformers
```

### Q2: 不同提供商的向量可以混用吗？

**答案**: 不可以。不同模型生成的向量维度和语义空间不同，不能在同一个 Collection 中混用。

### Q3: 如何迁移到不同的提供商？

```python
# 1. 导出数据
old_results = milvus_client.query(
    collection_name="old_collection",
    filter="id > 0",
    output_fields=["document"]
)

# 2. 创建新 Collection（新提供商）
create_collection_with_provider("new_provider")

# 3. 重新插入数据（自动使用新提供商）
documents = [{"document": r['document']} for r in old_results]
milvus_client.insert("new_collection", documents)
```

---

## 最佳实践

### 1. 开发-生产分离

```python
# 开发环境：使用本地模型
if os.getenv("ENV") == "development":
    provider = "sentence_transformers"
# 生产环境：使用 API 服务
else:
    provider = "openai"
```

### 2. 成本优化

```python
# 使用缓存减少 API 调用
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    # 缓存嵌入结果
    pass
```

### 3. 降级策略

```python
try:
    # 尝试使用 OpenAI
    create_collection_with_provider("openai")
except Exception as e:
    print(f"OpenAI 失败，降级到本地模型: {e}")
    create_collection_with_provider("sentence_transformers")
```

---

## 下一步

- **场景3**: [批量数据操作](./07_实战代码_场景3_批量数据操作.md) - 大规模数据导入
- **场景4**: [高级检索模式](./07_实战代码_场景4_高级检索模式.md) - 混合搜索
- **场景5**: [端到端RAG应用](./07_实战代码_场景5_端到端RAG应用.md) - 完整系统

---

## 参考资料

- [Embedding Overview](https://milvus.io/docs/embeddings.md)
- [Embedding Function Overview](https://milvus.io/docs/embedding-function-overview.md)
- temp/core_concepts/embedding_providers_docs.md

**版本**: v1.0 (基于 Milvus 2.6)
**最后更新**: 2026-02-22
