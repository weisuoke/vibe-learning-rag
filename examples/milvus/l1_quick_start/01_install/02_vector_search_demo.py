"""
Milvus 2.6 向量检索完整示例
"""

from pymilvus import DataType, MilvusClient
import random
import time

def vector_search_demo():
    """向量检索完整示例"""
    print("=" * 60)
    print("Milvus 2.6 向量检索示例")
    print("=" * 60)
    print()

    # 1. 连接到 Milvus
    print("[1/6] 连接到 Milvus...")
    client = MilvusClient(uri="http://localhost:19530")
    print("✅ 连接成功")
    print()

    # 2. 创建 Collection
    print("[2/6] 创建 Collection...")
    collection_name = "vector_search_demo"

    # 如果存在则删除
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 注意：MilvusClient.create_collection 在不传 schema 时会自动创建 AUTOINDEX，
    # 后续再 create_index(同字段) 会报：
    # "at most one distinct index is allowed per field"
    # 为了演示“手动建索引”，这里用 schema 方式创建 collection，不自动建索引/加载。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=128)
    schema.add_field("text", DataType.VARCHAR, max_length=200)
    schema.add_field("category", DataType.VARCHAR, max_length=50)
    schema.add_field("score", DataType.INT64)

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' 创建成功")
    print()

    # 3. 插入数据
    print("[3/6] 插入数据...")
    num_entities= 1000
    data = [
        {
            "id": i,
            "vector": [random.random() for _ in range(128)],
            "text": f"这是第 {i} 条数据",
            "category": f"类别_{i % 10}",
            "score": random.randint(1, 100)
        }
        for i in range(num_entities)
    ]

    # 批量插入
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        client.insert(collection_name=collection_name, data=batch)
        print(f"   已插入 {min(i+batch_size, len(data))}/{len(data)}")

    print(f"✅ 插入 {num_entities} 条数据完成")
    print()

    # 4. 创建索引
    print("[4/6] 创建索引...")
    client.flush(collection_name=collection_name)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )

    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    client.load_collection(collection_name=collection_name)
    print("✅ 索引创建成功")
    print()

    # 5. 向量检索
    print("[5/6] 向量检索...")

    # 生成查询向量
    query_vector = [random.random() for _ in range(128)]

    # 基础检索
    print("   5.1 基础检索 (Top-5):")
    start = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        output_fields=["id", "text", "category", "score"]
    )
    elapsed = time.time() - start

    for i, hit in enumerate(results[0], 1):
            print(f"      {i}. ID: {hit['id']}, 相似度: {hit['distance']:.4f}, "
                  f"类别: {hit['entity']['category']}, 分数: {hit['entity']['score']}")
    print(f"   ⏱️  检索耗时: {elapsed*1000:.2f}ms")
    print()

    # 带标量过滤的检索
    print("   5.2 带标量过滤的检索 (score > 50):")
    start = time.time()
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        filter="score > 50",
        output_fields=["id", "text", "category", "score"]
    )
    elapsed = time.time() - start

    for i, hit in enumerate(results[0], 1):
        print(f"      {i}. ID: {hit['id']}, 相似度: {hit['distance']:.4f}, "
              f"分数: {hit['entity']['score']}")
    print(f"   ⏱️  检索耗时: {elapsed*1000:.2f}ms")
    print()

    # 6. 清理
    print("[6/6] 清理...")
    client.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' 已删除")
    print()

    print("=" * 60)
    print("🎉 向量检索示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    vector_search_demo()
