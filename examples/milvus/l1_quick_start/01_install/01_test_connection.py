"""
Milvus 2.6 连接测试脚本
"""

from pymilvus import MilvusClient
import sys
import random

def test_connection():
    """测试 Milvus 连接 """
    print("=" * 60)
    print("Milvus 2.6 连接测试")
    print("=" * 60)
    print()

    try:
        # 连接到 Milvus
        print("[1/3] 连接到 Milvus...")
        client = MilvusClient(uri="http://localhost:19530")
        print("✅ 连接成功")
        print()

        # 列出 Collection
        print("[2/3] 列出 Collection...")
        collections = client.list_collections()
        print(f"✅ Collection 数量: {len(collections)}")
        if collections:
            print(f"   Collection 列表: {', '.join(collections)}")
        print()

        # 创建测试 Collection
        print("[3/3] 创建测试 Collection...")
        test_collection = "test_connection"

        # 如果存在则删除
        if client.has_collection(test_collection):
            client.drop_collection(test_collection)
        
        # 创建 Collection
        client.create_collection(
            collection_name=test_collection,
            dimension=128,
            metric_type="COSINE"
        )
        print(f"✅ Collection '{test_collection}' 创建成功")

        # 插入测试数据
        test_data = [{
            "id": i,
            "vector": [random.random() for _ in range(128)],
            "text": f"测试数据 {i}"
        } for i in range(10)]

        client.insert(collection_name=test_collection, data=test_data)
        print(f"✅ 插入 {len(test_data)} 条测试数据")

        client.flush(collection_name=test_collection)

        # 查询测试数据
        results = client.query(
            collection_name=test_collection,
            filter="id >= 0",
            output_fields=["id", "text"],
            limit=5,
            consistency_level="Strong"
        )
        print(f"✅ 查询成功,返回 {len(results)} 条数据")

        # 清理测试 Collection
        client.drop_collection(test_collection)
        print(f"✅ 清理测试 Collection")
        print()

        print("=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print()
        return False
    
if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)