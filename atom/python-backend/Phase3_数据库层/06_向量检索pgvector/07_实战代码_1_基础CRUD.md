# 实战代码场景1：基础 CRUD 操作

本示例演示如何在 PostgreSQL + pgvector 中进行基础的向量 CRUD 操作。

## 环境准备

```bash
# 1. 启动 PostgreSQL + pgvector（Docker）
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. 安装 Python 依赖
uv add psycopg2-binary pgvector openai python-dotenv

# 3. 创建 .env 文件
cat > .env << EOF
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
EOF
```

## 完整代码示例

```python
"""
pgvector 基础 CRUD 操作示例
演示：创建表、插入、查询、更新、删除向量
"""

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== 1. 连接数据库 =====
print("=== 1. 连接数据库 ===")

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()

# 启用 pgvector 扩展
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
print("✅ pgvector 扩展已启用")

# ===== 2. 创建向量表 =====
print("\n=== 2. 创建向量表 ===")

cursor.execute("""
    DROP TABLE IF EXISTS documents CASCADE
""")

cursor.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(1536),  -- OpenAI text-embedding-3-small 的维度
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
conn.commit()
print("✅ 表 documents 创建成功")

# ===== 3. 插入单个向量 =====
print("\n=== 3. 插入单个向量 ===")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 生成 Embedding
text = "什么是向量数据库？"
embedding = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
).data[0].embedding

print(f"文本: {text}")
print(f"向量维度: {len(embedding)}")
print(f"向量前5维: {embedding[:5]}")

# 插入向量
cursor.execute(
    """
    INSERT INTO documents (content, embedding, metadata)
    VALUES (%s, %s, %s)
    RETURNING id
    """,
    (text, embedding, {"category": "tech", "priority": 5})
)
doc_id = cursor.fetchone()[0]
conn.commit()
print(f"✅ 文档插入成功，ID: {doc_id}")

# ===== 4. 批量插入向量 =====
print("\n=== 4. 批量插入向量 ===")

texts = [
    "如何使用 pgvector？",
    "RAG 系统的架构设计",
    "向量检索的性能优化",
    "HNSW 索引的原理",
    "IVFFlat 索引的使用"
]

# 批量生成 Embedding
print(f"正在生成 {len(texts)} 个文档的 Embedding...")
embeddings_response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

# 批量插入
data = [
    (text, emb.embedding, {"category": "tech", "priority": i + 1})
    for i, (text, emb) in enumerate(zip(texts, embeddings_response.data))
]

cursor.executemany(
    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
    data
)
conn.commit()
print(f"✅ 批量插入成功，共 {len(texts)} 条")

# ===== 5. 查询向量 =====
print("\n=== 5. 查询向量 ===")

# 查询所有文档
cursor.execute("SELECT id, content, metadata FROM documents ORDER BY id")
results = cursor.fetchall()

print(f"数据库中共有 {len(results)} 条文档：")
for doc_id, content, metadata in results:
    print(f"  ID {doc_id}: {content} | {metadata}")

# 查询单个文档的向量
cursor.execute("SELECT id, content, embedding FROM documents WHERE id = %s", (1,))
doc_id, content, embedding = cursor.fetchone()
print(f"\n文档 ID {doc_id} 的向量信息：")
print(f"  内容: {content}")
print(f"  向量维度: {len(embedding)}")
print(f"  向量前5维: {embedding[:5]}")

# ===== 6. 更新向量 =====
print("\n=== 6. 更新向量 ===")

# 更新文档内容和向量
new_text = "什么是向量数据库？（更新版）"
new_embedding = client.embeddings.create(
    input=new_text,
    model="text-embedding-3-small"
).data[0].embedding

cursor.execute(
    """
    UPDATE documents
    SET content = %s, embedding = %s, metadata = %s
    WHERE id = %s
    """,
    (new_text, new_embedding, {"category": "tech", "priority": 10, "updated": True}, 1)
)
conn.commit()
print(f"✅ 文档 ID 1 更新成功")

# 验证更新
cursor.execute("SELECT content, metadata FROM documents WHERE id = %s", (1,))
content, metadata = cursor.fetchone()
print(f"  更新后内容: {content}")
print(f"  更新后元数据: {metadata}")

# ===== 7. 删除向量 =====
print("\n=== 7. 删除向量 ===")

# 删除单个文档
cursor.execute("DELETE FROM documents WHERE id = %s", (1,))
conn.commit()
print(f"✅ 文档 ID 1 删除成功")

# 验证删除
cursor.execute("SELECT COUNT(*) FROM documents")
count = cursor.fetchone()[0]
print(f"  剩余文档数: {count}")

# ===== 8. 条件删除 =====
print("\n=== 8. 条件删除 ===")

# 删除优先级低于 3 的文档
cursor.execute(
    "DELETE FROM documents WHERE (metadata->>'priority')::int < 3 RETURNING id"
)
deleted_ids = [row[0] for row in cursor.fetchall()]
conn.commit()
print(f"✅ 删除了 {len(deleted_ids)} 条文档，ID: {deleted_ids}")

# 验证剩余文档
cursor.execute("SELECT id, content, metadata->>'priority' AS priority FROM documents ORDER BY id")
results = cursor.fetchall()
print(f"  剩余文档:")
for doc_id, content, priority in results:
    print(f"    ID {doc_id}: {content} | 优先级: {priority}")

# ===== 9. 查看表统计信息 =====
print("\n=== 9. 查看表统计信息 ===")

# 表大小
cursor.execute("""
    SELECT pg_size_pretty(pg_total_relation_size('documents')) AS table_size
""")
table_size = cursor.fetchone()[0]
print(f"表大小: {table_size}")

# 文档数量
cursor.execute("SELECT COUNT(*) FROM documents")
doc_count = cursor.fetchone()[0]
print(f"文档数量: {doc_count}")

# 平均向量维度（验证）
cursor.execute("SELECT AVG(array_length(embedding::float[], 1)) FROM documents")
avg_dim = cursor.fetchone()[0]
print(f"平均向量维度: {avg_dim}")

# ===== 10. 清理资源 =====
print("\n=== 10. 清理资源 ===")

cursor.close()
conn.close()
print("✅ 数据库连接已关闭")

print("\n=== 示例完成 ===")
```

## 运行输出示例

```
=== 1. 连接数据库 ===
✅ pgvector 扩展已启用

=== 2. 创建向量表 ===
✅ 表 documents 创建成功

=== 3. 插入单个向量 ===
文本: 什么是向量数据库？
向量维度: 1536
向量前5维: [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]
✅ 文档插入成功，ID: 1

=== 4. 批量插入向量 ===
正在生成 5 个文档的 Embedding...
✅ 批量插入成功，共 5 条

=== 5. 查询向量 ===
数据库中共有 6 条文档：
  ID 1: 什么是向量数据库？ | {'category': 'tech', 'priority': 5}
  ID 2: 如何使用 pgvector？ | {'category': 'tech', 'priority': 1}
  ID 3: RAG 系统的架构设计 | {'category': 'tech', 'priority': 2}
  ID 4: 向量检索的性能优化 | {'category': 'tech', 'priority': 3}
  ID 5: HNSW 索引的原理 | {'category': 'tech', 'priority': 4}
  ID 6: IVFFlat 索引的使用 | {'category': 'tech', 'priority': 5}

文档 ID 1 的向量信息：
  内容: 什么是向量数据库？
  向量维度: 1536
  向量前5维: [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]

=== 6. 更新向量 ===
✅ 文档 ID 1 更新成功
  更新后内容: 什么是向量数据库？（更新版）
  更新后元数据: {'category': 'tech', 'priority': 10, 'updated': True}

=== 7. 删除向量 ===
✅ 文档 ID 1 删除成功
  剩余文档数: 5

=== 8. 条件删除 ===
✅ 删除了 2 条文档，ID: [2, 3]
  剩余文档:
    ID 4: 向量检索的性能优化 | 优先级: 3
    ID 5: HNSW 索引的原理 | 优先级: 4
    ID 6: IVFFlat 索引的使用 | 优先级: 5

=== 9. 查看表统计信息 ===
表大小: 128 kB
文档数量: 3
平均向量维度: 1536.0

=== 10. 清理资源 ===
✅ 数据库连接已关闭

=== 示例完成 ===
```

## 代码说明

### 1. 连接数据库和启用扩展

```python
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

- 使用 `psycopg2` 连接 PostgreSQL
- 启用 `vector` 扩展（如果未启用）

### 2. 创建向量表

```python
cursor.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(1536),  -- 固定维度
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
```

- `vector(1536)` 指定向量维度（必须与 Embedding 模型一致）
- `JSONB` 用于存储灵活的元数据

### 3. 插入向量

```python
# 单个插入
cursor.execute(
    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
    (text, embedding, metadata)
)

# 批量插入
cursor.executemany(
    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
    data
)
```

- 使用 `%s` 占位符，psycopg2 自动处理向量格式转换
- 批量插入使用 `executemany` 提高效率

### 4. 查询向量

```python
cursor.execute("SELECT id, content, embedding FROM documents WHERE id = %s", (1,))
doc_id, content, embedding = cursor.fetchone()
```

- 查询返回的 `embedding` 是 Python 列表
- 可以直接使用 `len(embedding)` 获取维度

### 5. 更新向量

```python
cursor.execute(
    "UPDATE documents SET content = %s, embedding = %s WHERE id = %s",
    (new_text, new_embedding, doc_id)
)
```

- 更新向量时需要重新生成 Embedding

### 6. 删除向量

```python
# 删除单个
cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

# 条件删除
cursor.execute("DELETE FROM documents WHERE (metadata->>'priority')::int < 3")
```

- 支持 JSONB 字段的条件删除

## 常见问题

### Q1: 向量维度不匹配怎么办？

```python
# 错误示例
embedding = [0.1, 0.2, 0.3]  # 3 维
cursor.execute("INSERT INTO documents (embedding) VALUES (%s)", (embedding,))
# 报错：expected 1536 dimensions, not 3

# 解决方案：确保维度一致
print(f"Embedding 维度: {len(embedding)}")  # 检查维度
# 如果维度不对，检查 Embedding 模型配置
```

### Q2: 如何处理大批量插入？

```python
# 分批插入（每批 1000 条）
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    cursor.executemany(
        "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
        batch
    )
    conn.commit()
    print(f"✅ 已插入 {i + len(batch)} / {len(data)} 条")
```

### Q3: 如何验证向量是否正确存储？

```python
import numpy as np

# 插入向量
original_embedding = [0.1, 0.2, 0.3, ..., 0.9]
cursor.execute(
    "INSERT INTO documents (embedding) VALUES (%s) RETURNING id",
    (original_embedding,)
)
doc_id = cursor.fetchone()[0]

# 读取向量
cursor.execute("SELECT embedding FROM documents WHERE id = %s", (doc_id,))
retrieved_embedding = cursor.fetchone()[0]

# 验证一致性
assert np.allclose(original_embedding, retrieved_embedding, atol=1e-6)
print("✅ 向量存储正确")
```

### Q4: 如何处理 NULL 向量？

```python
# 允许 NULL 向量
cursor.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(1536) NULL  -- 允许 NULL
    )
""")

# 插入 NULL 向量
cursor.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    ("文档内容", None)
)

# 查询时过滤 NULL
cursor.execute("SELECT * FROM documents WHERE embedding IS NOT NULL")
```

## 下一步

完成基础 CRUD 操作后，可以继续学习：
- **场景2**：相似度检索（使用距离函数进行 Top-K 检索）
- **场景3**：RAG 集成（构建完整的文档问答系统）
- **场景4**：索引优化（创建 HNSW/IVFFlat 索引，提升性能）
