# 核心概念 03：Neo4j与Cypher

## Neo4j简介

**Neo4j**是世界上最流行的图数据库，采用原生图存储和处理引擎。

**核心特点**：
- 原生图存储：数据以图结构存储
- ACID事务支持：保证数据一致性
- Cypher查询语言：声明式图查询语言
- 高性能：针对图遍历优化
- 丰富的生态：工具、插件、社区支持

**2025-2026新特性**：
- AI函数集成：直接在Cypher中调用LLM
- 向量索引：支持向量相似度搜索
- GraphRAG支持：官方GraphRAG Python包
- 性能提升：查询优化和并行处理

---

## Cypher查询语言基础

### 1. CREATE - 创建节点和关系

**创建节点**：
```cypher
// 创建单个节点
CREATE (p:Person {name: '张三', age: 30})

// 创建多个节点
CREATE (p1:Person {name: '张三'}),
       (p2:Person {name: '李四'}),
       (c:Company {name: '阿里巴巴'})

// 创建节点和关系
CREATE (p:Person {name: '张三'})
       -[:WORKS_AT {since: '2020-01-01'}]->
       (c:Company {name: '阿里巴巴'})
```

**Python示例**：
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def create_person(name: str, age: int):
    with driver.session() as session:
        result = session.run("""
            CREATE (p:Person {name: $name, age: $age})
            RETURN p
        """, name=name, age=age)
        return result.single()

create_person("张三", 30)
```

### 2. MATCH - 查询节点和关系

**基本查询**：
```cypher
// 查询所有Person节点
MATCH (p:Person)
RETURN p

// 查询特定Person
MATCH (p:Person {name: '张三'})
RETURN p

// 查询关系
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
RETURN p.name, c.name

// 查询路径
MATCH path = (p:Person)-[:WORKS_AT]->(:Company)-[:LOCATED_IN]->(city:City)
RETURN path
```

**Python示例**：
```python
def find_person(name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person {name: $name})
            RETURN p
        """, name=name)
        return [dict(record["p"]) for record in result]

persons = find_person("张三")
print(persons)
```

### 3. WHERE - 条件过滤

```cypher
// 属性过滤
MATCH (p:Person)
WHERE p.age > 30
RETURN p

// 多条件
MATCH (p:Person)
WHERE p.age > 30 AND p.city = '杭州'
RETURN p

// 正则表达式
MATCH (p:Person)
WHERE p.name =~ '张.*'
RETURN p

// 关系过滤
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
WHERE r.since > '2020-01-01'
RETURN p, c
```

### 4. RETURN - 返回结果

```cypher
// 返回节点
MATCH (p:Person)
RETURN p

// 返回属性
MATCH (p:Person)
RETURN p.name, p.age

// 返回计数
MATCH (p:Person)
RETURN count(p) AS person_count

// 返回去重
MATCH (p:Person)
RETURN DISTINCT p.city

// 返回限制
MATCH (p:Person)
RETURN p
ORDER BY p.age DESC
LIMIT 10
```

---

## 模式匹配

### 1. 基本模式

```cypher
// 单向关系
MATCH (a)-[:KNOWS]->(b)
RETURN a, b

// 双向关系
MATCH (a)-[:KNOWS]-(b)
RETURN a, b

// 多跳关系
MATCH (a)-[:KNOWS*2]-(b)
RETURN a, b

// 可变长度路径
MATCH (a)-[:KNOWS*1..3]-(b)
RETURN a, b
```

### 2. 复杂模式

```cypher
// 查询同事
MATCH (p:Person {name: '张三'})
      -[:WORKS_AT]->(c:Company)
      <-[:WORKS_AT]-(coworker:Person)
WHERE coworker <> p
RETURN coworker.name

// 查询朋友的朋友
MATCH (p:Person {name: '张三'})
      -[:FRIENDS_WITH*2]-(friend_of_friend:Person)
WHERE friend_of_friend <> p
RETURN DISTINCT friend_of_friend.name

// 查询最短路径
MATCH path = shortestPath(
    (p1:Person {name: '张三'})
    -[*]-(p2:Person {name: '李四'})
)
RETURN path
```

**Python示例**：
```python
def find_coworkers(person_name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person {name: $name})
                  -[:WORKS_AT]->(c:Company)
                  <-[:WORKS_AT]-(coworker:Person)
            WHERE coworker <> p
            RETURN coworker.name AS name
        """, name=person_name)
        return [record["name"] for record in result]

coworkers = find_coworkers("张三")
print(coworkers)
```

---

## 聚合与排序

### 1. 聚合函数

```cypher
// 计数
MATCH (p:Person)
RETURN count(p) AS total

// 求和
MATCH (p:Person)
RETURN sum(p.age) AS total_age

// 平均值
MATCH (p:Person)
RETURN avg(p.age) AS avg_age

// 最大最小值
MATCH (p:Person)
RETURN max(p.age) AS max_age, min(p.age) AS min_age

// 分组聚合
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, count(p) AS employee_count
ORDER BY employee_count DESC
```

### 2. 排序和分页

```cypher
// 排序
MATCH (p:Person)
RETURN p
ORDER BY p.age DESC

// 分页
MATCH (p:Person)
RETURN p
ORDER BY p.age DESC
SKIP 10
LIMIT 10
```

---

## MERGE - 创建或更新

```cypher
// 如果不存在则创建
MERGE (p:Person {name: '张三'})
RETURN p

// 创建时设置属性
MERGE (p:Person {name: '张三'})
ON CREATE SET p.created = timestamp()
RETURN p

// 更新时设置属性
MERGE (p:Person {name: '张三'})
ON MATCH SET p.updated = timestamp()
ON CREATE SET p.created = timestamp()
RETURN p

// 创建关系
MATCH (p:Person {name: '张三'})
MATCH (c:Company {name: '阿里巴巴'})
MERGE (p)-[r:WORKS_AT]->(c)
RETURN r
```

**Python示例**：
```python
def upsert_person(name: str, age: int):
    with driver.session() as session:
        result = session.run("""
            MERGE (p:Person {name: $name})
            ON CREATE SET p.age = $age, p.created = timestamp()
            ON MATCH SET p.age = $age, p.updated = timestamp()
            RETURN p
        """, name=name, age=age)
        return dict(result.single()["p"])

person = upsert_person("张三", 30)
print(person)
```

---

## 索引和约束

### 1. 创建索引

```cypher
// 单属性索引
CREATE INDEX person_name FOR (p:Person) ON (p.name)

// 复合索引
CREATE INDEX person_name_age FOR (p:Person) ON (p.name, p.age)

// 全文索引
CREATE FULLTEXT INDEX person_fulltext FOR (p:Person) ON EACH [p.name, p.bio]

// 查看索引
SHOW INDEXES
```

### 2. 创建约束

```cypher
// 唯一性约束
CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE

// 存在性约束（企业版）
CREATE CONSTRAINT person_name_exists FOR (p:Person) REQUIRE p.name IS NOT NULL

// 查看约束
SHOW CONSTRAINTS
```

---

## 2025-2026新特性

### 1. 向量索引

```cypher
// 创建向量索引
CREATE VECTOR INDEX person_embedding FOR (p:Person) ON (p.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}

// 向量相似度搜索
MATCH (p:Person)
WHERE p.embedding IS NOT NULL
WITH p, vector.similarity.cosine(p.embedding, $query_embedding) AS score
WHERE score > 0.8
RETURN p.name, score
ORDER BY score DESC
LIMIT 10
```

**Python示例**：
```python
from openai import OpenAI

client = OpenAI()

def vector_search(query: str, top_k: int = 10):
    # 1. 生成查询向量
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    # 2. 向量搜索
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person)
            WHERE p.embedding IS NOT NULL
            WITH p, vector.similarity.cosine(p.embedding, $query_embedding) AS score
            WHERE score > 0.8
            RETURN p.name AS name, score
            ORDER BY score DESC
            LIMIT $top_k
        """, query_embedding=query_embedding, top_k=top_k)
        return [{"name": record["name"], "score": record["score"]} for record in result]

results = vector_search("高级工程师")
print(results)
```

### 2. AI函数集成

```cypher
// 调用LLM生成文本
MATCH (p:Person {name: '张三'})
WITH p, ai.openai.chat([
    {role: 'user', content: '生成一段关于' + p.name + '的简介'}
]) AS bio
SET p.bio = bio
RETURN p

// 生成Embedding
MATCH (p:Person)
WHERE p.embedding IS NULL
WITH p, ai.openai.embedding(p.name + ' ' + p.bio) AS embedding
SET p.embedding = embedding
RETURN count(p) AS updated_count
```

---

## 在RAG中的应用

### 1. 知识图谱构建

```python
"""从文档构建知识图谱"""

def build_kg_from_doc(doc: str):
    # 1. 提取三元组（使用LLM）
    triples = extract_triples_llm(doc)

    # 2. 批量写入Neo4j
    with driver.session() as session:
        for triple in triples:
            session.run("""
                MERGE (s:Entity {name: $subject})
                SET s.type = $subject_type
                MERGE (o:Entity {name: $object})
                SET o.type = $object_type
                MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            """,
                subject=triple["subject"]["name"],
                subject_type=triple["subject"]["type"],
                predicate=triple["predicate"],
                object=triple["object"]["name"],
                object_type=triple["object"]["type"]
            )

doc = "张三在阿里巴巴工作，阿里巴巴位于杭州。"
build_kg_from_doc(doc)
```

### 2. 图检索

```python
"""基于图的检索"""

def graph_retrieval(query: str) -> list[str]:
    # 1. 提取实体
    entities = extract_entities(query)

    # 2. 图遍历
    with driver.session() as session:
        results = []
        for entity in entities:
            result = session.run("""
                MATCH path = (e:Entity {name: $entity})-[*1..2]-(related)
                RETURN path
                LIMIT 10
            """, entity=entity)

            for record in result:
                path = record["path"]
                # 转换为文本
                path_text = format_path(path)
                results.append(path_text)

        return results

def format_path(path):
    """格式化路径为文本"""
    nodes = [dict(node) for node in path.nodes]
    rels = [rel.type for rel in path.relationships]

    text_parts = []
    for i, node in enumerate(nodes):
        text_parts.append(node["name"])
        if i < len(rels):
            text_parts.append(rels[i])

    return " ".join(text_parts)

context = graph_retrieval("张三在哪工作？")
print(context)
```

### 3. 混合检索（向量+图）

```python
"""混合检索：向量检索 + 图检索"""

def hybrid_retrieval(query: str, top_k: int = 10):
    # 1. 向量检索
    vector_results = vector_search(query, top_k)

    # 2. 图检索
    graph_results = []
    for result in vector_results:
        entity_name = result["name"]
        paths = graph_retrieval_by_entity(entity_name)
        graph_results.extend(paths)

    # 3. 结果融合
    combined_results = {
        "vector_results": vector_results,
        "graph_results": graph_results
    }

    return combined_results

def graph_retrieval_by_entity(entity_name: str):
    with driver.session() as session:
        result = session.run("""
            MATCH path = (e:Entity {name: $entity})-[*1..2]-(related)
            RETURN path
            LIMIT 5
        """, entity=entity_name)

        return [format_path(record["path"]) for record in result]

results = hybrid_retrieval("张三的工作地点")
print(results)
```

---

## 性能优化

### 1. 使用索引

```cypher
// ❌ 慢：全表扫描
MATCH (p:Person)
WHERE p.name = '张三'
RETURN p

// ✅ 快：使用索引
MATCH (p:Person {name: '张三'})
RETURN p
```

### 2. 限制返回结果

```cypher
// ❌ 慢：返回所有结果
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN p, c

// ✅ 快：限制返回数量
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN p, c
LIMIT 100
```

### 3. 使用PROFILE分析

```cypher
PROFILE
MATCH (p:Person {name: '张三'})
      -[:WORKS_AT]->(c:Company)
RETURN p, c
```

---

## 总结

### Cypher的核心优势

1. **声明式语法**：描述"要什么"而非"怎么做"
2. **可视化查询**：查询语句像画图一样直观
3. **强大的模式匹配**：支持复杂的图模式
4. **性能优化**：针对图遍历优化

### 最佳实践

1. **使用参数化查询**：防止注入攻击
2. **创建适当的索引**：加速查询
3. **使用MERGE而非CREATE**：避免重复数据
4. **限制返回结果**：避免内存溢出
5. **使用PROFILE分析**：优化慢查询

---

**引用来源**：
- [Neo4j官方文档](https://neo4j.com/docs/)
- [Cypher查询语言](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j 2025新特性](https://neo4j.com/blog/2025-year-of-ai-scalability/)
- [Neo4j GraphRAG](https://neo4j.com/docs/neo4j-graphrag-python)

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
