# 核心概念 04：SPARQL与RDF

## RDF模型简介

**RDF**（Resource Description Framework）是W3C制定的语义网标准，用于描述Web资源。

**核心特点**：
- 严格的三元组模型
- 基于URI的资源标识
- 支持语义推理
- 开放数据标准

**与属性图的区别**：

| 特性 | RDF | 属性图 |
|------|-----|--------|
| 数据模型 | 三元组 | 节点+边+属性 |
| 属性表示 | 额外三元组 | 直接附加 |
| 标准化 | W3C标准 | 无统一标准 |
| 推理能力 | 强 | 弱 |
| 应用场景 | 语义网、学术 | 应用开发 |

---

## RDF三元组

### 基本结构

```
三元组 = (主语, 谓语, 宾语)
       = (Subject, Predicate, Object)
```

**示例**：
```turtle
# Turtle格式（RDF的一种序列化格式）
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:张三 foaf:name "张三" .
ex:张三 foaf:age 30 .
ex:张三 ex:worksFor ex:阿里巴巴 .
ex:阿里巴巴 ex:locatedIn ex:杭州 .
```

### RDF的三种节点类型

**1. IRI（国际化资源标识符）**：
```turtle
ex:张三  # IRI节点
```

**2. 字面量（Literal）**：
```turtle
"张三"  # 字符串字面量
30  # 数字字面量
"2026-02-14"^^xsd:date  # 带类型的字面量
```

**3. 空白节点（Blank Node）**：
```turtle
_:b1  # 匿名节点
```

---

## SPARQL查询语言

### 1. 基本查询

**SELECT查询**：
```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# 查询所有人的姓名
SELECT ?name
WHERE {
  ?person foaf:name ?name .
}

# 查询特定人
SELECT ?age
WHERE {
  ex:张三 foaf:age ?age .
}

# 查询工作关系
SELECT ?person ?company
WHERE {
  ?person ex:worksFor ?company .
}
```

**Python示例**：
```python
from rdflib import Graph, Namespace

# 创建图
g = Graph()

# 定义命名空间
EX = Namespace("http://example.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

# 添加三元组
g.add((EX.张三, FOAF.name, Literal("张三")))
g.add((EX.张三, FOAF.age, Literal(30)))
g.add((EX.张三, EX.worksFor, EX.阿里巴巴))

# SPARQL查询
query = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?age
WHERE {
  ?person foaf:name ?name .
  ?person foaf:age ?age .
}
"""

results = g.query(query)
for row in results:
    print(f"{row.name}, {row.age}岁")
```

### 2. FILTER过滤

```sparql
# 年龄过滤
SELECT ?name ?age
WHERE {
  ?person foaf:name ?name .
  ?person foaf:age ?age .
  FILTER (?age > 30)
}

# 字符串匹配
SELECT ?name
WHERE {
  ?person foaf:name ?name .
  FILTER (REGEX(?name, "张.*"))
}

# 多条件
SELECT ?name ?age
WHERE {
  ?person foaf:name ?name .
  ?person foaf:age ?age .
  FILTER (?age > 25 && ?age < 35)
}
```

### 3. OPTIONAL可选匹配

```sparql
# 查询人员信息，邮箱可选
SELECT ?name ?email
WHERE {
  ?person foaf:name ?name .
  OPTIONAL { ?person foaf:email ?email }
}
```

### 4. UNION联合查询

```sparql
# 查询朋友或同事
SELECT ?person ?relation ?other
WHERE {
  {
    ?person foaf:knows ?other .
    BIND("朋友" AS ?relation)
  }
  UNION
  {
    ?person ex:worksFor ?company .
    ?other ex:worksFor ?company .
    BIND("同事" AS ?relation)
  }
}
```

---

## Python操作RDF

### 1. 使用rdflib

```python
"""RDF基础操作"""

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, FOAF

# 创建图
g = Graph()

# 定义命名空间
EX = Namespace("http://example.org/")
g.bind("ex", EX)
g.bind("foaf", FOAF)

# 添加三元组
g.add((EX.张三, RDF.type, FOAF.Person))
g.add((EX.张三, FOAF.name, Literal("张三", lang="zh")))
g.add((EX.张三, FOAF.age, Literal(30)))
g.add((EX.张三, EX.worksFor, EX.阿里巴巴))

g.add((EX.阿里巴巴, RDF.type, EX.Company))
g.add((EX.阿里巴巴, FOAF.name, Literal("阿里巴巴")))
g.add((EX.阿里巴巴, EX.locatedIn, EX.杭州))

# 序列化为Turtle格式
print(g.serialize(format="turtle"))

# 查询
query = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?person ?company ?city
WHERE {
  ?person ex:worksFor ?company .
  ?company ex:locatedIn ?city .
}
"""

results = g.query(query)
for row in results:
    print(f"{row.person} 在 {row.company} 工作，位于 {row.city}")
```

### 2. 使用pyoxigraph（高性能）

```python
"""使用pyoxigraph进行快速查询"""

from pyoxigraph import Store

# 创建存储
store = Store()

# 添加三元组
store.add((
    "<http://example.org/张三>",
    "<http://xmlns.com/foaf/0.1/name>",
    '"张三"'
))

store.add((
    "<http://example.org/张三>",
    "<http://example.org/worksFor>",
    "<http://example.org/阿里巴巴>"
))

# SPARQL查询
query = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?name
WHERE {
  ?person foaf:name ?name .
}
"""

results = store.query(query)
for result in results:
    print(result)
```

---

## SPARQL vs Cypher对比

### 查询对比

**场景：查询张三的同事**

**SPARQL**：
```sparql
PREFIX ex: <http://example.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?coworkerName
WHERE {
  ex:张三 ex:worksFor ?company .
  ?coworker ex:worksFor ?company .
  ?coworker foaf:name ?coworkerName .
  FILTER (?coworker != ex:张三)
}
```

**Cypher**：
```cypher
MATCH (p:Person {name: '张三'})
      -[:WORKS_AT]->(c:Company)
      <-[:WORKS_AT]-(coworker:Person)
WHERE coworker <> p
RETURN coworker.name
```

**对比**：
- SPARQL：更接近逻辑语言，需要显式声明所有变量
- Cypher：更接近可视化图，模式匹配更直观

### 性能对比

| 操作 | SPARQL | Cypher | 说明 |
|------|--------|--------|------|
| 简单查询 | 中等 | 快 | Cypher优化更好 |
| 复杂推理 | 快 | 慢 | SPARQL支持推理 |
| 路径查询 | 慢 | 快 | Cypher原生支持 |
| 聚合统计 | 快 | 快 | 两者都支持 |

---

## 在RAG中的应用

### 1. 开放数据集成

```python
"""从DBpedia查询知识"""

from SPARQLWrapper import SPARQLWrapper, JSON

def query_dbpedia(entity: str):
    """查询DBpedia"""
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>

    SELECT ?abstract
    WHERE {{
      dbr:{entity} dbo:abstract ?abstract .
      FILTER (lang(?abstract) = 'zh')
    }}
    LIMIT 1
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if results["results"]["bindings"]:
        return results["results"]["bindings"][0]["abstract"]["value"]
    return None

# 使用
abstract = query_dbpedia("Alibaba_Group")
print(abstract)
```

### 2. 语义推理

```python
"""使用RDFS推理"""

from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS

g = Graph()
EX = Namespace("http://example.org/")

# 定义类层次
g.add((EX.Employee, RDFS.subClassOf, EX.Person))
g.add((EX.Manager, RDFS.subClassOf, EX.Employee))

# 添加实例
g.add((EX.张三, RDF.type, EX.Manager))

# 推理查询
query = """
PREFIX ex: <http://example.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?type
WHERE {
  ex:张三 rdf:type ?class .
  ?class rdfs:subClassOf* ?type .
}
"""

results = g.query(query)
for row in results:
    print(row.type)
# 输出：Manager, Employee, Person
```

---

## 何时使用SPARQL？

### 适合SPARQL的场景

1. **开放数据查询**：
   - DBpedia、Wikidata等
   - 需要与外部知识库集成

2. **语义推理**：
   - 需要类层次推理
   - 需要本体（Ontology）支持

3. **学术研究**：
   - 知识图谱研究
   - 语义网应用

### 适合Cypher的场景

1. **应用开发**：
   - 企业知识图谱
   - 推荐系统
   - 社交网络

2. **路径查询**：
   - 多跳关系查询
   - 最短路径
   - 社区发现

3. **高性能需求**：
   - 大规模图遍历
   - 实时查询

---

## 混合使用

```python
"""混合使用SPARQL和Cypher"""

class HybridKnowledgeGraph:
    """混合知识图谱"""

    def __init__(self, neo4j_driver, rdf_graph):
        self.neo4j = neo4j_driver
        self.rdf = rdf_graph

    def query_internal(self, entity: str):
        """查询内部知识（Cypher）"""
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity})-[r]->(related)
                RETURN related.name AS name, type(r) AS relation
            """, entity=entity)
            return [dict(record) for record in result]

    def query_external(self, entity: str):
        """查询外部知识（SPARQL）"""
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?abstract
        WHERE {{
          <http://dbpedia.org/resource/{entity}> dbo:abstract ?abstract .
          FILTER (lang(?abstract) = 'zh')
        }}
        """
        results = self.rdf.query(query)
        return [str(row.abstract) for row in results]

    def hybrid_search(self, entity: str):
        """混合检索"""
        internal = self.query_internal(entity)
        external = self.query_external(entity)

        return {
            "internal_knowledge": internal,
            "external_knowledge": external
        }
```

---

## 总结

### SPARQL的核心价值

1. **标准化**：W3C标准，互操作性强
2. **语义推理**：支持RDFS/OWL推理
3. **开放数据**：与DBpedia、Wikidata等集成

### 选择建议

**选择SPARQL如果**：
- 需要语义推理
- 需要与开放数据集成
- 学术研究项目

**选择Cypher如果**：
- 应用开发
- 需要高性能图遍历
- 需要丰富的工具支持

**混合使用如果**：
- 需要内部和外部知识
- 需要推理和性能兼顾

---

**引用来源**：
- [RDF规范](https://www.w3.org/TR/rdf11-concepts/)
- [SPARQL规范](https://www.w3.org/TR/sparql11-query/)
- [rdflib文档](https://rdflib.readthedocs.io/)
- [pyoxigraph文档](https://pyoxigraph.readthedocs.io/)
- [DBpedia](https://www.dbpedia.org/)

---

**版本**：v1.0
**最后更新**：2026-02-14
**维护者**：Claude Code
