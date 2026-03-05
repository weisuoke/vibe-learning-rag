---
type: context7_documentation
library: Chroma
library_id: /chroma-core/chroma
fetched_at: 2026-02-25
knowledge_point: 06_VectorStore后端选择
context7_query: Chroma vector database for LangChain
---

# Context7 文档：Chroma 官方文档

## 文档来源
- 库名称：Chroma
- Library ID：/chroma-core/chroma
- 官方文档链接：https://github.com/chroma-core/chroma
- Source Reputation：High
- Benchmark Score：79.9

## 关键信息提取

### 1. Collection 查询操作

#### 基础查询

```python
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("articles")

# 添加示例数据
collection.add(
    ids=["a1", "a2", "a3", "a4"],
    documents=[
        "Quantum computing advances in 2024",
        "New discoveries in renewable energy",
        "AI models break performance records",
        "Climate change impacts on agriculture"
    ],
    metadatas=[
        {"category": "technology", "year": 2024},
        {"category": "energy", "year": 2024},
        {"category": "technology", "year": 2024},
        {"category": "environment", "year": 2023}
    ]
)

# 基础文本查询
results = collection.query(
    query_texts=["latest AI developments"],
    n_results=3
)
print(results["documents"])
print(results["distances"])
```

**关键点**：
- 使用 `query_texts` 进行文本查询
- `n_results` 指定返回结果数量
- 返回文档内容和距离分数

#### 元数据过滤查询

```python
# 单条件过滤
results = collection.query(
    query_texts=["technology news"],
    n_results=5,
    where={"category": "technology"}
)

# 多条件过滤（$and）
results = collection.query(
    query_texts=["recent advances"],
    n_results=10,
    where={
        "$and": [
            {"category": "technology"},
            {"year": {"$gte": 2024}}
        ]
    }
)

# 文档内容过滤
results = collection.query(
    query_texts=["scientific breakthroughs"],
    n_results=5,
    where_document={"$contains": "2024"}
)
```

**关键点**：
- `where` 参数用于元数据过滤
- 支持布尔运算符（`$and`, `$or`）
- 支持比较运算符（`$gte`, `$lte`, `$eq`）
- `where_document` 用于文档内容过滤

### 2. Embedding 函数集成

#### Cohere Embedding

```typescript
import { ChromaClient } from 'chromadb';
import { CohereEmbeddingFunction } from '@chroma-core/cohere';

// 初始化 embedder
const embedder = new CohereEmbeddingFunction({
  apiKey: 'your-api-key',  // 或设置 COHERE_API_KEY 环境变量
  modelName: 'embed-english-v3.0',  // 可选，默认为 'embed-english-v3.0'
});

// 创建 ChromaClient
const client = new ChromaClient({
  path: 'http://localhost:8000',
});

// 创建 collection 并指定 embedder
const collection = await client.createCollection({
  name: 'my-collection',
  embeddingFunction: embedder,
});

// 添加文档
await collection.add({
  ids: ["1", "2", "3"],
  documents: ["Document 1", "Document 2", "Document 3"],
});

// 查询文档
const results = await collection.query({
  queryTexts: ["Sample query"],
  nResults: 2,
});
```

**关键点**：
- 支持多种 embedding 函数（Cohere, OpenAI, Ollama, Mistral）
- 可以在创建 collection 时指定 embedding 函数
- 支持自定义 embedding 函数

#### OpenAI 和 Cohere Embedding 配置

```typescript
import { OpenAIEmbeddingFunction } from "@chroma-core/openai";
import { CohereEmbeddingFunction } from "@chroma-core/cohere";

// 方式1：通过 embedding_function 参数
const openAICollection = await client.createCollection({
  name: "my_openai_collection",
  embedding_function: new OpenAIEmbeddingFunction({
    model_name: "text-embedding-3-small",
  }),
  configuration: { hnsw: { space: "cosine" } },
});

// 方式2：在 configuration 中设置
const cohereCollection = await client.getOrCreateCollection({
  name: "my_cohere_collection",
  configuration: {
    embeddingFunction: new CohereEmbeddingFunction({
      modelName: "embed-english-light-v2.0",
      truncate: "NONE",
    }),
    hnsw: { space: "cosine" },
  },
});
```

**关键点**：
- 两种方式指定 embedding 函数
- 支持 HNSW 配置（space: cosine, l2, ip）
- 支持模型参数配置

#### Ollama Embedding（本地模型）

```typescript
import { ChromaClient } from 'chromadb';
import { OllamaEmbeddingFunction } from '@chroma-core/ollama';

// 初始化 Ollama embedder
const embedder = new OllamaEmbeddingFunction({
  url: 'http://localhost:11434',  // 默认 Ollama 服务器 URL
  model: 'chroma/all-minilm-l6-v2-f32',  // 默认模型
});

// 创建 ChromaClient
const client = new ChromaClient({
  path: 'http://localhost:8000',
});

// 创建 collection
const collection = await client.createCollection({
  name: 'my-collection',
  embeddingFunction: embedder,
});

// 添加文档
await collection.add({
  ids: ["1", "2", "3"],
  documents: ["Document 1", "Document 2", "Document 3"],
});

// 查询文档
const results = await collection.query({
  queryTexts: ["Sample query"],
  nResults: 2,
});
```

**关键点**：
- 支持本地 Ollama 模型
- 无需 API key
- 适合离线或隐私敏感场景

#### Mistral Embedding

```typescript
import { ChromaClient } from 'chromadb';
import { MistralEmbeddingFunction } from '@chroma-core/mistral';

// 初始化 Mistral embedder
const embedder = new MistralEmbeddingFunction({
  apiKey: 'your-api-key',  // 或设置 MISTRAL_API_KEY 环境变量
  model: 'mistral-embed',
});

// 创建 ChromaClient
const client = new ChromaClient({
  path: 'http://localhost:8000',
});

// 创建 collection
const collection = await client.createCollection({
  name: 'my-collection',
  embeddingFunction: embedder,
});

// 添加文档
await collection.add({
  ids: ["1", "2", "3"],
  documents: ["Document 1", "Document 2", "Document 3"],
});

// 查询文档
const results = await collection.query({
  queryTexts: ["Sample query"],
  nResults: 2,
});
```

**关键点**：
- 支持 Mistral AI embedding 模型
- 需要 API key
- 集成方式与其他 embedding 函数一致

## 支持的 Embedding 函数

| Embedding 函数 | 提供商 | 需要 API Key | 适用场景 |
|---------------|--------|-------------|---------|
| OpenAI | OpenAI | ✅ | 高质量文本 embedding |
| Cohere | Cohere | ✅ | 多语言支持 |
| Ollama | 本地 | ❌ | 离线/隐私敏感 |
| Mistral | Mistral AI | ✅ | 欧洲数据合规 |

## 查询过滤运算符

| 运算符 | 说明 | 示例 |
|--------|------|------|
| `$eq` | 等于 | `{"category": {"$eq": "tech"}}` |
| `$ne` | 不等于 | `{"year": {"$ne": 2023}}` |
| `$gt` | 大于 | `{"year": {"$gt": 2023}}` |
| `$gte` | 大于等于 | `{"year": {"$gte": 2024}}` |
| `$lt` | 小于 | `{"year": {"$lt": 2025}}` |
| `$lte` | 小于等于 | `{"year": {"$lte": 2024}}` |
| `$and` | 逻辑与 | `{"$and": [{...}, {...}]}` |
| `$or` | 逻辑或 | `{"$or": [{...}, {...}]}` |
| `$contains` | 包含（文档内容） | `{"$contains": "keyword"}` |

## 与 LangChain 的集成要点

1. **包依赖**：`langchain-chroma`, `chromadb`
2. **初始化方式**：
   - `Chroma.from_texts()` - 从文本列表创建
   - `Chroma.from_documents()` - 从文档列表创建
3. **持久化**：
   - `persist_directory` 参数指定持久化目录
   - 数据保存到本地文件系统
4. **Embedding 函数**：
   - 支持多种 embedding 提供商
   - 可以自定义 embedding 函数
5. **查询功能**：
   - 支持元数据过滤
   - 支持文档内容过滤
   - 支持 top-k 检索

## 适用场景

- ✅ 本地开发和测试
- ✅ 原型验证
- ✅ 中小规模应用（< 100K 文档）
- ✅ 需要持久化的场景
- ✅ 需要灵活 embedding 函数选择
- ❌ 大规模生产环境（> 100K 文档）
- ❌ 高并发场景
- ❌ 分布式部署
