---
type: context7_documentation
library: langchain-openai
version: latest
fetched_at: 2026-02-25
knowledge_point: Embedding模型集成
context7_query: OpenAI embeddings integration, text-embedding-3, dimensions parameter, caching embeddings, batch processing
---

# Context7 文档：LangChain OpenAI Embeddings

## 文档来源
- 库名称：langchain-openai
- Context7 ID：/websites/langchain_oss_python_langchain
- 官方文档链接：https://docs.langchain.com/oss/python/langchain/

## 关键信息提取

### 1. 安装与初始化

**安装**：
```shell
pip install -U "langchain-openai"
```

**初始化**：
```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### 2. 基础使用

**嵌入单个查询**：
```python
vector_1 = embeddings.embed_query(all_splits[0].page_content)
```

**嵌入多个文档**：
```python
vector_2 = embeddings.embed_query(all_splits[1].page_content)
```

**验证向量长度**：
```python
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}")
print(vector_1[:10])
```

**输出示例**：
```text
Generated vectors of length 1536

[-0.008586574345827103, -0.03341241180896759, -0.008936782367527485, -0.0036674530711025, 0.010564599186182022, 0.009598285891115665, -0.028587326407432556, -0.015824200585484505, 0.0030416189692914486, -0.012899317778646946]
```

### 3. 批量处理

**批量请求**：
```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

**并发控制**：
```python
model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # 限制并发数
    }
)
```

### 4. 模型选择

**text-embedding-3-large**：
- 维度：1536（默认）
- 性能：最好
- 成本：较高

**text-embedding-3-small**：
- 维度：512（默认）
- 性能：良好
- 成本：较低

### 5. 与 RAG 集成

**典型流程**：
1. 加载文档
2. 分块（split）
3. 嵌入（embed）
4. 存储到向量数据库
5. 检索

## 与源码分析的对应

1. **API 密钥管理**：与源码中的 `secret_from_env` 对应
2. **模型选择**：与源码中的 `model` 参数对应
3. **批量处理**：与源码中的 `batch()` 方法对应
4. **并发控制**：与源码中的 `max_concurrency` 配置对应

## 最佳实践

1. **环境变量**：使用环境变量管理 API 密钥
2. **模型选择**：根据性能和成本需求选择模型
3. **批量处理**：使用 `batch()` 方法提高效率
4. **并发控制**：设置 `max_concurrency` 避免速率限制
