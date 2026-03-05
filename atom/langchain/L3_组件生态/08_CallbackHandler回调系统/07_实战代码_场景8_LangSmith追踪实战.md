# 实战代码 - 场景8：LangSmith追踪实战

> 演示如何使用 LangSmith 追踪系统实现 LangChain 应用的可观测性

---

## 场景概述

**目标**：使用 LangSmith 官方追踪系统监控 RAG 应用的完整执行流程

**核心技术**：
- `@traceable` 装饰器 - 自动追踪函数执行
- `wrap_openai` - 包装 OpenAI 客户端
- 自定义 metadata 和 tags - 丰富追踪信息
- LangSmith Web UI - 可视化追踪查看

**适用场景**：
- RAG 应用开发调试
- 生产环境性能监控
- LLM 调用成本追踪
- 多步骤流程可视化

---

## 完整代码示例

```python
"""
LangSmith 追踪实战
演示：RAG 应用的完整追踪流程
"""

import os
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== 1. 环境配置 =====
print("=== 1. LangSmith 环境配置 ===")

# 必需的环境变量
os.environ["LANGSMITH_TRACING"] = "true"  # 启用追踪
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_api_key")
os.environ["LANGSMITH_PROJECT"] = "rag-callback-demo"  # 项目名称

print(f"✓ LangSmith 追踪已启用")
print(f"✓ 项目名称: {os.environ['LANGSMITH_PROJECT']}")


# ===== 2. 使用 @traceable 装饰器追踪自定义函数 =====
print("\n=== 2. @traceable 装饰器基础使用 ===")

from langsmith import traceable

@traceable(
    name="document_retriever",  # 追踪中显示的名称
    run_type="retriever",  # 运行类型
    tags=["rag", "retrieval"],  # 标签
    metadata={"version": "1.0"}  # 元数据
)
def retrieve_documents(query: str, top_k: int = 3) -> List[str]:
    """模拟文档检索"""
    # 实际应用中这里会调用向量数据库
    mock_docs = [
        f"文档1：关于 {query} 的详细说明...",
        f"文档2：{query} 的实践案例...",
        f"文档3：{query} 的最佳实践..."
    ]
    return mock_docs[:top_k]

# 测试追踪
query = "LangChain 回调系统"
docs = retrieve_documents(query)
print(f"✓ 检索到 {len(docs)} 个文档")
print(f"✓ 追踪信息已发送到 LangSmith")


# ===== 3. wrap_openai 自动追踪 LLM 调用 =====
print("\n=== 3. wrap_openai 自动追踪 ===")

from openai import OpenAI
from langsmith.wrappers import wrap_openai

# 包装 OpenAI 客户端（自动追踪所有调用）
client = wrap_openai(OpenAI())

def generate_answer(question: str, context: str) -> str:
    """使用 LLM 生成答案"""
    system_message = (
        "你是一个专业的技术助手。根据提供的上下文回答用户问题。\n"
        f"上下文：\n{context}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# 测试 LLM 调用追踪
context = "\n".join(docs)
answer = generate_answer(query, context)
print(f"✓ LLM 生成答案: {answer[:100]}...")
print(f"✓ LLM 调用已自动追踪（包含 token 用量）")


# ===== 4. 嵌套追踪：完整 RAG 流程 =====
print("\n=== 4. 嵌套追踪：完整 RAG 流程 ===")

@traceable(
    name="rag_pipeline",
    run_type="chain",
    tags=["rag", "production"],
    metadata={"pipeline_version": "2.0"}
)
def rag_pipeline(question: str) -> dict:
    """完整的 RAG 流程（自动嵌套追踪）"""

    # 步骤1：检索文档（子追踪）
    documents = retrieve_documents(question, top_k=3)

    # 步骤2：生成答案（子追踪）
    context = "\n".join(documents)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "num_docs": len(documents)
    }

# 执行完整流程
result = rag_pipeline("什么是 CallbackHandler？")
print(f"✓ RAG 流程完成")
print(f"✓ 问题: {result['question']}")
print(f"✓ 答案: {result['answer'][:100]}...")
print(f"✓ 完整追踪链已生成（包含所有子步骤）")


# ===== 5. 流式输出追踪 =====
print("\n=== 5. 流式输出追踪 ===")

def _reduce_chunks(chunks: list) -> dict:
    """聚合流式输出的 chunks"""
    all_text = "".join([
        chunk.choices[0].delta.content or ""
        for chunk in chunks
    ])
    return {"content": all_text}

@traceable(
    name="streaming_llm",
    run_type="llm",
    reduce_fn=_reduce_chunks,  # 聚合函数
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"}
)
def streaming_generate(prompt: str):
    """流式生成（追踪会聚合所有 chunks）"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        yield chunk

# 测试流式追踪
print("流式输出: ", end="", flush=True)
for chunk in streaming_generate("简单介绍 LangSmith"):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n✓ 流式输出已聚合并追踪")


# ===== 6. 自定义 metadata 和 tags =====
print("\n=== 6. 自定义 metadata 和 tags ===")

@traceable(
    tags=["production", "user-query", "high-priority"],
    metadata={
        "user_id": "user_12345",
        "session_id": "session_abc",
        "environment": "production",
        "feature_flag": "new_retrieval_v2"
    }
)
def production_rag(question: str, user_id: str) -> dict:
    """生产环境 RAG（带丰富的追踪信息）"""

    # 检索
    docs = retrieve_documents(question)

    # 生成
    context = "\n".join(docs)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "user_id": user_id,
        "timestamp": "2026-02-25T10:00:00Z"
    }

# 执行生产环境查询
result = production_rag("如何使用 LangSmith？", user_id="user_12345")
print(f"✓ 生产环境查询完成")
print(f"✓ 追踪包含自定义 metadata（user_id, session_id 等）")
print(f"✓ 可在 LangSmith UI 中按 tags 过滤")


# ===== 7. LangChain 集成追踪 =====
print("\n=== 7. LangChain 集成追踪 ===")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain 组件会自动使用 LangSmith 追踪
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的技术助手。"),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 执行链（自动追踪）
response = chain.invoke({"question": "什么是 LCEL？"})
print(f"✓ LangChain 链执行完成")
print(f"✓ 响应: {response[:100]}...")
print(f"✓ 所有组件（Prompt, LLM, Parser）都已自动追踪")


# ===== 8. 错误追踪 =====
print("\n=== 8. 错误追踪 ===")

@traceable(name="error_handling_example")
def function_with_error():
    """演示错误追踪"""
    try:
        # 模拟错误
        result = 1 / 0
    except Exception as e:
        # 错误会自动记录到 LangSmith
        print(f"✗ 捕获错误: {e}")
        raise

try:
    function_with_error()
except:
    print(f"✓ 错误已追踪到 LangSmith（包含堆栈信息）")


# ===== 9. 查看追踪结果 =====
print("\n=== 9. 查看追踪结果 ===")
print("""
访问 LangSmith Web UI 查看追踪：
1. 打开 https://smith.langchain.com/
2. 登录你的账号
3. 选择项目 'rag-callback-demo'
4. 查看所有追踪记录

追踪信息包含：
- 执行时间和延迟
- 输入输出数据
- Token 用量和成本
- 错误和异常
- 自定义 metadata 和 tags
- 嵌套调用关系（父子追踪）
""")


# ===== 10. 最佳实践总结 =====
print("\n=== 10. 最佳实践总结 ===")
print("""
LangSmith 追踪最佳实践：

1. 环境配置
   - 开发环境：LANGSMITH_TRACING=true
   - 生产环境：使用独立的 PROJECT 名称
   - 敏感信息：不要在 metadata 中包含密码等

2. 装饰器使用
   - 为关键函数添加 @traceable
   - 设置有意义的 name 和 run_type
   - 使用 tags 分类（环境、功能、优先级）

3. Metadata 设计
   - 用户信息：user_id, session_id
   - 环境信息：environment, version
   - 业务信息：feature_flag, experiment_id

4. 性能优化
   - 避免追踪高频低价值函数
   - 使用 reduce_fn 聚合流式输出
   - 定期清理旧追踪数据

5. 调试技巧
   - 使用 tags 快速过滤
   - 查看嵌套追踪理解执行流程
   - 对比不同版本的性能差异
""")

print("\n✓ LangSmith 追踪实战完成！")
print(f"✓ 所有追踪已发送到项目: {os.environ['LANGSMITH_PROJECT']}")
```

---

## 运行输出示例

```
=== 1. LangSmith 环境配置 ===
✓ LangSmith 追踪已启用
✓ 项目名称: rag-callback-demo

=== 2. @traceable 装饰器基础使用 ===
✓ 检索到 3 个文档
✓ 追踪信息已发送到 LangSmith

=== 3. wrap_openai 自动追踪 ===
✓ LLM 生成答案: CallbackHandler 是 LangChain 的回调系统核心接口...
✓ LLM 调用已自动追踪（包含 token 用量）

=== 4. 嵌套追踪：完整 RAG 流程 ===
✓ RAG 流程完成
✓ 问题: 什么是 CallbackHandler？
✓ 答案: CallbackHandler 是 LangChain 中用于监控和追踪执行流程的核心机制...
✓ 完整追踪链已生成（包含所有子步骤）

=== 5. 流式输出追踪 ===
流式输出: LangSmith 是 LangChain 的官方可观测性平台，提供追踪、调试和监控功能...
✓ 流式输出已聚合并追踪

=== 6. 自定义 metadata 和 tags ===
✓ 生产环境查询完成
✓ 追踪包含自定义 metadata（user_id, session_id 等）
✓ 可在 LangSmith UI 中按 tags 过滤

=== 7. LangChain 集成追踪 ===
✓ LangChain 链执行完成
✓ 响应: LCEL（LangChain Expression Language）是 LangChain 的声明式组合语法...
✓ 所有组件（Prompt, LLM, Parser）都已自动追踪

=== 8. 错误追踪 ===
✗ 捕获错误: division by zero
✓ 错误已追踪到 LangSmith（包含堆栈信息）

=== 9. 查看追踪结果 ===
访问 LangSmith Web UI 查看追踪：
1. 打开 https://smith.langchain.com/
2. 登录你的账号
3. 选择项目 'rag-callback-demo'
4. 查看所有追踪记录

=== 10. 最佳实践总结 ===
✓ LangSmith 追踪实战完成！
✓ 所有追踪已发送到项目: rag-callback-demo
```

---

## 代码说明

### 1. 环境配置

```python
os.environ["LANGSMITH_TRACING"] = "true"  # 启用追踪
os.environ["LANGSMITH_API_KEY"] = "your_api_key"
os.environ["LANGSMITH_PROJECT"] = "rag-callback-demo"
```

**关键点**：
- `LANGSMITH_TRACING` 必须设置为 `"true"`
- `LANGSMITH_API_KEY` 从 LangSmith 网站获取
- `LANGSMITH_PROJECT` 用于组织不同的追踪

### 2. @traceable 装饰器

```python
@traceable(
    name="document_retriever",  # 追踪中显示的名称
    run_type="retriever",  # 运行类型
    tags=["rag", "retrieval"],  # 标签
    metadata={"version": "1.0"}  # 元数据
)
def retrieve_documents(query: str, top_k: int = 3) -> List[str]:
    ...
```

**参数说明**：
- `name`: 追踪中显示的函数名称
- `run_type`: 运行类型（llm, chain, tool, retriever）
- `tags`: 标签列表（用于过滤和分类）
- `metadata`: 自定义元数据（任意键值对）

### 3. wrap_openai 包装

```python
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())
```

**作用**：
- 自动追踪所有 OpenAI API 调用
- 记录输入、输出和 token 用量
- 无需修改现有代码逻辑

### 4. 嵌套追踪

```python
@traceable(name="rag_pipeline")
def rag_pipeline(question: str) -> dict:
    documents = retrieve_documents(question)  # 子追踪
    answer = generate_answer(question, context)  # 子追踪
    return {"answer": answer}
```

**特性**：
- 自动建立父子追踪关系
- 可视化完整执行流程
- 计算每个步骤的耗时

### 5. 流式输出追踪

```python
@traceable(
    run_type="llm",
    reduce_fn=_reduce_chunks  # 聚合函数
)
def streaming_generate(prompt: str):
    for chunk in stream:
        yield chunk
```

**关键点**：
- `reduce_fn` 用于聚合流式输出
- 追踪会记录完整的输出内容
- 不会丢失中间的 chunks

---

## 在 RAG 开发中的应用

### 1. 调试 RAG 流程

**场景**：检索结果不准确

```python
@traceable(name="debug_retrieval")
def debug_retrieval(query: str):
    # 追踪会记录：
    # - 查询向量化结果
    # - 检索到的文档
    # - 相似度分数
    ...
```

**优势**：
- 可视化检索过程
- 对比不同查询的结果
- 发现检索策略问题

### 2. 性能监控

**场景**：生产环境性能优化

```python
@traceable(
    tags=["production", "performance"],
    metadata={"region": "us-west-1"}
)
def production_rag(query: str):
    ...
```

**监控指标**：
- 端到端延迟
- LLM 调用次数
- Token 用量和成本
- 错误率

### 3. A/B 测试

**场景**：对比不同的 RAG 策略

```python
@traceable(
    tags=["experiment"],
    metadata={"variant": "strategy_a"}
)
def rag_strategy_a(query: str):
    ...

@traceable(
    tags=["experiment"],
    metadata={"variant": "strategy_b"}
)
def rag_strategy_b(query: str):
    ...
```

**对比维度**：
- 答案质量
- 响应速度
- 成本效益

---

## 常见问题

### Q1: 追踪数据会影响性能吗？

**A**: 影响很小（< 5ms）
- 追踪是异步发送的
- 不会阻塞主流程
- 可以通过环境变量关闭

### Q2: 如何避免追踪敏感信息？

**A**: 使用过滤器

```python
@traceable(
    metadata={
        "user_id": user_id,  # ✓ 可以追踪
        # "password": password  # ✗ 不要追踪
    }
)
def secure_function(user_id: str, password: str):
    ...
```

### Q3: 追踪数据保存多久？

**A**: 根据 LangSmith 计划
- 免费版：7 天
- 付费版：可自定义（30-90 天）

### Q4: 可以在本地查看追踪吗？

**A**: 不可以
- LangSmith 是云服务
- 需要网络连接
- 可以使用 AgentReplay 等本地工具

---

## 与其他追踪工具对比

| 特性 | LangSmith | Langfuse | Phoenix |
|------|-----------|----------|---------|
| 官方支持 | ✓ | ✗ | ✗ |
| 自动追踪 | ✓ | ✓ | ✓ |
| 开源 | ✗ | ✓ | ✓ |
| 本地部署 | ✗ | ✓ | ✓ |
| 成本 | 付费 | 免费/付费 | 免费 |
| 易用性 | ★★★★★ | ★★★★☆ | ★★★☆☆ |

**选择建议**：
- 官方支持 + 易用性 → LangSmith
- 开源 + 本地部署 → Langfuse
- 完全免费 → Phoenix

---

## 学习检查

完成以下任务，确保掌握 LangSmith 追踪：

- [ ] 配置 LangSmith 环境变量
- [ ] 使用 @traceable 装饰器追踪自定义函数
- [ ] 使用 wrap_openai 追踪 LLM 调用
- [ ] 实现嵌套追踪（父子关系）
- [ ] 追踪流式输出
- [ ] 添加自定义 metadata 和 tags
- [ ] 在 LangSmith UI 中查看追踪结果
- [ ] 使用 tags 过滤追踪记录
- [ ] 对比不同版本的性能差异

---

## 下一步

- 学习 **Langfuse 集成**（开源替代方案）
- 探索 **自定义回调处理器**（更灵活的追踪）
- 实践 **生产环境监控**（告警和仪表盘）
- 研究 **成本优化策略**（基于追踪数据）

---

**相关文档**：
- [LangSmith 官方文档](https://docs.smith.langchain.com/)
- [LangChain 可观测性指南](https://python.langchain.com/docs/guides/observability)
- [CallbackHandler 核心概念](./03_核心概念_7_LangSmith追踪系统.md)
