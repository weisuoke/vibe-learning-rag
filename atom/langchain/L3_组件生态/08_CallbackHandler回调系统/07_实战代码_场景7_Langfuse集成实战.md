# 实战代码 - 场景7：Langfuse集成实战

## 场景说明

**应用场景**：使用 Langfuse 开源可观测性平台追踪 LangChain 应用，实现 Prompt 版本控制、追踪管理和评估集成，适用于需要完整可观测性的生产环境。

**核心价值**：
- 完整追踪链路，可视化 LLM 调用
- Prompt 版本控制，支持 A/B 测试
- 评估集成，量化系统性能
- 开源免费，支持自部署

**RAG 开发价值**：
- RAG 系统涉及多个环节，需要完整追踪
- Prompt 版本控制帮助优化检索和生成策略
- 评估集成支持 RAG 系统质量监控
- 支持多租户和团队协作

---

## 完整可运行代码

### 方案1：基础 Langfuse 集成

```python
"""
Langfuse 基础集成
演示：使用 Langfuse CallbackHandler 追踪 LangChain 应用
适用场景：快速集成 Langfuse 可观测性
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Langfuse 集成
from langfuse.callback import CallbackHandler

load_dotenv()

# ===== 1. 初始化 Langfuse CallbackHandler =====
print("=== Langfuse 基础集成测试 ===\n")

# 配置 Langfuse（从环境变量读取）
# LANGFUSE_PUBLIC_KEY=pk-lf-***
# LANGFUSE_SECRET_KEY=sk-lf-***
# LANGFUSE_HOST=https://cloud.langfuse.com

langfuse_handler = CallbackHandler(
    # 可选：自定义配置
    # public_key="pk-lf-***",
    # secret_key="sk-lf-***",
    # host="https://cloud.langfuse.com"
)

# 验证连接
print("验证 Langfuse 连接...")
langfuse_handler.auth_check()
print("✓ Langfuse 连接成功\n")


# ===== 2. 简单 LLM 调用追踪 =====
print("=== 测试1：简单 LLM 调用 ===")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 传入 Langfuse 回调
response = llm.invoke(
    [HumanMessage(content="什么是 RAG？")],
    config={"callbacks": [langfuse_handler]}
)

print(f"回答: {response.content[:80]}...")
print(f"✓ 追踪已发送到 Langfuse\n")


# ===== 3. Chain 追踪 =====
print("=== 测试2：Chain 追踪 ===")

# 创建 Chain
prompt = ChatPromptTemplate.from_template("简单解释：{topic}")
chain = prompt | llm | StrOutputParser()

# 执行 Chain（自动追踪）
result = chain.invoke(
    {"topic": "向量数据库"},
    config={"callbacks": [langfuse_handler]}
)

print(f"回答: {result[:80]}...")
print(f"✓ Chain 追踪已发送到 Langfuse\n")


# ===== 4. 多次调用追踪（同一 Trace） =====
print("=== 测试3：多次调用追踪 ===")

from langfuse import Langfuse

# 创建 Langfuse 客户端
langfuse = Langfuse()

# 创建一个 Trace
trace = langfuse.trace(
    name="multi_call_example",
    user_id="user-123",
    metadata={"environment": "development"}
)

# 获取该 Trace 的 CallbackHandler
trace_handler = trace.get_langchain_handler()

# 第一次调用
print("第一次调用...")
response1 = llm.invoke(
    [HumanMessage(content="什么是 Embedding？")],
    config={"callbacks": [trace_handler]}
)
print(f"回答1: {response1.content[:60]}...")

# 第二次调用（同一 Trace）
print("第二次调用...")
response2 = llm.invoke(
    [HumanMessage(content="Embedding 有什么用？")],
    config={"callbacks": [trace_handler]}
)
print(f"回答2: {response2.content[:60]}...")

print(f"✓ 两次调用已记录到同一 Trace\n")
print(f"Trace ID: {trace.id}")


# ===== 5. 添加自定义标签和元数据 =====
print("\n=== 测试4：自定义标签和元数据 ===")

# 创建带标签的 Trace
trace_with_tags = langfuse.trace(
    name="rag_query",
    user_id="user-456",
    tags=["rag", "production", "gpt-4o-mini"],
    metadata={
        "query_type": "semantic_search",
        "collection": "docs",
        "top_k": 5
    }
)

handler_with_tags = trace_with_tags.get_langchain_handler()

response = llm.invoke(
    [HumanMessage(content="如何优化 RAG 检索性能？")],
    config={"callbacks": [handler_with_tags]}
)

print(f"回答: {response.content[:80]}...")
print(f"✓ 带标签和元数据的追踪已发送\n")


# ===== 6. 获取 Trace URL =====
print("=== Trace 链接 ===")
print(f"查看追踪详情：")
print(f"Trace 1: https://cloud.langfuse.com/trace/{trace.id}")
print(f"Trace 2: https://cloud.langfuse.com/trace/{trace_with_tags.id}")
```

---

### 方案2：Prompt 版本控制 + 评估集成

```python
"""
Langfuse Prompt 版本控制 + 评估集成
演示：使用 Langfuse 管理 Prompt 版本和添加评估分数
适用场景：生产环境的 Prompt 管理和质量监控
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

load_dotenv()


# ===== 1. Prompt 版本控制 =====
print("=== Langfuse Prompt 版本控制 ===\n")

# 初始化 Langfuse 客户端
langfuse = Langfuse()

# 创建 Prompt 模板（首次）
print("创建 Prompt 模板...")
prompt_template = """你是一个 RAG 系统专家。

用户问题：{question}

请提供简洁、准确的回答。"""

# 推送 Prompt 到 Langfuse
langfuse.create_prompt(
    name="rag_expert_prompt",
    prompt=prompt_template,
    is_active=True,  # 设为生产版本
    labels=["rag", "expert", "v1"]
)

print("✓ Prompt 已推送到 Langfuse\n")


# ===== 2. 从 Langfuse 获取 Prompt =====
print("=== 从 Langfuse 获取 Prompt ===\n")

# 获取生产版本的 Prompt（带缓存）
langfuse_prompt = langfuse.get_prompt(
    "rag_expert_prompt",
    cache_ttl_seconds=300  # 缓存 5 分钟
)

print(f"Prompt 名称: {langfuse_prompt.name}")
print(f"Prompt 版本: {langfuse_prompt.version}")
print(f"Prompt 内容:\n{langfuse_prompt.prompt}\n")


# ===== 3. 使用 Langfuse Prompt 执行 Chain =====
print("=== 使用 Langfuse Prompt 执行 Chain ===\n")

# 创建 LangChain Prompt（使用 Langfuse 的 Prompt）
lc_prompt = ChatPromptTemplate.from_template(langfuse_prompt.prompt)

# 创建 Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = lc_prompt | llm | StrOutputParser()

# 创建 Trace
trace = langfuse.trace(
    name="rag_query_with_prompt_version",
    user_id="user-789",
    metadata={
        "prompt_name": langfuse_prompt.name,
        "prompt_version": langfuse_prompt.version
    }
)

handler = trace.get_langchain_handler()

# 执行 Chain
question = "如何选择合适的 Embedding 模型？"
print(f"问题: {question}")

result = chain.invoke(
    {"question": question},
    config={"callbacks": [handler]}
)

print(f"回答: {result[:100]}...\n")


# ===== 4. 添加评估分数 =====
print("=== 添加评估分数 ===\n")

# 获取 Trace ID
trace_id = trace.id

# 添加多个评估分数
print("添加评估分数...")

# 分数1：回答质量（人工评估）
langfuse.score(
    trace_id=trace_id,
    name="answer_quality",
    value=0.9,  # 0-1 之间
    comment="回答准确且全面"
)

# 分数2：相关性（自动评估）
langfuse.score(
    trace_id=trace_id,
    name="relevance",
    value=0.95,
    comment="高度相关"
)

# 分数3：用户反馈
langfuse.score(
    trace_id=trace_id,
    name="user_feedback",
    value=1,  # 1 = 满意, 0 = 不满意
    comment="用户点赞"
)

print("✓ 评估分数已添加\n")


# ===== 5. Prompt A/B 测试 =====
print("=== Prompt A/B 测试 ===\n")

# 创建 Prompt 版本 B
prompt_template_v2 = """你是一个 RAG 系统专家，擅长用简单语言解释复杂概念。

用户问题：{question}

请用通俗易懂的语言回答，并举例说明。"""

# 推送版本 B（不设为生产版本）
langfuse.create_prompt(
    name="rag_expert_prompt_v2",
    prompt=prompt_template_v2,
    is_active=False,  # 测试版本
    labels=["rag", "expert", "v2", "ab_test"]
)

print("✓ Prompt 版本 B 已创建\n")

# 模拟 A/B 测试
import random

def run_ab_test(question: str, user_id: str):
    """运行 A/B 测试"""
    # 随机选择版本
    use_v2 = random.random() < 0.5

    if use_v2:
        prompt_name = "rag_expert_prompt_v2"
        version_label = "B"
    else:
        prompt_name = "rag_expert_prompt"
        version_label = "A"

    # 获取 Prompt
    prompt = langfuse.get_prompt(prompt_name)

    # 创建 Trace
    trace = langfuse.trace(
        name="ab_test_query",
        user_id=user_id,
        tags=["ab_test", f"version_{version_label}"],
        metadata={
            "prompt_version": version_label,
            "prompt_name": prompt_name
        }
    )

    # 执行 Chain
    lc_prompt = ChatPromptTemplate.from_template(prompt.prompt)
    chain = lc_prompt | llm | StrOutputParser()

    result = chain.invoke(
        {"question": question},
        config={"callbacks": [trace.get_langchain_handler()]}
    )

    print(f"版本 {version_label}: {result[:80]}...")
    return trace.id, version_label

# 运行 A/B 测试
print("运行 A/B 测试...")
trace_id_1, version_1 = run_ab_test("什么是向量数据库？", "user-001")
trace_id_2, version_2 = run_ab_test("什么是向量数据库？", "user-002")

print(f"\n✓ A/B 测试完成")
print(f"Trace 1 (版本 {version_1}): {trace_id_1}")
print(f"Trace 2 (版本 {version_2}): {trace_id_2}")


# ===== 6. 查看统计信息 =====
print("\n=== 查看 Langfuse Dashboard ===")
print(f"访问 https://cloud.langfuse.com 查看：")
print(f"- 所有 Traces 和调用详情")
print(f"- Prompt 版本历史")
print(f"- 评估分数统计")
print(f"- A/B 测试结果对比")
```

---

## 运行输出示例

### 方案1 输出：

```
=== Langfuse 基础集成测试 ===

验证 Langfuse 连接...
✓ Langfuse 连接成功

=== 测试1：简单 LLM 调用 ===
回答: RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的 AI 架构...
✓ 追踪已发送到 Langfuse

=== 测试2：Chain 追踪 ===
回答: 向量数据库是专门用于存储和检索高维向量的数据库系统...
✓ Chain 追踪已发送到 Langfuse

=== 测试3：多次调用追踪 ===
第一次调用...
回答1: Embedding 是将文本转换为稠密向量的技术...
第二次调用...
回答2: Embedding 主要用于语义相似度计算和检索...
✓ 两次调用已记录到同一 Trace

Trace ID: 550e8400-e29b-41d4-a716-446655440000

=== 测试4：自定义标签和元数据 ===
回答: 优化 RAG 检索性能的策略包括：1. 使用混合检索...
✓ 带标签和元数据的追踪已发送

=== Trace 链接 ===
查看追踪详情：
Trace 1: https://cloud.langfuse.com/trace/550e8400-e29b-41d4-a716-446655440000
Trace 2: https://cloud.langfuse.com/trace/660e8400-e29b-41d4-a716-446655440001
```

### 方案2 输出：

```
=== Langfuse Prompt 版本控制 ===

创建 Prompt 模板...
✓ Prompt 已推送到 Langfuse

=== 从 Langfuse 获取 Prompt ===

Prompt 名称: rag_expert_prompt
Prompt 版本: 1
Prompt 内容:
你是一个 RAG 系统专家。

用户问题：{question}

请提供简洁、准确的回答。

=== 使用 Langfuse Prompt 执行 Chain ===

问题: 如何选择合适的 Embedding 模型？
回答: 选择 Embedding 模型需要考虑以下因素：1. 向量维度（影响性能和精度）...

=== 添加评估分数 ===

添加评估分数...
✓ 评估分数已添加

=== Prompt A/B 测试 ===

✓ Prompt 版本 B 已创建

运行 A/B 测试...
版本 A: 向量数据库是专门用于存储和检索高维向量的数据库系统...
版本 B: 向量数据库就像一个专门存储"数字指纹"的仓库...

✓ A/B 测试完成
Trace 1 (版本 A): 770e8400-e29b-41d4-a716-446655440000
Trace 2 (版本 B): 880e8400-e29b-41d4-a716-446655440001

=== 查看 Langfuse Dashboard ===
访问 https://cloud.langfuse.com 查看：
- 所有 Traces 和调用详情
- Prompt 版本历史
- 评估分数统计
- A/B 测试结果对比
```

---

## 关键技术点

### 1. Langfuse CallbackHandler 初始化

```python
from langfuse.callback import CallbackHandler

# 方式1：从环境变量读取
handler = CallbackHandler()

# 方式2：显式配置
handler = CallbackHandler(
    public_key="pk-lf-***",
    secret_key="sk-lf-***",
    host="https://cloud.langfuse.com"
)
```

### 2. Trace 管理

```python
# 创建 Trace
trace = langfuse.trace(
    name="my_trace",
    user_id="user-123",
    tags=["production"],
    metadata={"key": "value"}
)

# 获取 Trace 的 CallbackHandler
handler = trace.get_langchain_handler()
```

### 3. Prompt 版本控制

```python
# 创建 Prompt
langfuse.create_prompt(
    name="my_prompt",
    prompt="template",
    is_active=True  # 设为生产版本
)

# 获取 Prompt（带缓存）
prompt = langfuse.get_prompt(
    "my_prompt",
    cache_ttl_seconds=300
)
```

### 4. 评估分数

```python
# 添加分数
langfuse.score(
    trace_id=trace_id,
    name="quality",
    value=0.9,
    comment="excellent"
)
```

---

## 与 RAG 开发的联系

### 1. RAG 系统追踪

```python
# 追踪完整 RAG 流程
trace = langfuse.trace(
    name="rag_query",
    metadata={
        "query": "用户问题",
        "retrieved_docs": 5,
        "rerank": True
    }
)

# 检索阶段
retrieval_span = trace.span(name="retrieval")
# ... 检索逻辑

# 生成阶段
generation_span = trace.span(name="generation")
# ... 生成逻辑
```

### 2. Prompt 优化

```python
# A/B 测试不同的 RAG Prompt
# 版本 A：简洁风格
# 版本 B：详细风格

# 通过评估分数对比效果
```

### 3. 质量监控

```python
# 自动评估 RAG 输出质量
langfuse.score(
    trace_id=trace_id,
    name="groundedness",  # 基于事实
    value=calculate_groundedness(answer, docs)
)

langfuse.score(
    trace_id=trace_id,
    name="relevance",  # 相关性
    value=calculate_relevance(answer, query)
)
```

---

## 常见问题

### 问题1：如何在生产环境使用 Langfuse？

**最佳实践**：
```python
# 1. 使用环境变量配置
# .env 文件
LANGFUSE_PUBLIC_KEY=pk-lf-***
LANGFUSE_SECRET_KEY=sk-lf-***
LANGFUSE_HOST=https://cloud.langfuse.com

# 2. 每个请求新建 Trace
def handle_request(user_id, query):
    trace = langfuse.trace(
        name="user_query",
        user_id=user_id,
        metadata={"query": query}
    )
    handler = trace.get_langchain_handler()

    # 执行 RAG 流程
    result = rag_chain.invoke(
        {"query": query},
        config={"callbacks": [handler]}
    )

    return result

# 3. 异步刷新（避免阻塞）
langfuse.flush()  # 确保数据发送
```

### 问题2：Langfuse vs LangSmith 如何选择？

**对比**：

| 特性 | Langfuse | LangSmith |
|------|----------|-----------|
| 开源 | ✅ 是 | ❌ 否 |
| 自部署 | ✅ 支持 | ❌ 仅云服务 |
| 价格 | 免费/自定义 | 付费 |
| Prompt 管理 | ✅ 支持 | ✅ 支持 |
| 评估集成 | ✅ 支持 | ✅ 支持 |
| 社区 | 活跃 | 官方支持 |

**选择建议**：
- 需要自部署 → Langfuse
- 需要官方支持 → LangSmith
- 预算有限 → Langfuse
- 深度集成 LangChain → LangSmith

---

## 总结

Langfuse 是一个强大的开源可观测性平台，通过 CallbackHandler 可以轻松集成到 LangChain 应用中。它提供了完整的追踪、Prompt 版本控制和评估功能，特别适合需要自部署和成本控制的生产环境。
