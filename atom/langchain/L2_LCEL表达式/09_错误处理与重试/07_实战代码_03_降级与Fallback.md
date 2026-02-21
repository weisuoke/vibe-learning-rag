# 实战代码3：降级与Fallback

## 概述

本文提供降级与Fallback的完整可运行示例，涵盖：
- 模型降级策略
- 多级fallback链
- 成本优化降级
- 优雅降级实践

所有代码都可以直接复制运行。

---

## 示例1：基础模型降级

**场景：** GPT-4 失败时自动切换到 GPT-3.5

```python
"""
基础模型降级
演示：with_fallbacks() 的基础用法
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("=== 基础模型降级 ===\n")

# ===== 1. 创建主模型和备选模型 =====
primary_llm = ChatOpenAI(model="gpt-4")
fallback_llm = ChatOpenAI(model="gpt-3.5-turbo")

print("主模型: GPT-4")
print("备选模型: GPT-3.5-turbo\n")

# ===== 2. 添加降级 =====
llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

print("✅ 降级配置完成\n")

# ===== 3. 测试 =====
test_prompts = [
    "什么是 Python？",
    "什么是 JavaScript？",
    "什么是 Go？"
]

print(f"处理 {len(test_prompts)} 个请求...\n")

for i, prompt in enumerate(test_prompts):
    print(f"[{i+1}/{len(test_prompts)}] {prompt}")
    try:
        response = llm_with_fallback.invoke(prompt)
        print(f"✅ 成功: {response.content[:50]}...\n")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}\n")

# ===== 4. 执行流程说明 =====
print("=== 执行流程 ===")
print("1. 尝试 GPT-4")
print("2. 如果 GPT-4 失败，自动切换到 GPT-3.5")
print("3. 如果 GPT-3.5 也失败，抛出异常")
```

---

## 示例2：多级降级链

**场景：** GPT-4 → GPT-3.5 → GPT-3.5-mini 三级降级

```python
"""
多级降级链
演示：3层降级策略
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("=== 多级降级链 ===\n")

# ===== 1. 创建三级模型 =====
# 第1级：GPT-4（最好，最贵）
primary = ChatOpenAI(model="gpt-4")
print("第1级: GPT-4 ($0.03/1K tokens)")

# 第2级：GPT-3.5-turbo（平衡）
fallback1 = ChatOpenAI(model="gpt-3.5-turbo")
print("第2级: GPT-3.5-turbo ($0.002/1K tokens)")

# 第3级：GPT-3.5-turbo-mini（保底）
fallback2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("第3级: GPT-3.5-turbo (temperature=0, 保底)\n")

# ===== 2. 构建降级链 =====
llm = primary.with_fallbacks([fallback1, fallback2])

print("✅ 三级降级链构建完成\n")

# ===== 3. 测试 =====
test_cases = [
    ("简单问题", "你好"),
    ("中等问题", "解释一下什么是机器学习"),
    ("复杂问题", "详细分析深度学习的发展历史和未来趋势")
]

for name, prompt in test_cases:
    print(f"--- {name} ---")
    print(f"Prompt: {prompt[:30]}...")
    try:
        response = llm.invoke(prompt)
        print(f"✅ 成功")
        print(f"响应长度: {len(response.content)} 字符\n")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}\n")

# ===== 4. 降级流程 =====
print("=== 降级流程 ===")
print("GPT-4 失败")
print("  ↓")
print("GPT-3.5-turbo")
print("  ↓ 仍失败")
print("GPT-3.5-turbo (temperature=0)")
print("  ↓ 仍失败")
print("抛出异常")
```

---

## 示例3：成本优化降级

**场景：** 先试便宜的模型，失败再用贵的

```python
"""
成本优化降级
演示：从便宜到贵的降级策略
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

print("=== 成本优化降级 ===\n")

# ===== 1. 创建模型（从便宜到贵）=====
# 第1级：GPT-3.5-turbo（便宜，先试）
cheap_llm = ChatOpenAI(model="gpt-3.5-turbo").with_retry(stop_after_attempt=2)
print("第1级: GPT-3.5-turbo ($0.002/1K tokens)")

# 第2级：GPT-4（贵，备用）
expensive_llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)
print("第2级: GPT-4 ($0.03/1K tokens)\n")

# ===== 2. 构建成本优化链 =====
llm = cheap_llm.with_fallbacks([expensive_llm])

print("✅ 成本优化链构建完成\n")

# ===== 3. 批量处理测试 =====
prompts = [
    "什么是 Python？",
    "什么是 JavaScript？",
    "什么是 Go？",
    "什么是 Rust？",
    "什么是 Java？"
]

print(f"批量处理 {len(prompts)} 个请求...\n")

start_time = time.time()
results = []
cost_stats = {"cheap": 0, "expensive": 0, "failed": 0}

for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] {prompt}")
    try:
        response = llm.invoke(prompt)
        results.append(response.content)
        # 简化版：实际需要追踪哪个模型被使用
        cost_stats["cheap"] += 1
        print(f"✅ 成功\n")
    except Exception as e:
        results.append(None)
        cost_stats["failed"] += 1
        print(f"❌ 失败: {type(e).__name__}\n")

elapsed_time = time.time() - start_time

# ===== 4. 成本分析 =====
print("=== 成本分析 ===")
print(f"总请求: {len(prompts)}")
print(f"成功: {len([r for r in results if r])}")
print(f"失败: {cost_stats['failed']}")
print(f"总耗时: {elapsed_time:.2f} 秒\n")

print("成本估算（假设每个请求 500 tokens）:")
print(f"如果全用 GPT-4: ${len(prompts) * 0.5 * 0.03:.4f}")
print(f"如果全用 GPT-3.5: ${len(prompts) * 0.5 * 0.002:.4f}")
print(f"使用降级策略: 成本介于两者之间，但可用性更高")
```

---

## 示例4：跨云服务商降级

**场景：** OpenAI → Anthropic → Google 多云部署

```python
"""
跨云服务商降级
演示：多云高可用部署
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

print("=== 跨云服务商降级 ===\n")

# ===== 1. 创建多云模型 =====
# 第1级：OpenAI GPT-4
openai_llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)
print("第1级: OpenAI GPT-4")

# 第2级：Anthropic Claude
anthropic_llm = ChatAnthropic(
    model="claude-3-sonnet-20240229"
).with_retry(stop_after_attempt=2)
print("第2级: Anthropic Claude-3-Sonnet")

# 注意：需要配置 ANTHROPIC_API_KEY 环境变量

# ===== 2. 构建多云降级链 =====
llm = openai_llm.with_fallbacks([anthropic_llm])

print("\n✅ 多云降级链构建完成\n")

# ===== 3. 测试 =====
test_prompts = [
    "什么是 LangChain？",
    "解释一下 LCEL 表达式",
    "什么是 Runnable 协议？"
]

for i, prompt in enumerate(test_prompts):
    print(f"[{i+1}/{len(test_prompts)}] {prompt}")
    try:
        response = llm.invoke(prompt)
        print(f"✅ 成功: {response.content[:50]}...\n")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}\n")

# ===== 4. 多云优势 =====
print("=== 多云部署优势 ===")
print("1. 高可用性: 单个云服务商故障不影响服务")
print("2. 避免供应商锁定: 可以灵活切换")
print("3. 成本优化: 根据价格和性能选择")
print("4. 合规性: 满足不同地区的数据合规要求")
```

---

## 示例5：功能降级

**场景：** 复杂功能失败时降级到简单功能

```python
"""
功能降级
演示：从复杂功能降级到简单功能
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

print("=== 功能降级 ===\n")

# ===== 1. 主链：复杂的 JSON 输出 =====
print("--- 主链: 复杂 JSON 输出 ---")

primary_prompt = ChatPromptTemplate.from_template(
    "以 JSON 格式返回详细分析: {query}\n"
    "格式: {{\"summary\": \"...\", \"details\": [...], \"confidence\": 0.0-1.0}}"
)
primary_llm = ChatOpenAI(model="gpt-4")
primary_parser = JsonOutputParser()

primary_chain = primary_prompt | primary_llm | primary_parser

print("输出格式: JSON (summary, details, confidence)")

# ===== 2. 备选链：简单的文本输出 =====
print("\n--- 备选链: 简单文本输出 ---")

fallback_prompt = ChatPromptTemplate.from_template("简单回答: {query}")
fallback_llm = ChatOpenAI(model="gpt-3.5-turbo")
fallback_parser = StrOutputParser()

fallback_chain = fallback_prompt | fallback_llm | fallback_parser

print("输出格式: 纯文本")

# ===== 3. 构建功能降级链 =====
print("\n✅ 功能降级链构建完成\n")

chain = primary_chain.with_fallbacks([fallback_chain])

# ===== 4. 测试 =====
queries = [
    "Python 的优点",
    "机器学习的应用",
    "区块链技术"
]

for i, query in enumerate(queries):
    print(f"[{i+1}/{len(queries)}] 查询: {query}")
    try:
        result = chain.invoke({"query": query})

        # 检查结果类型
        if isinstance(result, dict):
            print(f"✅ 复杂输出 (JSON)")
            print(f"   Summary: {result.get('summary', 'N/A')[:50]}...")
        else:
            print(f"✅ 简单输出 (文本)")
            print(f"   {result[:50]}...")
        print()
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}\n")

# ===== 5. 功能降级说明 =====
print("=== 功能降级说明 ===")
print("主功能: 结构化 JSON 输出（更丰富）")
print("  ↓ 失败")
print("备选功能: 简单文本输出（更可靠）")
print("\n优点:")
print("- 用户仍能得到答案")
print("- 降级对用户透明")
print("- 保证服务可用性")
```

---

## 示例6：RAG 系统的降级策略

**场景：** RAG 系统的多层降级

```python
"""
RAG 系统的降级策略
演示：检索器和生成器的双重降级
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

print("=== RAG 系统降级策略 ===\n")

# ===== 1. 模拟向量检索器 =====
def vector_retriever(query):
    """模拟向量检索"""
    print(f"  [向量检索] 查询: {query}")
    # 模拟检索结果
    return [
        "LangChain 是一个用于构建 AI 应用的框架",
        "LCEL 是 LangChain 的表达式语言",
        "Runnable 是 LangChain 的核心协议"
    ]

def keyword_retriever(query):
    """模拟关键词检索（降级）"""
    print(f"  [关键词检索] 查询: {query}")
    # 模拟简单的关键词匹配
    return [
        "LangChain 相关文档",
        "LCEL 相关文档"
    ]

# ===== 2. 创建检索器降级 =====
primary_retriever = RunnableLambda(vector_retriever)
fallback_retriever = RunnableLambda(keyword_retriever)

retriever = primary_retriever.with_fallbacks([fallback_retriever])

print("✅ 检索器降级配置完成")

# ===== 3. 创建生成器降级 =====
primary_llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)
fallback_llm = ChatOpenAI(model="gpt-3.5-turbo").with_retry(stop_after_attempt=2)

llm = primary_llm.with_fallbacks([fallback_llm])

print("✅ 生成器降级配置完成\n")

# ===== 4. 构建 RAG 链 =====
prompt = ChatPromptTemplate.from_template(
    "基于以下上下文回答问题:\n{context}\n\n问题: {question}"
)

def format_docs(docs):
    """格式化文档"""
    return "\n".join(docs)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

print("✅ RAG 链构建完成\n")

# ===== 5. 测试 =====
questions = [
    "什么是 LangChain？",
    "什么是 LCEL？",
    "什么是 Runnable？"
]

for i, question in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] 问题: {question}")
    try:
        response = rag_chain.invoke(question)
        print(f"✅ 回答: {response.content[:80]}...\n")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}\n")

# ===== 6. 降级层次 =====
print("=== RAG 系统降级层次 ===")
print("第1层: 向量检索 + GPT-4")
print("  ↓")
print("第2层: 向量检索 + GPT-3.5")
print("  ↓")
print("第3层: 关键词检索 + GPT-4")
print("  ↓")
print("第4层: 关键词检索 + GPT-3.5")
```

---

## 示例7：优雅降级最佳实践

**场景：** 生产级降级配置

```python
"""
优雅降级最佳实践
演示：生产级降级配置
"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import logging

# ===== 1. 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

print("=== 优雅降级最佳实践 ===\n")

# ===== 2. 创建带日志的降级链 =====
def log_fallback(input_data):
    """记录降级事件"""
    logger.warning(f"主服务失败，使用降级方案")
    return input_data

# 主模型 + 重试
primary = ChatOpenAI(model="gpt-4").with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

# 备选模型 + 重试 + 日志
fallback = (
    RunnableLambda(log_fallback)
    | ChatOpenAI(model="gpt-3.5-turbo").with_retry(stop_after_attempt=2)
)

# 最后的保底
def emergency_response(input_data):
    """紧急响应"""
    logger.error("所有服务都失败，返回默认响应")
    return "抱歉，服务暂时不可用，请稍后重试"

emergency = RunnableLambda(emergency_response)

# ===== 3. 构建三级降级 =====
llm = primary.with_fallbacks([fallback, emergency])

print("✅ 三级降级配置完成")
print("  第1级: GPT-4 (重试3次)")
print("  第2级: GPT-3.5 (重试2次)")
print("  第3级: 默认响应\n")

# ===== 4. 测试 =====
test_prompts = [
    "你好",
    "什么是 Python？",
    "解释一下机器学习"
]

for i, prompt in enumerate(test_prompts):
    logger.info(f"[{i+1}/{len(test_prompts)}] 处理: {prompt}")
    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            logger.info(f"成功，响应长度: {len(response.content)} 字符")
            print(f"✅ [{i+1}] {prompt}")
            print(f"   {response.content[:50]}...\n")
        else:
            logger.info(f"使用默认响应")
            print(f"⚠️  [{i+1}] {prompt}")
            print(f"   {response}\n")
    except Exception as e:
        logger.error(f"失败: {type(e).__name__}: {e}")
        print(f"❌ [{i+1}] {prompt}\n")

# ===== 5. 最佳实践总结 =====
print("=== 最佳实践总结 ===")
print("1. 多层防护: 主服务 + 备选服务 + 默认响应")
print("2. 重试 + 降级: 每层都有重试机制")
print("3. 日志记录: 记录所有降级事件")
print("4. 用户友好: 降级对用户透明")
print("5. 监控告警: 降级率超过阈值时告警")
```

---

## 运行环境要求

### 依赖安装

```bash
uv add langchain langchain-openai langchain-anthropic python-dotenv
```

### 环境变量配置

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # 可选
```

---

## 最佳实践总结

### 1. 降级层级设计

| 场景 | 推荐层级 | 配置 |
|------|----------|------|
| 非关键服务 | 1-2层 | 主服务 + 1个备选 |
| 标准服务 | 2-3层 | 主服务 + 2个备选 |
| 关键服务 | 3-5层 | 主服务 + 多个备选 + 默认响应 |

### 2. 降级策略选择

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 质量降级 | 追求最高质量 | 质量优先 | 成本高 |
| 成本优化 | 成本敏感 | 成本低 | 质量可能略低 |
| 多云部署 | 高可用要求 | 避免单点故障 | 配置复杂 |
| 功能降级 | 功能灵活 | 保证可用性 | 功能受限 |

### 3. 监控指标

- **降级率**: 降级次数 / 总请求数
- **最终失败率**: 最终失败次数 / 总请求数
- **平均响应时间**: 包含降级的平均时间

### 4. 告警阈值

- 降级率 > 5%: 警告
- 降级率 > 10%: 严重警告
- 最终失败率 > 1%: 立即介入

---

**记住：降级不是"备胎"，而是"保险"。**
