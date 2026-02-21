# 实战代码4：LangSmith追踪

## 概述

本文档提供LangSmith平台的完整实战代码，展示如何使用LangSmith进行生产追踪、调试和成本分析。

**涵盖场景**：
1. LangSmith快速集成
2. 生产追踪和调试
3. 成本追踪和分析
4. 性能优化实践

**前置要求**：
- Python 3.13+
- LangChain v0.3+
- LangSmith账号和API密钥

---

## 场景1：LangSmith快速集成

### 目标

零代码集成LangSmith，自动追踪所有链执行。

### 步骤1：注册并获取API密钥

1. 访问 https://smith.langchain.com/
2. 注册账号（支持GitHub登录）
3. 创建API密钥：Settings → API Keys → Create API Key

### 步骤2：配置环境变量

```bash
# .env 文件
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=my_project  # 可选，项目名称
```

### 步骤3：运行代码

```python
"""
场景1：LangSmith快速集成
零代码集成，自动追踪
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 加载环境变量（包含LangSmith配置）
load_dotenv()

# 构建链
prompt = ChatPromptTemplate.from_template("回答问题: {question}")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# 正常使用，自动追踪
result = chain.invoke({"question": "什么是LCEL?"})
print(result)

print("\n✅ 追踪已自动上传到LangSmith")
print("访问 https://smith.langchain.com/ 查看详情")
```

### 输出示例

```
LCEL是LangChain Expression Language的缩写...

✅ 追踪已自动上传到LangSmith
访问 https://smith.langchain.com/ 查看详情
```

### 在LangSmith中查看

访问 https://smith.langchain.com/，你会看到：

1. **执行流程图**：
   ```
   RunnableSequence
   ├─ ChatPromptTemplate (2ms)
   ├─ ChatOpenAI (1234ms)
   └─ StrOutputParser (1ms)
   ```

2. **详细信息**：
   - 输入：`{"question": "什么是LCEL?"}`
   - 输出：`"LCEL是..."`
   - 延迟：1.237s
   - Token：135 (prompt: 15, completion: 120)
   - 成本：$0.000074

---

## 场景2：生产追踪和调试

### 目标

在生产环境中使用LangSmith追踪和调试问题。

### 完整代码

```python
"""
场景2：生产追踪和调试
添加标签和元数据
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import uuid

load_dotenv()

prompt = ChatPromptTemplate.from_template("回答问题: {question}")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()


def process_user_request(user_id, session_id, question):
    """处理用户请求，添加追踪信息"""
    # 生成请求ID
    request_id = str(uuid.uuid4())

    # 配置追踪
    config = {
        "tags": [
            "production",
            f"user_{user_id}",
            f"session_{session_id}"
        ],
        "metadata": {
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "environment": "production",
            "version": "v1.0.0"
        }
    }

    try:
        # 执行链
        result = chain.invoke(
            {"question": question},
            config=config
        )

        print(f"✅ Request {request_id} completed")
        return result

    except Exception as e:
        print(f"❌ Request {request_id} failed: {e}")
        raise


# 使用示例
if __name__ == "__main__":
    result = process_user_request(
        user_id="user_123",
        session_id="session_abc",
        question="什么是LCEL?"
    )
    print(result)
```

### 在LangSmith中过滤

使用标签和元数据过滤追踪：

1. **按用户过滤**：`tag:user_123`
2. **按环境过滤**：`metadata.environment:production`
3. **按版本过滤**：`metadata.version:v1.0.0`
4. **按时间过滤**：选择时间范围

---

## 场景3：成本追踪和分析

### 目标

追踪和分析LLM调用的成本。

### 完整代码

```python
"""
场景3：成本追踪和分析
追踪Token使用和成本
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class CostTracker:
    """成本追踪器"""

    # 价格表（2026年价格）
    PRICES = {
        "gpt-4o": {
            "prompt": 2.50 / 1_000_000,
            "completion": 10.00 / 1_000_000
        },
        "gpt-4o-mini": {
            "prompt": 0.15 / 1_000_000,
            "completion": 0.60 / 1_000_000
        },
        "gpt-3.5-turbo": {
            "prompt": 0.50 / 1_000_000,
            "completion": 1.50 / 1_000_000
        }
    }

    def __init__(self):
        self.total_cost = 0
        self.total_tokens = 0
        self.requests = []

    def track_request(self, model, prompt_tokens, completion_tokens):
        """追踪单次请求"""
        prices = self.PRICES.get(model, self.PRICES["gpt-4o-mini"])

        cost = (
            prompt_tokens * prices["prompt"] +
            completion_tokens * prices["completion"]
        )

        self.total_cost += cost
        self.total_tokens += (prompt_tokens + completion_tokens)

        self.requests.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost
        })

        return cost

    def print_report(self):
        """打印成本报告"""
        print("\n=== 成本报告 ===")
        print(f"总请求数: {len(self.requests)}")
        print(f"总Token: {self.total_tokens:,}")
        print(f"总成本: ${self.total_cost:.6f}")
        print(f"平均成本/请求: ${self.total_cost / len(self.requests):.6f}")


def main():
    """成本追踪示例"""
    tracker = CostTracker()

    prompt = ChatPromptTemplate.from_template("回答问题: {question}")
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model

    questions = [
        "什么是LCEL?",
        "LCEL有什么优势?",
        "如何使用LCEL?"
    ]

    for question in questions:
        result = chain.invoke({"question": question})

        # 获取Token使用
        tokens = result.response_metadata.get("token_usage", {})
        prompt_tokens = tokens.get("prompt_tokens", 0)
        completion_tokens = tokens.get("completion_tokens", 0)

        # 追踪成本
        cost = tracker.track_request(
            "gpt-4o-mini",
            prompt_tokens,
            completion_tokens
        )

        print(f"Q: {question}")
        print(f"Cost: ${cost:.6f}")
        print()

    tracker.print_report()


if __name__ == "__main__":
    main()
```

### 输出示例

```
Q: 什么是LCEL?
Cost: $0.000074

Q: LCEL有什么优势?
Cost: $0.000082

Q: 如何使用LCEL?
Cost: $0.000091

=== 成本报告 ===
总请求数: 3
总Token: 405
总成本: $0.000247
平均成本/请求: $0.000082
```

### LangSmith成本分析

在LangSmith中查看成本分析：

1. **总成本**：所有请求的总成本
2. **按模型分组**：不同模型的成本占比
3. **按项目分组**：不同项目的成本
4. **成本趋势**：每天/每周/每月的成本变化
5. **成本预警**：设置预算告警

---

## 场景4：性能优化实践

### 目标

使用LangSmith分析性能瓶颈并优化。

### 完整代码

```python
"""
场景4：性能优化实践
使用LangSmith分析和优化性能
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import time

load_dotenv()


def build_rag_chain_v1():
    """RAG链 v1（未优化）"""
    texts = [
        "LCEL是LangChain Expression Language的缩写",
        "LCEL使用管道符|连接组件",
        "LCEL支持流式处理和异步执行"
    ]
    vectorstore = Chroma.from_texts(texts, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 检索3个文档

    template = """基于以下上下文回答问题:

上下文: {context}
问题: {question}
回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    return chain


def build_rag_chain_v2():
    """RAG链 v2（优化后）"""
    texts = [
        "LCEL是LangChain Expression Language的缩写",
        "LCEL使用管道符|连接组件",
        "LCEL支持流式处理和异步执行"
    ]
    vectorstore = Chroma.from_texts(texts, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 只检索2个文档

    # 优化：更简洁的prompt
    template = """上下文: {context}

问题: {question}
回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 降低temperature
        | StrOutputParser()
    )

    return chain


def benchmark_chain(chain, name, question):
    """性能测试"""
    config = {
        "tags": [name, "benchmark"],
        "metadata": {"version": name}
    }

    start = time.time()
    result = chain.invoke(question, config=config)
    duration = time.time() - start

    print(f"{name}:")
    print(f"  延迟: {duration:.2f}s")
    print(f"  结果: {result[:50]}...")
    print()

    return duration


def main():
    """性能优化示例"""
    question = "什么是LCEL?"

    # 测试v1
    chain_v1 = build_rag_chain_v1()
    duration_v1 = benchmark_chain(chain_v1, "v1_unoptimized", question)

    # 测试v2
    chain_v2 = build_rag_chain_v2()
    duration_v2 = benchmark_chain(chain_v2, "v2_optimized", question)

    # 对比
    improvement = (duration_v1 - duration_v2) / duration_v1 * 100
    print(f"性能提升: {improvement:.1f}%")


if __name__ == "__main__":
    main()
```

### 输出示例

```
v1_unoptimized:
  延迟: 1.45s
  结果: LCEL是LangChain Expression Language的缩写...

v2_optimized:
  延迟: 1.12s
  结果: LCEL是LangChain Expression Language的缩写...

性能提升: 22.8%
```

### 在LangSmith中对比

1. 过滤两个版本：`tag:v1_unoptimized` vs `tag:v2_optimized`
2. 对比延迟：查看P50、P95、P99
3. 对比Token使用：查看平均Token数
4. 对比成本：查看平均成本

---

## 高级功能：数据集和评估

### 完整代码

```python
"""
高级功能：数据集和评估
创建测试数据集，评估链性能
"""

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 初始化LangSmith客户端
client = Client()


def create_dataset():
    """创建测试数据集"""
    # 创建数据集
    dataset_name = "lcel_qa_dataset"

    try:
        dataset = client.create_dataset(dataset_name)
        print(f"✅ 创建数据集: {dataset_name}")
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✅ 使用现有数据集: {dataset_name}")

    # 添加测试用例
    examples = [
        {
            "inputs": {"question": "什么是LCEL?"},
            "outputs": {"answer": "LCEL是LangChain Expression Language的缩写"}
        },
        {
            "inputs": {"question": "LCEL有什么优势?"},
            "outputs": {"answer": "LCEL支持流式处理、异步执行和组件组合"}
        },
        {
            "inputs": {"question": "如何使用LCEL?"},
            "outputs": {"answer": "使用管道符|连接组件"}
        }
    ]

    for example in examples:
        try:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )
        except Exception:
            pass  # 示例已存在

    print(f"✅ 添加了 {len(examples)} 个测试用例")
    return dataset_name


def evaluate_chain(dataset_name):
    """评估链性能"""
    # 构建链
    prompt = ChatPromptTemplate.from_template("回答问题: {question}")
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model | StrOutputParser()

    # 运行评估
    from langchain.smith import RunEvalConfig

    eval_config = RunEvalConfig(
        evaluators=["qa"],  # 使用QA评估器
    )

    results = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda: chain,
        evaluation=eval_config,
        project_name="lcel_evaluation"
    )

    print(f"✅ 评估完成")
    print(f"结果: {results}")


def main():
    """数据集和评估示例"""
    # 创建数据集
    dataset_name = create_dataset()

    # 评估链
    evaluate_chain(dataset_name)


if __name__ == "__main__":
    main()
```

---

## 采样策略

### 完整代码

```python
"""
采样策略
只追踪部分请求，减少成本
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import random
import os

load_dotenv()


def should_trace(error=None, duration=None, sample_rate=0.1):
    """决定是否追踪"""
    # 总是追踪错误
    if error:
        return True

    # 总是追踪慢请求（>5s）
    if duration and duration > 5.0:
        return True

    # 随机采样
    return random.random() < sample_rate


def execute_with_sampling(chain, inputs, sample_rate=0.1):
    """执行链，使用采样策略"""
    import time

    start = time.time()
    error = None

    try:
        # 临时禁用追踪
        original_tracing = os.getenv("LANGCHAIN_TRACING_V2")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        result = chain.invoke(inputs)
        duration = time.time() - start

    except Exception as e:
        duration = time.time() - start
        error = e
        result = None

    finally:
        # 恢复追踪设置
        if original_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = original_tracing

    # 决定是否追踪
    if should_trace(error, duration, sample_rate):
        # 重新执行并追踪
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        try:
            result = chain.invoke(inputs)
        except Exception as e:
            error = e
        finally:
            if original_tracing:
                os.environ["LANGCHAIN_TRACING_V2"] = original_tracing

    if error:
        raise error

    return result


def main():
    """采样示例"""
    prompt = ChatPromptTemplate.from_template("回答问题: {question}")
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model

    # 执行多次请求，只追踪10%
    for i in range(10):
        result = execute_with_sampling(
            chain,
            {"question": f"问题{i+1}"},
            sample_rate=0.1
        )
        print(f"请求{i+1}完成")


if __name__ == "__main__":
    main()
```

---

## 最佳实践

### 1. 使用有意义的标签

```python
config = {
    "tags": [
        "production",
        f"user_{user_id}",
        f"feature_{feature_name}"
    ]
}
```

### 2. 添加丰富的元数据

```python
config = {
    "metadata": {
        "user_id": user_id,
        "session_id": session_id,
        "version": "v1.0.0",
        "environment": "production"
    }
}
```

### 3. 使用采样减少成本

```python
# 只追踪10%的请求
if random.random() < 0.1:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

### 4. 创建测试数据集

```python
# 创建数据集用于回归测试
dataset = client.create_dataset("my_dataset")
client.create_example(
    dataset_id=dataset.id,
    inputs={"question": "..."},
    outputs={"answer": "..."}
)
```

---

## 总结

**核心技巧**：

1. **零代码集成**：环境变量配置即可
2. **标签和元数据**：方便过滤和分析
3. **成本追踪**：实时监控Token和成本
4. **性能优化**：对比不同版本的性能
5. **采样策略**：减少追踪成本

**最佳实践**：

- 使用有意义的标签
- 添加丰富的元数据
- 使用采样减少成本
- 创建测试数据集
- 定期审查追踪数据

---

**版本信息**
- LangChain: v0.3+ (2025-2026)
- Python: 3.13+
- LangSmith: 最新版
- 最后更新: 2026-02-20
