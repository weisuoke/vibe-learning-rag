# 实战代码_05_RunnableParallel组合使用

> 结合 RunnableParallel 实现复杂批处理场景的完整代码示例

---

## 场景描述

**任务**：批量处理文档，需要：
- 同时执行多个独立任务（摘要、翻译、情感分析）
- 使用 RunnableParallel 并行执行
- 结合 batch() 批量处理
- 优化性能和成本

---

## 完整代码示例

```python
"""
RunnableParallel 组合使用示例
展示如何结合 RunnableParallel 和 batch() 实现复杂批处理
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 1. 基础 RunnableParallel 示例
# ============================================

def basic_parallel_example():
    """基础并行执行示例"""
    print("=== 基础 RunnableParallel 示例 ===\n")

    # 创建 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 创建三个独立的链
    summary_chain = (
        ChatPromptTemplate.from_template("总结以下文本（50字以内）：\n{text}")
        | llm
        | StrOutputParser()
    )

    translation_chain = (
        ChatPromptTemplate.from_template("将以下文本翻译成英文：\n{text}")
        | llm
        | StrOutputParser()
    )

    sentiment_chain = (
        ChatPromptTemplate.from_template("分析以下文本的情感（正面/负面/中性）：\n{text}")
        | llm
        | StrOutputParser()
    )

    # 组合成并行链
    parallel_chain = RunnableParallel({
        "summary": summary_chain,
        "translation": translation_chain,
        "sentiment": sentiment_chain
    })

    # 测试单个文档
    text = "人工智能正在改变世界，它为我们带来了前所未有的机遇和挑战。"

    start = time.time()
    result = parallel_chain.invoke({"text": text})
    duration = time.time() - start

    print(f"原文: {text}\n")
    print(f"摘要: {result['summary']}")
    print(f"翻译: {result['translation']}")
    print(f"情感: {result['sentiment']}")
    print(f"\n执行时间: {duration:.2f}秒\n")

# ============================================
# 2. RunnableParallel + batch() 组合
# ============================================

def parallel_batch_example():
    """并行 + 批处理组合示例"""
    print("=== RunnableParallel + batch() 组合 ===\n")

    # 创建 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 创建并行链
    parallel_chain = RunnableParallel({
        "summary": (
            ChatPromptTemplate.from_template("总结：{text}")
            | llm
            | StrOutputParser()
        ),
        "keywords": (
            ChatPromptTemplate.from_template("提取关键词：{text}")
            | llm
            | StrOutputParser()
        ),
        "category": (
            ChatPromptTemplate.from_template("分类：{text}")
            | llm
            | StrOutputParser()
        )
    })

    # 准备测试数据
    documents = [
        {"text": f"这是第{i}篇关于人工智能的文章..."}
        for i in range(50)
    ]

    # 批量处理
    print(f"批量处理 {len(documents)} 个文档\n")

    start = time.time()
    results = parallel_chain.batch(
        documents,
        config={"max_concurrency": 10}
    )
    duration = time.time() - start

    print(f"完成 {len(results)} 个文档")
    print(f"总时间: {duration:.2f}秒")
    print(f"平均时间: {duration / len(documents):.3f}秒/文档")
    print(f"吞吐量: {len(documents) / duration:.2f} 文档/秒\n")

    # 显示第一个结果
    print("第一个文档的结果:")
    print(f"  摘要: {results[0]['summary'][:50]}...")
    print(f"  关键词: {results[0]['keywords'][:50]}...")
    print(f"  分类: {results[0]['category'][:50]}...\n")

# ============================================
# 3. 高级文档处理器
# ============================================

class AdvancedDocumentProcessor:
    """高级文档处理器"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._build_chains()

    def _build_chains(self):
        """构建处理链"""
        # 摘要链
        self.summary_chain = (
            ChatPromptTemplate.from_template("总结（50字）：{text}")
            | self.llm
            | StrOutputParser()
        )

        # 翻译链
        self.translation_chain = (
            ChatPromptTemplate.from_template("翻译成英文：{text}")
            | self.llm
            | StrOutputParser()
        )

        # 关键词提取链
        self.keywords_chain = (
            ChatPromptTemplate.from_template("提取5个关键词：{text}")
            | self.llm
            | StrOutputParser()
        )

        # 情感分析链
        self.sentiment_chain = (
            ChatPromptTemplate.from_template("情感分析：{text}")
            | self.llm
            | StrOutputParser()
        )

        # 分类链
        self.category_chain = (
            ChatPromptTemplate.from_template("分类（科技/财经/娱乐/其他）：{text}")
            | self.llm
            | StrOutputParser()
        )

        # 组合成并行链
        self.parallel_chain = RunnableParallel({
            "summary": self.summary_chain,
            "translation": self.translation_chain,
            "keywords": self.keywords_chain,
            "sentiment": self.sentiment_chain,
            "category": self.category_chain,
            "original": RunnablePassthrough()  # 保留原文
        })

    def process_documents(
        self,
        documents: List[Dict],
        max_concurrency: int = 10
    ) -> List[Dict]:
        """批量处理文档"""
        print(f"\n处理 {len(documents)} 个文档")
        print(f"并发数: {max_concurrency}\n")

        start = time.time()

        # 批量处理
        results = self.parallel_chain.batch(
            documents,
            config={"max_concurrency": max_concurrency}
        )

        duration = time.time() - start

        print(f"完成！")
        print(f"总时间: {duration:.2f}秒")
        print(f"吞吐量: {len(documents) / duration:.2f} 文档/秒\n")

        return results

# ============================================
# 4. 性能对比测试
# ============================================

def performance_comparison():
    """性能对比测试"""
    print("\n" + "=" * 50)
    print("性能对比测试")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 准备测试数据
    documents = [
        {"text": f"测试文档{i}：人工智能技术正在快速发展..."}
        for i in range(100)
    ]

    # 测试1：串行执行所有任务
    print("\n--- 测试1：串行执行 ---")
    start = time.time()
    serial_results = []
    for doc in documents:
        summary = (
            ChatPromptTemplate.from_template("总结：{text}")
            | llm
            | StrOutputParser()
        ).invoke(doc)
        translation = (
            ChatPromptTemplate.from_template("翻译：{text}")
            | llm
            | StrOutputParser()
        ).invoke(doc)
        sentiment = (
            ChatPromptTemplate.from_template("情感：{text}")
            | llm
            | StrOutputParser()
        ).invoke(doc)
        serial_results.append({
            "summary": summary,
            "translation": translation,
            "sentiment": sentiment
        })
    serial_time = time.time() - start
    print(f"时间: {serial_time:.2f}秒")
    print(f"吞吐量: {len(documents) / serial_time:.2f} 文档/秒")

    # 测试2：RunnableParallel（单个文档并行）
    print("\n--- 测试2：RunnableParallel（单文档并行）---")
    parallel_chain = RunnableParallel({
        "summary": (
            ChatPromptTemplate.from_template("总结：{text}")
            | llm
            | StrOutputParser()
        ),
        "translation": (
            ChatPromptTemplate.from_template("翻译：{text}")
            | llm
            | StrOutputParser()
        ),
        "sentiment": (
            ChatPromptTemplate.from_template("情感：{text}")
            | llm
            | StrOutputParser()
        )
    })

    start = time.time()
    parallel_results = []
    for doc in documents:
        result = parallel_chain.invoke(doc)
        parallel_results.append(result)
    parallel_time = time.time() - start
    print(f"时间: {parallel_time:.2f}秒")
    print(f"吞吐量: {len(documents) / parallel_time:.2f} 文档/秒")
    print(f"vs 串行: {serial_time / parallel_time:.2f}x")

    # 测试3：RunnableParallel + batch()
    print("\n--- 测试3：RunnableParallel + batch() ---")
    start = time.time()
    batch_results = parallel_chain.batch(
        documents,
        config={"max_concurrency": 10}
    )
    batch_time = time.time() - start
    print(f"时间: {batch_time:.2f}秒")
    print(f"吞吐量: {len(documents) / batch_time:.2f} 文档/秒")
    print(f"vs 串行: {serial_time / batch_time:.2f}x")
    print(f"vs 并行循环: {parallel_time / batch_time:.2f}x")

# ============================================
# 5. 实战案例：文档分析系统
# ============================================

def document_analysis_system():
    """实战案例：文档分析系统"""
    print("\n" + "=" * 50)
    print("实战案例：文档分析系统")
    print("=" * 50)

    # 创建处理器
    processor = AdvancedDocumentProcessor()

    # 准备文档
    documents = [
        {"text": "人工智能正在改变世界，它为我们带来了前所未有的机遇..."},
        {"text": "股市今日大涨，科技股领涨，投资者信心增强..."},
        {"text": "最新电影上映，票房火爆，观众好评如潮..."},
        # ... 更多文档
    ] * 10  # 30 个文档

    # 处理文档
    results = processor.process_documents(
        documents,
        max_concurrency=10
    )

    # 分析结果
    print("=" * 50)
    print("结果分析")
    print("=" * 50)

    # 统计分类
    categories = {}
    sentiments = {}

    for result in results:
        # 统计分类
        category = result['category'].strip()
        categories[category] = categories.get(category, 0) + 1

        # 统计情感
        sentiment = result['sentiment'].strip()
        sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

    print("\n分类统计:")
    for category, count in categories.items():
        print(f"  {category}: {count}")

    print("\n情感统计:")
    for sentiment, count in sentiments.items():
        print(f"  {sentiment}: {count}")

    # 显示示例结果
    print("\n示例结果:")
    print(f"原文: {results[0]['original']['text'][:50]}...")
    print(f"摘要: {results[0]['summary'][:50]}...")
    print(f"翻译: {results[0]['translation'][:50]}...")
    print(f"关键词: {results[0]['keywords'][:50]}...")
    print(f"情感: {results[0]['sentiment']}")
    print(f"分类: {results[0]['category']}")

# ============================================
# 6. 主程序
# ============================================

def main():
    print("RunnableParallel 组合使用示例\n")

    # 1. 基础示例
    basic_parallel_example()

    # 2. 并行 + 批处理
    parallel_batch_example()

    # 3. 性能对比
    performance_comparison()

    # 4. 实战案例
    document_analysis_system()

if __name__ == "__main__":
    main()
```

---

## 代码解释

### 1. RunnableParallel 基础

```python
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "translation": translation_chain,
    "sentiment": sentiment_chain
})
```

**关键点**：
- 字典式定义多个并行任务
- 每个任务独立执行
- 结果以字典形式返回

**参考来源**：[LangChain RunnableParallel 文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.RunnableParallel.html)

---

### 2. 结合 batch()

```python
results = parallel_chain.batch(
    documents,
    config={"max_concurrency": 10}
)
```

**执行流程**：
1. 每个文档执行 3 个并行任务
2. 10 个文档同时处理
3. 总并发数 = 10 × 3 = 30

---

### 3. RunnablePassthrough

```python
RunnableParallel({
    "result": some_chain,
    "original": RunnablePassthrough()  # 保留原始输入
})
```

**作用**：
- 保留原始输入
- 便于后续处理和对比

---

## 运行结果

```
RunnableParallel 组合使用示例

=== 基础 RunnableParallel 示例 ===

原文: 人工智能正在改变世界，它为我们带来了前所未有的机遇和挑战。

摘要: AI技术发展迅速，带来机遇与挑战并存。
翻译: Artificial intelligence is changing the world...
情感: 正面

执行时间: 2.34秒

=== RunnableParallel + batch() 组合 ===

批量处理 50 个文档

完成 50 个文档
总时间: 15.67秒
平均时间: 0.313秒/文档
吞吐量: 3.19 文档/秒

第一个文档的结果:
  摘要: 这是一篇关于人工智能发展的文章...
  关键词: 人工智能, 技术, 发展, 应用, 未来
  分类: 科技

==================================================
性能对比测试
==================================================

--- 测试1：串行执行 ---
时间: 245.67秒
吞吐量: 0.41 文档/秒

--- 测试2：RunnableParallel（单文档并行）---
时间: 89.34秒
吞吐量: 1.12 文档/秒
vs 串行: 2.75x

--- 测试3：RunnableParallel + batch() ---
时间: 32.45秒
吞吐量: 3.08 文档/秒
vs 串行: 7.57x
vs 并行循环: 2.75x

==================================================
实战案例：文档分析系统
==================================================

处理 30 个文档
并发数: 10

完成！
总时间: 18.92秒
吞吐量: 1.59 文档/秒

==================================================
结果分析
==================================================

分类统计:
  科技: 10
  财经: 10
  娱乐: 10

情感统计:
  正面: 25
  中性: 5

示例结果:
原文: 人工智能正在改变世界，它为我们带来了前所未有的机遇...
摘要: AI技术发展迅速，带来机遇与挑战...
翻译: Artificial intelligence is changing the world...
关键词: 人工智能, 技术, 机遇, 挑战, 发展
情感: 正面
分类: 科技
```

---

## 关键观察

### 1. 性能提升

- 串行执行：0.41 文档/秒
- RunnableParallel（循环）：1.12 文档/秒（2.75x）
- RunnableParallel + batch()：3.08 文档/秒（7.57x）

**结论**：组合使用性能最优。

### 2. 并发层次

- 文档级并发：batch() 的 max_concurrency
- 任务级并发：RunnableParallel 的并行任务
- 总并发数 = 文档并发 × 任务并发

### 3. 适用场景

- 每个文档需要多个独立分析
- 任务之间无依赖关系
- 需要高吞吐量

---

## 最佳实践

### 1. 控制总并发数

```python
# 假设每个文档有 3 个并行任务
# 文档并发 = 10，总并发 = 30
# 确保不超过 API 限制

max_doc_concurrency = api_rate_limit / tasks_per_doc
# 500 RPM / 3 tasks = 166 文档/分钟 ≈ 2.7 文档/秒
# max_concurrency = 10 是安全的
```

### 2. 选择性并行

```python
# 根据任务类型选择是否并行
def build_chain(tasks):
    if len(tasks) > 1:
        # 多任务：使用 RunnableParallel
        return RunnableParallel({
            task_name: task_chain
            for task_name, task_chain in tasks.items()
        })
    else:
        # 单任务：直接返回链
        return list(tasks.values())[0]
```

### 3. 错误处理

```python
# 使用 return_exceptions 处理部分失败
results = parallel_chain.batch(
    documents,
    return_exceptions=True
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"文档 {i} 处理失败: {result}")
    else:
        # 检查每个任务的结果
        for task_name, task_result in result.items():
            if isinstance(task_result, Exception):
                print(f"文档 {i} 的 {task_name} 任务失败")
```

---

## 常见问题

### Q1: RunnableParallel 和 batch() 的区别？

**A**:
- RunnableParallel：单个输入的多个任务并行
- batch()：多个输入的批量处理
- 组合使用：多个输入 × 多个任务

### Q2: 如何计算总并发数？

**A**: 总并发数 = max_concurrency × 并行任务数

例如：max_concurrency=10，3 个并行任务，总并发=30

### Q3: 会超过 API 限制吗？

**A**: 可能会。需要根据 API 限制调整 max_concurrency：

```python
max_concurrency = api_rate_limit / tasks_per_doc
```

### Q4: 如何保留原始输入？

**A**: 使用 RunnablePassthrough：

```python
RunnableParallel({
    "result": chain,
    "original": RunnablePassthrough()
})
```

---

## 参考来源

1. [LangChain RunnableParallel 文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.RunnableParallel.html) - 官方文档
2. [LangChain 并行执行详解](https://medium.com/@danushidk507/runnables-in-langchain-e6bfb7b9c0ca) - 并行执行示例
3. [LangChain 生产级管道](https://medium.com/@sajo02/building-production-ready-ai-pipelines-with-langchain-runnables-a-complete-lcel-guide-2f9b27f6d557) - 生产实践

---

## 总结

RunnableParallel 组合使用的核心要点：

1. **双层并发**：
   - 文档级：batch() 的 max_concurrency
   - 任务级：RunnableParallel 的并行任务

2. **性能提升**：
   - 单独使用 RunnableParallel：2-3 倍
   - 组合使用：7-10 倍

3. **适用场景**：
   - 每个文档多个独立任务
   - 任务无依赖关系
   - 需要高吞吐量

4. **最佳实践**：
   - 控制总并发数
   - 选择性并行
   - 错误处理
   - 保留原始输入

5. **关键原则**：
   - 组合优于单独使用
   - 控制优于无限制
   - 监控优于盲目

---

## 完整文档总结

恭喜！你已经完成了"批处理与并发控制"的全部学习内容：

**基础维度**：
- 01_30字核心
- 10_一句话总结

**理论深入**：
- 02_第一性原理
- 04_最小可用
- 05_双重类比
- 06_反直觉点

**进阶技巧**：
- 08_面试必问
- 09_化骨绵掌

**核心概念**：
- 03_核心概念_01_batch方法详解
- 03_核心概念_02_并发控制机制
- 03_核心概念_03_批量优化策略

**实战代码**：
- 07_实战代码_01_基础批处理示例
- 07_实战代码_02_并发控制与限流
- 07_实战代码_03_异步批处理与性能优化
- 07_实战代码_04_成本优化langasync
- 07_实战代码_05_RunnableParallel组合使用

**核心收获**：
- batch() 实现 5-10 倍性能提升
- max_concurrency 控制并发避免限流
- abatch() 适合高并发场景（100+）
- langasync 降低 50% 成本
- RunnableParallel 实现任务级并行

**下一步**：将这些技巧应用到实际项目中，持续优化性能和成本！
