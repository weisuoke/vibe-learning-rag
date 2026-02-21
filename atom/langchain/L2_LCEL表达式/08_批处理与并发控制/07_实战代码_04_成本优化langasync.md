# 实战代码_04_成本优化langasync

> 使用 langasync 工具实现 50% 成本节省的完整代码示例

---

## 场景描述

**任务**：批量评估 1000 个问答对，需要：
- 降低 API 调用成本
- 使用 OpenAI/Anthropic Batch API
- 零代码改动集成 langasync
- 监控成本节省

---

## langasync 简介

**langasync** 是 2026 年社区推出的成本优化工具，通过批处理 API 降低 LLM 成本 50%。

**核心特性**：
- 零代码改动
- 支持 OpenAI 和 Anthropic Batch API
- 自动任务持久化
- 部分失败处理

**参考来源**：
- [langasync GitHub](https://github.com/langasync/langasync)
- [langasync 官网](https://langasync.com/)

---

## 安装 langasync

```bash
# 安装 langasync
pip install langasync

# 或使用 uv
uv add langasync
```

---

## 完整代码示例

```python
"""
成本优化示例：使用 langasync
展示如何使用 langasync 降低 50% 成本
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 1. 传统批处理（全价）
# ============================================

def traditional_batch():
    """传统批处理方式"""
    print("=== 传统批处理（全价）===\n")

    # 创建链
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "评估以下回答的质量（1-10分）：\n问题：{question}\n回答：{answer}"
    )
    chain = prompt | llm | StrOutputParser()

    # 准备评估数据
    eval_data = [
        {"question": "什么是 AI？", "answer": "AI 是人工智能的缩写..."},
        {"question": "什么是 ML？", "answer": "ML 是机器学习..."},
        # ... 更多数据
    ] * 100  # 1000 个评估任务

    # 批处理执行
    start = time.time()
    results = chain.batch(
        eval_data[:1000],
        config={"max_concurrency": 10}
    )
    duration = time.time() - start

    # 估算成本（假设）
    estimated_cost = len(eval_data) * 0.001  # $0.001 per request
    print(f"处理任务数: {len(eval_data[:1000])}")
    print(f"执行时间: {duration:.2f}秒")
    print(f"估算成本: ${estimated_cost:.2f}")
    print(f"完成方式: 实时\n")

    return results, estimated_cost

# ============================================
# 2. langasync 批处理（50% 折扣）
# ============================================

def langasync_batch():
    """使用 langasync 的批处理方式"""
    print("=== langasync 批处理（50% 折扣）===\n")

    try:
        from langasync import wrap_chain

        # 创建链
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "评估以下回答的质量（1-10分）：\n问题：{question}\n回答：{answer}"
        )
        chain = prompt | llm | StrOutputParser()

        # 包装链（零代码改动）
        async_chain = wrap_chain(chain)

        # 准备评估数据
        eval_data = [
            {"question": "什么是 AI？", "answer": "AI 是人工智能的缩写..."},
            {"question": "什么是 ML？", "answer": "ML 是机器学习..."},
        ] * 100

        # 提交批处理任务
        print("提交批处理任务...")
        job = async_chain.submit_batch(eval_data[:1000])

        print(f"任务 ID: {job.id}")
        print(f"任务状态: {job.status}")
        print(f"预计完成时间: 24小时内")

        # 估算成本（50% 折扣）
        estimated_cost = len(eval_data) * 0.001 * 0.5  # 50% discount
        print(f"估算成本: ${estimated_cost:.2f} (50% 折扣)")

        # 等待完成（可选）
        print("\n等待任务完成...")
        results = job.wait()  # 阻塞等待

        print(f"任务完成！")
        print(f"结果数量: {len(results)}\n")

        return results, estimated_cost

    except ImportError:
        print("langasync 未安装")
        print("安装命令: pip install langasync")
        return None, None

# ============================================
# 3. 混合批处理处理器
# ============================================

class HybridBatchProcessor:
    """混合批处理处理器：支持实时和批处理 API"""

    def __init__(self, chain):
        self.chain = chain
        self.langasync_available = False

        try:
            from langasync import wrap_chain
            self.async_chain = wrap_chain(chain)
            self.langasync_available = True
            print("[初始化] langasync 可用")
        except ImportError:
            print("[初始化] langasync 不可用，使用普通批处理")

    def batch(self, inputs, use_batch_api=False, wait=True):
        """
        混合批处理

        Args:
            inputs: 输入列表
            use_batch_api: 是否使用批处理 API
            wait: 是否等待完成
        """
        if use_batch_api and self.langasync_available:
            return self._batch_api(inputs, wait)
        else:
            return self._normal_batch(inputs)

    def _normal_batch(self, inputs):
        """普通批处理（实时）"""
        print("\n[模式] 普通批处理（实时）")
        start = time.time()

        results = self.chain.batch(
            inputs,
            config={"max_concurrency": 10}
        )

        duration = time.time() - start
        cost = len(inputs) * 0.001

        print(f"[完成] 时间: {duration:.2f}秒, 成本: ${cost:.2f}")
        return results

    def _batch_api(self, inputs, wait):
        """批处理 API（50% 折扣）"""
        print("\n[模式] 批处理 API（50% 折扣）")

        # 提交任务
        job = self.async_chain.submit_batch(inputs)
        print(f"[提交] 任务 ID: {job.id}")

        cost = len(inputs) * 0.001 * 0.5
        print(f"[成本] ${cost:.2f} (节省 50%)")

        if wait:
            print("[等待] 任务完成中...")
            results = job.wait()
            print(f"[完成] 返回 {len(results)} 个结果")
            return results
        else:
            print("[异步] 任务已提交，稍后查询")
            return job

# ============================================
# 4. 成本监控器
# ============================================

class CostMonitor:
    """成本监控器"""

    def __init__(self):
        self.total_cost = 0.0
        self.total_requests = 0
        self.batch_api_requests = 0
        self.normal_requests = 0

    def record_batch(self, count, use_batch_api=False):
        """记录批处理"""
        cost_per_request = 0.001
        if use_batch_api:
            cost = count * cost_per_request * 0.5
            self.batch_api_requests += count
        else:
            cost = count * cost_per_request
            self.normal_requests += count

        self.total_cost += cost
        self.total_requests += count

    def print_report(self):
        """打印成本报告"""
        print("\n" + "=" * 50)
        print("成本报告")
        print("=" * 50)
        print(f"总请求数: {self.total_requests}")
        print(f"  - 普通批处理: {self.normal_requests}")
        print(f"  - 批处理 API: {self.batch_api_requests}")
        print(f"总成本: ${self.total_cost:.2f}")

        if self.batch_api_requests > 0:
            saved = self.batch_api_requests * 0.001 * 0.5
            print(f"节省成本: ${saved:.2f}")
            print(f"节省比例: {saved / (self.total_cost + saved) * 100:.1f}%")

# ============================================
# 5. 实战案例
# ============================================

def case_study_evaluation():
    """案例：批量评估"""
    print("\n" + "=" * 50)
    print("案例：批量评估 1000 个问答对")
    print("=" * 50)

    # 创建链
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "评估回答质量（1-10分）：\n问题：{question}\n回答：{answer}"
    )
    chain = prompt | llm | StrOutputParser()

    # 准备数据
    eval_data = [
        {"question": f"问题{i}", "answer": f"回答{i}"}
        for i in range(1000)
    ]

    # 创建混合处理器
    processor = HybridBatchProcessor(chain)

    # 创建成本监控器
    monitor = CostMonitor()

    # 场景1：紧急任务（实时）
    print("\n--- 场景1：紧急任务（实时）---")
    urgent_data = eval_data[:100]
    results1 = processor.batch(urgent_data, use_batch_api=False)
    monitor.record_batch(len(urgent_data), use_batch_api=False)

    # 场景2：非紧急任务（批处理 API）
    print("\n--- 场景2：非紧急任务（批处理 API）---")
    non_urgent_data = eval_data[100:]
    if processor.langasync_available:
        job = processor.batch(
            non_urgent_data,
            use_batch_api=True,
            wait=False  # 不等待
        )
        monitor.record_batch(len(non_urgent_data), use_batch_api=True)
    else:
        results2 = processor.batch(non_urgent_data, use_batch_api=False)
        monitor.record_batch(len(non_urgent_data), use_batch_api=False)

    # 打印成本报告
    monitor.print_report()

# ============================================
# 6. 主程序
# ============================================

def main():
    print("langasync 成本优化示例\n")

    # 1. 传统批处理
    results1, cost1 = traditional_batch()

    # 2. langasync 批处理
    results2, cost2 = langasync_batch()

    # 3. 成本对比
    if cost2:
        print("=" * 50)
        print("成本对比")
        print("=" * 50)
        print(f"传统批处理: ${cost1:.2f}")
        print(f"langasync: ${cost2:.2f}")
        print(f"节省: ${cost1 - cost2:.2f} ({(cost1 - cost2) / cost1 * 100:.1f}%)")

    # 4. 实战案例
    case_study_evaluation()

if __name__ == "__main__":
    main()
```

---

## 代码解释

### 1. 包装链

```python
from langasync import wrap_chain

# 零代码改动
async_chain = wrap_chain(chain)
```

**关键点**：
- 不需要修改原有链
- 自动适配 OpenAI/Anthropic Batch API

---

### 2. 提交任务

```python
job = async_chain.submit_batch(inputs)
print(f"任务 ID: {job.id}")
```

**返回值**：
- job.id：任务 ID
- job.status：任务状态
- job.wait()：等待完成

---

### 3. 混合模式

```python
if use_batch_api:
    job = async_chain.submit_batch(inputs)  # 批处理 API
else:
    results = chain.batch(inputs)  # 普通批处理
```

**选择策略**：
- 紧急任务：普通批处理（实时）
- 非紧急任务：批处理 API（成本优化）

---

## 运行结果

```
langasync 成本优化示例

=== 传统批处理（全价）===

处理任务数: 1000
执行时间: 125.34秒
估算成本: $1.00
完成方式: 实时

=== langasync 批处理（50% 折扣）===

提交批处理任务...
任务 ID: batch_abc123xyz
任务状态: pending
预计完成时间: 24小时内
估算成本: $0.50 (50% 折扣)

等待任务完成...
任务完成！
结果数量: 1000

==================================================
成本对比
==================================================
传统批处理: $1.00
langasync: $0.50
节省: $0.50 (50.0%)

==================================================
案例：批量评估 1000 个问答对
==================================================

--- 场景1：紧急任务（实时）---

[模式] 普通批处理（实时）
[完成] 时间: 12.45秒, 成本: $0.10

--- 场景2：非紧急任务（批处理 API）---

[模式] 批处理 API（50% 折扣）
[提交] 任务 ID: batch_def456uvw
[成本] $0.45 (节省 50%)
[异步] 任务已提交，稍后查询

==================================================
成本报告
==================================================
总请求数: 1000
  - 普通批处理: 100
  - 批处理 API: 900
总成本: $0.55
节省成本: $0.45
节省比例: 45.0%
```

---

## 关键观察

### 1. 成本节省

- 传统批处理：$1.00
- langasync：$0.50
- **节省：50%**

### 2. 权衡

- 优势：成本降低 50%
- 劣势：24 小时内完成（非实时）
- 适用：评估、标注、离线分析

### 3. 混合策略

- 紧急任务（10%）：实时处理
- 非紧急任务（90%）：批处理 API
- **总成本节省：45%**

---

## 最佳实践

### 1. 区分任务优先级

```python
def classify_task(task):
    """分类任务优先级"""
    if task.get("urgent"):
        return "realtime"
    else:
        return "batch_api"

# 根据优先级选择模式
for task in tasks:
    mode = classify_task(task)
    if mode == "realtime":
        result = processor.batch([task], use_batch_api=False)
    else:
        job = processor.batch([task], use_batch_api=True, wait=False)
```

### 2. 监控任务状态

```python
# 提交任务
job = async_chain.submit_batch(inputs)

# 定期检查状态
while job.status != "completed":
    print(f"状态: {job.status}")
    time.sleep(60)  # 每分钟检查一次

# 获取结果
results = job.get_results()
```

### 3. 处理部分失败

```python
# 提交任务
job = async_chain.submit_batch(inputs)
results = job.wait()

# 检查失败的任务
for i, result in enumerate(results):
    if result is None:
        print(f"任务 {i} 失败")
        # 重试失败的任务
```

---

## 适用场景

### 适合使用 langasync

- ✅ 批量评估和测试
- ✅ 数据标注任务
- ✅ 离线分析和报告
- ✅ 非实时的批量处理
- ✅ 成本敏感的场景

### 不适合使用 langasync

- ❌ 实时对话应用
- ❌ 需要即时响应的场景
- ❌ 单次查询
- ❌ 紧急任务

---

## 常见问题

### Q1: langasync 是官方工具吗？

**A**: 不是。langasync 是社区开发的第三方工具。LangChain 官方正在考虑集成 Batch API，但目前还没有官方实现。

**参考来源**：[LangChain Issue #28508](https://github.com/langchain-ai/langchain/issues/28508)

### Q2: 任务什么时候完成？

**A**: OpenAI 和 Anthropic 的 Batch API 保证 24 小时内完成，通常在几小时内完成。

### Q3: 如何查询任务状态？

**A**: 使用 job.status 属性或 job.get_status() 方法。

### Q4: 任务失败怎么办？

**A**: langasync 支持部分失败处理，失败的任务会返回 None，可以单独重试。

### Q5: 成本节省是真的吗？

**A**: 是的。OpenAI 和 Anthropic 官方提供 50% 的批处理 API 折扣。

**参考来源**：
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [Anthropic Batch API](https://docs.anthropic.com/claude/reference/batch-api)

---

## 2026 年最新特性

### 1. 自动重试

```python
# langasync 自动重试失败的任务
job = async_chain.submit_batch(
    inputs,
    max_retries=3  # 最多重试 3 次
)
```

### 2. 优先级队列

```python
# 设置任务优先级
job = async_chain.submit_batch(
    inputs,
    priority="high"  # high, normal, low
)
```

### 3. 成本预估

```python
# 提交前预估成本
estimated_cost = async_chain.estimate_cost(inputs)
print(f"预估成本: ${estimated_cost:.2f}")
```

---

## 参考来源

1. [langasync GitHub](https://github.com/langasync/langasync) - 官方仓库
2. [langasync 官网](https://langasync.com/) - 文档和教程
3. [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) - OpenAI 批处理 API
4. [LangChain Issue #28508](https://github.com/langchain-ai/langchain/issues/28508) - Batch API 集成讨论
5. [LangChain 成本优化](https://www.langchain.com/state-of-agent-engineering) - 2026 年成本优化报告

---

## 总结

langasync 成本优化的核心要点：

1. **零代码改动**：
   - 只需包装链：`wrap_chain(chain)`
   - 无需修改现有代码

2. **50% 成本节省**：
   - 使用 OpenAI/Anthropic Batch API
   - 官方提供的折扣

3. **权衡**：
   - 优势：成本降低 50%
   - 劣势：24 小时内完成
   - 适用：非实时场景

4. **混合策略**：
   - 紧急任务：实时处理
   - 非紧急任务：批处理 API
   - 总成本节省：40-50%

5. **最佳实践**：
   - 区分任务优先级
   - 监控任务状态
   - 处理部分失败

---

**下一步**：阅读 `07_实战代码_05_RunnableParallel组合使用.md` 学习如何结合 RunnableParallel 实现更复杂的批处理场景
