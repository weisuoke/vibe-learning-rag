---
type: source_code_analysis
source: sourcecode/langchain
analyzed_files:
  - libs/core/langchain_core/runnables/base.py
  - libs/core/langchain_core/runnables/fallbacks.py
  - libs/core/langchain_core/runnables/retry.py
  - libs/core/langchain_core/runnables/branch.py
  - libs/core/langchain_core/runnables/configurable.py
  - libs/core/langchain_core/runnables/passthrough.py
analyzed_at: 2026-02-27
knowledge_point: 15_组件组合最佳实践
---

# 源码分析：LangChain 组件组合核心模式

## 分析的文件

### 1. base.py - 核心组合引擎 (6,261 行)

**关键类：**

#### Runnable (基类, line 124)
- 所有组件的基类，定义了 invoke/batch/stream/astream 等标准方法
- 组合原语：RunnableSequence (|) 和 RunnableParallel ({})
- 内置优化：batch 默认使用线程池并行执行，async 方法使用 asyncio

#### RunnableSequence (line 2817)
- **最重要的组合操作符**，几乎所有 chain 都使用
- 通过 `|` 操作符构建，输出作为下一个的输入
- 自动支持 sync/async/batch/streaming
- batch 和 abatch 使用线程池/asyncio gather 优化 IO 密集型操作
- **流式传播**：如果所有组件实现 transform，整个序列可以流式传输
- **注意**：RunnableLambda 默认不支持 transform，放置位置影响流式行为

#### RunnableParallel (line 3565)
- 并发执行多个 Runnable，提供相同输入
- 可通过 dict 字面量在序列中自动创建
- 三种构建方式：dict 字面量、RunnableParallel(dict)、RunnableParallel(key=value)

### 2. fallbacks.py - 降级策略 (664 行)

#### RunnableWithFallbacks (line 36)
- 当主 Runnable 失败时，按顺序尝试备选方案
- 支持 `exceptions_to_handle` 指定处理的异常类型
- 支持 `exception_key` 将异常传递给 fallback
- 通过 `.with_fallbacks()` 方法使用
- 支持 invoke/ainvoke/batch/abatch/stream/astream
- **__getattr__ 代理**：对 fallback 对象调用方法时，自动对所有 fallback 调用相同方法

### 3. retry.py - 重试机制 (379 行)

#### RunnableRetry (line 48)
- 基于 tenacity 库实现指数退避重试
- 通过 `.with_retry()` 方法使用
- 参数：retry_exception_types, wait_exponential_jitter, max_attempt_number
- **最佳实践**：重试范围尽量小，只重试可能失败的组件，不要重试整个链
- stream() 和 transform() 不支持重试（重试流不直观）
- batch 重试时只重试失败的元素，已成功的不会重复执行

### 4. branch.py - 条件路由 (461 行)

#### RunnableBranch (line 42)
- 基于条件选择执行分支
- 接受 (condition, runnable) 元组列表 + 默认分支
- 第一个为 True 的条件被选中
- 支持 invoke/ainvoke/stream/astream

### 5. configurable.py - 动态配置 (716 行)

#### DynamicRunnable (line 49)
- 运行时动态配置的 Runnable
- 通过 `configurable_fields()` 或 `configurable_alternatives()` 创建
- 支持字段级配置和替代方案选择

### 6. passthrough.py - 数据传递 (841 行)

#### RunnablePassthrough (line 74)
- 透传输入，可选添加额外键
- `RunnablePassthrough.assign()` 在保留原始输入的同时添加新字段
- 常用于 RAG 链中传递 question 同时检索 context

## 关键设计模式

### 组合模式总结

| 模式 | 类 | 操作符 | 用途 |
|------|-----|--------|------|
| 串行 | RunnableSequence | `\|` | 顺序执行 |
| 并行 | RunnableParallel | `{}` | 并发执行 |
| 条件 | RunnableBranch | - | 条件路由 |
| 降级 | RunnableWithFallbacks | `.with_fallbacks()` | 错误恢复 |
| 重试 | RunnableRetry | `.with_retry()` | 瞬态错误处理 |
| 配置 | DynamicRunnable | `.configurable_fields()` | 运行时配置 |
| 透传 | RunnablePassthrough | - | 数据传递 |
| 赋值 | RunnableAssign | `.assign()` | 添加字段 |

### 性能优化要点（从源码提取）

1. **batch 自动并行**：RunnableSequence.batch 对每个步骤调用 batch，利用线程池
2. **流式传播**：组件实现 transform 方法即可支持端到端流式
3. **RunnableLambda 流式限制**：不支持 transform，会阻断流式传播
4. **重试粒度**：只在可能失败的组件上添加重试，不要包装整个链
5. **fallback batch 优化**：只对失败的输入尝试 fallback，成功的不重复
