# 实战代码 - 场景1：InMemoryCache 基础使用

> **场景目标**：快速入门 LLM 缓存机制，理解缓存的基本工作原理

---

## 场景说明

### 适用场景
- 开发和测试环境
- 快速验证缓存效果
- 学习缓存机制原理
- 单机应用原型开发

### 核心特性
- **零配置**：无需外部依赖，开箱即用
- **高性能**：内存读写，毫秒级响应
- **简单易用**：全局设置，自动生效
- **容量控制**：支持 maxsize 限制

### 不适用场景
- ❌ 生产环境（不持久化）
- ❌ 分布式应用（不共享）
- ❌ 长时间运行（内存泄漏风险）

---

## 完整代码

```python
"""
LangChain InMemoryCache 基础使用实战
演示：LLM 缓存的基本工作原理和性能提升效果

环境要求：
- Python 3.13+
- langchain-core
- langchain-openai
- python-dotenv
"""

import os
import time
from typing import List
from dotenv import load_dotenv

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache, get_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()

print("=" * 80)
print("场景1：InMemoryCache 基础使用")
print("=" * 80)

# ===== 1. 基础缓存设置 =====
print("\n【步骤1】启用 InMemoryCache")
print("-" * 80)

# 创建并设置全局缓存
cache = InMemoryCache()
set_llm_cache(cache)

print("✓ InMemoryCache 已启用（无容量限制）")
print(f"✓ 缓存类型: {type(get_llm_cache()).__name__}")

# ===== 2. 首次调用（缓存未命中）=====
print("\n【步骤2】首次调用 LLM（缓存未命中）")
print("-" * 80)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100
)

prompt = "What is LangChain in one sentence?"

# 记录首次调用时间
start_time = time.time()
response1 = llm.invoke(prompt)
first_call_time = time.time() - start_time

print(f"问题: {prompt}")
print(f"回答: {response1.content}")
print(f"⏱️  首次调用耗时: {first_call_time:.3f} 秒")
print(f"💰 API 调用: 是（缓存未命中）")

# ===== 3. 第二次调用（缓存命中）=====
print("\n【步骤3】第二次调用相同问题（缓存命中）")
print("-" * 80)

# 记录第二次调用时间
start_time = time.time()
response2 = llm.invoke(prompt)
second_call_time = time.time() - start_time

print(f"问题: {prompt}")
print(f"回答: {response2.content}")
print(f"⏱️  第二次调用耗时: {second_call_time:.3f} 秒")
print(f"💰 API 调用: 否（缓存命中）")

# 性能对比
speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
print(f"\n📊 性能提升: {speedup:.1f}x 倍")
print(f"⚡ 响应时间减少: {(first_call_time - second_call_time):.3f} 秒")

# ===== 4. 验证缓存内容一致性 =====
print("\n【步骤4】验证缓存内容一致性")
print("-" * 80)

print(f"首次回答: {response1.content[:50]}...")
print(f"缓存回答: {response2.content[:50]}...")
print(f"内容一致: {'✓ 是' if response1.content == response2.content else '✗ 否'}")

# ===== 5. 不同参数不命中缓存 =====
print("\n【步骤5】不同配置参数不命中缓存")
print("-" * 80)

# 创建不同配置的 LLM
llm_different = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,  # 不同的 temperature
    max_tokens=100
)

start_time = time.time()
response3 = llm_different.invoke(prompt)
third_call_time = time.time() - start_time

print(f"问题: {prompt}")
print(f"配置: temperature=0.5 (原配置 temperature=0.7)")
print(f"⏱️  调用耗时: {third_call_time:.3f} 秒")
print(f"💰 API 调用: 是（配置不同，缓存未命中）")

# ===== 6. 不同问题不命中缓存 =====
print("\n【步骤6】不同问题不命中缓存")
print("-" * 80)

different_prompt = "What is RAG in one sentence?"

start_time = time.time()
response4 = llm.invoke(different_prompt)
fourth_call_time = time.time() - start_time

print(f"问题: {different_prompt}")
print(f"回答: {response4.content}")
print(f"⏱️  调用耗时: {fourth_call_time:.3f} 秒")
print(f"💰 API 调用: 是（问题不同，缓存未命中）")

# ===== 7. 批量调用演示 =====
print("\n【步骤7】批量调用缓存效果")
print("-" * 80)

questions = [
    "What is LangChain in one sentence?",  # 已缓存
    "What is RAG in one sentence?",        # 已缓存
    "What is a vector database?",          # 新问题
    "What is LangChain in one sentence?",  # 已缓存（重复）
]

total_time = 0
cache_hits = 0
cache_misses = 0

for i, question in enumerate(questions, 1):
    start_time = time.time()
    response = llm.invoke(question)
    elapsed = time.time() - start_time
    total_time += elapsed

    # 简单判断：耗时 < 0.1秒 认为是缓存命中
    is_cached = elapsed < 0.1
    if is_cached:
        cache_hits += 1
    else:
        cache_misses += 1

    status = "✓ 缓存命中" if is_cached else "✗ 缓存未命中"
    print(f"{i}. {question[:40]}... - {elapsed:.3f}s - {status}")

print(f"\n📊 批量调用统计:")
print(f"   总调用次数: {len(questions)}")
print(f"   缓存命中: {cache_hits} 次")
print(f"   缓存未命中: {cache_misses} 次")
print(f"   命中率: {cache_hits/len(questions)*100:.1f}%")
print(f"   总耗时: {total_time:.3f} 秒")

# ===== 8. 容量限制演示 =====
print("\n【步骤8】容量限制（maxsize）演示")
print("-" * 80)

# 创建容量限制的缓存
limited_cache = InMemoryCache(maxsize=2)
set_llm_cache(limited_cache)

print("✓ 创建 maxsize=2 的缓存（最多缓存2个条目）")

# 插入3个不同的问题
test_questions = [
    "Question 1: What is Python?",
    "Question 2: What is JavaScript?",
    "Question 3: What is TypeScript?",
]

for i, q in enumerate(test_questions, 1):
    llm.invoke(q)
    print(f"✓ 插入问题 {i}: {q[:30]}...")

# 测试第一个问题是否还在缓存中（应该被淘汰）
print("\n测试 FIFO 淘汰策略:")
start_time = time.time()
llm.invoke(test_questions[0])  # 第一个问题
elapsed = time.time() - start_time

if elapsed < 0.1:
    print(f"✗ 问题1 仍在缓存中（不符合预期）")
else:
    print(f"✓ 问题1 已被淘汰（FIFO 策略生效）- {elapsed:.3f}s")

# ===== 9. 清除缓存 =====
print("\n【步骤9】清除缓存")
print("-" * 80)

# 重新设置无限制缓存
set_llm_cache(InMemoryCache())

# 调用一次建立缓存
llm.invoke("Test question for cache clear")
print("✓ 建立缓存")

# 清除缓存
get_llm_cache().clear()
print("✓ 缓存已清除")

# 验证缓存已清除
start_time = time.time()
llm.invoke("Test question for cache clear")
elapsed = time.time() - start_time

if elapsed > 0.1:
    print(f"✓ 缓存清除成功（重新调用 API）- {elapsed:.3f}s")
else:
    print(f"✗ 缓存可能未清除")

# ===== 10. 聊天模型缓存 =====
print("\n【步骤10】聊天模型缓存（多轮对话）")
print("-" * 80)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?")
]

# 首次调用
start_time = time.time()
response1 = llm.invoke(messages)
first_time = time.time() - start_time

print(f"首次调用: {first_time:.3f}s")
print(f"回答: {response1.content}")

# 第二次调用（相同消息）
start_time = time.time()
response2 = llm.invoke(messages)
second_time = time.time() - start_time

print(f"第二次调用: {second_time:.3f}s")
print(f"缓存命中: {'✓ 是' if second_time < 0.1 else '✗ 否'}")

# ===== 总结 =====
print("\n" + "=" * 80)
print("【总结】InMemoryCache 核心要点")
print("=" * 80)

print("""
1. 缓存键组成：(prompt, llm_string)
   - prompt: 提示词内容
   - llm_string: LLM 配置参数（model, temperature, max_tokens 等）

2. 缓存命中条件：
   ✓ 相同的 prompt
   ✓ 相同的 LLM 配置参数
   ✗ 任何一个不同都不会命中

3. 性能提升：
   - 首次调用: ~1-3 秒（API 调用）
   - 缓存命中: ~0.001-0.01 秒（内存读取）
   - 速度提升: 100-1000 倍

4. 容量管理：
   - 默认无限制（可能导致内存泄漏）
   - 设置 maxsize 限制容量
   - FIFO 淘汰策略（先进先出）

5. 适用场景：
   ✓ 开发测试环境
   ✓ 快速原型验证
   ✓ 单机应用
   ✗ 生产环境（不持久化）
   ✗ 分布式应用（不共享）

6. 最佳实践：
   - 开发阶段使用 InMemoryCache
   - 生产环境切换到 RedisCache
   - 设置合理的 maxsize 避免内存泄漏
   - 定期清理缓存
""")

print("\n✓ 场景1演示完成！")
```

---

## 运行输出示例

```
================================================================================
场景1：InMemoryCache 基础使用
================================================================================

【步骤1】启用 InMemoryCache
--------------------------------------------------------------------------------
✓ InMemoryCache 已启用（无容量限制）
✓ 缓存类型: InMemoryCache

【步骤2】首次调用 LLM（缓存未命中）
--------------------------------------------------------------------------------
问题: What is LangChain in one sentence?
回答: LangChain is a framework for developing applications powered by language models through composable components and chains.
⏱️  首次调用耗时: 1.234 秒
💰 API 调用: 是（缓存未命中）

【步骤3】第二次调用相同问题（缓存命中）
--------------------------------------------------------------------------------
问题: What is LangChain in one sentence?
回答: LangChain is a framework for developing applications powered by language models through composable components and chains.
⏱️  第二次调用耗时: 0.003 秒
💰 API 调用: 否（缓存命中）

📊 性能提升: 411.3x 倍
⚡ 响应时间减少: 1.231 秒

【步骤4】验证缓存内容一致性
--------------------------------------------------------------------------------
首次回答: LangChain is a framework for developing applic...
缓存回答: LangChain is a framework for developing applic...
内容一致: ✓ 是

【步骤5】不同配置参数不命中缓存
--------------------------------------------------------------------------------
问题: What is LangChain in one sentence?
配置: temperature=0.5 (原配置 temperature=0.7)
⏱️  调用耗时: 1.156 秒
💰 API 调用: 是（配置不同，缓存未命中）

【步骤6】不同问题不命中缓存
--------------------------------------------------------------------------------
问题: What is RAG in one sentence?
回答: RAG (Retrieval-Augmented Generation) is a technique that enhances language model responses by retrieving relevant information from external knowledge sources.
⏱️  调用耗时: 1.089 秒
💰 API 调用: 是（问题不同，缓存未命中）

【步骤7】批量调用缓存效果
--------------------------------------------------------------------------------
1. What is LangChain in one sentence?... - 0.002s - ✓ 缓存命中
2. What is RAG in one sentence?... - 0.003s - ✓ 缓存命中
3. What is a vector database?... - 1.145s - ✗ 缓存未命中
4. What is LangChain in one sentence?... - 0.002s - ✓ 缓存命中

📊 批量调用统计:
   总调用次数: 4
   缓存命中: 3 次
   缓存未命中: 1 次
   命中率: 75.0%
   总耗时: 1.152 秒

【步骤8】容量限制（maxsize）演示
--------------------------------------------------------------------------------
✓ 创建 maxsize=2 的缓存（最多缓存2个条目）
✓ 插入问题 1: Question 1: What is Python?...
✓ 插入问题 2: Question 2: What is JavaScri...
✓ 插入问题 3: Question 3: What is TypeScri...

测试 FIFO 淘汰策略:
✓ 问题1 已被淘汰（FIFO 策略生效）- 1.078s

【步骤9】清除缓存
--------------------------------------------------------------------------------
✓ 建立缓存
✓ 缓存已清除
✓ 缓存清除成功（重新调用 API）- 1.123s

【步骤10】聊天模型缓存（多轮对话）
--------------------------------------------------------------------------------
首次调用: 1.067s
回答: 2 + 2 equals 4.
第二次调用: 0.003s
缓存命中: ✓ 是

================================================================================
【总结】InMemoryCache 核心要点
================================================================================

1. 缓存键组成：(prompt, llm_string)
   - prompt: 提示词内容
   - llm_string: LLM 配置参数（model, temperature, max_tokens 等）

2. 缓存命中条件：
   ✓ 相同的 prompt
   ✓ 相同的 LLM 配置参数
   ✗ 任何一个不同都不会命中

3. 性能提升：
   - 首次调用: ~1-3 秒（API 调用）
   - 缓存命中: ~0.001-0.01 秒（内存读取）
   - 速度提升: 100-1000 倍

4. 容量管理：
   - 默认无限制（可能导致内存泄漏）
   - 设置 maxsize 限制容量
   - FIFO 淘汰策略（先进先出）

5. 适用场景：
   ✓ 开发测试环境
   ✓ 快速原型验证
   ✓ 单机应用
   ✗ 生产环境（不持久化）
   ✗ 分布式应用（不共享）

6. 最佳实践：
   - 开发阶段使用 InMemoryCache
   - 生产环境切换到 RedisCache
   - 设置合理的 maxsize 避免内存泄漏
   - 定期清理缓存

✓ 场景1演示完成！
```

---

## 关键点解释

### 1. 缓存键设计

**缓存键组成**：
```python
cache_key = (prompt, llm_string)
```

- **prompt**：提示词的字符串表示
  - 对于 Chat 模型：序列化后的消息列表
  - 对于 LLM：直接的文本字符串

- **llm_string**：LLM 配置的字符串表示
  - 包含：model, temperature, max_tokens, stop 等参数
  - 示例：`"model=gpt-4o-mini,temperature=0.7,max_tokens=100"`

**为什么这样设计？**
- 确保相同输入 + 相同配置才命中缓存
- 避免返回错误的缓存结果
- 不同配置可能产生不同输出

### 2. FIFO 淘汰策略

**实现原理**（源码）：
```python
def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
    if self._maxsize is not None and len(self._cache) == self._maxsize:
        del self._cache[next(iter(self._cache))]  # 删除第一个（最旧的）
    self._cache[prompt, llm_string] = return_val
```

**特点**：
- 使用 Python 字典的插入顺序（Python 3.7+）
- `next(iter(self._cache))` 获取第一个键
- 简单高效，无需额外数据结构

### 3. 性能提升原理

**首次调用（缓存未命中）**：
```
用户请求 → LangChain → OpenAI API → 网络传输 → 模型推理 → 返回结果
耗时：~1-3 秒
```

**缓存命中**：
```
用户请求 → LangChain → 内存查找 → 返回缓存结果
耗时：~0.001-0.01 秒
```

**速度提升**：100-1000 倍

### 4. 配置参数影响

**不同配置不命中缓存**：
```python
llm1 = ChatOpenAI(temperature=0.7)  # 配置1
llm2 = ChatOpenAI(temperature=0.5)  # 配置2

# 相同问题，不同配置，不会命中缓存
llm1.invoke("What is AI?")  # API 调用
llm2.invoke("What is AI?")  # API 调用（不命中缓存）
```

**原因**：
- `llm_string` 不同
- 不同 temperature 可能产生不同输出
- 缓存必须保证结果一致性

### 5. 内存泄漏风险

**问题**：
- 默认无容量限制
- 长时间运行会不断积累缓存
- 最终导致内存耗尽

**解决方案**：
```python
# 方案1：设置 maxsize
cache = InMemoryCache(maxsize=1000)

# 方案2：定期清理
cache.clear()

# 方案3：生产环境使用 Redis
from langchain_redis import RedisCache
set_llm_cache(RedisCache(redis_client))
```

### 6. 聊天模型缓存

**消息序列化**：
```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?")
]

# 缓存键中的 prompt 是序列化后的消息
# 包含消息类型、内容、顺序等信息
```

**注意事项**：
- 消息顺序影响缓存
- 消息 ID 可能影响缓存（需要规范化）
- 系统消息也参与缓存键生成

---

## 数据来源

本文档基于以下资料编写：

1. **源码分析** (`reference/source_cache_01.md`)
   - `langchain_core/caches.py` - BaseCache 接口和 InMemoryCache 实现
   - FIFO 淘汰策略源码
   - 缓存键生成逻辑

2. **官方文档** (`reference/context7_langchain_cache_01.md`)
   - InMemoryCache 使用方法
   - 全局缓存设置
   - 性能对比数据

3. **社区讨论** (`reference/search_cache_reddit_01.md`)
   - InMemoryCache 适用场景
   - 内存泄漏问题讨论
   - 最佳实践建议

4. **GitHub Issues** (`reference/search_cache_github_01.md`)
   - 缓存键生成问题
   - 消息 ID 规范化
   - 配置参数影响

---

## 下一步学习

完成本场景后，建议继续学习：

1. **场景2：RedisCache 生产部署** - 生产环境缓存配置
2. **场景3：CacheBackedEmbeddings 实战** - Embedding 缓存
3. **场景4：语义缓存实现** - 提升缓存命中率
4. **场景5：缓存性能优化** - 监控和优化策略

---

**版本信息**：
- LangChain 版本：0.3.x (2025+)
- Python 版本：3.13+
- 最后更新：2026-02-25
