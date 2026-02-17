# 核心概念 01：Deque 基础操作

> 深入理解 Deque 的实现原理、API 和性能特性

---

## 概述

Deque（Double-Ended Queue，双端队列）是 Python `collections` 模块提供的高性能数据结构，支持从两端进行 O(1) 时间复杂度的插入和删除操作。

**核心特性：**
- 双端 O(1) 操作
- 线程安全的原子操作
- 固定大小自动淘汰（maxlen）
- 内存高效的实现

---

## 1. Deque 的实现原理

### 1.1 底层数据结构

Python 的 deque 使用**双向链表**实现，每个节点包含：
- 数据块（block）：存储多个元素的数组
- 前向指针：指向前一个块
- 后向指针：指向后一个块

```
双向链表结构：
[Block 1] ←→ [Block 2] ←→ [Block 3]
   ↓            ↓            ↓
[e1,e2,e3]  [e4,e5,e6]  [e7,e8,e9]
```

**为什么不是简单的双向链表？**
- 简单双向链表：每个元素一个节点，内存开销大
- 块状双向链表：每个块存储多个元素，减少内存开销

### 1.2 性能分析

| 操作 | 时间复杂度 | 原因 |
|------|-----------|------|
| `append(x)` | O(1) | 在右端块添加元素 |
| `appendleft(x)` | O(1) | 在左端块添加元素 |
| `pop()` | O(1) | 从右端块删除元素 |
| `popleft()` | O(1) | 从左端块删除元素 |
| `d[i]` | O(n) | 需要遍历块找到索引 |
| `len(d)` | O(1) | 维护计数器 |
| `x in d` | O(n) | 需要遍历所有元素 |

---

## 2. Deque 的核心 API

### 2.1 创建 Deque

```python
from collections import deque

# 空 deque
d = deque()

# 从可迭代对象创建
d = deque([1, 2, 3, 4, 5])

# 固定大小 deque
d = deque(maxlen=10)

# 从可迭代对象创建固定大小 deque
d = deque([1, 2, 3], maxlen=5)
```

### 2.2 添加元素

```python
from collections import deque

d = deque([1, 2, 3])

# 右端添加单个元素
d.append(4)  # [1, 2, 3, 4]

# 左端添加单个元素
d.appendleft(0)  # [0, 1, 2, 3, 4]

# 右端添加多个元素
d.extend([5, 6])  # [0, 1, 2, 3, 4, 5, 6]

# 左端添加多个元素
d.extendleft([-2, -1])  # [-1, -2, 0, 1, 2, 3, 4, 5, 6]
# 注意：extendleft 会反转元素顺序
```

### 2.3 删除元素

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

# 右端删除并返回
x = d.pop()  # x = 5, d = [1, 2, 3, 4]

# 左端删除并返回
x = d.popleft()  # x = 1, d = [2, 3, 4]

# 删除指定值（第一次出现）
d.remove(3)  # d = [2, 4]

# 清空 deque
d.clear()  # d = []
```

### 2.4 访问元素

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

# 索引访问
print(d[0])   # 1
print(d[-1])  # 5

# 不支持切片
# d[1:3]  # TypeError

# 转换为列表后切片
print(list(d)[1:3])  # [2, 3]
```

### 2.5 旋转操作

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

# 向右旋转 2 步
d.rotate(2)  # [4, 5, 1, 2, 3]

# 向左旋转 1 步
d.rotate(-1)  # [5, 1, 2, 3, 4]
```

### 2.6 其他操作

```python
from collections import deque

d = deque([1, 2, 3, 2, 4])

# 计数
print(d.count(2))  # 2

# 反转
d.reverse()  # [4, 2, 3, 2, 1]

# 复制
d2 = d.copy()

# 长度
print(len(d))  # 5

# 检查是否为空
if d:
    print("非空")
```

---

## 3. maxlen 参数详解

### 3.1 基本用法

```python
from collections import deque

# 创建固定大小 deque
d = deque(maxlen=3)

d.append(1)  # [1]
d.append(2)  # [1, 2]
d.append(3)  # [1, 2, 3]
d.append(4)  # [2, 3, 4]  ← 自动移除最左边的 1

print(d.maxlen)  # 3
```

### 3.2 自动淘汰机制

```python
from collections import deque

d = deque(maxlen=5)

# append 时自动淘汰左端
for i in range(10):
    d.append(i)
print(d)  # deque([5, 6, 7, 8, 9], maxlen=5)

# appendleft 时自动淘汰右端
d = deque(maxlen=5)
for i in range(10):
    d.appendleft(i)
print(d)  # deque([9, 8, 7, 6, 5], maxlen=5)
```

### 3.3 maxlen 的不可变性

```python
from collections import deque

d = deque(maxlen=3)

# maxlen 是只读属性
# d.maxlen = 5  # AttributeError

# 如果需要改变大小，必须重新创建
d = deque(d, maxlen=5)
```

---

## 4. 线程安全性

### 4.1 原子操作

Deque 的以下操作是线程安全的（原子操作）：
- `append()`
- `appendleft()`
- `pop()`
- `popleft()`
- `extend()`
- `extendleft()`

```python
from collections import deque
import threading

d = deque()

def producer():
    for i in range(1000):
        d.append(i)

def consumer():
    for _ in range(1000):
        try:
            d.popleft()
        except IndexError:
            pass

# 多线程安全
threads = [
    threading.Thread(target=producer),
    threading.Thread(target=consumer)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

### 4.2 非原子操作

以下操作不是线程安全的：
- 索引访问 `d[i]`
- 切片操作
- `len(d)` 与其他操作的组合

```python
from collections import deque
import threading

d = deque([1, 2, 3])

def unsafe_operation():
    # ❌ 不是原子操作
    if len(d) > 0:  # 检查长度
        d.pop()     # 删除元素
    # 在多线程环境下，可能在检查和删除之间被其他线程修改

# 需要使用锁保护
import threading

lock = threading.Lock()

def safe_operation():
    with lock:
        if len(d) > 0:
            d.pop()
```

---

## 5. 性能对比

### 5.1 Deque vs List

```python
import time
from collections import deque

n = 10000

# 测试左端删除
# List: O(n)
start = time.time()
lst = list(range(n))
for _ in range(n):
    lst.pop(0)
list_time = time.time() - start

# Deque: O(1)
start = time.time()
d = deque(range(n))
for _ in range(n):
    d.popleft()
deque_time = time.time() - start

print(f"List:  {list_time:.4f}s")
print(f"Deque: {deque_time:.4f}s")
print(f"Deque 快 {list_time / deque_time:.0f} 倍")

# 输出示例：
# List:  2.5000s
# Deque: 0.0010s
# Deque 快 2500 倍
```

### 5.2 内存使用

```python
import sys
from collections import deque

# List 内存使用
lst = list(range(1000))
print(f"List 内存: {sys.getsizeof(lst)} bytes")

# Deque 内存使用
d = deque(range(1000))
print(f"Deque 内存: {sys.getsizeof(d)} bytes")

# Deque 通常使用更多内存（因为块状结构）
# 但在频繁双端操作时，性能优势远超内存开销
```

---

## 6. AI Agent 应用场景

### 6.1 对话历史管理

```python
from collections import deque
from typing import Dict, List

class ConversationMemory:
    """AI Agent 对话记忆管理器"""

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 保留的最大对话轮数
        """
        self.messages = deque(maxlen=max_turns * 2)

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.messages.append({
            "role": "assistant",
            "content": content
        })

    def get_context(self) -> List[Dict]:
        """获取当前上下文"""
        return list(self.messages)

    def clear(self):
        """清空历史"""
        self.messages.clear()

# 使用示例
memory = ConversationMemory(max_turns=3)

# 添加 5 轮对话
for i in range(5):
    memory.add_user_message(f"问题 {i}")
    memory.add_assistant_message(f"回答 {i}")

# 只保留最近 3 轮（6 条消息）
context = memory.get_context()
print(f"消息数: {len(context)}")  # 6
```

**2025-2026 实际应用：**

**OpenAI Agents SDK (2026):**
```python
from collections import deque
from openai import OpenAI

class TrimmingSession:
    """OpenAI SDK 风格的会话管理"""

    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)
        self.client = OpenAI()

    def chat(self, user_message: str) -> str:
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})

        # 调用 LLM
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=list(self.messages)
        )

        # 添加助手消息
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message
```

**来源**: [OpenAI Agents SDK - Session Memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
**时间**: 2026

### 6.2 流式输出缓冲

```python
from collections import deque
import asyncio

class StreamBuffer:
    """流式输出缓冲区"""

    def __init__(self, buffer_size: int = 100):
        self.buffer = deque(maxlen=buffer_size)

    async def add_chunk(self, chunk: str):
        """添加输出块"""
        self.buffer.append(chunk)

    def get_recent_output(self, n: int = 10) -> str:
        """获取最近 n 个输出块"""
        recent = list(self.buffer)[-n:]
        return "".join(recent)

    def get_all_output(self) -> str:
        """获取所有输出"""
        return "".join(self.buffer)

# 使用示例
buffer = StreamBuffer(buffer_size=50)

async def stream_response():
    chunks = ["Hello", " ", "world", "!", " ", "How", " ", "are", " ", "you", "?"]
    for chunk in chunks:
        await buffer.add_chunk(chunk)
        await asyncio.sleep(0.1)

asyncio.run(stream_response())
print(buffer.get_all_output())  # "Hello world! How are you?"
```

### 6.3 API 限流器

```python
from collections import deque
import time

class RateLimiter:
    """滑动窗口限流器"""

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Args:
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def allow_request(self) -> bool:
        """检查是否允许请求"""
        now = time.time()

        # 移除窗口外的请求
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()

        # 检查是否超过限制
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

# 使用示例
limiter = RateLimiter(max_requests=3, window_seconds=1)

for i in range(5):
    if limiter.allow_request():
        print(f"请求 {i}: 允许")
    else:
        print(f"请求 {i}: 拒绝（超过限流）")
    time.sleep(0.2)
```

---

## 7. 常见错误与最佳实践

### 7.1 错误：忘记 Deque 不支持切片

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

# ❌ 错误
# result = d[1:3]  # TypeError

# ✅ 正确
result = list(d)[1:3]  # [2, 3]

# ✅ 或使用 itertools
from itertools import islice
result = list(islice(d, 1, 3))  # [2, 3]
```

### 7.2 错误：混淆 append 和 appendleft

```python
from collections import deque

# ❌ 错误：想在左边添加，却用了 append
d = deque([1, 2, 3])
d.append(0)  # [1, 2, 3, 0]  ← 添加到右边了

# ✅ 正确
d = deque([1, 2, 3])
d.appendleft(0)  # [0, 1, 2, 3]
```

### 7.3 错误：空 Deque 调用 pop

```python
from collections import deque

d = deque()

# ❌ 错误
# d.popleft()  # IndexError: pop from an empty deque

# ✅ 正确：先检查
if d:
    d.popleft()

# ✅ 或使用 try-except
try:
    d.popleft()
except IndexError:
    print("deque 为空")
```

### 7.4 最佳实践：选择合适的数据结构

```python
# 场景1：需要频繁双端操作 → Deque
from collections import deque
queue = deque()
queue.append(item)
queue.popleft()

# 场景2：需要随机访问 → List
lst = [1, 2, 3, 4, 5]
print(lst[2])  # O(1)

# 场景3：需要固定大小 + 自动淘汰 → Deque with maxlen
memory = deque(maxlen=100)
```

---

## 8. 性能优化技巧

### 8.1 批量操作

```python
from collections import deque

# ❌ 低效：逐个添加
d = deque()
for i in range(1000):
    d.append(i)

# ✅ 高效：批量添加
d = deque(range(1000))

# ✅ 或使用 extend
d = deque()
d.extend(range(1000))
```

### 8.2 避免不必要的转换

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

# ❌ 低效：频繁转换
for _ in range(100):
    lst = list(d)
    # 处理 lst

# ✅ 高效：只在必要时转换
# 直接在 deque 上操作
for item in d:
    # 处理 item
    pass
```

### 8.3 使用 maxlen 而非手动管理

```python
from collections import deque

# ❌ 低效：手动管理大小
d = deque()
max_size = 100
for item in items:
    d.append(item)
    if len(d) > max_size:
        d.popleft()

# ✅ 高效：使用 maxlen
d = deque(maxlen=100)
for item in items:
    d.append(item)  # 自动淘汰
```

---

## 学习检查清单

- [ ] 理解 Deque 的块状双向链表实现
- [ ] 掌握所有核心 API（append, appendleft, pop, popleft）
- [ ] 理解 maxlen 的自动淘汰机制
- [ ] 知道哪些操作是线程安全的
- [ ] 能够选择合适的数据结构（Deque vs List）
- [ ] 理解 Deque 在 AI Agent 中的应用
- [ ] 避免常见错误（切片、空 deque 等）

---

## 下一步学习

### 深入算法
→ **03_核心概念_02_单调队列原理.md** - 学习单调队列算法

### AI Agent 应用
→ **03_核心概念_05_AI_Agent短期记忆.md** - AI Agent 记忆管理

### 实战代码
→ **07_实战代码_01_Deque基础实现.md** - 完整代码示例

---

**版本**: v1.0
**最后更新**: 2026-02-13
**适用于**: Python 3.13+, AI Agent 开发, RAG 系统
