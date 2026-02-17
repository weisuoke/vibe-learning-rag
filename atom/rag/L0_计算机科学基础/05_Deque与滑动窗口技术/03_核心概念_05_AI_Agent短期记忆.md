# 核心概念 05：AI Agent 短期记忆

> 深入理解 AI Agent 短期记忆系统的设计与实现

---

## 概述

AI Agent 的短期记忆（Short-Term Memory）是指 Agent 在对话过程中保留的最近交互历史。由于 LLM 的上下文窗口有限，短期记忆管理是 AI Agent 开发的核心挑战之一。

**核心问题：**
- LLM 上下文窗口有限（4K-200K tokens）
- 对话可能无限长
- 需要保留最相关的上下文

**解决方案：** 使用 Deque + 滑动窗口实现高效的短期记忆管理。

---

## 1. 短期记忆的基本概念

### 1.1 什么是短期记忆？

**定义：** AI Agent 在当前会话中保留的最近对话历史，用于维持对话连贯性和上下文理解。

```
短期记忆 vs 长期记忆：

短期记忆（Short-Term Memory）：
- 保留最近 N 轮对话
- 存储在内存中
- 快速访问（O(1)）
- 自动淘汰旧对话

长期记忆（Long-Term Memory）：
- 保留所有历史对话
- 存储在数据库/向量库
- 需要检索（O(log n)）
- 手动管理
```

### 1.2 为什么需要短期记忆？

**核心原因：** LLM 的上下文窗口限制

```
GPT-4:     8K / 32K / 128K tokens
Claude 3:  200K tokens
Gemini:    1M tokens

实际对话：可能有数千轮
→ 必须管理上下文窗口
```

**示例：**
```python
# ❌ 错误：无限增长
messages = []
for turn in range(1000):
    messages.append({"role": "user", "content": f"问题 {turn}"})
    messages.append({"role": "assistant", "content": f"回答 {turn}"})
# 内存持续增长，最终超过 LLM 上下文窗口

# ✅ 正确：固定大小
from collections import deque

messages = deque(maxlen=20)  # 只保留最近 10 轮对话
for turn in range(1000):
    messages.append({"role": "user", "content": f"问题 {turn}"})
    messages.append({"role": "assistant", "content": f"回答 {turn}"})
# 内存固定，自动淘汰旧对话
```

---

## 2. 基于 Deque 的短期记忆实现

### 2.1 基础实现

```python
from collections import deque
from typing import Dict, List

class ShortTermMemory:
    """AI Agent 短期记忆管理器"""

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 保留的最大对话轮数
        """
        # 每轮对话包含 user + assistant 两条消息
        self.messages = deque(maxlen=max_turns * 2)
        self.max_turns = max_turns

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
        """清空记忆"""
        self.messages.clear()

    def get_turn_count(self) -> int:
        """获取当前对话轮数"""
        return len(self.messages) // 2

# 使用示例
memory = ShortTermMemory(max_turns=3)

# 添加 5 轮对话
for i in range(5):
    memory.add_user_message(f"问题 {i}")
    memory.add_assistant_message(f"回答 {i}")

# 只保留最近 3 轮（6 条消息）
context = memory.get_context()
print(f"消息数: {len(context)}")  # 6
print(f"轮数: {memory.get_turn_count()}")  # 3
```

### 2.2 带系统提示的实现

```python
from collections import deque
from typing import Dict, List, Optional

class ShortTermMemoryWithSystem:
    """带系统提示的短期记忆管理器"""

    def __init__(self, system_prompt: str, max_turns: int = 10):
        """
        Args:
            system_prompt: 系统提示词
            max_turns: 保留的最大对话轮数
        """
        self.system_prompt = system_prompt
        self.messages = deque(maxlen=max_turns * 2)
        self.max_turns = max_turns

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
        """获取当前上下文（包含系统提示）"""
        return [
            {"role": "system", "content": self.system_prompt},
            *list(self.messages)
        ]

    def update_system_prompt(self, new_prompt: str):
        """更新系统提示"""
        self.system_prompt = new_prompt

# 使用示例
memory = ShortTermMemoryWithSystem(
    system_prompt="你是一个友好的 AI 助手。",
    max_turns=5
)

memory.add_user_message("你好")
memory.add_assistant_message("你好！有什么可以帮你的？")

context = memory.get_context()
print(f"上下文长度: {len(context)}")  # 3 (system + user + assistant)
```

---

## 3. 2025-2026 年主流框架实现

### 3.1 OpenAI Agents SDK (2026)

```python
from collections import deque
from openai import OpenAI

class TrimmingSession:
    """OpenAI Agents SDK 风格的会话管理"""

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: 保留的最大对话轮数
        """
        self.messages = deque(maxlen=max_turns * 2)
        self.client = OpenAI()

    def chat(self, user_message: str, model: str = "gpt-4") -> str:
        """
        发送消息并获取响应

        Args:
            user_message: 用户消息
            model: 模型名称

        Returns:
            助手响应
        """
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})

        # 调用 LLM
        response = self.client.chat.completions.create(
            model=model,
            messages=list(self.messages)
        )

        # 添加助手消息
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def get_history(self) -> list:
        """获取对话历史"""
        return list(self.messages)

# 使用示例
session = TrimmingSession(max_turns=5)

response1 = session.chat("什么是 Python？")
print(response1)

response2 = session.chat("它有什么特点？")
print(response2)

print(f"历史消息数: {len(session.get_history())}")
```

**来源**: [OpenAI Agents SDK - Session Memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
**时间**: 2026
**关键点**: 使用 `deque(maxlen=N)` 实现自动淘汰的对话历史

### 3.2 LangGraph (2025-2026)

```python
from collections import deque
from typing import Dict, List

class SlidingWindowMemory:
    """LangGraph 风格的滑动窗口记忆"""

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 窗口大小（消息数量）
        """
        self.messages = deque(maxlen=window_size)

    def add_message(self, message: Dict):
        """添加消息"""
        self.messages.append(message)

    def get_context(self) -> List[Dict]:
        """获取当前上下文"""
        return list(self.messages)

    def clear(self):
        """清空记忆"""
        self.messages.clear()

# 使用示例
memory = SlidingWindowMemory(window_size=10)

for i in range(15):
    memory.add_message({"role": "user", "content": f"消息 {i}"})

print(f"消息数: {len(memory.get_context())}")  # 10
```

**来源**: [LangGraph Message History with Sliding Windows](https://aiproduct.engineer/tutorials/langgraph-tutorial-message-history-management-with-sliding-windows-unit-12-exercise-3)
**时间**: 2025-2026
**关键点**: 滑动窗口模式管理对话历史

### 3.3 LangChain ConversationBufferWindowMemory

```python
from collections import deque
from typing import Dict, List

class ConversationBufferWindowMemory:
    """LangChain 风格的窗口记忆"""

    def __init__(self, k: int = 5):
        """
        Args:
            k: 保留的对话轮数
        """
        self.messages = deque(maxlen=k * 2)
        self.k = k

    def save_context(self, inputs: Dict, outputs: Dict):
        """保存对话上下文"""
        self.messages.append({"role": "user", "content": inputs["input"]})
        self.messages.append({"role": "assistant", "content": outputs["output"]})

    def load_memory_variables(self) -> Dict:
        """加载记忆变量"""
        return {"history": list(self.messages)}

    def clear(self):
        """清空记忆"""
        self.messages.clear()

# 使用示例
memory = ConversationBufferWindowMemory(k=3)

memory.save_context(
    {"input": "你好"},
    {"output": "你好！有什么可以帮你的？"}
)

memory.save_context(
    {"input": "今天天气怎么样？"},
    {"output": "今天天气晴朗。"}
)

history = memory.load_memory_variables()
print(f"历史消息数: {len(history['history'])}")  # 4
```

---

## 4. 高级特性

### 4.1 带元数据的记忆管理

```python
from collections import deque
from typing import Dict, List, Optional
import time

class MemoryWithMetadata:
    """带元数据的记忆管理器"""

    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """添加带元数据的消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.messages.append(message)

    def get_context(self, include_metadata: bool = False) -> List[Dict]:
        """获取上下文"""
        if include_metadata:
            return list(self.messages)
        else:
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.messages
            ]

# 使用示例
memory = MemoryWithMetadata(max_turns=5)

memory.add_message(
    "user",
    "什么是 Python？",
    metadata={"source": "web", "user_id": "123"}
)

memory.add_message(
    "assistant",
    "Python 是一种编程语言。",
    metadata={"model": "gpt-4", "tokens": 50}
)

context = memory.get_context(include_metadata=True)
print(context[0]["metadata"])  # {'source': 'web', 'user_id': '123'}
```

### 4.2 条件性记忆保留

```python
from collections import deque
from typing import Dict, List, Callable

class ConditionalMemory:
    """条件性记忆保留"""

    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)
        self.important_messages = []  # 永久保留的重要消息

    def add_message(
        self,
        role: str,
        content: str,
        is_important: bool = False
    ):
        """添加消息"""
        message = {"role": role, "content": content}

        if is_important:
            self.important_messages.append(message)
        else:
            self.messages.append(message)

    def get_context(self) -> List[Dict]:
        """获取上下文（重要消息 + 最近消息）"""
        return self.important_messages + list(self.messages)

# 使用示例
memory = ConditionalMemory(max_turns=3)

# 添加重要消息（永久保留）
memory.add_message(
    "system",
    "你是一个专业的 Python 开发助手。",
    is_important=True
)

# 添加普通消息（滑动窗口）
for i in range(5):
    memory.add_message("user", f"问题 {i}")
    memory.add_message("assistant", f"回答 {i}")

context = memory.get_context()
print(f"上下文长度: {len(context)}")  # 7 (1 important + 6 recent)
```

### 4.3 分层记忆管理

```python
from collections import deque
from typing import Dict, List

class HierarchicalMemory:
    """分层记忆管理"""

    def __init__(
        self,
        short_term_size: int = 5,
        mid_term_size: int = 20
    ):
        """
        Args:
            short_term_size: 短期记忆大小（最近对话）
            mid_term_size: 中期记忆大小（摘要）
        """
        self.short_term = deque(maxlen=short_term_size * 2)
        self.mid_term = deque(maxlen=mid_term_size)

    def add_message(self, role: str, content: str):
        """添加消息"""
        message = {"role": role, "content": content}
        self.short_term.append(message)

    def summarize_and_archive(self, summary: str):
        """摘要并归档到中期记忆"""
        self.mid_term.append({
            "role": "system",
            "content": f"[摘要] {summary}"
        })
        self.short_term.clear()

    def get_context(self) -> List[Dict]:
        """获取完整上下文（中期 + 短期）"""
        return list(self.mid_term) + list(self.short_term)

# 使用示例
memory = HierarchicalMemory(short_term_size=3, mid_term_size=5)

# 第一阶段对话
for i in range(5):
    memory.add_message("user", f"问题 {i}")
    memory.add_message("assistant", f"回答 {i}")

# 摘要并归档
memory.summarize_and_archive("讨论了 Python 基础知识")

# 第二阶段对话
memory.add_message("user", "新问题")
memory.add_message("assistant", "新回答")

context = memory.get_context()
print(f"上下文长度: {len(context)}")  # 3 (1 summary + 2 recent)
```

---

## 5. 性能优化

### 5.1 内存使用对比

```python
import sys
from collections import deque

# 方案1：列表（无限增长）
messages_list = []
for i in range(1000):
    messages_list.append({"role": "user", "content": f"消息 {i}"})

print(f"List 内存: {sys.getsizeof(messages_list)} bytes")

# 方案2：deque（固定大小）
messages_deque = deque(maxlen=100)
for i in range(1000):
    messages_deque.append({"role": "user", "content": f"消息 {i}"})

print(f"Deque 内存: {sys.getsizeof(messages_deque)} bytes")

# 输出示例：
# List 内存:  9016 bytes
# Deque 内存: 1120 bytes
# 内存节省：8 倍
```

### 5.2 操作性能对比

```python
import time
from collections import deque

n = 10000

# 方案1：列表 + 手动管理
start = time.time()
messages = []
for i in range(n):
    messages.append({"role": "user", "content": f"消息 {i}"})
    if len(messages) > 100:
        messages.pop(0)  # O(n)
list_time = time.time() - start

# 方案2：deque + 自动管理
start = time.time()
messages = deque(maxlen=100)
for i in range(n):
    messages.append({"role": "user", "content": f"消息 {i}"})  # O(1)
deque_time = time.time() - start

print(f"List:  {list_time:.4f}s")
print(f"Deque: {deque_time:.4f}s")
print(f"Deque 快 {list_time / deque_time:.0f} 倍")

# 输出示例：
# List:  2.5000s
# Deque: 0.0050s
# Deque 快 500 倍
```

---

## 6. 最佳实践

### 6.1 选择合适的窗口大小

```python
# 根据 LLM 上下文窗口选择
context_window_tokens = 4096  # GPT-4 8K
avg_message_tokens = 100      # 平均每条消息 100 tokens
system_prompt_tokens = 200    # 系统提示 200 tokens

# 计算最大消息数
max_messages = (context_window_tokens - system_prompt_tokens) // avg_message_tokens
max_turns = max_messages // 2

print(f"推荐最大轮数: {max_turns}")  # 19

# 实际使用时留出余量
recommended_turns = int(max_turns * 0.8)
print(f"实际推荐轮数: {recommended_turns}")  # 15
```

### 6.2 处理长消息

```python
from collections import deque

class SmartMemory:
    """智能记忆管理器 - 处理长消息"""

    def __init__(self, max_turns: int = 10, max_message_length: int = 1000):
        self.messages = deque(maxlen=max_turns * 2)
        self.max_message_length = max_message_length

    def add_message(self, role: str, content: str):
        """添加消息，自动截断过长内容"""
        if len(content) > self.max_message_length:
            content = content[:self.max_message_length] + "..."

        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list:
        return list(self.messages)
```

### 6.3 线程安全

```python
import threading
from collections import deque

class ThreadSafeMemory:
    """线程安全的记忆管理器"""

    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)
        self.lock = threading.Lock()

    def add_message(self, role: str, content: str):
        """线程安全地添加消息"""
        with self.lock:
            self.messages.append({"role": role, "content": content})

    def get_context(self) -> list:
        """线程安全地获取上下文"""
        with self.lock:
            return list(self.messages)
```

---

## 7. 常见问题

### Q1: 如何选择窗口大小？

**答案：** 根据以下因素综合考虑：
1. LLM 上下文窗口大小
2. 平均消息长度
3. 对话复杂度
4. 性能要求

**推荐：**
- 简单对话：5-10 轮
- 复杂对话：10-20 轮
- 技术支持：20-50 轮

### Q2: 如何处理重要消息？

**答案：** 使用条件性记忆保留（见 4.2 节）

### Q3: 如何与长期记忆结合？

**答案：**
```python
class HybridMemory:
    """混合记忆系统"""

    def __init__(self, short_term_size: int = 10):
        self.short_term = deque(maxlen=short_term_size * 2)
        self.long_term_db = []  # 实际应该是数据库

    def add_message(self, role: str, content: str):
        # 添加到短期记忆
        self.short_term.append({"role": role, "content": content})

        # 同时保存到长期记忆
        self.long_term_db.append({"role": role, "content": content})

    def get_context(self, include_long_term: bool = False) -> list:
        if include_long_term:
            # 从长期记忆检索相关内容
            relevant = self._retrieve_relevant()
            return relevant + list(self.short_term)
        return list(self.short_term)
```

---

## 学习检查清单

- [ ] 理解短期记忆的基本概念
- [ ] 能够使用 Deque 实现短期记忆管理
- [ ] 了解 2025-2026 年主流框架的实现方式
- [ ] 能够实现带元数据的记忆管理
- [ ] 能够实现条件性记忆保留
- [ ] 理解性能优化的关键点
- [ ] 知道如何选择合适的窗口大小

---

## 下一步学习

### 深入理解
→ **03_核心概念_06_上下文窗口管理.md** - Token 窗口管理

### 实战代码
→ **07_实战代码_04_AI_Agent记忆管理.md** - 完整代码示例

### 生产实践
→ **07_实战代码_06_生产级实践.md** - 生产级实现

---

**版本**: v1.0
**最后更新**: 2026-02-13
**适用于**: Python 3.13+, AI Agent 开发, RAG 系统
