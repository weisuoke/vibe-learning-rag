# 实战代码 04：AI Agent 记忆管理

> OpenAI SDK 风格实现 + LangGraph 集成 + 生产级对话管理

---

## 1. OpenAI Agents SDK 风格实现

```python
from collections import deque
from openai import OpenAI
from typing import List, Dict, Optional

class TrimmingSession:
    """OpenAI Agents SDK 风格的会话管理"""
    
    def __init__(self, max_turns: int = 10, model: str = "gpt-4"):
        self.messages = deque(maxlen=max_turns * 2)
        self.client = OpenAI()
        self.model = model
    
    def chat(self, user_message: str) -> str:
        """发送消息并获取响应"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_message})
        
        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(self.messages)
        )
        
        # 添加助手消息
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return list(self.messages)
    
    def clear(self):
        """清空历史"""
        self.messages.clear()

# 使用示例
session = TrimmingSession(max_turns=5)

response1 = session.chat("什么是 Python？")
print(f"助手: {response1}")

response2 = session.chat("它有什么特点？")
print(f"助手: {response2}")

print(f"\n历史消息数: {len(session.get_history())}")
```

---

## 2. 带系统提示的会话管理

```python
from collections import deque
from typing import List, Dict

class ConversationManager:
    """带系统提示的会话管理器"""
    
    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 10
    ):
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
        """获取完整上下文（包含系统提示）"""
        return [
            {"role": "system", "content": self.system_prompt},
            *list(self.messages)
        ]
    
    def update_system_prompt(self, new_prompt: str):
        """更新系统提示"""
        self.system_prompt = new_prompt
    
    def get_turn_count(self) -> int:
        """获取对话轮数"""
        return len(self.messages) // 2

# 使用示例
manager = ConversationManager(
    system_prompt="你是一个友好的 Python 编程助手。",
    max_turns=5
)

manager.add_user_message("如何定义函数？")
manager.add_assistant_message("使用 def 关键字定义函数...")

context = manager.get_context()
print(f"上下文长度: {len(context)}")  # 3 (system + user + assistant)
```

---

## 3. LangGraph 风格的滑动窗口记忆

```python
from collections import deque
from typing import Dict, List, Optional

class SlidingWindowMemory:
    """LangGraph 风格的滑动窗口记忆"""
    
    def __init__(self, window_size: int = 5):
        self.messages = deque(maxlen=window_size)
        self.window_size = window_size
    
    def add_message(self, message: Dict):
        """添加消息"""
        self.messages.append(message)
    
    def get_context(self) -> List[Dict]:
        """获取当前上下文"""
        return list(self.messages)
    
    def clear(self):
        """清空记忆"""
        self.messages.clear()
    
    def get_last_message(self) -> Optional[Dict]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.messages)

# 使用示例
memory = SlidingWindowMemory(window_size=10)

for i in range(15):
    memory.add_message({
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"消息 {i}"
    })

print(f"消息数: {memory.get_message_count()}")  # 10
print(f"最后一条: {memory.get_last_message()}")
```

---

## 4. 带元数据的记忆管理

```python
from collections import deque
from typing import Dict, List, Optional
import time
import uuid

class MemoryWithMetadata:
    """带元数据的记忆管理器"""
    
    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """添加带元数据的消息"""
        message_id = str(uuid.uuid4())
        
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        return message_id
    
    def get_context(self, include_metadata: bool = False) -> List[Dict]:
        """获取上下文"""
        if include_metadata:
            return list(self.messages)
        else:
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.messages
            ]
    
    def get_message_by_id(self, message_id: str) -> Optional[Dict]:
        """根据 ID 获取消息"""
        for msg in self.messages:
            if msg["id"] == message_id:
                return msg
        return None
    
    def search_by_metadata(self, key: str, value: any) -> List[Dict]:
        """根据元数据搜索消息"""
        return [
            msg for msg in self.messages
            if msg["metadata"].get(key) == value
        ]

# 使用示例
memory = MemoryWithMetadata(max_turns=5)

msg_id = memory.add_message(
    "user",
    "什么是 Python？",
    metadata={"source": "web", "user_id": "123"}
)

memory.add_message(
    "assistant",
    "Python 是一种编程语言。",
    metadata={"model": "gpt-4", "tokens": 50}
)

# 获取带元数据的上下文
context = memory.get_context(include_metadata=True)
print(f"第一条消息元数据: {context[0]['metadata']}")

# 根据元数据搜索
web_messages = memory.search_by_metadata("source", "web")
print(f"来自 web 的消息数: {len(web_messages)}")
```

---

## 5. 条件性记忆保留

```python
from collections import deque
from typing import Dict, List

class ConditionalMemory:
    """条件性记忆保留"""
    
    def __init__(self, max_turns: int = 10):
        self.messages = deque(maxlen=max_turns * 2)
        self.important_messages = []  # 永久保留
    
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
    
    def clear_recent(self):
        """清空最近消息，保留重要消息"""
        self.messages.clear()
    
    def clear_all(self):
        """清空所有消息"""
        self.messages.clear()
        self.important_messages.clear()

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

---

## 6. 分层记忆管理

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
        self.short_term = deque(maxlen=short_term_size * 2)
        self.mid_term = deque(maxlen=mid_term_size)
        self.long_term = []  # 永久存储
    
    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        message = {"role": role, "content": content}
        self.short_term.append(message)
    
    def summarize_and_archive(self, summary: str):
        """摘要并归档到中期记忆"""
        self.mid_term.append({
            "role": "system",
            "content": f"[摘要] {summary}"
        })
        self.short_term.clear()
    
    def archive_to_long_term(self, content: str):
        """归档到长期记忆"""
        self.long_term.append({
            "role": "system",
            "content": f"[长期记忆] {content}"
        })
    
    def get_context(self, include_long_term: bool = False) -> List[Dict]:
        """获取完整上下文"""
        context = []
        
        if include_long_term:
            context.extend(self.long_term)
        
        context.extend(list(self.mid_term))
        context.extend(list(self.short_term))
        
        return context

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

# 归档重要信息到长期记忆
memory.archive_to_long_term("用户是 Python 初学者")

context = memory.get_context(include_long_term=True)
print(f"上下文长度: {len(context)}")
```

---

## 7. 完整的生产级实现

```python
from collections import deque
from typing import Dict, List, Optional
import time
import json

class ProductionMemoryManager:
    """生产级记忆管理器"""
    
    def __init__(
        self,
        max_turns: int = 10,
        system_prompt: Optional[str] = None,
        enable_metadata: bool = True
    ):
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.enable_metadata = enable_metadata
        self.messages = deque(maxlen=max_turns * 2)
        self.stats = {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0
        }
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """添加消息"""
        message = {
            "role": role,
            "content": content
        }
        
        if self.enable_metadata:
            message["timestamp"] = time.time()
            message["metadata"] = metadata or {}
        
        self.messages.append(message)
        
        # 更新统计
        self.stats["total_messages"] += 1
        if role == "user":
            self.stats["user_messages"] += 1
        elif role == "assistant":
            self.stats["assistant_messages"] += 1
    
    def get_context(self) -> List[Dict]:
        """获取上下文"""
        context = []
        
        if self.system_prompt:
            context.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # 移除元数据（如果启用）
        for msg in self.messages:
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            context.append(clean_msg)
        
        return context
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "current_messages": len(self.messages),
            "turn_count": len(self.messages) // 2
        }
    
    def export_history(self) -> str:
        """导出历史（JSON 格式）"""
        return json.dumps(list(self.messages), indent=2)
    
    def import_history(self, history_json: str):
        """导入历史"""
        messages = json.loads(history_json)
        self.messages.clear()
        for msg in messages:
            self.messages.append(msg)

# 使用示例
manager = ProductionMemoryManager(
    max_turns=5,
    system_prompt="你是一个友好的 AI 助手。",
    enable_metadata=True
)

manager.add_message(
    "user",
    "你好",
    metadata={"source": "web"}
)

manager.add_message(
    "assistant",
    "你好！有什么可以帮你的？",
    metadata={"model": "gpt-4"}
)

print("统计信息:")
print(json.dumps(manager.get_stats(), indent=2))

print("\n导出历史:")
print(manager.export_history())
```

---

**版本**: v1.0
**最后更新**: 2026-02-13
