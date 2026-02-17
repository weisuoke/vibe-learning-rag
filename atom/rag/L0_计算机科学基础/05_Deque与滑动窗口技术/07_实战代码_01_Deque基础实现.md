# 实战代码 01：Deque 基础实现

> 手写 Deque 实现 + Python collections.deque 完整使用示例

---

## 1. 手写 Deque：循环数组实现

```python
class CircularArrayDeque:
    """使用循环数组实现的 Deque"""
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.data = [None] * capacity
        self.front = 0
        self.size = 0
    
    def _resize(self):
        """扩容"""
        new_capacity = self.capacity * 2
        new_data = [None] * new_capacity
        
        # 复制元素到新数组
        for i in range(self.size):
            new_data[i] = self.data[(self.front + i) % self.capacity]
        
        self.data = new_data
        self.front = 0
        self.capacity = new_capacity
    
    def append(self, value):
        """右端添加"""
        if self.size == self.capacity:
            self._resize()
        
        rear = (self.front + self.size) % self.capacity
        self.data[rear] = value
        self.size += 1
    
    def appendleft(self, value):
        """左端添加"""
        if self.size == self.capacity:
            self._resize()
        
        self.front = (self.front - 1) % self.capacity
        self.data[self.front] = value
        self.size += 1
    
    def pop(self):
        """右端删除"""
        if self.size == 0:
            raise IndexError("pop from empty deque")
        
        rear = (self.front + self.size - 1) % self.capacity
        value = self.data[rear]
        self.data[rear] = None
        self.size -= 1
        return value
    
    def popleft(self):
        """左端删除"""
        if self.size == 0:
            raise IndexError("pop from empty deque")
        
        value = self.data[self.front]
        self.data[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return value
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        items = [self.data[(self.front + i) % self.capacity] for i in range(self.size)]
        return f"CircularArrayDeque({items})"

# 测试
d = CircularArrayDeque(capacity=3)
d.append(1)
d.append(2)
d.append(3)
print(d)  # CircularArrayDeque([1, 2, 3])

d.append(4)  # 触发扩容
print(d)  # CircularArrayDeque([1, 2, 3, 4])

d.appendleft(0)
print(d)  # CircularArrayDeque([0, 1, 2, 3, 4])

print(d.pop())      # 4
print(d.popleft())  # 0
print(d)  # CircularArrayDeque([1, 2, 3])
```

---

## 2. 手写 Deque：双向链表实现

```python
class Node:
    """双向链表节点"""
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class LinkedListDeque:
    """使用双向链表实现的 Deque"""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, value):
        """右端添加"""
        new_node = Node(value)
        
        if self.size == 0:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def appendleft(self, value):
        """左端添加"""
        new_node = Node(value)
        
        if self.size == 0:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def pop(self):
        """右端删除"""
        if self.size == 0:
            raise IndexError("pop from empty deque")
        
        value = self.tail.value
        
        if self.size == 1:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        
        self.size -= 1
        return value
    
    def popleft(self):
        """左端删除"""
        if self.size == 0:
            raise IndexError("pop from empty deque")
        
        value = self.head.value
        
        if self.size == 1:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        
        self.size -= 1
        return value
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        items = []
        current = self.head
        while current:
            items.append(current.value)
            current = current.next
        return f"LinkedListDeque({items})"

# 测试
d = LinkedListDeque()
d.append(1)
d.append(2)
d.append(3)
print(d)  # LinkedListDeque([1, 2, 3])

d.appendleft(0)
print(d)  # LinkedListDeque([0, 1, 2, 3])

print(d.pop())      # 3
print(d.popleft())  # 0
print(d)  # LinkedListDeque([1, 2])
```

---

## 3. Python collections.deque 完整使用

```python
from collections import deque

# ===== 创建 =====
d1 = deque()                    # 空 deque
d2 = deque([1, 2, 3])          # 从列表创建
d3 = deque(maxlen=5)           # 固定大小
d4 = deque([1, 2, 3], maxlen=5)  # 固定大小 + 初始值

# ===== 添加元素 =====
d = deque([1, 2, 3])
d.append(4)         # [1, 2, 3, 4]
d.appendleft(0)     # [0, 1, 2, 3, 4]
d.extend([5, 6])    # [0, 1, 2, 3, 4, 5, 6]
d.extendleft([-2, -1])  # [-1, -2, 0, 1, 2, 3, 4, 5, 6]

# ===== 删除元素 =====
d.pop()             # 删除右端：6
d.popleft()         # 删除左端：-1
d.remove(0)         # 删除指定值
d.clear()           # 清空

# ===== 访问元素 =====
d = deque([1, 2, 3, 4, 5])
print(d[0])         # 1
print(d[-1])        # 5
print(2 in d)       # True

# ===== 旋转 =====
d.rotate(2)         # 向右旋转 2 步：[4, 5, 1, 2, 3]
d.rotate(-1)        # 向左旋转 1 步：[5, 1, 2, 3, 4]

# ===== 其他操作 =====
d.reverse()         # 反转
d.count(2)          # 计数
len(d)              # 长度
d.copy()            # 复制
```

---

## 4. 性能测试

```python
import time
from collections import deque

def benchmark_deque_vs_list():
    """性能对比：Deque vs List"""
    n = 10000
    
    # 测试 List 左端删除
    print("=== List 左端删除 ===")
    start = time.time()
    lst = list(range(n))
    for _ in range(n):
        lst.pop(0)
    list_time = time.time() - start
    print(f"时间: {list_time:.4f}s")
    
    # 测试 Deque 左端删除
    print("\n=== Deque 左端删除 ===")
    start = time.time()
    d = deque(range(n))
    for _ in range(n):
        d.popleft()
    deque_time = time.time() - start
    print(f"时间: {deque_time:.4f}s")
    
    print(f"\n性能提升: {list_time / deque_time:.0f} 倍")

benchmark_deque_vs_list()

# 输出示例：
# === List 左端删除 ===
# 时间: 2.5000s
#
# === Deque 左端删除 ===
# 时间: 0.0010s
#
# 性能提升: 2500 倍
```

---

## 5. AI Agent 应用：对话历史管理

```python
from collections import deque
from typing import Dict, List

class ConversationMemory:
    """AI Agent 对话记忆管理器"""
    
    def __init__(self, max_turns: int = 10):
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
memory = ConversationMemory(max_turns=3)

# 添加 5 轮对话
for i in range(5):
    memory.add_user_message(f"问题 {i}")
    memory.add_assistant_message(f"回答 {i}")

# 只保留最近 3 轮
context = memory.get_context()
print(f"消息数: {len(context)}")  # 6
print(f"轮数: {memory.get_turn_count()}")  # 3

# 打印最近对话
for msg in context:
    print(f"{msg['role']}: {msg['content']}")
```

---

**版本**: v1.0
**最后更新**: 2026-02-13
