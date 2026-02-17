# 核心概念1：LIFO与栈操作

> 理解 Stack 的核心特性：后进先出（LIFO）和基本操作

---

## 概念定义

**LIFO（Last In First Out）**：最后放入的元素最先被取出

**Stack 的本质**：一端受限的线性表，只能在栈顶进行插入和删除操作

```
栈底 → [A] [B] [C] [D] ← 栈顶
       ↑               ↑
    不可操作        可操作
```

---

## 1. LIFO 原理详解

### 1.1 为什么是 LIFO？

**根本原因：** 程序执行的嵌套特性

```python
def main():
    print("1. main 开始")
    func_a()
    print("4. main 结束")

def func_a():
    print("2. func_a 开始")
    func_b()
    print("3. func_a 结束")

def func_b():
    print("  2.1 func_b 执行")

main()
```

**输出：**
```
1. main 开始
2. func_a 开始
  2.1 func_b 执行
3. func_a 结束
4. main 结束
```

**调用栈变化：**
```
[main]
[main, func_a]
[main, func_a, func_b]  ← func_b 最后进入
[main, func_a]          ← func_b 最先退出（LIFO）
[main]
[]
```

### 1.2 LIFO 的数学表示

**栈的状态转换：**
```
S₀ = []                    # 初始状态
S₁ = push(S₀, A) = [A]
S₂ = push(S₁, B) = [A, B]
S₃ = push(S₂, C) = [A, B, C]

pop(S₃) → (C, S₂)          # 返回 C，状态变为 S₂
pop(S₂) → (B, S₁)          # 返回 B，状态变为 S₁
pop(S₁) → (A, S₀)          # 返回 A，状态变为 S₀
```

### 1.3 LIFO 在 AI Agent 中的体现

**场景：Agent 工具调用链**

```python
# Agent 调用栈
call_stack = []

# 1. 主 Agent 开始
call_stack.append('main_agent')
print(f"调用栈: {call_stack}")  # ['main_agent']

# 2. 调用搜索工具
call_stack.append('search_tool')
print(f"调用栈: {call_stack}")  # ['main_agent', 'search_tool']

# 3. 搜索工具调用 API
call_stack.append('api_call')
print(f"调用栈: {call_stack}")  # ['main_agent', 'search_tool', 'api_call']

# 4. API 调用完成（最后进入，最先退出）
result_3 = call_stack.pop()
print(f"完成: {result_3}, 调用栈: {call_stack}")

# 5. 搜索工具完成
result_2 = call_stack.pop()
print(f"完成: {result_2}, 调用栈: {call_stack}")

# 6. 主 Agent 完成
result_1 = call_stack.pop()
print(f"完成: {result_1}, 调用栈: {call_stack}")
```

**输出：**
```
调用栈: ['main_agent']
调用栈: ['main_agent', 'search_tool']
调用栈: ['main_agent', 'search_tool', 'api_call']
完成: api_call, 调用栈: ['main_agent', 'search_tool']
完成: search_tool, 调用栈: ['main_agent']
完成: main_agent, 调用栈: []
```

---

## 2. 栈的基本操作

### 2.1 Push（压栈）

**定义：** 将元素添加到栈顶

**时间复杂度：** O(1)

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        """压栈操作 - O(1)"""
        self.items.append(item)
        print(f"Push: {item}, 栈: {self.items}")

# 示例
stack = Stack()
stack.push('A')  # ['A']
stack.push('B')  # ['A', 'B']
stack.push('C')  # ['A', 'B', 'C']
```

**内存操作：**
```
初始: []
      ↓ push('A')
     [A]
      ↓ push('B')
     [A][B]
      ↓ push('C')
     [A][B][C]
```

### 2.2 Pop（出栈）

**定义：** 移除并返回栈顶元素

**时间复杂度：** O(1)

```python
def pop(self):
    """出栈操作 - O(1)"""
    if self.is_empty():
        raise IndexError("Stack is empty")
    item = self.items.pop()
    print(f"Pop: {item}, 栈: {self.items}")
    return item

# 示例
item1 = stack.pop()  # 'C', 栈: ['A', 'B']
item2 = stack.pop()  # 'B', 栈: ['A']
item3 = stack.pop()  # 'A', 栈: []
```

### 2.3 Peek（查看栈顶）

**定义：** 查看栈顶元素但不移除

**时间复杂度：** O(1)

```python
def peek(self):
    """查看栈顶 - O(1)"""
    if self.is_empty():
        raise IndexError("Stack is empty")
    return self.items[-1]

# 示例
stack.push('X')
stack.push('Y')
top = stack.peek()  # 'Y'，栈不变: ['X', 'Y']
```

### 2.4 辅助操作

```python
def is_empty(self):
    """检查栈是否为空 - O(1)"""
    return len(self.items) == 0

def size(self):
    """返回栈的大小 - O(1)"""
    return len(self.items)

def clear(self):
    """清空栈 - O(1)"""
    self.items = []
```

---

## 3. 栈帧（Stack Frame）

### 3.1 什么是栈帧？

**栈帧**：函数调用时在栈上分配的内存区域，包含：
- 局部变量
- 函数参数
- 返回地址
- 保存的寄存器

```python
def calculate(a, b):
    """
    栈帧包含：
    - 参数: a, b
    - 局部变量: result
    - 返回地址: 调用者的下一条指令
    """
    result = a + b
    return result

def main():
    x = 5
    y = 3
    z = calculate(x, y)  # 创建新栈帧
    print(z)

main()
```

**栈帧结构：**
```
调用 calculate(5, 3) 时的栈：

+-------------------+
| main 栈帧         |
| - x = 5          |
| - y = 3          |
| - z = ?          |
| - 返回地址: OS   |
+-------------------+
| calculate 栈帧    |
| - a = 5          |
| - b = 3          |
| - result = 8     |
| - 返回地址: main |
+-------------------+ ← 栈顶
```

### 3.2 栈帧的生命周期

```python
import sys

def show_stack_frame():
    """展示栈帧信息"""
    frame = sys._getframe()
    print(f"函数名: {frame.f_code.co_name}")
    print(f"局部变量: {frame.f_locals}")
    print(f"行号: {frame.f_lineno}")

def func_a(x):
    y = x * 2
    show_stack_frame()
    return y

def func_b():
    result = func_a(10)
    return result

func_b()
```

**输出：**
```
函数名: show_stack_frame
局部变量: {'frame': <frame object>}
行号: 5
```

### 3.3 AI Agent 中的栈帧应用

```python
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

@dataclass
class AgentStackFrame:
    """Agent 执行的栈帧"""
    function_name: str
    arguments: Dict[str, Any]
    local_vars: Dict[str, Any]
    start_time: datetime
    parent_frame: 'AgentStackFrame' = None

class AgentExecutor:
    def __init__(self):
        self.call_stack = []

    def enter_function(self, func_name: str, args: Dict[str, Any]):
        """进入函数，创建栈帧"""
        parent = self.call_stack[-1] if self.call_stack else None
        frame = AgentStackFrame(
            function_name=func_name,
            arguments=args,
            local_vars={},
            start_time=datetime.now(),
            parent_frame=parent
        )
        self.call_stack.append(frame)
        print(f"{'  ' * len(self.call_stack)}→ 进入 {func_name}")

    def exit_function(self, return_value: Any = None):
        """退出函数，销毁栈帧"""
        if not self.call_stack:
            raise RuntimeError("调用栈为空")

        frame = self.call_stack.pop()
        duration = (datetime.now() - frame.start_time).total_seconds()
        print(f"{'  ' * len(self.call_stack)}← 退出 {frame.function_name} (耗时: {duration:.3f}s)")
        return return_value

    def get_current_frame(self) -> AgentStackFrame:
        """获取当前栈帧"""
        return self.call_stack[-1] if self.call_stack else None

# 使用示例
executor = AgentExecutor()

executor.enter_function('research_agent', {'query': 'AI 2026'})
executor.enter_function('search_tool', {'query': 'AI 2026'})
executor.exit_function(['result1', 'result2'])
executor.enter_function('analyze_tool', {'results': ['result1', 'result2']})
executor.exit_function('analysis')
executor.exit_function('final_result')
```

**输出：**
```
  → 进入 research_agent
    → 进入 search_tool
  ← 退出 search_tool (耗时: 0.001s)
    → 进入 analyze_tool
  ← 退出 analyze_tool (耗时: 0.001s)
← 退出 research_agent (耗时: 0.002s)
```

---

## 4. 栈的实现方式

### 4.1 数组实现

```python
class ArrayStack:
    def __init__(self, capacity=10):
        self.items = []
        self.capacity = capacity

    def push(self, item):
        if len(self.items) >= self.capacity:
            # 动态扩容（加倍策略）
            self.capacity *= 2
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
```

**优点：**
- 实现简单
- 缓存友好（连续内存）
- 空间效率高

**缺点：**
- 扩容时需要复制（偶尔 O(n)）
- 预分配空间可能浪费

### 4.2 链表实现

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    def __init__(self):
        self.top = None
        self._size = 0

    def push(self, item):
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        data = self.top.data
        self.top = self.top.next
        self._size -= 1
        return data

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.top.data

    def is_empty(self):
        return self.top is None

    def size(self):
        return self._size
```

**优点：**
- 无需扩容（真正的 O(1) 插入）
- 内存灵活分配

**缺点：**
- 指针开销
- 缓存不友好

---

## 5. 栈操作的时间复杂度

| 操作 | 数组实现 | 链表实现 | 说明 |
|------|---------|---------|------|
| Push | O(1)* | O(1) | *数组扩容时 O(n) |
| Pop | O(1) | O(1) | 直接移除栈顶 |
| Peek | O(1) | O(1) | 只读取不修改 |
| Is Empty | O(1) | O(1) | 检查标志位 |
| Size | O(1) | O(1) | 维护计数器 |

**摊还分析：**
- 数组扩容虽然偶尔 O(n)，但摊还后仍是 O(1)
- 扩容策略：每次加倍，n 次操作最多扩容 log(n) 次

---

## 6. 在 AI Agent 中的应用

### 6.1 调用链追踪

```python
class CallChainTracker:
    def __init__(self):
        self.stack = []

    def enter(self, func_name: str):
        self.stack.append(func_name)

    def exit(self):
        return self.stack.pop()

    def get_chain(self):
        return ' → '.join(self.stack)

# 使用
tracker = CallChainTracker()
tracker.enter('agent')
tracker.enter('tool1')
tracker.enter('tool2')
print(tracker.get_chain())  # agent → tool1 → tool2
```

### 6.2 状态快照管理

```python
class StateManager:
    def __init__(self):
        self.snapshots = []

    def save_state(self, state: dict):
        """保存当前状态"""
        self.snapshots.append(state.copy())

    def restore_state(self) -> dict:
        """恢复上一个状态"""
        if not self.snapshots:
            raise RuntimeError("没有可恢复的状态")
        return self.snapshots.pop()

    def peek_state(self) -> dict:
        """查看当前状态但不恢复"""
        if not self.snapshots:
            raise RuntimeError("没有保存的状态")
        return self.snapshots[-1]
```

### 6.3 Prompt 嵌套管理

```python
class PromptStack:
    def __init__(self):
        self.prompts = []

    def push_prompt(self, prompt: str, context: dict):
        """压入新的 Prompt"""
        self.prompts.append({
            'prompt': prompt,
            'context': context,
            'depth': len(self.prompts)
        })

    def pop_prompt(self):
        """弹出当前 Prompt"""
        return self.prompts.pop()

    def get_full_context(self):
        """获取完整上下文（所有层级）"""
        return [p['context'] for p in self.prompts]
```

---

## 总结

**LIFO 的本质：**
- 后进先出是程序执行的自然规律
- 最后调用的函数最先返回
- 最近的状态最先恢复

**栈操作的特点：**
- 所有基本操作都是 O(1)
- 只能在栈顶操作
- 限制保证了 LIFO 特性

**在 AI Agent 中的价值：**
- 追踪调用链
- 管理执行状态
- 实现回溯机制

**记住：** Stack 的核心就是 LIFO + 栈顶操作，理解这两点就掌握了 Stack 的本质！
