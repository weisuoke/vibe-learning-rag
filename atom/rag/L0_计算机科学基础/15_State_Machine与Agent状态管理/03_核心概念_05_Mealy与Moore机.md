# 核心概念05：Mealy与Moore机

> **定义**：Mealy机和Moore机是两种不同的有限状态机输出模式，区别在于输出是否依赖输入

---

## 一、形式化定义

### 1.1 Mealy机

**定义**：
```
M = (Q, Σ, Ω, δ, λ, q0)

其中：
- Q：有限状态集合
- Σ：输入字母表
- Ω：输出字母表
- δ：状态转移函数 δ: Q × Σ → Q
- λ：输出函数 λ: Q × Σ → Ω（输出依赖状态和输入）
- q0：初始状态
```

**关键特征**：
- **输出依赖输入**：λ(q, a) = o
- **响应快**：输入立即产生输出
- **状态数少**：相同功能需要的状态更少

---

### 1.2 Moore机

**定义**：
```
M = (Q, Σ, Ω, δ, λ, q0)

其中：
- Q：有限状态集合
- Σ：输入字母表
- Ω：输出字母表
- δ：状态转移函数 δ: Q × Σ → Q
- λ：输出函数 λ: Q → Ω（输出仅依赖状态）
- q0：初始状态
```

**关键特征**：
- **输出仅依赖状态**：λ(q) = o
- **输出稳定**：状态不变则输出不变
- **状态数多**：相同功能需要更多状态

---

## 二、核心区别

### 2.1 输出模式对比

| 维度 | Mealy机 | Moore机 |
|------|---------|---------|
| **输出函数** | λ: Q × Σ → Ω | λ: Q → Ω |
| **输出依赖** | 状态 + 输入 | 仅状态 |
| **响应速度** | 快（输入立即输出） | 慢（需要状态转移） |
| **状态数** | 少 | 多 |
| **输出稳定性** | 可能抖动 | 稳定 |
| **调试难度** | 较难 | 较易 |
| **生产环境** | 较少使用 | 更常用 |

---

### 2.2 图形表示

**Mealy机**：
```
输出标注在边上

    a/0        b/1
q0 -----> q1 -----> q2
 ↑                   |
 |       c/0         |
 +-------------------+

读取：
- 在q0读'a' → 输出'0'，转到q1
- 在q1读'b' → 输出'1'，转到q2
```

**Moore机**：
```
输出标注在状态上

q0/0 --a--> q1/1 --b--> q2/0
 ↑                       |
 |          c            |
 +-----------------------+

读取：
- 在q0 → 输出'0'
- 读'a'转到q1 → 输出'1'
- 读'b'转到q2 → 输出'0'
```

---

## 三、Python实现

### 3.1 Mealy机实现

```python
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class MealyMachine:
    """Mealy机实现"""
    states: set[str]
    input_alphabet: set[str]
    output_alphabet: set[str]
    transitions: Dict[Tuple[str, str], str]  # δ: (state, input) → next_state
    outputs: Dict[Tuple[str, str], str]      # λ: (state, input) → output
    initial_state: str

    def process(self, input_string: str) -> list[str]:
        """
        处理输入字符串，返回输出序列

        Returns:
            输出序列（与输入长度相同）
        """
        current_state = self.initial_state
        output_sequence = []

        for symbol in input_string:
            # 输出依赖当前状态和输入
            output = self.outputs[(current_state, symbol)]
            output_sequence.append(output)

            # 状态转移
            current_state = self.transitions[(current_state, symbol)]

        return output_sequence

    def trace(self, input_string: str) -> list[Tuple[str, str, str]]:
        """
        追踪执行过程

        Returns:
            [(state, input, output), ...]
        """
        current_state = self.initial_state
        trace = []

        for symbol in input_string:
            output = self.outputs[(current_state, symbol)]
            trace.append((current_state, symbol, output))
            current_state = self.transitions[(current_state, symbol)]

        return trace


# 示例：奇偶性检测器（Mealy机）
def create_parity_checker_mealy() -> MealyMachine:
    """
    创建奇偶性检测器（Mealy机）

    功能：输出当前1的个数的奇偶性
    - 输入'1' → 输出当前奇偶性
    - 输入'0' → 输出当前奇偶性
    """
    return MealyMachine(
        states={'even', 'odd'},
        input_alphabet={'0', '1'},
        output_alphabet={'E', 'O'},
        transitions={
            ('even', '0'): 'even',
            ('even', '1'): 'odd',
            ('odd', '0'): 'odd',
            ('odd', '1'): 'even',
        },
        outputs={
            # 输出依赖状态和输入
            ('even', '0'): 'E',  # 偶数个1，读0 → 输出E
            ('even', '1'): 'O',  # 偶数个1，读1 → 输出O（变成奇数）
            ('odd', '0'): 'O',   # 奇数个1，读0 → 输出O
            ('odd', '1'): 'E',   # 奇数个1，读1 → 输出E（变成偶数）
        },
        initial_state='even'
    )


# 测试Mealy机
mealy = create_parity_checker_mealy()
input_str = "1101"
outputs = mealy.process(input_str)
print(f"输入: {input_str}")
print(f"输出: {''.join(outputs)}")
print("\n执行追踪:")
for state, inp, out in mealy.trace(input_str):
    print(f"  状态:{state}, 输入:{inp} → 输出:{out}")
```

**输出**：
```
输入: 1101
输出: OEEO

执行追踪:
  状态:even, 输入:1 → 输出:O
  状态:odd, 输入:1 → 输出:E
  状态:even, 输入:0 → 输出:E
  状态:even, 输入:1 → 输出:O
```

---

### 3.2 Moore机实现

```python
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class MooreMachine:
    """Moore机实现"""
    states: set[str]
    input_alphabet: set[str]
    output_alphabet: set[str]
    transitions: Dict[Tuple[str, str], str]  # δ: (state, input) → next_state
    outputs: Dict[str, str]                  # λ: state → output
    initial_state: str

    def process(self, input_string: str) -> list[str]:
        """
        处理输入字符串，返回输出序列

        Returns:
            输出序列（长度 = 输入长度 + 1，包含初始输出）
        """
        current_state = self.initial_state
        output_sequence = []

        # 初始输出
        output_sequence.append(self.outputs[current_state])

        for symbol in input_string:
            # 状态转移
            current_state = self.transitions[(current_state, symbol)]
            # 输出仅依赖新状态
            output_sequence.append(self.outputs[current_state])

        return output_sequence

    def trace(self, input_string: str) -> list[Tuple[str, str, str]]:
        """
        追踪执行过程

        Returns:
            [(state, output, input), ...]
        """
        current_state = self.initial_state
        trace = [(current_state, self.outputs[current_state], '')]

        for symbol in input_string:
            current_state = self.transitions[(current_state, symbol)]
            output = self.outputs[current_state]
            trace.append((current_state, output, symbol))

        return trace


# 示例：奇偶性检测器（Moore机）
def create_parity_checker_moore() -> MooreMachine:
    """
    创建奇偶性检测器（Moore机）

    功能：输出当前1的个数的奇偶性
    """
    return MooreMachine(
        states={'even', 'odd'},
        input_alphabet={'0', '1'},
        output_alphabet={'E', 'O'},
        transitions={
            ('even', '0'): 'even',
            ('even', '1'): 'odd',
            ('odd', '0'): 'odd',
            ('odd', '1'): 'even',
        },
        outputs={
            # 输出仅依赖状态
            'even': 'E',
            'odd': 'O',
        },
        initial_state='even'
    )


# 测试Moore机
moore = create_parity_checker_moore()
input_str = "1101"
outputs = moore.process(input_str)
print(f"输入: {input_str}")
print(f"输出: {''.join(outputs)}")
print("\n执行追踪:")
for state, out, inp in moore.trace(input_str):
    if inp:
        print(f"  读入:{inp} → 状态:{state}, 输出:{out}")
    else:
        print(f"  初始状态:{state}, 输出:{out}")
```

**输出**：
```
输入: 1101
输出: EOEEO

执行追踪:
  初始状态:even, 输出:E
  读入:1 → 状态:odd, 输出:O
  读入:1 → 状态:even, 输出:E
  读入:0 → 状态:even, 输出:E
  读入:1 → 状态:odd, 输出:O
```

---

## 四、Mealy与Moore的转换

### 4.1 Mealy转Moore

**算法**：
1. 为Mealy机的每个(状态, 输出)对创建Moore机的新状态
2. 新状态的输出 = Mealy机的输出
3. 转移关系保持不变

**Python实现**：
```python
def mealy_to_moore(mealy: MealyMachine) -> MooreMachine:
    """
    将Mealy机转换为等价的Moore机

    注意：Moore机的状态数可能增加
    """
    # 新状态：(原状态, 输出)
    new_states = set()
    new_transitions = {}
    new_outputs = {}

    # 计算初始状态的输出（取任意输入的输出）
    first_input = next(iter(mealy.input_alphabet))
    initial_output = mealy.outputs[(mealy.initial_state, first_input)]
    new_initial_state = (mealy.initial_state, initial_output)
    new_states.add(new_initial_state)

    # BFS构建新状态
    queue = [new_initial_state]
    visited = {new_initial_state}

    while queue:
        (state, output) = queue.pop(0)
        new_outputs[(state, output)] = output

        for symbol in mealy.input_alphabet:
            # Mealy转移
            next_state = mealy.transitions[(state, symbol)]
            next_output = mealy.outputs[(next_state, symbol)]
            new_state = (next_state, next_output)

            # 添加Moore转移
            new_transitions[((state, output), symbol)] = new_state

            if new_state not in visited:
                visited.add(new_state)
                new_states.add(new_state)
                queue.append(new_state)

    return MooreMachine(
        states=new_states,
        input_alphabet=mealy.input_alphabet,
        output_alphabet=mealy.output_alphabet,
        transitions=new_transitions,
        outputs=new_outputs,
        initial_state=new_initial_state
    )
```

---

### 4.2 Moore转Mealy

**算法**：
1. Mealy机的状态 = Moore机的状态
2. Mealy机的输出 = 转移后状态的Moore输出
3. 状态数不变

**Python实现**：
```python
def moore_to_mealy(moore: MooreMachine) -> MealyMachine:
    """
    将Moore机转换为等价的Mealy机

    注意：Mealy机的状态数与Moore机相同
    """
    new_outputs = {}

    for (state, symbol), next_state in moore.transitions.items():
        # Mealy输出 = 转移后状态的Moore输出
        new_outputs[(state, symbol)] = moore.outputs[next_state]

    return MealyMachine(
        states=moore.states,
        input_alphabet=moore.input_alphabet,
        output_alphabet=moore.output_alphabet,
        transitions=moore.transitions,
        outputs=new_outputs,
        initial_state=moore.initial_state
    )
```

---

## 五、选择标准

### 5.1 何时使用Mealy机

**适用场景**：
1. **快速响应**：需要输入立即产生输出
2. **状态数限制**：内存受限，需要减少状态数
3. **输出多样化**：相同状态对不同输入有不同输出

**示例**：
- 键盘输入处理（按键 → 立即反馈）
- 网络协议（收到包 → 立即响应）
- 实时控制系统

---

### 5.2 何时使用Moore机

**适用场景**：
1. **输出稳定性**：需要稳定的输出（无抖动）
2. **易于调试**：输出仅依赖状态，易于追踪
3. **生产环境**：可靠性优先于响应速度

**示例**：
- 交通灯控制（状态 → 灯色）
- 工作流系统（状态 → 显示信息）
- AI Agent状态管理

---

### 5.3 实际选择建议

| 需求 | 推荐 | 原因 |
|------|------|------|
| **生产环境** | Moore | 稳定可靠 |
| **实时系统** | Mealy | 响应快 |
| **调试优先** | Moore | 易于追踪 |
| **内存受限** | Mealy | 状态少 |
| **AI Agent** | Moore | 输出稳定 |

---

## 六、AI Agent中的应用

### 6.1 Moore机：Agent状态输出

**场景**：RAG系统的状态显示

```python
from enum import Enum

class RAGState(Enum):
    IDLE = "idle"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    GENERATING = "generating"
    COMPLETED = "completed"

class RAGMooreAgent:
    """使用Moore机的RAG Agent"""

    def __init__(self):
        self.current_state = RAGState.IDLE

        # 状态转移
        self.transitions = {
            (RAGState.IDLE, 'query'): RAGState.RETRIEVING,
            (RAGState.RETRIEVING, 'retrieved'): RAGState.REASONING,
            (RAGState.REASONING, 'reasoned'): RAGState.GENERATING,
            (RAGState.GENERATING, 'generated'): RAGState.COMPLETED,
        }

        # 输出仅依赖状态（Moore机）
        self.outputs = {
            RAGState.IDLE: "💤 等待用户输入",
            RAGState.RETRIEVING: "🔍 正在检索文档...",
            RAGState.REASONING: "🧠 正在推理...",
            RAGState.GENERATING: "✍️ 正在生成答案...",
            RAGState.COMPLETED: "✅ 完成",
        }

    def get_output(self) -> str:
        """获取当前输出（仅依赖状态）"""
        return self.outputs[self.current_state]

    def transition(self, event: str):
        """执行状态转移"""
        key = (self.current_state, event)
        if key in self.transitions:
            self.current_state = self.transitions[key]


# 使用示例
agent = RAGMooreAgent()
print(agent.get_output())  # 💤 等待用户输入

agent.transition('query')
print(agent.get_output())  # 🔍 正在检索文档...

agent.transition('retrieved')
print(agent.get_output())  # 🧠 正在推理...
```

---

### 6.2 Mealy机：事件驱动响应

**场景**：聊天机器人的快速响应

```python
class ChatBotMealyAgent:
    """使用Mealy机的聊天机器人"""

    def __init__(self):
        self.current_state = 'idle'

        # 状态转移
        self.transitions = {
            ('idle', 'greeting'): 'chatting',
            ('chatting', 'question'): 'chatting',
            ('chatting', 'bye'): 'idle',
        }

        # 输出依赖状态和输入（Mealy机）
        self.outputs = {
            ('idle', 'greeting'): "你好！我是AI助手，有什么可以帮你的吗？",
            ('chatting', 'question'): "让我想想...",
            ('chatting', 'bye'): "再见！期待下次见面！",
        }

    def respond(self, event: str) -> str:
        """响应事件（输出依赖状态和输入）"""
        key = (self.current_state, event)
        if key not in self.outputs:
            return "抱歉，我没理解。"

        # 输出（Mealy：立即响应）
        output = self.outputs[key]

        # 状态转移
        self.current_state = self.transitions[key]

        return output


# 使用示例
bot = ChatBotMealyAgent()
print(bot.respond('greeting'))  # 你好！我是AI助手...
print(bot.respond('question'))  # 让我想想...
print(bot.respond('bye'))       # 再见！期待下次见面！
```

---

## 七、实战对比

### 7.1 相同功能的两种实现

**需求**：识别二进制字符串中1的个数是否为3的倍数

**Mealy机实现**：
```python
mealy_mod3 = MealyMachine(
    states={'s0', 's1', 's2'},  # 余数0, 1, 2
    input_alphabet={'0', '1'},
    output_alphabet={'Y', 'N'},
    transitions={
        ('s0', '0'): 's0', ('s0', '1'): 's1',
        ('s1', '0'): 's1', ('s1', '1'): 's2',
        ('s2', '0'): 's2', ('s2', '1'): 's0',
    },
    outputs={
        ('s0', '0'): 'Y', ('s0', '1'): 'N',
        ('s1', '0'): 'N', ('s1', '1'): 'N',
        ('s2', '0'): 'N', ('s2', '1'): 'Y',
    },
    initial_state='s0'
)
```

**Moore机实现**：
```python
moore_mod3 = MooreMachine(
    states={'s0', 's1', 's2'},  # 余数0, 1, 2
    input_alphabet={'0', '1'},
    output_alphabet={'Y', 'N'},
    transitions={
        ('s0', '0'): 's0', ('s0', '1'): 's1',
        ('s1', '0'): 's1', ('s1', '1'): 's2',
        ('s2', '0'): 's2', ('s2', '1'): 's0',
    },
    outputs={
        's0': 'Y',  # 余数0 → 是3的倍数
        's1': 'N',  # 余数1 → 不是
        's2': 'N',  # 余数2 → 不是
    },
    initial_state='s0'
)
```

**对比**：
- **状态数**：相同（3个状态）
- **输出时机**：Mealy立即，Moore延迟一步
- **输出长度**：Mealy = 输入长度，Moore = 输入长度 + 1

---

## 八、总结

### 核心要点

1. **输出模式**：
   - Mealy：λ(q, a) = o（依赖状态和输入）
   - Moore：λ(q) = o（仅依赖状态）

2. **状态数**：
   - Mealy：通常更少
   - Moore：可能更多

3. **响应速度**：
   - Mealy：快（输入立即输出）
   - Moore：慢（需要状态转移）

4. **稳定性**：
   - Mealy：可能抖动
   - Moore：稳定

5. **生产环境**：
   - Mealy：较少使用
   - Moore：更常用

### 选择建议

**优先使用Moore机**（生产环境）：
- ✅ 输出稳定
- ✅ 易于调试
- ✅ 易于测试
- ✅ 适合AI Agent

**特殊场景使用Mealy机**：
- ✅ 需要快速响应
- ✅ 内存受限
- ✅ 实时系统

### 学习建议

1. **理解输出模式**：λ的定义差异
2. **手写实现**：两种机器的完整实现
3. **掌握转换**：Mealy ⟺ Moore
4. **实践应用**：AI Agent状态管理
5. **对比分析**：理解权衡取舍

---

## 参考资料

1. **经典理论**：
   - GeeksforGeeks - Mealy Machine vs Moore Machine (2026)
   - Medium - Mealy vs. Moore... You decide.
   - Electronics Stack Exchange - How to choose between Mealy and Moore
   - Wikipedia - Mealy machine, Moore machine

2. **AI Agent应用**：
   - LangGraph - State-based output
   - Agent state management patterns

---

**版本**: v1.0
**最后更新**: 2026-02-14
**代码行数**: ~450行
