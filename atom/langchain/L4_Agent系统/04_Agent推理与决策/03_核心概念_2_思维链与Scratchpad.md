# 核心概念 2：思维链与 Scratchpad

> Agent 的推理不是"灵光一闪"，而是"一步步写在草稿纸上"的迭代过程

---

## 什么是思维链（Chain-of-Thought）？

**思维链（CoT）的核心思想：让 LLM "展示工作过程"，而不是直接给答案。**

人类解决复杂问题时，不会直接跳到答案，而是一步步推导。CoT 就是让 LLM 也这样做 —— 把中间推理步骤写出来，最终答案的质量会显著提升。

**前端类比：** 就像 Redux DevTools 的 action log。你不会只看最终 state，而是回放每一个 action 和 state 变化，才能理解"为什么 state 变成了这样"。CoT 就是 LLM 的 action log。

**日常生活类比：** 就像数学考试要求"写出解题过程"。老师不只看最终答案，还要看你怎么推导的。写过程不仅方便检查，还能帮你理清思路、减少错误。

---

## CoT 的三种形态

### 1. Zero-Shot CoT："Let's think step by step"

最简单的 CoT —— 只需要在 prompt 末尾加一句话：

```python
# Zero-Shot CoT：不需要任何示例，一句话触发推理
prompt = """
问题：一个商店有 23 个苹果，卖掉了 17 个，又进货了 12 个，现在有多少个？

Let's think step by step.
"""

# LLM 会自动展开推理：
# Step 1: 初始有 23 个苹果
# Step 2: 卖掉 17 个，剩 23 - 17 = 6 个
# Step 3: 进货 12 个，现在 6 + 12 = 18 个
# 答案：18 个
```

**为什么有效？** "Let's think step by step" 激活了 LLM 在训练数据中学到的"分步推理"模式。就像告诉一个学生"别急，慢慢想"。

### 2. Few-Shot CoT：提供推理示例

给 LLM 看几个"带推理过程的示例"，它就会模仿这种推理风格：

```python
# Few-Shot CoT：通过示例教 LLM 怎么推理
few_shot_prompt = """
示例 1：
问题：小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
推理：初始 5 个 → 给出 2 个剩 3 个 → 买入 3 个变 6 个
答案：6 个

示例 2：
问题：一根绳子 10 米，剪掉 3 米，再接上 5 米，多长？
推理：初始 10 米 → 剪掉 3 米剩 7 米 → 接上 5 米变 12 米
答案：12 米

现在回答：
问题：{user_question}
推理：
"""
```

**Few-Shot 比 Zero-Shot 更强的原因：** 示例不仅触发了推理模式，还教会了 LLM 推理的"格式"和"粒度"。

### 3. Agent 中的 CoT：迭代推理（最特殊）

**Agent 中的 CoT 和普通 CoT 有本质区别：**

| 维度 | 普通 CoT | Agent CoT |
|------|----------|-----------|
| 推理次数 | 一次性完成 | 多轮迭代 |
| 信息来源 | 仅靠已有知识 | 每轮获取新信息 |
| 推理依据 | 固定上下文 | 上下文不断增长 |
| 类比 | 闭卷考试 | 开卷考试 + 可以查资料 |

```
普通 CoT：
  问题 → 推理步骤1 → 推理步骤2 → 推理步骤3 → 答案
  （一次性完成，中间不获取新信息）

Agent CoT：
  问题 → 推理"需要查天气" → [调用工具] → 拿到天气数据
       → 推理"还需要查酒店" → [调用工具] → 拿到酒店数据
       → 推理"信息够了" → 最终答案
  （每轮推理都基于新获取的信息）
```

**这就是 Scratchpad 存在的意义 —— 它记录了每一轮的推理和观察，让 LLM 在下一轮推理时能"看到"之前做了什么。**

---

## Scratchpad 机制：Agent 的草稿纸

### 什么是 Agent Scratchpad？

**Scratchpad（草稿纸）是 Agent 推理历史的累积记录。** 每次 LLM 做出决策并执行工具后，结果会被追加到 Scratchpad 中，下一轮推理时 LLM 能看到完整的历史。

```
┌─────────────────────────────────────────────────────┐
│                  Agent Scratchpad                     │
│                                                       │
│  第 1 轮记录：                                        │
│  ┌─────────────────────────────────────────────┐     │
│  │ Thought: 我需要查询北京天气                    │     │
│  │ Action: get_weather                          │     │
│  │ Action Input: {"city": "北京"}               │     │
│  │ Observation: 北京今天晴，22°C                 │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  第 2 轮记录：                                        │
│  ┌─────────────────────────────────────────────┐     │
│  │ Thought: 天气查到了，还需要查酒店              │     │
│  │ Action: search_hotels                        │     │
│  │ Action Input: {"city": "北京"}               │     │
│  │ Observation: 推荐：国贸大酒店 ¥800/晚         │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  Thought:  ← LLM 从这里继续推理                      │
└─────────────────────────────────────────────────────┘
```

在 LangChain 中，Scratchpad 的原始数据是 `intermediate_steps: List[Tuple[AgentAction, str]]` —— 一个 (动作, 观察结果) 的列表。但 LLM 不能直接读这个 Python 列表，需要格式化成文本或消息。

---

## 源码解析：两种格式化方式

### 方式 1：format_log_to_str() —— 格式化为字符串

```python
# 来源: langchain_classic/agents/format_scratchpad/log.py

def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """将推理历史格式化为字符串，适用于纯文本 LLM"""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log  # 包含 Thought + Action + Action Input
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
```

**逐行解读：**

1. `thoughts = ""` —— 初始化空字符串，准备累积所有历史
2. `for action, observation in intermediate_steps` —— 遍历每一步
3. `thoughts += action.log` —— 追加 LLM 的推理文本（包含 Thought 和 Action）
4. `thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"` —— 追加工具返回的观察结果，并在末尾加上 `Thought: ` 前缀

**生成的文本长这样：**

```
Thought: 我需要查询北京天气
Action: get_weather
Action Input: {"city": "北京"}
Observation: 北京今天晴，22°C
Thought: 天气查到了，还需要查酒店
Action: search_hotels
Action Input: {"city": "北京"}
Observation: 推荐：国贸大酒店 ¥800/晚
Thought:
```

**注意末尾的 `Thought:`** —— 这不是多余的！它是一个"引导前缀"，告诉 LLM："轮到你继续推理了"。就像考试卷子上印好了"解："，学生自然会接着写解题过程。

### 方式 2：format_log_to_messages() —— 格式化为消息列表

```python
# 来源: langchain_classic/agents/format_scratchpad/log_to_messages.py

def format_log_to_messages(
    intermediate_steps: List[Tuple[AgentAction, str]],
    template_tool_response: str = "{observation}",
) -> List[BaseMessage]:
    """将推理历史格式化为消息列表，适用于 Chat API"""
    messages: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        # LLM 的推理和决策 → AI 消息
        messages.append(AIMessage(content=action.log))
        # 工具的返回结果 → Human 消息（模拟用户反馈）
        messages.append(HumanMessage(
            content=template_tool_response.format(
                observation=observation
            )
        ))
    return messages
```

**生成的消息列表长这样：**

```python
[
    AIMessage(content="Thought: 我需要查天气\nAction: get_weather\n..."),
    HumanMessage(content="北京今天晴，22°C"),
    AIMessage(content="Thought: 还需要查酒店\nAction: search_hotels\n..."),
    HumanMessage(content="推荐：国贸大酒店 ¥800/晚"),
]
```

### 两种格式的对比

| 维度 | format_log_to_str | format_log_to_messages |
|------|-------------------|----------------------|
| 输出类型 | 单个字符串 | 消息列表 `List[BaseMessage]` |
| 适用模型 | 纯文本 LLM（Completion API） | Chat 模型（Chat API） |
| 注入方式 | 作为 `{agent_scratchpad}` 变量 | 作为消息序列插入对话 |
| 引导方式 | 末尾 `Thought:` 前缀 | 无需显式引导（Chat 模型自动续写） |
| 使用场景 | `create_react_agent` | `create_tool_calling_agent` |
| 前端类比 | 拼接 HTML 字符串 | 使用 React 组件列表 |

**选择建议：**
- 用 `create_react_agent`（文本推理）→ 用 `format_log_to_str`
- 用 `create_tool_calling_agent`（函数调用）→ 用 `format_to_tool_messages`（消息格式）

---

## Scratchpad 在 LCEL 链中的位置

回顾 `create_react_agent` 的源码，Scratchpad 是这样嵌入的：

```python
# 来源: langchain_classic/agents/react/agent.py

agent = (
    RunnablePassthrough.assign(
        # 关键！每次调用都重新格式化 intermediate_steps
        agent_scratchpad=lambda x: format_log_to_str(
            x["intermediate_steps"]
        ),
    )
    | prompt      # Scratchpad 被注入到 prompt 的 {agent_scratchpad} 位置
    | llm_with_stop
    | output_parser
)
```

**数据流：**

```
输入: {
    "input": "北京天气怎么样？",
    "intermediate_steps": [(action1, obs1), (action2, obs2)]
}
    │
    ▼ RunnablePassthrough.assign()
{
    "input": "北京天气怎么样？",
    "intermediate_steps": [...],
    "agent_scratchpad": "Thought: ...\nObservation: ...\nThought: "
}
    │
    ▼ prompt（模板替换）
"Question: 北京天气怎么样？
Thought: 我需要查天气
Action: get_weather
...
Observation: 北京今天晴，22°C
Thought: "
    │
    ▼ llm_with_stop（LLM 推理，遇到 Observation 停止）
"天气数据已获取，可以回答了。
Final Answer: 北京今天晴天，22°C"
    │
    ▼ output_parser（提取决策）
AgentFinish(return_values={"output": "北京今天晴天，22°C"})
```

---

## Scratchpad 的三大作用

### 作用 1：维持推理上下文

没有 Scratchpad，LLM 每轮推理都是"失忆"的 —— 不知道之前做了什么。

```python
# 没有 Scratchpad 的灾难场景
# 第 1 轮：LLM 决定查天气 → 查到了 22°C
# 第 2 轮：LLM 不知道已经查过天气 → 又去查天气（死循环！）

# 有 Scratchpad 的正常场景
# 第 1 轮：LLM 决定查天气 → 查到了 22°C
# 第 2 轮：LLM 看到 Scratchpad 里已经有天气数据 → 决定查酒店
# 第 3 轮：LLM 看到天气和酒店都有了 → 给出最终答案
```

**前端类比：** 就像 React 组件的 state。没有 state，组件每次渲染都不知道之前发生了什么。Scratchpad 就是 Agent 的 state。

### 作用 2：避免重复推理

Scratchpad 让 LLM 能看到"已经做过什么"，从而避免重复调用同一个工具：

```
Scratchpad 内容：
  Thought: 查询北京天气
  Action: get_weather("北京")
  Observation: 晴天 22°C
  Thought:  ← LLM 从这里继续

LLM 看到已经查过天气，不会再查一次，而是决定下一步该做什么。
```

### 作用 3：引导 LLM 继续推理

`format_log_to_str` 在末尾添加的 `Thought:` 前缀是一个精妙的设计：

```python
# 末尾的 "Thought: " 前缀
thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
#                                                  ^^^^^^^^^^^^^^^^
#                                                  默认是 "Thought: "
```

这个前缀的作用：
1. **格式引导**：告诉 LLM "接下来应该输出 Thought 格式的内容"
2. **角色暗示**：LLM 看到 `Thought:` 就知道该"思考"了，而不是直接输出答案
3. **防止格式错乱**：没有这个前缀，LLM 可能输出任意格式的文本

**日常生活类比：** 就像填空题的提示。"请分析原因：____" 比 "请回答：____" 更能引导出分析性的回答。

---

## CoT 增强技术

### 技术 1：Few-Shot 推理示例注入

在 Prompt 中提供完整的推理示例，教 LLM 怎么"想"：

```python
# ReAct wiki_prompt.py 中的 Few-Shot 示例（简化版）
EXAMPLES = """
Question: What is the elevation range for the area that the
eastern sector of the Colorado orogeny extends into?

Thought 1: I need to search Colorado orogeny, find the area
that the eastern sector extends into, then find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building...

Thought 2: It does not mention the eastern sector.
So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: The eastern sector extends into the High Plains.

Thought 3: High Plains rise in elevation from around 1,800 to 7,000 ft.
Action 3: Finish[1,800 to 7,000 ft]
"""
```

**LangChain 的 wiki_prompt.py 包含 6 个完整的推理示例**，覆盖了多种推理模式：
- 多步搜索 + 查找
- 信息不足时的重新搜索
- 直接从搜索结果中提取答案
- 需要比较多个信息源

### 技术 2：System Prompt 中的推理指令

通过 System Prompt 明确要求 LLM 展示推理过程：

```python
system_prompt = """You are a helpful assistant.
When solving problems:
1. Always think step by step
2. Show your reasoning before taking action
3. After each observation, reflect on what you learned
4. Only give a final answer when you have enough information

Use the following format:
Thought: your reasoning about what to do next
Action: the tool to use
Action Input: the input for the tool
"""
```

### 技术 3：自我反思（Self-Reflection）模式

让 LLM 在每轮推理时不仅决定下一步，还反思之前的推理是否正确：

```python
# 增强版 Scratchpad 格式化（概念示例）
def format_with_reflection(intermediate_steps):
    """在标准 Scratchpad 基础上添加反思提示"""
    base = format_log_to_str(intermediate_steps)
    if intermediate_steps:
        # 在末尾添加反思引导
        base += (
            "Before deciding the next action, "
            "reflect on what I've learned so far "
            "and whether my approach is correct.\n"
            "Reflection: "
        )
    return base
```

**Self-Reflection 的价值：**
- 减少"惯性推理"（LLM 沿着错误方向一路走到底）
- 提高工具选择的准确性
- 在 RAG 场景中，帮助 LLM 判断检索结果是否真正相关

---

## 完整示例：Scratchpad 的生命周期

```python
"""
Scratchpad 生命周期演示
展示 intermediate_steps 如何在多轮推理中累积
"""
from langchain_core.agents import AgentAction, AgentFinish

# ===== 初始状态 =====
intermediate_steps = []  # 空的 Scratchpad

# ===== 第 1 轮：LLM 推理 =====
# Scratchpad 为空，LLM 只看到用户问题
# format_log_to_str([]) → ""（空字符串）

action_1 = AgentAction(
    tool="get_weather",
    tool_input={"city": "北京"},
    log=(
        "I need to find the weather in Beijing.\n"
        "Action: get_weather\n"
        "Action Input: {\"city\": \"北京\"}"
    )
)
observation_1 = "北京今天晴，22°C，PM2.5: 35"

# 追加到 Scratchpad
intermediate_steps.append((action_1, observation_1))

# ===== 第 2 轮：LLM 推理 =====
# Scratchpad 包含第 1 轮的记录
# format_log_to_str(intermediate_steps) →
#   "I need to find the weather in Beijing.
#    Action: get_weather
#    Action Input: {"city": "北京"}
#    Observation: 北京今天晴，22°C，PM2.5: 35
#    Thought: "

action_2 = AgentAction(
    tool="get_air_quality_advice",
    tool_input={"pm25": 35},
    log=(
        "Weather data obtained. PM2.5 is 35, "
        "I should check if it's suitable for outdoor activities.\n"
        "Action: get_air_quality_advice\n"
        "Action Input: {\"pm25\": 35}"
    )
)
observation_2 = "PM2.5 35 属于优良级别，适合户外运动"

intermediate_steps.append((action_2, observation_2))

# ===== 第 3 轮：LLM 推理 =====
# Scratchpad 包含第 1、2 轮的完整记录
# LLM 看到天气和空气质量都查到了 → 给出最终答案

finish = AgentFinish(
    return_values={
        "output": "北京今天晴天 22°C，空气质量优良（PM2.5: 35），非常适合户外运动！"
    },
    log="I now have both weather and air quality data. I can answer the question."
)

print(f"最终答案: {finish.return_values['output']}")
print(f"总推理轮数: {len(intermediate_steps) + 1}")  # +1 是最终回答那轮
print(f"工具调用次数: {len(intermediate_steps)}")
```

**运行输出：**
```
最终答案: 北京今天晴天 22°C，空气质量优良（PM2.5: 35），非常适合户外运动！
总推理轮数: 3
工具调用次数: 2
```

---

## 与 RAG 的关系

Scratchpad 在 RAG Agent 中尤其重要：

```
传统 RAG（无 Scratchpad）：
  查询 → 检索 → 生成
  （一次检索，结果好坏全靠运气）

Agent RAG（有 Scratchpad）：
  查询 → 推理"用什么关键词检索" → 检索 → 观察结果
       → Scratchpad 记录：检索了 X，结果不太相关
       → 推理"换个关键词" → 再次检索 → 观察结果
       → Scratchpad 记录：第二次检索结果更好
       → 推理"信息够了" → 生成最终答案
```

Scratchpad 让 RAG Agent 具备了：
- **自适应检索**：根据第一次检索结果调整策略
- **多轮检索**：不满意就换关键词再查
- **信息整合**：综合多次检索的结果给出答案

---

## 常见问题

**Q1: Scratchpad 会不会越来越长，超出 Context Window？**

会。这是 Agent 的一个实际限制。解决方案：
- `max_iterations` 限制最大轮数（默认 15）
- 高级方案：对历史步骤做摘要压缩（trim older steps）
- 选择 Context Window 更大的模型

**Q2: format_log_to_str 和 format_log_to_messages 能混用吗？**

不建议。它们对应不同的 Agent 类型：
- `create_react_agent` 用 `format_log_to_str`（文本格式）
- `create_tool_calling_agent` 用 `format_to_tool_messages`（消息格式）
混用会导致格式不匹配，LLM 无法正确解析历史。

**Q3: 为什么 format_log_to_messages 用 HumanMessage 而不是 ToolMessage？**

`format_log_to_messages` 是为 ReAct 文本推理设计的，用 HumanMessage 模拟"用户反馈观察结果"。而 `format_to_tool_messages`（用于 tool_calling_agent）使用 ToolMessage，因为现代 Chat API 有专门的 tool 消息类型。

---

[来源: `langchain_classic/agents/format_scratchpad/log.py`, `log_to_messages.py`, `react/agent.py`, `react/wiki_prompt.py`]
