# Agent推理与决策 - 核心概念1：ReAct推理模式

> ReAct = Reasoning + Acting，让 LLM 在"思考"和"行动"之间交替循环，是 Agent 推理的经典范式

---

## 一句话定义

**ReAct 是一种让 LLM 交替进行推理（Thought）和行动（Action）的框架，通过 Thought→Action→Observation 循环实现多步问题求解。**

---

## 1. ReAct 的起源与核心思想

### 1.1 论文背景

ReAct 由 Yao et al. 在 2022 年论文《ReAct: Synergizing Reasoning and Acting in Language Models》中提出。核心发现：

- **纯推理（Chain-of-Thought）**：LLM 只思考不行动，容易产生幻觉
- **纯行动（Action-only）**：LLM 只调用工具不思考，缺乏规划能力
- **ReAct**：推理指导行动，行动的结果反馈推理，两者协同效果最好

```
纯 CoT：    Thought → Thought → Thought → Answer（可能幻觉）
纯 Action： Action → Action → Action → Answer（缺乏规划）
ReAct：     Thought → Action → Observation → Thought → Action → ... → Answer
```

### 1.2 Thought→Action→Observation 循环

这是 ReAct 的核心机制，每一轮包含三个阶段：

```
┌─────────────────────────────────────────────┐
│                 ReAct 循环                    │
│                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Thought  │───→│  Action  │───→│Observe │ │
│  │ (LLM想)  │    │ (执行工具)│    │(看结果) │ │
│  └──────────┘    └──────────┘    └────────┘ │
│       ↑                                │     │
│       └────────────────────────────────┘     │
│                  循环继续                      │
│                                              │
│  终止条件：LLM 输出 Final Answer / Finish     │
└─────────────────────────────────────────────┘
```

**三个阶段的职责：**

| 阶段 | 谁执行 | 做什么 | 示例 |
|------|--------|--------|------|
| Thought | LLM | 分析当前状态，决定下一步 | "我需要搜索北京天气" |
| Action | 工具 | 执行 LLM 选择的操作 | `weather_tool("北京")` |
| Observation | 系统 | 将工具返回值反馈给 LLM | "北京 5°C，多云" |

### 1.3 与纯 CoT 和纯 Action 的关键区别

```python
# 纯 CoT（Chain-of-Thought）—— 只想不做
"""
Question: 北京今天天气如何？
Thought: 北京在华北地区，3月份通常比较冷...
Thought: 大概5-10度左右吧...
Answer: 北京今天大约5-10度  ← 完全是猜的！
"""

# 纯 Action —— 只做不想
"""
Question: 对比北京和上海的天气
Action: weather("北京")  ← 没有规划，不知道还需要查上海
Observation: 北京 5°C
Answer: 北京5°C  ← 忘了上海！
"""

# ReAct —— 边想边做
"""
Question: 对比北京和上海的天气
Thought: 需要分别查两个城市的天气，先查北京
Action: weather("北京")
Observation: 北京 5°C，多云
Thought: 北京查到了，现在查上海
Action: weather("上海")
Observation: 上海 12°C，晴
Thought: 两个城市都查到了，可以对比了
Final Answer: 北京5°C多云，上海12°C晴，上海比北京暖7度
"""
```

---

## 2. LangChain 中的 ReAct 实现

### 2.1 create_react_agent() 工厂函数

[来源: source_react_agent_01.md - libs/langchain/langchain_classic/agents/react/agent.py]

```python
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: AgentOutputParser | None = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: bool | list[str] = True,
) -> Runnable
```

**参数解析：**

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `llm` | 语言模型 | 必填 |
| `tools` | 可用工具列表 | 必填 |
| `prompt` | 推理提示模板 | 必填（通常用 `hub.pull("hwchase17/react")`） |
| `output_parser` | 输出解析器 | `ReActSingleInputOutputParser` |
| `tools_renderer` | 工具描述渲染器 | `render_text_description` |
| `stop_sequence` | 停止序列 | `True`（即 `["\nObservation"]`） |

### 2.2 LCEL 链结构

[来源: source_react_agent_01.md] 这是 `create_react_agent` 内部构建的 LCEL 链：

```python
agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm_with_stop
    | output_parser
)
```

**数据流详解：**

```
输入: {"input": "用户问题", "intermediate_steps": [...]}
  │
  ▼
Step 1: RunnablePassthrough.assign()
  │  将 intermediate_steps 格式化为字符串形式的 agent_scratchpad
  │  输出: {"input": "...", "agent_scratchpad": "Thought: ...\nAction: ...\nObservation: ..."}
  │
  ▼
Step 2: prompt
  │  将 input、agent_scratchpad、tools、tool_names 填入模板
  │  输出: 完整的 Prompt 字符串
  │
  ▼
Step 3: llm_with_stop
  │  LLM 生成文本，遇到 "\nObservation" 就停止
  │  输出: "Thought: 我需要...\nAction: search[query]"
  │
  ▼
Step 4: output_parser
  │  解析 LLM 输出，提取 Action 和参数
  │  输出: AgentAction("search", "query", ...) 或 AgentFinish(...)
```

### 2.3 Prompt 配置

[来源: source_react_agent_01.md, context7_langchain_01.md]

`create_react_agent` 在内部对 prompt 做了两件事：

```python
# 1. 注入工具描述和工具名称
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# 2. 绑定 stop sequence
llm_with_stop = llm.bind(stop=["\nObservation"])
```

**标准 ReAct Prompt 模板**（`hwchase17/react`）的核心结构：

```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

---

## 3. ReAct 推理示例

### 3.1 完整多步推理过程

[来源: source_react_agent_01.md - libs/langchain/langchain_classic/agents/react/wiki_prompt.py]

LangChain 源码中包含 6 个经典推理示例。以下是一个典型的多步推理：

```
Question: What is the elevation range for the area that the eastern sector
          of the Colorado orogeny extends into?

Thought 1: I need to search Colorado orogeny, find the area that the eastern
           sector extends into, then find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building
               (an orogeny) in Colorado and surrounding areas.

Thought 2: It does not mention the eastern sector. So I need to look up
           eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains.

Thought 3: The eastern sector of the Colorado orogeny extends into the
           High Plains. I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of the two distinct land regions
               of the Great Plains... elevation from around 1,800 to 7,000 ft.

Thought 4: High Plains rise in elevation from around 1,800 to 7,000 ft.
Action 4: Finish[1,800 to 7,000 ft]
```

**推理过程分析：**

| 步骤 | Thought 做了什么 | Action 做了什么 | 推理质量 |
|------|-----------------|----------------|---------|
| 1 | 制定计划：搜索→查找→获取 | 搜索主题 | 有规划 |
| 2 | 发现信息不足，调整策略 | 换个角度查找 | 自我修正 |
| 3 | 提取关键信息，继续深入 | 搜索新目标 | 信息链接 |
| 4 | 确认信息充分，给出答案 | 终止循环 | 知道何时停止 |

### 3.2 推理模式总结

从源码示例中可以提炼出 ReAct 的几种推理模式：

```
模式1：直接搜索 → 找到答案 → 结束（1-2步）
模式2：搜索 → 信息不足 → 换关键词 → 找到 → 结束（3-4步）
模式3：搜索A → 发现需要B → 搜索B → 对比 → 结束（4-6步）
模式4：搜索 → 结果有歧义 → 消歧 → 确认 → 结束（3-5步）
```

---

## 4. ReAct 的核心设计要素

### 4.1 Format Instructions（格式指令）

格式指令是 ReAct 的"规则手册"，告诉 LLM 必须按什么格式输出。

[来源: source_reasoning_strategies_03.md - libs/langchain/langchain_classic/agents/mrkl/prompt.py]

```
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

**为什么格式指令如此重要？**

- LLM 是文本生成器，没有格式约束就会自由发挥
- 格式指令让输出**可解析**——OutputParser 能从中提取结构化信息
- `can repeat N times` 告诉 LLM 可以多步推理
- `Final Answer` 提供了明确的终止信号

### 4.2 Tools Description（工具描述注入）

[来源: source_reasoning_strategies_03.md - ZeroShotAgent.create_prompt]

工具描述决定了 LLM "知道自己能做什么"：

```python
# 源码中的工具描述生成
tool_strings = "\n".join(
    [f"{tool.name}: {tool.description}" for tool in tools]
)
tool_names = ", ".join([tool.name for tool in tools])
```

生成的描述注入到 Prompt 中：

```
You have access to the following tools:

search: Search for information on the internet
calculator: Perform mathematical calculations
weather: Get current weather for a location

Use the following format:
...
Action: the action to take, should be one of [search, calculator, weather]
```

**工具描述的质量直接影响推理质量**：
- 描述太模糊 → LLM 不知道什么时候该用这个工具
- 描述太冗长 → 占用 Context Window，降低推理效率
- 描述有歧义 → LLM 可能选错工具

### 4.3 Agent Scratchpad（推理历史）

[来源: source_react_agent_01.md - libs/langchain/langchain_classic/agents/format_scratchpad/log.py]

Scratchpad 是 Agent 的"草稿纸"，记录之前所有的推理步骤：

```python
def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
```

**工作原理：**

```
第1轮后的 scratchpad:
""（空的，第一次推理）

第2轮后的 scratchpad:
"Thought: 我需要搜索...
Action: search[query]
Observation: 搜索结果...
Thought: "  ← 注意末尾的 "Thought: "，引导 LLM 继续推理

第3轮后的 scratchpad:
"Thought: 我需要搜索...
Action: search[query]
Observation: 搜索结果...
Thought: 信息不够，继续查...
Action: search[another query]
Observation: 更多结果...
Thought: "
```

**关键设计**：末尾的 `Thought: ` 前缀是一个巧妙的设计——它引导 LLM 自然地接着"想"下去，而不是输出其他格式的内容。

### 4.4 Stop Sequence（停止序列）

[来源: source_react_agent_01.md]

Stop Sequence 是防止 LLM 幻觉的关键机制：

```python
# 默认 stop sequence
llm_with_stop = llm.bind(stop=["\nObservation"])
```

**为什么需要 Stop Sequence？**

没有 stop sequence 时，LLM 可能会这样：

```
Thought: 我需要搜索天气
Action: weather[北京]
Observation: 北京今天5°C，多云    ← LLM 自己编的！不是真实工具返回的！
Thought: 好的，我知道了
Final Answer: 北京今天5°C，多云   ← 基于幻觉的答案
```

有了 `stop=["\nObservation"]`，LLM 生成到 `Action: weather[北京]` 后就会停止，等待真实的工具执行结果。

**这是 ReAct 模式中最精妙的设计之一**：用一个简单的停止条件，就解决了 LLM 自说自话的问题。

### 4.5 Output Parser（输出解析器）

[来源: source_react_agent_01.md - libs/langchain/langchain_classic/agents/react/output_parser.py]

OutputParser 负责从 LLM 的文本输出中提取结构化的决策：

```python
class ReActOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action_prefix = "Action: "
        # 找到最后一行以 "Action: " 开头的内容
        action_block = text.strip().split("\n")[-1]
        action_str = action_block[len(action_prefix):]

        # 用正则提取 Action 名称和参数
        # 格式：ActionName[ActionInput]
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        action = re_matches.group(1)      # 工具名
        action_input = re_matches.group(2) # 工具参数

        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)
```

**两种解析结果：**

| LLM 输出 | 解析结果 | 含义 |
|----------|---------|------|
| `Action: Search[LangChain]` | `AgentAction("Search", "LangChain")` | 继续循环，执行工具 |
| `Action: Finish[最终答案]` | `AgentFinish({"output": "最终答案"})` | 停止循环，返回答案 |

---

## 5. 完整代码示例

### 5.1 基础 ReAct Agent

```python
"""
ReAct 推理模式实战
演示：构建一个能搜索和计算的 ReAct Agent
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.tools import tool


# ===== 1. 定义工具 =====

@tool
def search(query: str) -> str:
    """Search for information about a topic. Use this when you need to find facts."""
    # 模拟搜索结果
    mock_results = {
        "langchain": "LangChain is a framework for developing applications powered by LLMs.",
        "react pattern": "ReAct combines reasoning and acting in LLM agents.",
        "python": "Python is a high-level programming language.",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Input should be a valid math expression."""
    try:
        result = eval(expression)  # 仅用于演示
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ===== 2. 创建 ReAct Agent =====

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 拉取标准 ReAct prompt
prompt = hub.pull("hwchase17/react")

# 创建 agent（LCEL 链）
tools = [search, calculator]
agent = create_react_agent(llm, tools, prompt)

# 包装为 AgentExecutor（提供循环执行能力）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # 打印推理过程
    max_iterations=5,   # 最多5轮推理
    handle_parsing_errors=True,  # 解析错误时自动重试
)


# ===== 3. 执行推理 =====

result = executor.invoke({"input": "What is LangChain and how many characters are in its name?"})
print(f"\n最终答案: {result['output']}")
```

**预期输出（verbose=True 时）：**

```
> Entering new AgentExecutor chain...

Thought: I need to find out what LangChain is and count the characters in its name.
Let me first search for information about LangChain.
Action: search
Action Input: langchain
Observation: LangChain is a framework for developing applications powered by LLMs.

Thought: Now I know what LangChain is. Let me count the characters in "LangChain".
I can use the calculator for this.
Action: calculator
Action Input: len("LangChain")
Observation: 9

Thought: I now know the final answer.
Final Answer: LangChain is a framework for developing applications powered by LLMs.
The name "LangChain" has 9 characters.

> Finished chain.

最终答案: LangChain is a framework for developing applications powered by LLMs.
The name "LangChain" has 9 characters.
```

### 5.2 理解内部数据流

```python
"""
深入理解 ReAct 内部机制
演示：手动模拟 ReAct 的每个步骤
"""

from langchain.agents.react.agent import create_react_agent
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.schema import AgentAction, AgentFinish


# ===== 模拟 Scratchpad 格式化 =====

# 模拟两轮推理后的 intermediate_steps
intermediate_steps = [
    (
        AgentAction(
            tool="search",
            tool_input="LangChain",
            log="Thought: I need to search for LangChain\nAction: search[LangChain]"
        ),
        "LangChain is a framework for LLM applications."  # Observation
    ),
    (
        AgentAction(
            tool="calculator",
            tool_input="len('LangChain')",
            log="Thought: Now count the characters\nAction: calculator[len('LangChain')]"
        ),
        "9"  # Observation
    ),
]

# 格式化为 scratchpad 字符串
scratchpad = format_log_to_str(intermediate_steps)
print("=== Agent Scratchpad ===")
print(scratchpad)
print("========================")
```

**输出：**

```
=== Agent Scratchpad ===
Thought: I need to search for LangChain
Action: search[LangChain]
Observation: LangChain is a framework for LLM applications.
Thought: Now count the characters
Action: calculator[len('LangChain')]
Observation: 9
Thought:
========================
```

注意最后的 `Thought: `——这就是引导 LLM 继续推理的"钩子"。

---

## 6. ReAct 在 RAG 中的应用

ReAct 模式在 RAG 系统中特别有价值，因为检索往往需要多步：

```python
@tool
def retrieve(query: str) -> str:
    """Retrieve relevant documents from the knowledge base."""
    # 向量检索逻辑
    ...

@tool
def rerank(query: str, documents: str) -> str:
    """Rerank documents by relevance to the query."""
    # 重排序逻辑
    ...

# Agent 的推理过程：
"""
Question: 公司的退款政策是什么？超过30天还能退吗？

Thought: 用户问了两个问题：1) 退款政策 2) 超过30天的情况。
         我先检索退款政策的基本信息。
Action: retrieve[退款政策]
Observation: 公司支持7天无理由退款，30天内有质量问题可退...

Thought: 找到了基本政策，但用户特别问了超过30天的情况。
         让我专门检索这个。
Action: retrieve[超过30天退款 特殊情况]
Observation: 超过30天的退款需要提供质量检测报告，走特殊审批流程...

Thought: 现在两个问题都有答案了，可以综合回答。
Final Answer: 公司退款政策分三档：7天无理由退款；30天内质量问题可退；
             超过30天需提供质量检测报告走特殊审批流程。
"""
```

**ReAct 让 RAG 从"一次检索"升级为"智能多轮检索"。**

---

## 7. ReAct 的局限性与演进

### 7.1 已知局限

| 局限 | 说明 | 影响 |
|------|------|------|
| 文本解析脆弱 | 依赖正则匹配，LLM 格式稍有偏差就解析失败 | 生产环境不够稳定 |
| 推理开销大 | 每步都要完整的 LLM 调用 | 延迟高、成本高 |
| 单工具限制 | 经典 ReAct 每步只能调用一个工具 | 无法并行执行 |
| Context 膨胀 | Scratchpad 越来越长，占用 Context Window | 长推理链性能下降 |

### 7.2 演进方向（2025-2026）

[来源: search_agent_reasoning_01.md, context7_langchain_01.md]

```
ReAct（2022）
  ↓
Tool Calling（2023-2024）：模型原生函数调用，更可靠
  ↓
混合模式（2025）：CoT + Tool Calling，兼顾推理和可靠性
  ↓
LangGraph 状态机（2025-2026）：更灵活的推理流程控制
  ↓
Structured Output（2026）：主循环集成结构化输出
```

**ReAct 仍然有价值的场景**：
- 需要显式推理过程（可解释性要求高）
- 模型不支持原生函数调用
- 教学和调试（推理过程完全透明）

---

## 8. 核心架构总结

[来源: source_react_agent_01.md]

```
Input → Prompt(tools + scratchpad) → LLM(stop=Observation) → OutputParser
                                                                    ↓
                                                          AgentAction / AgentFinish
                                                                    ↓
                                                    ┌───────────────┴───────────────┐
                                                    ↓                               ↓
                                              AgentAction                     AgentFinish
                                                    ↓                               ↓
                                            Tool Execution                    返回最终答案
                                                    ↓
                                              Observation
                                                    ↓
                                          Append to scratchpad
                                                    ↓
                                            Loop back to Prompt
```

**一句话记住 ReAct**：LLM 想一步做一步，做完看结果，看完再想，想明白了就回答。

---

## 学习检查清单

- [ ] 能解释 ReAct 的三个阶段（Thought/Action/Observation）
- [ ] 理解 `create_react_agent()` 的 LCEL 链结构
- [ ] 知道 Stop Sequence 的作用（防止 LLM 自己编造 Observation）
- [ ] 理解 Scratchpad 如何累积推理历史
- [ ] 能区分 `AgentAction`（继续）和 `AgentFinish`（停止）
- [ ] 了解 ReAct 与纯 CoT、纯 Action 的区别
- [ ] 知道 ReAct 在 RAG 中的应用价值

---

## 下一步学习

→ **03_核心概念_2_思维链与Scratchpad.md**：深入理解 Scratchpad 的格式化机制和 CoT 原理

---

[来源: source_react_agent_01.md, context7_langchain_01.md, source_reasoning_strategies_03.md, search_agent_reasoning_01.md]
