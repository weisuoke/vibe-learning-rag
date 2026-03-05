# 实战代码 场景4：动态工具选择与 Agent 集成

> 构建一个多工具 Agent，演示动态工具选择策略和完整的 Agent 工具调用循环——从工具定义、智能筛选到多步推理，覆盖 RAG 场景中的检索工具集成

---

## 场景概述

前三个场景分别解决了 Tool 定义、函数调用链路、Schema 验证。本场景把它们串起来，回答一个核心问题：**当 Agent 拥有多个工具时，如何动态选择最相关的工具，并驱动完整的多步推理循环？**

你将学到：
- 定义一组功能各异的工具
- 基于关键词的简单工具筛选
- 手动实现完整的 Agent 循环（bind_tools → tool_calls → ToolMessage → 循环）
- 多轮对话中工具的持续调用
- RAG 场景下"检索工具 + 生成"的集成模式

每个示例都是完整可运行的 Python 代码，需要 OpenAI API Key。

---

## 环境准备

```python
# 安装依赖（只需执行一次）
# pip install langchain-core langchain-openai python-dotenv

# 配置 API Key
# 方式 1: 环境变量
# export OPENAI_API_KEY=your_key_here

# 方式 2: .env 文件
# OPENAI_API_KEY=your_key_here
# OPENAI_BASE_URL=https://your-proxy.com/v1  # 可选

import os
from dotenv import load_dotenv
load_dotenv()
```

---

## 示例1：定义多个工具

**解决的问题：** 为 Agent 准备一组功能各异的工具——搜索、计算、时间查询、知识库检索，模拟真实场景中的工具集。

```python
"""
定义多个工具，模拟真实 Agent 的工具集
每个工具的 docstring 是 LLM 选择工具的唯一依据，必须写清楚适用场景
"""
from langchain_core.tools import tool
from datetime import datetime


@tool
def search_web(query: str) -> str:
    """搜索网页获取最新信息。适用于需要实时数据、新闻、公开知识的问题。"""
    # 模拟搜索结果
    return f"[网页搜索] 关于'{query}'的结果: 找到 3 篇相关网页，最新更新于今天。"


@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库。适用于公司内部文档、政策规定、技术文档的查询。"""
    mock_kb = {
        "请假": "请假流程: 提前3天在OA系统提交，直属领导审批，超过3天需部门总监审批。",
        "报销": "报销流程: 保留原始发票，在财务系统提交，金额超5000需VP审批。",
        "部署": "部署流程: 代码合并到main分支，CI通过后自动部署到staging，手动确认后发布到production。",
    }
    for key, value in mock_kb.items():
        if key in query:
            return f"[知识库] {value}"
    return f"[知识库] 未找到与'{query}'直接相关的文档，建议尝试其他关键词。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。适用于需要精确数值计算的问题，如加减乘除、百分比等。"""
    try:
        result = eval(expression)  # 示例用，生产环境应使用安全解析器
        return f"[计算] {expression} = {result}"
    except Exception as e:
        return f"[计算错误] 表达式 '{expression}' 无法计算: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。适用于需要知道当前时间的问题。"""
    now = datetime.now()
    return f"[时间] 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


# ===== 汇总所有工具 =====
all_tools = [search_web, search_knowledge_base, calculate, get_current_time]

# ===== 验证工具定义 =====
for t in all_tools:
    print(f"  {t.name}: {t.description[:40]}...")
```

### 预期输出

```
  search_web: 搜索网页获取最新信息。适用于需要实时数据、新闻、公开知识的问题。...
  search_knowledge_base: 搜索内部知识库。适用于公司内部文档、政策规定、技术文档的查询。...
  calculate: 计算数学表达式。适用于需要精确数值计算的问题，如加减乘除、百分比...
  get_current_time: 获取当前日期和时间。适用于需要知道当前时间的问题。...
```

> **关键点：** 每个工具的 docstring 必须写清楚"适用于什么场景"。LLM 完全依赖这段描述来决定是否调用该工具。描述模糊 = 调用混乱。

---

## 示例2：基于关键词的简单工具选择

**解决的问题：** 当工具数量较多时，不需要每次都把所有工具发给 LLM。用关键词匹配先筛选一轮，减少 token 消耗，提高选择准确率。

```python
"""
基于关键词的工具选择器
思路：维护一个 关键词 → 工具名 的映射表，根据用户输入筛选相关工具
"""
from langchain_core.tools import tool
from datetime import datetime

# ===== 工具定义（复用示例1） =====
@tool
def search_web(query: str) -> str:
    """搜索网页获取最新信息。适用于需要实时数据、新闻、公开知识的问题。"""
    return f"[网页搜索] 关于'{query}'的结果"

@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库。适用于公司内部文档、政策规定、技术文档的查询。"""
    return f"[知识库] 关于'{query}'的文档"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。适用于需要精确数值计算的问题。"""
    return f"[计算] {expression} = {eval(expression)}"

@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return f"[时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

all_tools = [search_web, search_knowledge_base, calculate, get_current_time]


# ===== 关键词 → 工具名映射 =====
KEYWORD_TOOL_MAP = {
    "搜索": ["search_web"],
    "查找": ["search_web"],
    "新闻": ["search_web"],
    "最新": ["search_web"],
    "知识库": ["search_knowledge_base"],
    "内部": ["search_knowledge_base"],
    "文档": ["search_knowledge_base"],
    "政策": ["search_knowledge_base"],
    "请假": ["search_knowledge_base"],
    "报销": ["search_knowledge_base"],
    "计算": ["calculate"],
    "多少": ["calculate"],
    "加": ["calculate"],
    "乘": ["calculate"],
    "时间": ["get_current_time"],
    "几点": ["get_current_time"],
    "日期": ["get_current_time"],
    "今天": ["get_current_time"],
}


def select_tools_by_keyword(query: str, all_tools: list) -> list:
    """根据关键词选择相关工具"""
    tools_map = {t.name: t for t in all_tools}
    selected_names = set()

    for keyword, tool_names in KEYWORD_TOOL_MAP.items():
        if keyword in query:
            selected_names.update(tool_names)

    if not selected_names:
        # 没匹配到任何关键词 → 返回全部工具
        return all_tools

    return [tools_map[name] for name in selected_names if name in tools_map]


# ===== 测试工具选择 =====
test_queries = [
    "公司请假流程是什么？",
    "帮我计算 125 * 0.8 + 50",
    "今天几点了？",
    "搜索一下 Python 3.13 的新特性",
    "你好，介绍一下你自己",  # 无关键词命中
]

for q in test_queries:
    selected = select_tools_by_keyword(q, all_tools)
    names = [t.name for t in selected]
    print(f"问题: {q}")
    print(f"  选中工具: {names}")
    print()
```

### 预期输出

```
问题: 公司请假流程是什么？
  选中工具: ['search_knowledge_base']

问题: 帮我计算 125 * 0.8 + 50
  选中工具: ['calculate']

问题: 今天几点了？
  选中工具: ['get_current_time']

问题: 搜索一下 Python 3.13 的新特性
  选中工具: ['search_web']

问题: 你好，介绍一下你自己
  选中工具: ['search_web', 'search_knowledge_base', 'calculate', 'get_current_time']
```

> **关键词选择的局限：** 最后一个问题没有命中任何关键词，只能返回全部工具。用户表达方式千变万化，关键词很难穷举。这就是为什么生产环境更推荐用 LLM 做工具选择（见核心概念5中的 `LLMToolSelectorMiddleware`）。

---

## 示例3：手动实现完整的 Agent 循环（核心）

**解决的问题：** 这是本场景的核心——手动实现 Agent 的"思考-行动-观察"循环。理解这个循环，就理解了所有 Agent 框架的底层原理。

```
Agent 循环的本质：

用户问题 → LLM 思考 → 需要工具？
                         ├─ YES → 调用工具 → 把结果喂回 LLM → 继续思考
                         └─ NO  → 直接输出最终回答（退出循环）
```

```python
"""
手动实现完整的 Agent 循环
核心：bind_tools() + tool_calls 检测 + ToolMessage 回传 + 循环
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from datetime import datetime


# ===== 工具定义 =====
@tool
def search_web(query: str) -> str:
    """搜索网页获取最新信息。适用于需要实时数据、新闻、公开知识的问题。"""
    return (
        f"[网页搜索] 关于'{query}': Python 3.13 于 2024年10月发布，"
        "新增实验性自由线程模式和改进的错误消息。"
    )

@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库。适用于公司内部文档、政策规定、技术文档的查询。"""
    mock_kb = {
        "请假": "请假流程: 提前3天在OA系统提交，直属领导审批，超过3天需部门总监审批。",
        "报销": "报销流程: 保留原始发票，在财务系统提交，金额超5000需VP审批。",
    }
    for key, value in mock_kb.items():
        if key in query:
            return f"[知识库] {value}"
    return f"[知识库] 未找到与'{query}'直接相关的文档。"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。适用于需要精确数值计算的问题。"""
    try:
        return f"[计算] {expression} = {eval(expression)}"
    except Exception as e:
        return f"[计算错误] {e}"

@tool
def get_current_time() -> str:
    """获取当前日期和时间。适用于需要知道当前时间的问题。"""
    return f"[时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


all_tools = [search_web, search_knowledge_base, calculate, get_current_time]


# ===== Agent 循环实现 =====
def simple_agent(question: str, tools: list, model, max_steps: int = 5) -> str:
    """
    简单的 Agent 实现，展示工具调用循环。

    参数:
        question: 用户问题
        tools: 可用工具列表
        model: ChatModel 实例（未绑定工具）
        max_steps: 最大循环步数，防止无限循环
    """
    # 构建工具名 → 工具实例的映射
    tools_map = {t.name: t for t in tools}

    # 将工具绑定到模型
    model_with_tools = model.bind_tools(tools)

    # 初始化消息列表
    messages = [HumanMessage(content=question)]

    print(f"{'='*60}")
    print(f"用户问题: {question}")
    print(f"可用工具: {list(tools_map.keys())}")
    print(f"{'='*60}")

    for step in range(1, max_steps + 1):
        print(f"\n--- 第 {step} 步 ---")

        # 调用模型
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # 检查：模型是否请求调用工具？
        if not response.tool_calls:
            # 没有 tool_calls → 模型给出了最终回答
            print(f"[最终回答] {response.content}")
            return response.content

        # 有 tool_calls → 逐个执行工具
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            print(f"[调用工具] {tool_name}({tool_args})")

            # 执行工具
            if tool_name in tools_map:
                tool_result = tools_map[tool_name].invoke(tool_args)
            else:
                tool_result = f"错误: 未知工具 '{tool_name}'"

            print(f"[工具结果] {tool_result}")

            # 将结果作为 ToolMessage 回传
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id,
            ))

    return "达到最大步数限制，未能得出最终回答。"


# ===== 运行 Agent =====
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 测试 1: 需要知识库检索的问题
result1 = simple_agent("公司请假流程是什么？", all_tools, model)

print("\n")

# 测试 2: 需要计算的问题
result2 = simple_agent(
    "如果商品原价 299 元，打 8 折后再减 30 元，最终价格是多少？",
    all_tools, model,
)

print("\n")

# 测试 3: 不需要工具的简单问题
result3 = simple_agent("你好，请用一句话介绍自己。", all_tools, model)
```

### 预期输出

```
============================================================
用户问题: 公司请假流程是什么？
可用工具: ['search_web', 'search_knowledge_base', 'calculate', 'get_current_time']
============================================================

--- 第 1 步 ---
[调用工具] search_knowledge_base({'query': '请假流程'})
[工具结果] [知识库] 请假流程: 提前3天在OA系统提交，直属领导审批，超过3天需部门总监审批。

--- 第 2 步 ---
[最终回答] 公司请假流程如下：提前3天在OA系统提交请假申请，由直属领导审批。如果请假超过3天，还需要部门总监审批。


============================================================
用户问题: 如果商品原价 299 元，打 8 折后再减 30 元，最终价格是多少？
可用工具: ['search_web', 'search_knowledge_base', 'calculate', 'get_current_time']
============================================================

--- 第 1 步 ---
[调用工具] calculate({'expression': '299 * 0.8 - 30'})
[工具结果] [计算] 299 * 0.8 - 30 = 209.20000000000002

--- 第 2 步 ---
[最终回答] 商品原价 299 元，打 8 折后为 239.2 元，再减 30 元，最终价格为 209.2 元。


============================================================
用户问题: 你好，请用一句话介绍自己。
可用工具: ['search_web', 'search_knowledge_base', 'calculate', 'get_current_time']
============================================================

--- 第 1 步 ---
[最终回答] 你好！我是一个AI助手，可以帮你搜索信息、查询知识库、进行计算和查看时间。
```

### Agent 循环流程图

```
messages = [HumanMessage]
        |
        v
+-> model_with_tools.invoke(messages)
|       |
|       v
|   response 有 tool_calls？
|       |           |
|      YES         NO --> 返回 response.content（最终回答）
|       |
|       v
|   遍历 tool_calls:
|     1. 取出 name, args, id
|     2. tools_map[name].invoke(args)
|     3. 构造 ToolMessage(content=结果, tool_call_id=id)
|     4. 追加到 messages
|       |
+-------+  （回到循环顶部，带着工具结果再次调用模型）
```

> **这就是所有 Agent 框架的底层原理。** LangGraph、AutoGen、CrewAI 等框架的核心都是这个循环，只是加了状态管理、并发控制、错误恢复等生产级特性。

---

## 示例4：多轮对话中的工具调用

**解决的问题：** 真实场景中用户不会只问一个问题。Agent 需要在多轮对话中保持上下文，并在每一轮根据需要调用不同的工具。

```python
"""
多轮对话 Agent：在对话历史中持续调用工具
关键：messages 列表在多轮之间持续累积，模型能看到之前的工具调用和结果
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from datetime import datetime


# ===== 工具定义 =====
@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库。适用于公司内部文档、政策规定、技术文档的查询。"""
    mock_kb = {
        "请假": "年假15天，病假需医院证明，事假每月不超过3天。",
        "报销": "差旅报销标准: 火车二等座、经济型酒店，餐补每天100元。",
        "薪资": "薪资每月15日发放，五险一金按当地标准缴纳。",
    }
    for key, value in mock_kb.items():
        if key in query:
            return f"[知识库] {value}"
    return f"[知识库] 未找到'{query}'相关文档。"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。适用于需要精确数值计算的问题。"""
    try:
        return f"[计算] {expression} = {eval(expression)}"
    except Exception as e:
        return f"[计算错误] {e}"


tools = [search_knowledge_base, calculate]


def agent_step(messages: list, model_with_tools, tools_map) -> str | None:
    """执行一个 Agent 步骤：调用模型，处理工具调用，返回最终回答或 None"""
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        return response.content

    for tc in response.tool_calls:
        result = tools_map[tc["name"]].invoke(tc["args"])
        print(f"  [工具] {tc['name']}({tc['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # 工具调用后再次调用模型获取最终回答
    final = model_with_tools.invoke(messages)
    messages.append(final)
    return final.content


# ===== 多轮对话 =====
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools_map = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

# 对话历史在多轮之间共享
messages = []

# 第 1 轮
print("=" * 50)
print("[用户] 公司差旅报销标准是什么？")
messages.append(HumanMessage(content="公司差旅报销标准是什么？"))
answer1 = agent_step(messages, model_with_tools, tools_map)
print(f"[助手] {answer1}")

# 第 2 轮（基于上一轮的上下文追问）
print("\n" + "=" * 50)
print("[用户] 如果出差5天，餐补总共多少钱？")
messages.append(HumanMessage(content="如果出差5天，餐补总共多少钱？"))
answer2 = agent_step(messages, model_with_tools, tools_map)
print(f"[助手] {answer2}")

# 第 3 轮（继续追问，不需要工具）
print("\n" + "=" * 50)
print("[用户] 谢谢，还有其他注意事项吗？")
messages.append(HumanMessage(content="谢谢，还有其他注意事项吗？"))
answer3 = agent_step(messages, model_with_tools, tools_map)
print(f"[助手] {answer3}")

# 查看完整对话历史
print(f"\n对话历史共 {len(messages)} 条消息")
for i, msg in enumerate(messages):
    print(f"  [{i}] {type(msg).__name__}: {str(msg.content)[:60]}...")
```

### 预期输出

```
==================================================
[用户] 公司差旅报销标准是什么？
  [工具] search_knowledge_base({'query': '差旅报销标准'}) -> [知识库] 差旅报销标准: 火车二等座、经济型酒店，餐补每天100元。
[助手] 公司差旅报销标准如下：交通选择火车二等座，住宿选择经济型酒店，餐补每天100元。

==================================================
[用户] 如果出差5天，餐补总共多少钱？
  [工具] calculate({'expression': '100 * 5'}) -> [计算] 100 * 5 = 500
[助手] 出差5天的餐补总共是 500 元（每天100元 × 5天）。

==================================================
[用户] 谢谢，还有其他注意事项吗？
[助手] 建议注意以下几点：1. 保留所有原始发票和票据；2. 出差前提前在系统中提交申请；3. 超出标准的费用需要提前审批。

对话历史共 9 条消息
  [0] HumanMessage: 公司差旅报销标准是什么？...
  [1] AIMessage: ...
  [2] ToolMessage: [知识库] 差旅报销标准: 火车二等座、经济型酒店，餐补每天100元。...
  [3] AIMessage: 公司差旅报销标准如下：交通选择火车二等座，住宿选择经济型酒店，餐补每天...
  [4] HumanMessage: 如果出差5天，餐补总共多少钱？...
  [5] AIMessage: ...
  [6] ToolMessage: [计算] 100 * 5 = 500...
  [7] AIMessage: 出差5天的餐补总共是 500 元（每天100元 × 5天）。...
  [8] HumanMessage: 谢谢，还有其他注意事项吗？...
```

> **多轮对话的关键：** `messages` 列表在多轮之间持续累积。第 2 轮问"餐补总共多少钱"时，模型能从第 1 轮的工具结果中知道"餐补每天100元"，所以它调用 `calculate("100 * 5")` 而不是先查知识库。这就是上下文的力量。

---

## 示例5：RAG 场景——检索工具 + 生成

**解决的问题：** 在 RAG 系统中，检索器本身就是一个工具。本示例展示如何把"文档检索"包装成 Tool，让 Agent 自主决定何时检索、检索什么，然后基于检索结果生成回答。

```python
"""
RAG 场景：把检索器包装成 Tool，Agent 自主决定何时检索
模式：retriever-as-tool，Agent 在需要时调用检索工具获取上下文
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# ===== 模拟文档知识库 =====
DOCUMENTS = [
    {
        "id": 1,
        "title": "RAG 架构概述",
        "content": "RAG（检索增强生成）由三个核心阶段组成：索引、检索、生成。"
                   "索引阶段将文档分块并向量化存入向量数据库；"
                   "检索阶段根据用户查询找到最相关的文档片段；"
                   "生成阶段将检索结果作为上下文注入 Prompt，由 LLM 生成回答。",
        "tags": ["RAG", "架构", "基础"],
    },
    {
        "id": 2,
        "title": "Chunking 分块策略",
        "content": "文本分块是 RAG 的关键步骤。常见策略包括：固定大小分块（按字符数切分）、"
                   "语义分块（按段落或语义边界切分）、递归分块（先按大分隔符切，再按小分隔符细分）。"
                   "推荐 chunk_size=500，overlap=50 作为起始参数。",
        "tags": ["Chunking", "分块", "优化"],
    },
    {
        "id": 3,
        "title": "向量检索优化",
        "content": "提升检索质量的方法：1. 使用混合检索（向量+关键词）；"
                   "2. 添加 ReRank 重排序；3. 优化 Embedding 模型选择；"
                   "4. 调整 top_k 参数（通常 3-5 效果最佳）；"
                   "5. 使用元数据过滤缩小检索范围。",
        "tags": ["检索", "优化", "向量"],
    },
    {
        "id": 4,
        "title": "Prompt Engineering for RAG",
        "content": "RAG 场景的 Prompt 设计要点：1. 明确指示模型基于提供的上下文回答；"
                   "2. 要求模型在无法从上下文找到答案时明确说明；"
                   "3. 使用分隔符区分上下文和问题；"
                   "4. 控制回答的格式和长度。",
        "tags": ["Prompt", "生成", "优化"],
    },
]


def simple_retriever(query: str, top_k: int = 2) -> list[dict]:
    """简单的关键词检索器（生产环境应替换为向量检索）"""
    scored = []
    query_lower = query.lower()
    for doc in DOCUMENTS:
        score = 0
        # 标题匹配权重高
        for word in query_lower.split():
            if word in doc["title"].lower():
                score += 3
            if word in doc["content"].lower():
                score += 1
            if word in " ".join(doc["tags"]).lower():
                score += 2
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ===== 把检索器包装成 Tool =====
@tool
def rag_retrieve(query: str) -> str:
    """检索 RAG 知识库中的相关文档。当用户询问 RAG、检索、分块、向量等技术问题时使用。"""
    results = simple_retriever(query, top_k=2)
    if not results:
        return "未找到相关文档。"

    output = []
    for i, doc in enumerate(results, 1):
        output.append(f"[文档{i}] {doc['title']}\n{doc['content']}")
    return "\n\n".join(output)


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。"""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


# ===== RAG Agent =====
def rag_agent(question: str, max_steps: int = 3) -> str:
    """RAG Agent：自主决定是否需要检索，基于检索结果生成回答"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [rag_retrieve, calculate]
    tools_map = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    system_prompt = (
        "你是一个 RAG 技术专家助手。回答问题时：\n"
        "1. 如果问题涉及 RAG 技术知识，先使用 rag_retrieve 工具检索相关文档\n"
        "2. 基于检索到的文档内容回答，不要编造文档中没有的信息\n"
        "3. 如果检索结果不足以回答问题，明确告知用户\n"
        "4. 回答要简洁、结构化"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    print(f"[问题] {question}")

    for step in range(1, max_steps + 1):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"[回答] {response.content}")
            return response.content

        for tc in response.tool_calls:
            result = tools_map[tc["name"]].invoke(tc["args"])
            print(f"  [检索] {tc['name']}({tc['args']}) -> {result[:80]}...")
            messages.append(ToolMessage(
                content=str(result), tool_call_id=tc["id"]
            ))

    return "未能生成回答。"


# ===== 测试 RAG Agent =====
print("=" * 60)
rag_agent("RAG 系统的核心架构是什么？")

print("\n" + "=" * 60)
rag_agent("如何优化 RAG 的检索质量？")

print("\n" + "=" * 60)
rag_agent("推荐的 chunk_size 是多少？为什么？")

print("\n" + "=" * 60)
rag_agent("你好，今天天气怎么样？")  # 不需要检索的问题
```

### 预期输出

```
============================================================
[问题] RAG 系统的核心架构是什么？
  [检索] rag_retrieve({'query': 'RAG 核心架构'}) -> [文档1] RAG 架构概述
RAG（检索增强生成）由三个核心阶段组成：索引、检索、生成...
[回答] RAG 系统的核心架构由三个阶段组成：
1. **索引阶段**：将文档分块并向量化，存入向量数据库
2. **检索阶段**：根据用户查询找到最相关的文档片段
3. **生成阶段**：将检索结果作为上下文注入 Prompt，由 LLM 生成回答

============================================================
[问题] 如何优化 RAG 的检索质量？
  [检索] rag_retrieve({'query': '检索质量优化'}) -> [文档1] 向量检索优化
提升检索质量的方法：1. 使用混合检索（向量+关键词）...
[回答] 优化 RAG 检索质量的方法包括：
1. 使用混合检索（向量检索 + 关键词检索）
2. 添加 ReRank 重排序
3. 优化 Embedding 模型选择
4. 调整 top_k 参数（推荐 3-5）
5. 使用元数据过滤缩小检索范围

============================================================
[问题] 推荐的 chunk_size 是多少？为什么？
  [检索] rag_retrieve({'query': 'chunk_size 推荐'}) -> [文档1] Chunking 分块策略
文本分块是 RAG 的关键步骤...
[回答] 推荐的起始参数是 chunk_size=500，overlap=50。这是因为 500 字符的分块大小能在保留足够上下文信息和检索精度之间取得平衡，而 50 字符的重叠确保分块边界处的信息不会丢失。

============================================================
[问题] 你好，今天天气怎么样？
[回答] 你好！我是 RAG 技术专家助手，主要回答 RAG 相关的技术问题。天气查询不在我的能力范围内，建议使用天气应用查看。有 RAG 相关问题随时问我！
```

> **retriever-as-tool 模式的优势：** Agent 自主判断是否需要检索。对于"你好"这样的闲聊，它直接回答而不浪费检索资源。对于技术问题，它先检索再回答，确保回答有据可依。这比"每次都检索"的 Naive RAG 更智能、更省资源。

---

## 关键要点总结

1. **工具描述决定一切**：LLM 根据 `description` 选择工具。描述要写清楚"适用于什么场景"，而不只是"这个工具做什么"。
2. **关键词选择是起点**：简单、零成本、可解释，但覆盖率有限。适合工具少、场景明确的情况。
3. **Agent 循环是核心模式**：`bind_tools() → 检测 tool_calls → 执行工具 → ToolMessage 回传 → 循环`，所有 Agent 框架的底层都是这个循环。
4. **多轮对话靠 messages 累积**：对话历史中包含之前的工具调用和结果，模型能利用这些上下文做出更智能的决策。
5. **retriever-as-tool 是 RAG 的高级模式**：让 Agent 自主决定何时检索，比"每次都检索"更灵活、更省资源。
6. **max_steps 是安全阀**：防止 Agent 陷入无限循环，生产环境建议设为 5-10。

### 四个示例的递进关系

```
示例1: 定义工具集        → 准备"菜单"
示例2: 关键词工具选择    → 根据点菜内容筛选菜单
示例3: Agent 循环        → 服务员接单、下单、上菜的完整流程
示例4: 多轮对话          → 多次点菜，服务员记住之前点了什么
示例5: RAG 检索工具      → 服务员先查菜谱再推荐菜品
```

---

**上一篇**: [07_实战代码_场景3_自定义Schema与参数验证.md](./07_实战代码_场景3_自定义Schema与参数验证.md) — Pydantic 模型与 docstring 解析的参数验证实战
**下一篇**: [08_面试必问.md](./08_面试必问.md) — Tools 与函数调用的高频面试题
