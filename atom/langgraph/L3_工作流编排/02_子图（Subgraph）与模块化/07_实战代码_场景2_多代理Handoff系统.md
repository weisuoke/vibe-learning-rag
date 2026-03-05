# 实战代码 - 场景2：多代理 Handoff 系统

> 使用 Command.PARENT 构建客服系统，实现代理间的智能切换与状态传递

---

## 场景概述

**目标**：构建一个多代理客服系统，用户消息由通用代理接待，根据意图自动切换到专业代理处理。

**系统架构**：
```
用户消息 → 通用代理（意图判断）
              ├─ 技术问题 → 技术支持代理 → 返回结果
              └─ 退款请求 → 退款代理 → 返回结果
```

**核心技术**：
1. 每个代理是一个独立的子图
2. 使用 `Command(goto="target", graph=Command.PARENT)` 实现 Handoff
3. 状态在代理间通过父图传递

**适用场景**：
- 客服系统（多部门路由）
- 多代理协作系统
- RAG 系统中的多策略检索

---

## 完整代码示例

```python
"""
多代理 Handoff 系统 - 客服场景
演示：使用 Command.PARENT 实现代理间切换

核心机制：
- 每个代理是独立子图，有自己的内部状态
- 子图通过 Command(graph=Command.PARENT) 向父图发送导航指令
- 父图根据 goto 参数将控制权转交给目标代理

来源：
- Context7 官方文档：Command.PARENT 跨图导航
- 源码分析：types.py Command 类
"""

from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator


# ================================================================
# 第1部分：定义状态结构
# ================================================================

print("=" * 60)
print("多代理 Handoff 客服系统")
print("=" * 60)

print("\n=== 第1部分：定义状态结构 ===")


class AgentState(TypedDict):
    """
    父图状态：所有代理共享的全局状态

    关键设计：
    - messages 使用 Annotated + operator.add 作为 reducer
      这样子图通过 Command.PARENT 更新时能正确追加消息
    - current_agent 记录当前活跃的代理
    - user_intent 记录识别出的用户意图
    - resolution 记录最终处理结果

    来源：Context7 文档 - Command.PARENT 要求父图有 reducer
    """
    messages: Annotated[list[str], operator.add]  # reducer: 追加消息
    current_agent: str
    user_intent: str
    resolution: str


class TechSupportState(TypedDict):
    """技术支持代理的内部状态"""
    messages: Annotated[list[str], operator.add]
    tech_category: str        # 私有：技术问题分类
    troubleshoot_steps: list  # 私有：排查步骤


class RefundState(TypedDict):
    """退款代理的内部状态"""
    messages: Annotated[list[str], operator.add]
    order_id: str          # 私有：订单号
    refund_amount: float   # 私有：退款金额
    refund_status: str     # 私有：退款状态


print("  AgentState: 父图全局状态（messages 带 reducer）")
print("  TechSupportState: 技术支持代理内部状态")
print("  RefundState: 退款代理内部状态")


# ================================================================
# 第2部分：模拟 LLM 响应（不依赖外部 API）
# ================================================================

print("\n=== 第2部分：模拟 LLM 响应函数 ===")


def mock_intent_detection(user_message: str) -> str:
    """
    模拟意图识别（替代真实 LLM 调用）

    规则：
    - 包含 "退款/退货/退钱" → refund
    - 包含 "报错/bug/崩溃/无法" → tech_support
    - 其他 → general
    """
    msg = user_message.lower()
    refund_keywords = ["退款", "退货", "退钱", "refund", "return"]
    tech_keywords = ["报错", "bug", "崩溃", "无法", "error", "crash", "broken"]

    for kw in refund_keywords:
        if kw in msg:
            return "refund"
    for kw in tech_keywords:
        if kw in msg:
            return "tech_support"
    return "general"


def mock_tech_diagnosis(message: str) -> tuple[str, list[str]]:
    """模拟技术问题诊断"""
    if "登录" in message or "login" in message.lower():
        return "账号相关", ["检查密码是否正确", "清除浏览器缓存", "尝试重置密码"]
    elif "慢" in message or "slow" in message.lower():
        return "性能相关", ["检查网络连接", "清理缓存数据", "升级到最新版本"]
    else:
        return "一般技术问题", ["重启应用", "检查系统更新", "联系高级技术支持"]


def mock_refund_process(message: str) -> tuple[str, float, str]:
    """模拟退款处理"""
    # 简单模拟：从消息中提取订单号和金额
    order_id = "ORD-20260227-001"
    amount = 99.00
    status = "approved"
    return order_id, amount, status


print("  mock_intent_detection: 意图识别")
print("  mock_tech_diagnosis: 技术诊断")
print("  mock_refund_process: 退款处理")


# ================================================================
# 第3部分：构建通用代理子图
# ================================================================

print("\n=== 第3部分：构建通用代理子图 ===")


class GeneralAgentState(TypedDict):
    """通用代理内部状态"""
    messages: Annotated[list[str], operator.add]
    detected_intent: str


def general_greet(state: GeneralAgentState) -> dict:
    """通用代理：打招呼并识别意图"""
    messages = state["messages"]
    last_message = messages[-1] if messages else ""

    print(f"  [通用代理] 接收消息: '{last_message}'")

    # 识别意图
    intent = mock_intent_detection(last_message)
    print(f"  [通用代理] 识别意图: {intent}")

    greeting = f"[通用代理] 您好！我已收到您的问题，正在为您转接专业客服..."
    return {
        "messages": [greeting],
        "detected_intent": intent,
    }


def general_route(state: GeneralAgentState) -> Command:
    """
    通用代理：根据意图路由到对应的专业代理

    关键点：
    - 返回 Command 对象而非普通 dict
    - graph=Command.PARENT 表示命令发给父图
    - goto 指定父图中的目标节点（即目标代理子图）
    - update 更新父图的状态

    来源：Context7 文档 - Command.PARENT 跨图导航
    """
    intent = state["detected_intent"]
    print(f"  [通用代理] 准备 Handoff，目标意图: {intent}")

    if intent == "tech_support":
        return Command(
            update={
                "messages": [f"[系统] 已转接至技术支持代理"],
                "current_agent": "tech_support",
                "user_intent": "tech_support",
            },
            goto="tech_support_agent",
            graph=Command.PARENT,
        )
    elif intent == "refund":
        return Command(
            update={
                "messages": [f"[系统] 已转接至退款代理"],
                "current_agent": "refund",
                "user_intent": "refund",
            },
            goto="refund_agent",
            graph=Command.PARENT,
        )
    else:
        # 一般问题，通用代理直接处理
        return Command(
            update={
                "messages": ["[通用代理] 这是一个常见问题，让我直接为您解答..."],
                "current_agent": "general",
                "user_intent": "general",
                "resolution": "已由通用代理解答",
            },
            goto="final_response",
            graph=Command.PARENT,
        )


# 构建通用代理子图
general_builder = StateGraph(GeneralAgentState)
general_builder.add_node("greet", general_greet)
general_builder.add_node("route", general_route)
general_builder.add_edge(START, "greet")
general_builder.add_edge("greet", "route")
# 注意：route 节点返回 Command.PARENT，不需要连接到 END
# LangGraph 会在 Command 发送后自动结束子图

general_agent = general_builder.compile()

print("  通用代理子图: START -> greet -> route -> Command.PARENT")
print("  编译完成")


# ================================================================
# 第4部分：构建技术支持代理子图
# ================================================================

print("\n=== 第4部分：构建技术支持代理子图 ===")


def tech_diagnose(state: TechSupportState) -> dict:
    """技术支持代理：诊断问题"""
    messages = state["messages"]
    # 找到用户的原始消息（第一条）
    user_msg = messages[0] if messages else ""

    print(f"  [技术代理] 开始诊断问题...")

    category, steps = mock_tech_diagnosis(user_msg)

    print(f"  [技术代理] 问题分类: {category}")
    print(f"  [技术代理] 排查步骤: {steps}")

    return {
        "tech_category": category,
        "troubleshoot_steps": steps,
        "messages": [f"[技术代理] 问题诊断完成，分类为: {category}"],
    }


def tech_resolve(state: TechSupportState) -> Command:
    """
    技术支持代理：生成解决方案并 Handoff 回父图

    使用 Command.PARENT 将结果传回父图
    """
    category = state["tech_category"]
    steps = state["troubleshoot_steps"]

    solution = f"[技术代理] 解决方案（{category}）：\n"
    for i, step in enumerate(steps, 1):
        solution += f"  {i}. {step}\n"
    solution += "如果以上步骤无法解决，请提供更多信息。"

    print(f"  [技术代理] 生成解决方案完成")

    # Handoff 回父图
    return Command(
        update={
            "messages": [solution],
            "resolution": f"技术问题（{category}）已提供解决方案",
        },
        goto="final_response",
        graph=Command.PARENT,
    )


# 构建技术支持代理子图
tech_builder = StateGraph(TechSupportState)
tech_builder.add_node("diagnose", tech_diagnose)
tech_builder.add_node("resolve", tech_resolve)
tech_builder.add_edge(START, "diagnose")
tech_builder.add_edge("diagnose", "resolve")

tech_agent = tech_builder.compile()

print("  技术支持代理子图: START -> diagnose -> resolve -> Command.PARENT")
print("  编译完成")


# ================================================================
# 第5部分：构建退款代理子图
# ================================================================

print("\n=== 第5部分：构建退款代理子图 ===")


def refund_verify(state: RefundState) -> dict:
    """退款代理：验证退款请求"""
    messages = state["messages"]
    user_msg = messages[0] if messages else ""

    print(f"  [退款代理] 验证退款请求...")

    order_id, amount, status = mock_refund_process(user_msg)

    print(f"  [退款代理] 订单: {order_id}, 金额: {amount}, 状态: {status}")

    return {
        "order_id": order_id,
        "refund_amount": amount,
        "refund_status": status,
        "messages": [f"[退款代理] 已找到订单 {order_id}，退款金额 {amount} 元"],
    }


def refund_process(state: RefundState) -> Command:
    """
    退款代理：处理退款并 Handoff 回父图
    """
    order_id = state["order_id"]
    amount = state["refund_amount"]
    status = state["refund_status"]

    if status == "approved":
        result_msg = (
            f"[退款代理] 退款已批准！\n"
            f"  订单号: {order_id}\n"
            f"  退款金额: {amount} 元\n"
            f"  预计 3-5 个工作日到账"
        )
        resolution = f"订单 {order_id} 退款 {amount} 元已批准"
    else:
        result_msg = f"[退款代理] 抱歉，退款申请未通过审核，请联系人工客服"
        resolution = f"订单 {order_id} 退款申请被拒绝"

    print(f"  [退款代理] 处理完成: {resolution}")

    return Command(
        update={
            "messages": [result_msg],
            "resolution": resolution,
        },
        goto="final_response",
        graph=Command.PARENT,
    )


# 构建退款代理子图
refund_builder = StateGraph(RefundState)
refund_builder.add_node("verify", refund_verify)
refund_builder.add_node("process", refund_process)
refund_builder.add_edge(START, "verify")
refund_builder.add_edge("verify", "process")

refund_agent = refund_builder.compile()

print("  退款代理子图: START -> verify -> process -> Command.PARENT")
print("  编译完成")


# ================================================================
# 第6部分：构建父图（编排所有代理）
# ================================================================

print("\n=== 第6部分：构建父图 ===")


def final_response(state: AgentState) -> dict:
    """
    父图节点：生成最终响应

    当任意代理通过 Command.PARENT 路由到此节点时执行
    汇总所有消息，生成最终回复
    """
    messages = state["messages"]
    resolution = state["resolution"]
    agent = state["current_agent"]

    summary = (
        f"\n[系统总结] 处理完成\n"
        f"  处理代理: {agent}\n"
        f"  处理结果: {resolution}\n"
        f"  消息总数: {len(messages)} 条"
    )
    print(f"  [最终响应] {summary}")

    return {"messages": [summary]}


# 构建父图
parent_builder = StateGraph(AgentState)

# 添加所有代理子图和最终响应节点
parent_builder.add_node("general_agent", general_agent)
parent_builder.add_node("tech_support_agent", tech_agent)
parent_builder.add_node("refund_agent", refund_agent)
parent_builder.add_node("final_response", final_response)

# 入口：所有请求先到通用代理
parent_builder.add_edge(START, "general_agent")

# 最终响应到结束
parent_builder.add_edge("final_response", END)

# 注意：不需要从代理子图连边到 final_response
# 因为子图内部使用 Command.PARENT 直接跳转

# 编译父图
parent_graph = parent_builder.compile()

print("  父图结构:")
print("  START -> general_agent")
print("    ├─ Command.PARENT goto='tech_support_agent'")
print("    ├─ Command.PARENT goto='refund_agent'")
print("    └─ Command.PARENT goto='final_response'")
print("  *_agent -> Command.PARENT goto='final_response'")
print("  final_response -> END")
print("  编译完成")


# ================================================================
# 第7部分：测试不同场景
# ================================================================

print("\n" + "=" * 60)
print("开始测试")
print("=" * 60)


# ----- 测试1：技术支持场景 -----
print("\n" + "-" * 40)
print("测试1：技术支持场景")
print("-" * 40 + "\n")

result1 = parent_graph.invoke({
    "messages": ["我的应用登录页面一直报错，无法正常登录"],
    "current_agent": "",
    "user_intent": "",
    "resolution": "",
})

print(f"\n完整消息流:")
for i, msg in enumerate(result1["messages"]):
    print(f"  [{i}] {msg}")


# ----- 测试2：退款场景 -----
print("\n" + "-" * 40)
print("测试2：退款场景")
print("-" * 40 + "\n")

result2 = parent_graph.invoke({
    "messages": ["我想退款，上周买的产品有质量问题"],
    "current_agent": "",
    "user_intent": "",
    "resolution": "",
})

print(f"\n完整消息流:")
for i, msg in enumerate(result2["messages"]):
    print(f"  [{i}] {msg}")


# ----- 测试3：一般问题场景 -----
print("\n" + "-" * 40)
print("测试3：一般问题场景")
print("-" * 40 + "\n")

result3 = parent_graph.invoke({
    "messages": ["请问你们的营业时间是什么？"],
    "current_agent": "",
    "user_intent": "",
    "resolution": "",
})

print(f"\n完整消息流:")
for i, msg in enumerate(result3["messages"]):
    print(f"  [{i}] {msg}")


print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
多代理 Handoff 客服系统
============================================================

=== 第1部分：定义状态结构 ===
  AgentState: 父图全局状态（messages 带 reducer）
  TechSupportState: 技术支持代理内部状态
  RefundState: 退款代理内部状态

=== 第2部分：模拟 LLM 响应函数 ===
  mock_intent_detection: 意图识别
  mock_tech_diagnosis: 技术诊断
  mock_refund_process: 退款处理

=== 第3部分：构建通用代理子图 ===
  通用代理子图: START -> greet -> route -> Command.PARENT
  编译完成

=== 第4部分：构建技术支持代理子图 ===
  技术支持代理子图: START -> diagnose -> resolve -> Command.PARENT
  编译完成

=== 第5部分：构建退款代理子图 ===
  退款代理子图: START -> verify -> process -> Command.PARENT
  编译完成

=== 第6部分：构建父图 ===
  父图结构:
  START -> general_agent
    ├─ Command.PARENT goto='tech_support_agent'
    ├─ Command.PARENT goto='refund_agent'
    └─ Command.PARENT goto='final_response'
  *_agent -> Command.PARENT goto='final_response'
  final_response -> END
  编译完成

============================================================
开始测试
============================================================

----------------------------------------
测试1：技术支持场景
----------------------------------------

  [通用代理] 接收消息: '我的应用登录页面一直报错，无法正常登录'
  [通用代理] 识别意图: tech_support
  [通用代理] 准备 Handoff，目标意图: tech_support
  [技术代理] 开始诊断问题...
  [技术代理] 问题分类: 账号相关
  [技术代理] 排查步骤: ['检查密码是否正确', '清除浏览器缓存', '尝试重置密码']
  [技术代理] 生成解决方案完成
  [最终响应]
[系统总结] 处理完成
  处理代理: tech_support
  处理结果: 技术问题（账号相关）已提供解决方案
  消息总数: 5 条

完整消息流:
  [0] 我的应用登录页面一直报错，无法正常登录
  [1] [通用代理] 您好！我已收到您的问题，正在为您转接专业客服...
  [2] [系统] 已转接至技术支持代理
  [3] [技术代理] 问题诊断完成，分类为: 账号相关
  [4] [技术代理] 解决方案（账号相关）：
    1. 检查密码是否正确
    2. 清除浏览器缓存
    3. 尝试重置密码
  如果以上步骤无法解决，请提供更多信息。
  [5]
  [系统总结] 处理完成
    处理代理: tech_support
    处理结果: 技术问题（账号相关）已提供解决方案
    消息总数: 6 条

----------------------------------------
测试2：退款场景
----------------------------------------

  [通用代理] 接收消息: '我想退款，上周买的产品有质量问题'
  [通用代理] 识别意图: refund
  [通用代理] 准备 Handoff，目标意图: refund
  [退款代理] 验证退款请求...
  [退款代理] 订单: ORD-20260227-001, 金额: 99.0, 状态: approved
  [退款代理] 处理完成: 订单 ORD-20260227-001 退款 99.0 元已批准
  [最终响应]
[系统总结] 处理完成
  处理代理: refund
  处理结果: 订单 ORD-20260227-001 退款 99.0 元已批准
  消息总数: 6 条

完整消息流:
  [0] 我想退款，上周买的产品有质量问题
  [1] [通用代理] 您好！我已收到您的问题，正在为您转接专业客服...
  [2] [系统] 已转接至退款代理
  [3] [退款代理] 已找到订单 ORD-20260227-001，退款金额 99.0 元
  [4] [退款代理] 退款已批准！
    订单号: ORD-20260227-001
    退款金额: 99.0 元
    预计 3-5 个工作日到账
  [5]
  [系统总结] 处理完成
    处理代理: refund
    处理结果: 订单 ORD-20260227-001 退款 99.0 元已批准
    消息总数: 6 条

----------------------------------------
测试3：一般问题场景
----------------------------------------

  [通用代理] 接收消息: '请问你们的营业时间是什么？'
  [通用代理] 识别意图: general
  [通用代理] 准备 Handoff，目标意图: general
  [最终响应]
[系统总结] 处理完成
  处理代理: general
  处理结果: 已由通用代理解答
  消息总数: 4 条

完整消息流:
  [0] 请问你们的营业时间是什么？
  [1] [通用代理] 您好！我已收到您的问题，正在为您转接专业客服...
  [2] [通用代理] 这是一个常见问题，让我直接为您解答...
  [3]
  [系统总结] 处理完成
    处理代理: general
    处理结果: 已由通用代理解答
    消息总数: 4 条

============================================================
所有测试完成
============================================================
```

---

## 核心知识点解析

### 1. Command.PARENT 的工作机制

```
┌─────────────────────────────────────────────────────────┐
│                       父图                               │
│                                                         │
│  AgentState { messages, current_agent, resolution }     │
│                                                         │
│  ┌─────────────┐   Command.PARENT    ┌──────────────┐  │
│  │ 通用代理子图  │ ────────────────── > │ 技术支持子图   │  │
│  │             │   goto="tech_..."   │              │  │
│  └─────────────┘                     └──────────────┘  │
│        │                                    │           │
│        │ Command.PARENT                     │ Command   │
│        │ goto="final_response"              │ .PARENT   │
│        ▼                                    ▼           │
│  ┌─────────────────────────────────────────────────┐   │
│  │              final_response 节点                  │   │
│  └─────────────────────────────────────────────────┘   │
│        │                                                │
│        ▼                                                │
│       END                                               │
└─────────────────────────────────────────────────────────┘
```

**关键规则**：
- `graph=Command.PARENT` 将命令发送给最近的父图
- `goto` 指定父图中的目标节点名称
- `update` 更新父图的状态（必须匹配父图 State Schema）
- 父图中带 reducer 的字段（如 `messages`）才能被 Command.PARENT 更新

**来源：Context7 文档、源码 types.py**

### 2. 为什么 messages 需要 reducer

```python
# 没有 reducer 时：
messages: list[str]
# Command(update={"messages": ["新消息"]}) 会覆盖整个 messages

# 有 reducer 时：
messages: Annotated[list[str], operator.add]
# Command(update={"messages": ["新消息"]}) 会追加到 messages
```

**这是使用 Command.PARENT 的关键前提**：当子图通过 `Command(update={"messages": [...]}, graph=Command.PARENT)` 更新父图的 `messages` 时，如果没有 reducer，新值会直接覆盖旧值，导致历史消息丢失。使用 `operator.add` 作为 reducer 确保消息被追加。

**来源：Context7 文档 - "使用 Command.PARENT 更新共享 key 时，父图必须定义 reducer"**

### 3. Handoff 模式要点

| 要点 | 说明 |
|------|------|
| **子图独立性** | 每个代理有自己的 State，互不干扰 |
| **路由灵活性** | goto 可以指向任何父图节点（包括其他代理） |
| **状态传递** | 通过 update 将结果写入父图共享状态 |
| **无需显式边** | 子图到父图的跳转不需要 add_edge |
| **单向通信** | 子图只能向父图发送，不能向兄弟子图直接发送 |

### 4. 代理间不能直接通信

```python
# 子图 A 不能直接跳到子图 B
# 必须通过父图中转

# 正确的方式：
return Command(
    goto="agent_b",           # 父图中 agent_b 的节点名
    graph=Command.PARENT,     # 发送给父图
)

# 父图收到后，会执行 agent_b 子图
```

---

## 双重类比

### 类比1：Command.PARENT 是什么

**前端类比**：子组件向父组件 emit 事件
```javascript
// Vue 子组件
this.$emit('navigate', { target: 'tech-support', data: result });

// LangGraph 子图
return Command(goto="tech_support_agent", update={...}, graph=Command.PARENT)
```

子组件不直接操作路由，而是通知父组件"请帮我跳转到 X"。

**日常生活类比**：公司内部转接电话
- 前台（通用代理）接到客户电话
- 前台说"我帮您转接技术部"（Command.PARENT goto="tech"）
- 前台把客户信息（update）传给技术部
- 技术部处理完后说"请帮我转回总台"（Command.PARENT goto="final"）

### 类比2：多代理架构

**前端类比**：微前端架构
```
主应用（父图）
├── 子应用A：用户管理（通用代理）
├── 子应用B：技术文档（技术代理）
└── 子应用C：订单系统（退款代理）
```
每个子应用独立开发部署，通过主应用的路由进行切换。

**日常生活类比**：医院分诊系统
- 分诊台（通用代理）→ 判断病情
- 内科（技术代理）→ 处理内科问题
- 外科（退款代理）→ 处理外科问题
- 病历（共享状态）→ 各科室共享

---

## 常见问题

### Q1: Command.PARENT 的 goto 可以跳到哪些节点？

**答案**：可以跳到父图中任何已注册的节点，包括其他子图节点和普通节点。但不能跳到 START 或 END。

### Q2: 如果子图没有返回 Command.PARENT 会怎样？

**答案**：子图正常执行到 END，然后按照父图中定义的边继续执行下一个节点。Command.PARENT 是可选的，不是必须的。

### Q3: Command.PARENT 的 update 可以更新父图的所有字段吗？

**答案**：只能更新父图 State Schema 中存在的字段。如果父图字段使用了 reducer（如 `Annotated[list, operator.add]`），update 的值会通过 reducer 合并；如果没有 reducer，值会直接覆盖。

### Q4: 多层嵌套时 Command.PARENT 发给谁？

**答案**：发给最近的父图。如果需要发给更上层的祖父图，需要中间的父图做中转。

### Q5: 可以从一个代理直接切换到另一个代理吗？

**答案**：不能直接切换。必须通过 `Command(goto="other_agent", graph=Command.PARENT)` 让父图来调度。这保证了父图对控制流的统一管理。

---

## 最佳实践

### 1. 统一消息格式

```python
# 所有代理的消息带前缀，方便追踪
messages: Annotated[list[str], operator.add]

# 每条消息标注来源
f"[通用代理] 您好..."
f"[技术代理] 问题诊断..."
f"[系统] 已转接至..."
```

### 2. 代理状态最小化

```python
# 只在代理子图中定义它需要的私有状态
class TechSupportState(TypedDict):
    messages: Annotated[list[str], operator.add]  # 共享
    tech_category: str        # 仅技术代理需要
    troubleshoot_steps: list  # 仅技术代理需要
```

### 3. 确保父图有 reducer

```python
# Command.PARENT 更新的字段，父图必须定义 reducer
# 否则多次更新会互相覆盖

# 正确
messages: Annotated[list[str], operator.add]

# 错误 - 每次 Command 更新会覆盖之前的消息
messages: list[str]
```

### 4. 添加兜底路由

```python
def general_route(state: GeneralAgentState) -> Command:
    intent = state["detected_intent"]
    if intent == "tech_support":
        return Command(goto="tech_support_agent", graph=Command.PARENT)
    elif intent == "refund":
        return Command(goto="refund_agent", graph=Command.PARENT)
    else:
        # 兜底：通用代理自己处理
        return Command(goto="final_response", graph=Command.PARENT)
```

---

## 扩展阅读

### 相关知识点

1. **基础子图创建与调用**（场景1）
   - 共享状态键和状态转换两种基础集成方式
   - 子图的独立测试

2. **核心概念 - Command 跨图通信**（03_核心概念_4）
   - Command 的三个参数详解
   - 多层嵌套中的 Command 传递

3. **核心概念 - Send 动态分发**（03_核心概念_5）
   - 动态创建多个子图实例
   - MapReduce 模式

### 参考资源

**官方文档**：
- LangGraph 多代理文档：https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Command API：https://docs.langchain.com/oss/python/langgraph/types

**源码分析**：
- types.py：Command 类定义，graph 参数处理
- state.py：子图节点注册与执行

---

## 总结

**多代理 Handoff 系统的核心流程**：

1. **定义代理子图**：每个代理有独立的 State 和处理逻辑
2. **使用 Command.PARENT**：子图通过 Command 向父图发送导航和状态更新
3. **父图编排**：注册所有代理子图，定义入口
4. **Handoff 路由**：通用代理识别意图后，通过 Command.PARENT 切换到目标代理
5. **结果汇总**：专业代理处理完毕后，通过 Command.PARENT 跳到最终响应节点

**关键要点**：
- Command.PARENT 是子图到父图的通信通道
- goto 指定父图中的目标节点
- update 更新父图状态（需要 reducer）
- 代理间不能直接通信，必须通过父图中转
- 消息字段推荐使用 `Annotated[list, operator.add]`

**下一步**：
- 学习 Send 动态分发实现 MapReduce 模式
- 探索子图流式输出与调试
- 尝试多层嵌套子图

---

**文件信息**：
- 知识点：02_子图（Subgraph）与模块化
- 场景：场景2 - 多代理 Handoff 系统
- 代码行数：约 320 行
- 文档行数：约 500 行
- 最后更新：2026-02-27

**数据来源**：
- Context7 官方文档：Command.PARENT 跨图导航、多代理 Handoff
- 源码分析：types.py（Command 类）、state.py（子图执行）
- 社区实践：多代理架构模式、客服系统设计
