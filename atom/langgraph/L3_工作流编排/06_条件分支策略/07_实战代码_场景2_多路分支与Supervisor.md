# 实战代码 - 场景2：多路分支与 Supervisor

> 本文档演示 Supervisor 路由模式：中央调度节点根据用户意图分发到不同专家代理，并扩展优先级路由与默认回退

---

## 场景描述

**场景**：构建一个智能客服系统，Supervisor（调度员）分析用户消息的意图，将请求路由到对应的专家代理处理。

**特点**：
- Supervisor 模式：一个中央分类节点 + 多个专家节点
- 多路分支：4 条路由路径（账单 / 技术 / 通用 / VIP）
- 优先级路由：VIP 用户无论什么意图都走专属通道
- 默认回退：无法识别的意图路由到通用代理
- 不依赖外部 API，可直接运行

**流程图**：

```
START
  ↓
supervisor（意图分析 + 优先级判断）
  ↓
route_with_priority（路由决策）
  ├── VIP 用户     → vip_agent（VIP 专属服务）→ END
  ├── "billing"    → billing_agent（账单代理）→ END
  ├── "technical"  → technical_agent（技术代理）→ END
  └── "general"    → general_agent（通用代理）→ END
```

---

## 完整代码

```python
"""
条件分支策略 - 实战场景2：Supervisor 多路分支
演示：客服系统意图路由 + 优先级路由 + 默认回退

Supervisor 模式的核心思想：
  一个"调度员"节点负责分析和分类，
  多个"专家"节点各自处理特定类型的请求，
  路由函数根据分类结果（+ 优先级）选择目标专家。
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ============================================================
# 第一部分：状态定义
# ============================================================

class CustomerState(TypedDict):
    """客服系统状态

    字段说明：
    - message: 用户输入的消息
    - user_type: 用户类型（"vip" 或 "normal"）
    - intent: Supervisor 分析出的意图
    - response: 专家代理生成的回复
    - routed_to: 实际路由到的代理名称（用于追踪）
    - log: 处理日志
    """
    message: str
    user_type: str       # "vip" | "normal"
    intent: str          # "billing" | "technical" | "general"
    response: str
    routed_to: str
    log: list[str]


# ============================================================
# 第二部分：Supervisor 节点
# ============================================================

def supervisor(state: CustomerState) -> dict:
    """Supervisor 节点：分析用户意图

    这是整个系统的"大脑"——负责理解用户想要什么。
    实际项目中这里会调用 LLM 做意图识别，
    这里用关键词匹配模拟。
    """
    msg = state["message"].lower()

    # 意图识别规则
    billing_keywords = ["账单", "付款", "退款", "费用", "扣费", "充值", "余额"]
    technical_keywords = ["故障", "报错", "崩溃", "无法", "失败", "bug", "错误", "卡顿"]

    if any(w in msg for w in billing_keywords):
        intent = "billing"
    elif any(w in msg for w in technical_keywords):
        intent = "technical"
    else:
        intent = "general"

    print(f"[Supervisor] 消息: {state['message']}")
    print(f"[Supervisor] 用户类型: {state['user_type']}")
    print(f"[Supervisor] 识别意图: {intent}")

    return {
        "intent": intent,
        "log": state.get("log", []) + [
            f"Supervisor 分析完成: intent={intent}, user_type={state['user_type']}"
        ],
    }


# ============================================================
# 第三部分：专家代理节点
# ============================================================

def billing_agent(state: CustomerState) -> dict:
    """账单代理：处理账单相关问题"""
    print(f"[账单代理] 处理中...")
    response = (
        f"[账单代理] 您好，关于您的问题「{state['message']}」：\n"
        f"  我已查询您的账户信息，正在为您处理账单相关事宜。\n"
        f"  如需进一步帮助，请提供订单号。"
    )
    return {
        "response": response,
        "routed_to": "billing_agent",
        "log": state.get("log", []) + ["账单代理处理完成"],
    }


def technical_agent(state: CustomerState) -> dict:
    """技术代理：处理技术故障问题"""
    print(f"[技术代理] 处理中...")
    response = (
        f"[技术代理] 您好，关于您的问题「{state['message']}」：\n"
        f"  我已记录您的技术问题，正在排查故障原因。\n"
        f"  建议您先尝试：1) 清除缓存 2) 重启应用 3) 检查网络连接。"
    )
    return {
        "response": response,
        "routed_to": "technical_agent",
        "log": state.get("log", []) + ["技术代理处理完成"],
    }


def general_agent(state: CustomerState) -> dict:
    """通用代理：处理一般性问题（默认回退）"""
    print(f"[通用代理] 处理中...")
    response = (
        f"[通用代理] 您好，关于您的问题「{state['message']}」：\n"
        f"  感谢您的咨询，我会尽力为您解答。\n"
        f"  如果问题较复杂，我会为您转接专业客服。"
    )
    return {
        "response": response,
        "routed_to": "general_agent",
        "log": state.get("log", []) + ["通用代理处理完成"],
    }


def vip_agent(state: CustomerState) -> dict:
    """VIP 代理：VIP 用户专属服务"""
    print(f"[VIP 代理] 处理中...")
    response = (
        f"[VIP 代理] 尊敬的 VIP 用户，关于您的问题「{state['message']}」：\n"
        f"  您的问题已被标记为最高优先级。\n"
        f"  专属客服经理将在 5 分钟内与您联系。\n"
        f"  （原始意图: {state['intent']}，已升级为 VIP 通道）"
    )
    return {
        "response": response,
        "routed_to": "vip_agent",
        "log": state.get("log", []) + ["VIP 代理处理完成（优先级路由）"],
    }


# ============================================================
# 第四部分：路由函数
# ============================================================

def route_with_priority(
    state: CustomerState,
) -> Literal["vip_agent", "billing_agent", "technical_agent", "general_agent"]:
    """优先级路由函数

    路由逻辑：
    1. VIP 用户 → 无论什么意图，都走 VIP 专属通道
    2. 普通用户 → 根据意图路由到对应专家
    3. 未知意图 → 回退到通用代理（默认路径）

    这是 Supervisor 模式的核心：
    先检查优先级，再检查意图，最后兜底。
    """
    # 优先级1：VIP 用户走专属通道
    if state["user_type"] == "vip":
        print(f"[路由] VIP 用户 → vip_agent")
        return "vip_agent"

    # 优先级2：根据意图路由
    intent = state["intent"]
    intent_to_agent = {
        "billing": "billing_agent",
        "technical": "technical_agent",
        "general": "general_agent",
    }

    target = intent_to_agent.get(intent, "general_agent")  # 默认回退
    print(f"[路由] 意图={intent} → {target}")
    return target


# ============================================================
# 第五部分：构建图
# ============================================================

def build_supervisor_graph() -> object:
    """构建 Supervisor 路由图"""
    builder = StateGraph(CustomerState)

    # 添加节点：1 个 Supervisor + 4 个专家
    builder.add_node("supervisor", supervisor)
    builder.add_node("billing_agent", billing_agent)
    builder.add_node("technical_agent", technical_agent)
    builder.add_node("general_agent", general_agent)
    builder.add_node("vip_agent", vip_agent)

    # 入口边
    builder.add_edge(START, "supervisor")

    # 条件边：Supervisor → 专家代理
    # 使用 Literal 注解，LangGraph 自动推断 4 条路径
    builder.add_conditional_edges("supervisor", route_with_priority)

    # 所有专家处理完后 → END
    builder.add_edge("billing_agent", END)
    builder.add_edge("technical_agent", END)
    builder.add_edge("general_agent", END)
    builder.add_edge("vip_agent", END)

    return builder.compile()


# ============================================================
# 第六部分：测试
# ============================================================

def run_tests():
    """测试多种场景"""
    graph = build_supervisor_graph()

    test_cases = [
        # (消息, 用户类型, 预期路由)
        ("我的账单有问题，上个月多扣了费用", "normal", "billing_agent"),
        ("应用一直报错，无法正常使用", "normal", "technical_agent"),
        ("你们的营业时间是什么时候？", "normal", "general_agent"),
        ("我的账单有问题", "vip", "vip_agent"),          # VIP 优先级
        ("应用报错了", "vip", "vip_agent"),               # VIP 优先级
        ("随便问个问题", "normal", "general_agent"),       # 默认回退
    ]

    for i, (message, user_type, expected) in enumerate(test_cases, 1):
        print(f"{'=' * 60}")
        print(f"测试 {i}: {message} (用户类型: {user_type})")
        print(f"{'=' * 60}")

        result = graph.invoke({
            "message": message,
            "user_type": user_type,
            "intent": "",
            "response": "",
            "routed_to": "",
            "log": [],
        })

        actual = result["routed_to"]
        status = "PASS" if actual == expected else "FAIL"

        print(f"  路由到: {actual} (预期: {expected}) [{status}]")
        print(f"  回复: {result['response'][:60]}...")
        print(f"  日志: {result['log']}")
        print()


if __name__ == "__main__":
    run_tests()
```

---

## 运行输出

```
============================================================
测试 1: 我的账单有问题，上个月多扣了费用 (用户类型: normal)
============================================================
[Supervisor] 消息: 我的账单有问题，上个月多扣了费用
[Supervisor] 用户类型: normal
[Supervisor] 识别意图: billing
[路由] 意图=billing → billing_agent
[账单代理] 处理中...
  路由到: billing_agent (预期: billing_agent) [PASS]
  回复: [账单代理] 您好，关于您的问题「我的账单有问题，上个月多扣了费用」：...
  日志: ['Supervisor 分析完成: intent=billing, user_type=normal', '账单代理处理完成']

============================================================
测试 2: 应用一直报错，无法正常使用 (用户类型: normal)
============================================================
[Supervisor] 消息: 应用一直报错，无法正常使用
[Supervisor] 用户类型: normal
[Supervisor] 识别意图: technical
[路由] 意图=technical → technical_agent
[技术代理] 处理中...
  路由到: technical_agent (预期: technical_agent) [PASS]
  回复: [技术代理] 您好，关于您的问题「应用一直报错，无法正常使用」：...
  日志: ['Supervisor 分析完成: intent=technical, user_type=normal', '技术代理处理完成']

============================================================
测试 3: 你们的营业时间是什么时候？ (用户类型: normal)
============================================================
[Supervisor] 消息: 你们的营业时间是什么时候？
[Supervisor] 用户类型: normal
[Supervisor] 识别意图: general
[路由] 意图=general → general_agent
[通用代理] 处理中...
  路由到: general_agent (预期: general_agent) [PASS]
  回复: [通用代理] 您好，关于您的问题「你们的营业时间是什么时候？」：...
  日志: ['Supervisor 分析完成: intent=general, user_type=normal', '通用代理处理完成']

============================================================
测试 4: 我的账单有问题 (用户类型: vip)
============================================================
[Supervisor] 消息: 我的账单有问题
[Supervisor] 用户类型: vip
[Supervisor] 识别意图: billing
[路由] VIP 用户 → vip_agent
[VIP 代理] 处理中...
  路由到: vip_agent (预期: vip_agent) [PASS]
  回复: [VIP 代理] 尊敬的 VIP 用户，关于您的问题「我的账单有问题」：...
  日志: ['Supervisor 分析完成: intent=billing, user_type=vip', 'VIP 代理处理完成（优先级路由）']

============================================================
测试 5: 应用报错了 (用户类型: vip)
============================================================
[Supervisor] 消息: 应用报错了
[Supervisor] 用户类型: vip
[Supervisor] 识别意图: technical
[路由] VIP 用户 → vip_agent
[VIP 代理] 处理中...
  路由到: vip_agent (预期: vip_agent) [PASS]
  回复: [VIP 代理] 尊敬的 VIP 用户，关于您的问题「应用报错了」：...
  日志: ['Supervisor 分析完成: intent=technical, user_type=vip', 'VIP 代理处理完成（优先级路由）']

============================================================
测试 6: 随便问个问题 (用户类型: normal)
============================================================
[Supervisor] 消息: 随便问个问题
[Supervisor] 用户类型: normal
[Supervisor] 识别意图: general
[路由] 意图=general → general_agent
[通用代理] 处理中...
  路由到: general_agent (预期: general_agent) [PASS]
  回复: [通用代理] 您好，关于您的问题「随便问个问题」：...
  日志: ['Supervisor 分析完成: intent=general, user_type=normal', '通用代理处理完成']
```

---

## 代码详解

### 1. Supervisor 模式的核心结构

```
┌─────────────────────────────────────────────┐
│              Supervisor 模式                  │
│                                              │
│   ┌──────────┐                               │
│   │Supervisor│ ← 中央调度（分类 + 优先级）     │
│   └────┬─────┘                               │
│        │ add_conditional_edges               │
│   ┌────┼────────┼────────┼────┐              │
│   ▼    ▼        ▼        ▼    │              │
│  VIP  账单     技术     通用   │              │
│  代理  代理     代理     代理   │ ← 专家节点   │
│   │    │        │        │    │              │
│   └────┼────────┼────────┼────┘              │
│        ▼                                     │
│       END                                    │
└─────────────────────────────────────────────┘
```

**Supervisor 模式的三个角色**：
1. **Supervisor（调度员）**：分析意图，写入状态
2. **路由函数**：读取状态，选择目标专家
3. **专家代理**：处理具体请求，生成回复

### 2. 优先级路由的实现

```python
def route_with_priority(state) -> Literal["vip_agent", "billing_agent", ...]:
    # 优先级1：VIP 用户 → 无条件走 VIP 通道
    if state["user_type"] == "vip":
        return "vip_agent"

    # 优先级2：根据意图路由
    return intent_to_agent.get(state["intent"], "general_agent")
    #                                           ↑ 默认回退
```

**关键设计**：
- 优先级检查在意图路由之前
- VIP 用户的意图仍然会被 Supervisor 识别（写入 `intent` 字段），但路由时被优先级覆盖
- `dict.get(key, default)` 实现默认回退，未知意图自动路由到通用代理

### 3. 默认回退路径

```python
# 方式1：在路由函数中用 dict.get 兜底
target = intent_to_agent.get(intent, "general_agent")

# 方式2：在 Supervisor 中确保 intent 只有已知值
if intent not in ("billing", "technical"):
    intent = "general"  # 未知意图归入 general
```

两种方式都能实现回退，推荐同时使用（双重保险）。

### 4. path_map 替代写法

如果想用 path_map 实现同样的效果：

```python
def route_with_priority(state) -> str:
    """返回语义化的键"""
    if state["user_type"] == "vip":
        return "vip"
    return state["intent"]  # "billing" / "technical" / "general"

builder.add_conditional_edges(
    "supervisor",
    route_with_priority,
    {
        "vip": "vip_agent",
        "billing": "billing_agent",
        "technical": "technical_agent",
        "general": "general_agent",
    },
)
```

这种写法的优势：路由函数不需要知道实际节点名，只返回语义化的键。

---

## 扩展：添加二级路由

在实际系统中，专家代理处理完后可能还需要进一步路由（比如技术问题需要升级到高级工程师）：

```python
class CustomerStateV2(TypedDict):
    message: str
    user_type: str
    intent: str
    severity: str       # "low" | "medium" | "high"
    response: str
    routed_to: str
    log: list[str]


def technical_agent_v2(state: CustomerStateV2) -> dict:
    """技术代理 V2：增加严重程度评估"""
    msg = state["message"].lower()

    # 评估严重程度
    if any(w in msg for w in ["崩溃", "数据丢失", "无法启动"]):
        severity = "high"
    elif any(w in msg for w in ["报错", "失败"]):
        severity = "medium"
    else:
        severity = "low"

    return {
        "severity": severity,
        "routed_to": "technical_agent",
        "log": state.get("log", []) + [f"技术代理评估: severity={severity}"],
    }


def route_severity(
    state: CustomerStateV2,
) -> Literal["auto_resolve", "engineer", "senior_engineer"]:
    """二级路由：根据严重程度分配"""
    severity = state["severity"]
    if severity == "low":
        return "auto_resolve"
    elif severity == "medium":
        return "engineer"
    else:
        return "senior_engineer"


def auto_resolve(state: CustomerStateV2) -> dict:
    return {"response": "[自动解决] 已为您自动处理该问题。"}


def engineer(state: CustomerStateV2) -> dict:
    return {"response": "[工程师] 已分配工程师处理，预计 2 小时内回复。"}


def senior_engineer(state: CustomerStateV2) -> dict:
    return {"response": "[高级工程师] 已升级为紧急工单，高级工程师将立即介入。"}


def build_two_level_graph() -> object:
    """构建二级路由图"""
    builder = StateGraph(CustomerStateV2)

    builder.add_node("supervisor", supervisor)
    builder.add_node("technical_agent", technical_agent_v2)
    builder.add_node("auto_resolve", auto_resolve)
    builder.add_node("engineer", engineer)
    builder.add_node("senior_engineer", senior_engineer)

    builder.add_edge(START, "supervisor")

    # 一级路由（简化：只演示技术路径）
    builder.add_conditional_edges(
        "supervisor",
        lambda s: "technical_agent",  # 简化：直接路由到技术代理
        {"technical_agent": "technical_agent"},
    )

    # 二级路由：技术代理 → 根据严重程度分配
    builder.add_conditional_edges("technical_agent", route_severity)

    builder.add_edge("auto_resolve", END)
    builder.add_edge("engineer", END)
    builder.add_edge("senior_engineer", END)

    return builder.compile()
```

**二级路由的流程**：

```
START → supervisor → technical_agent → route_severity
                                         ├── auto_resolve    → END
                                         ├── engineer        → END
                                         └── senior_engineer → END
```

---

## 最佳实践

### 1. Supervisor 节点的职责边界

```python
# ✅ Supervisor 只负责"分析和分类"
def supervisor(state):
    intent = classify(state["message"])
    return {"intent": intent}  # 只写分类结果

# ❌ Supervisor 不应该直接处理请求
def supervisor(state):
    intent = classify(state["message"])
    if intent == "billing":
        response = handle_billing(state)  # 不要在这里处理！
    return {"intent": intent, "response": response}
```

### 2. 路由函数的优先级设计

```python
# ✅ 优先级从高到低，清晰的 if-elif-else
def route(state) -> Literal["emergency", "vip", "billing", "general"]:
    if state["is_emergency"]:     # 最高优先级
        return "emergency"
    if state["user_type"] == "vip":  # 次高优先级
        return "vip"
    return intent_map.get(state["intent"], "general")  # 正常路由 + 兜底
```

### 3. 专家节点的统一接口

```python
# ✅ 所有专家节点返回相同的字段结构
def billing_agent(state) -> dict:
    return {"response": "...", "routed_to": "billing_agent", "log": [...]}

def technical_agent(state) -> dict:
    return {"response": "...", "routed_to": "technical_agent", "log": [...]}

# 统一接口让下游处理更简单
```

---

## 与 RAG 的关联

Supervisor 模式在 RAG 系统中的典型应用：

| 场景 | Supervisor 职责 | 专家代理 |
|------|----------------|---------|
| 多知识库 RAG | 判断问题属于哪个领域 | 技术文档检索 / FAQ 检索 / 通用检索 |
| Adaptive RAG | 判断问题复杂度 | 简单检索 / 多步推理 / 网络搜索 |
| 多模态 RAG | 判断输入类型 | 文本检索 / 图片检索 / 表格检索 |
| 多 Agent 协作 | 分配任务给不同 Agent | 研究 Agent / 写作 Agent / 审核 Agent |

**核心思路**：Supervisor 是 RAG 系统的"路由器"，决定每个请求走哪条检索-生成管道。

---

## 总结

Supervisor 模式是多路分支的经典实现：一个中央调度节点负责分类，多个专家节点各司其职。优先级路由通过在路由函数中按优先级排列条件实现（VIP > 意图 > 默认回退）。`dict.get(key, default)` 是实现默认回退的最简方式。当业务复杂度增加时，可以叠加二级路由形成层级调度。

---

**下一步**：
- 场景3：Send 并行 Map-Reduce
- 场景4：Command 动态路由
- 场景5：错误处理与回退
