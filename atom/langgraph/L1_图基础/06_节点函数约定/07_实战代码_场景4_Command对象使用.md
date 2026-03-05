# 实战代码 - 场景4：Command 对象使用

> 本文档展示 Command 对象在 LangGraph 中的实战应用，包括控制流程跳转、动态路由、父图通信等核心场景。

---

## 场景概述

Command 对象是 LangGraph 中控制图执行流程的核心机制，允许节点函数动态决定下一步执行路径。

**核心能力**：
- 流程跳转控制（goto）
- 状态更新与跳转结合（update + goto）
- 动态路由决策
- 父图通信（ParentCommand）
- 多目标跳转

[来源: reference/source_节点函数约定_01.md]

---

## 场景1：基础流程跳转

### 业务场景

实现一个内容审核工作流，根据审核结果跳转到不同的处理节点。

### 完整代码

```python
"""
场景1：基础流程跳转
演示如何使用 Command 对象控制图的执行流程
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 定义状态
class ContentState(TypedDict):
    content: str
    status: str
    reason: str

# 审核节点 - 返回 Command 对象跳转
def review_content(state: ContentState) -> Command[Literal["approve", "reject"]]:
    """
    审核内容，根据结果跳转到不同节点

    返回 Command 对象指定下一个节点
    """
    content = state["content"]

    # 简单的审核逻辑
    if "违规" in content or "敏感" in content:
        print(f"❌ 审核不通过: {content}")
        # 跳转到 reject 节点
        return Command(
            goto="reject",
            update={"status": "rejected", "reason": "包含违规内容"}
        )
    else:
        print(f"✅ 审核通过: {content}")
        # 跳转到 approve 节点
        return Command(
            goto="approve",
            update={"status": "approved", "reason": "内容合规"}
        )

# 批准节点
def approve_content(state: ContentState) -> dict:
    """处理批准的内容"""
    print(f"📝 内容已发布: {state['content']}")
    return {"status": "published"}

# 拒绝节点
def reject_content(state: ContentState) -> dict:
    """处理拒绝的内容"""
    print(f"🚫 内容已拒绝: {state['reason']}")
    return {"status": "blocked"}

# 构建图
def create_review_graph():
    """创建审核工作流图"""
    builder = StateGraph(ContentState)

    # 添加节点
    builder.add_node("review", review_content)
    builder.add_node("approve", approve_content)
    builder.add_node("reject", reject_content)

    # 添加边
    builder.add_edge(START, "review")
    # Command 会自动处理从 review 到 approve/reject 的路由
    builder.add_edge("approve", END)
    builder.add_edge("reject", END)

    return builder.compile()

# 测试
if __name__ == "__main__":
    graph = create_review_graph()

    # 测试1：正常内容
    print("\n=== 测试1：正常内容 ===")
    result1 = graph.invoke({
        "content": "这是一篇正常的文章",
        "status": "pending",
        "reason": ""
    })
    print(f"最终状态: {result1['status']}\n")

    # 测试2：违规内容
    print("=== 测试2：违规内容 ===")
    result2 = graph.invoke({
        "content": "这是一篇包含违规词汇的文章",
        "status": "pending",
        "reason": ""
    })
    print(f"最终状态: {result2['status']}")
```

### 关键点解析

1. **Command 对象结构**：
   ```python
   Command(
       goto="target_node",  # 目标节点名称
       update={"key": "value"}  # 状态更新（可选）
   )
   ```
   [来源: reference/source_节点函数约定_01.md]

2. **类型注解**：使用 `Command[Literal["approve", "reject"]]` 提供类型安全

3. **自动路由**：Command 对象会自动处理节点间的跳转，无需手动配置条件边

---

## 场景2：动态路由决策

### 业务场景

实现一个智能客服路由系统，根据用户问题的复杂度动态选择处理路径。

### 完整代码

```python
"""
场景2：动态路由决策
演示如何根据运行时状态动态决定跳转目标
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 定义状态
class TicketState(TypedDict):
    question: str
    complexity: int  # 1-5 复杂度评分
    handler: str
    answer: str

# 分析节点 - 动态决定路由
def analyze_question(state: TicketState) -> Command[Literal["simple", "medium", "complex"]]:
    """
    分析问题复杂度，动态路由到不同处理节点
    """
    question = state["question"]

    # 简单的复杂度评估
    if len(question) < 20:
        complexity = 1
        target = "simple"
        handler = "AI助手"
    elif len(question) < 50:
        complexity = 3
        target = "medium"
        handler = "初级客服"
    else:
        complexity = 5
        target = "complex"
        handler = "高级专家"

    print(f"📊 问题分析: 复杂度={complexity}, 路由到={target}")

    return Command(
        goto=target,
        update={
            "complexity": complexity,
            "handler": handler
        }
    )

# 简单问题处理
def handle_simple(state: TicketState) -> dict:
    """AI 自动回答简单问题"""
    answer = f"[{state['handler']}] 这是一个简单问题的自动回复"
    print(f"🤖 {answer}")
    return {"answer": answer}

# 中等问题处理
def handle_medium(state: TicketState) -> dict:
    """初级客服处理中等问题"""
    answer = f"[{state['handler']}] 我来帮您解决这个问题"
    print(f"👤 {answer}")
    return {"answer": answer}

# 复杂问题处理
def handle_complex(state: TicketState) -> dict:
    """高级专家处理复杂问题"""
    answer = f"[{state['handler']}] 这需要深入分析，请稍等"
    print(f"👨‍💼 {answer}")
    return {"answer": answer}

# 构建图
def create_routing_graph():
    """创建动态路由图"""
    builder = StateGraph(TicketState)

    # 添加节点
    builder.add_node("analyze", analyze_question)
    builder.add_node("simple", handle_simple)
    builder.add_node("medium", handle_medium)
    builder.add_node("complex", handle_complex)

    # 添加边
    builder.add_edge(START, "analyze")
    # Command 自动处理动态路由
    builder.add_edge("simple", END)
    builder.add_edge("medium", END)
    builder.add_edge("complex", END)

    return builder.compile()

# 测试
if __name__ == "__main__":
    graph = create_routing_graph()

    test_cases = [
        "如何登录？",
        "我的账号无法登录，尝试了多次都失败，密码也重置过",
        "系统在高并发场景下出现性能瓶颈，数据库连接池配置需要优化，同时需要考虑缓存策略和负载均衡方案"
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n=== 测试{i} ===")
        print(f"问题: {question}")
        result = graph.invoke({
            "question": question,
            "complexity": 0,
            "handler": "",
            "answer": ""
        })
        print(f"处理结果: {result['answer']}\n")
```

### 关键点解析

1. **动态目标选择**：根据运行时计算的值决定 `goto` 参数

2. **状态更新与跳转结合**：`update` 参数在跳转前更新状态

3. **类型安全**：`Literal` 类型确保只能跳转到预定义的节点

---

## 场景3：父图通信

### 业务场景

实现一个包含子图的工作流，子图通过 Command.PARENT 向父图发送信号。

### 完整代码

```python
"""
场景3：父图通信
演示子图如何通过 Command.PARENT 与父图通信
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 父图状态
class ParentState(TypedDict):
    task: str
    subtask_result: str
    final_result: str

# 子图状态
class SubState(TypedDict):
    data: str
    processed: bool

# 子图节点 - 向父图发送结果
def process_subtask(state: SubState) -> Command:
    """
    子图处理节点，完成后向父图发送结果
    """
    data = state["data"]
    processed_data = data.upper()  # 简单处理

    print(f"🔧 子图处理: {data} -> {processed_data}")

    # 使用 Command.PARENT 向父图发送更新
    return Command(
        goto=Command.PARENT,  # 特殊标记：返回父图
        update={
            "subtask_result": processed_data,
            "processed": True
        }
    )

# 创建子图
def create_subgraph():
    """创建子图"""
    builder = StateGraph(SubState)
    builder.add_node("process", process_subtask)
    builder.add_edge(START, "process")
    # 不添加到 END 的边，因为 Command.PARENT 会处理返回
    return builder.compile()

# 父图节点 - 调用子图
def delegate_to_subgraph(state: ParentState) -> dict:
    """
    父图节点，将任务委托给子图
    """
    print(f"📤 父图委托任务: {state['task']}")

    # 调用子图
    subgraph = create_subgraph()
    result = subgraph.invoke({
        "data": state["task"],
        "processed": False
    })

    print(f"📥 父图接收结果: {result.get('subtask_result', '')}")

    return {
        "subtask_result": result.get("subtask_result", ""),
        "final_result": f"已完成: {result.get('subtask_result', '')}"
    }

# 创建父图
def create_parent_graph():
    """创建父图"""
    builder = StateGraph(ParentState)
    builder.add_node("delegate", delegate_to_subgraph)
    builder.add_edge(START, "delegate")
    builder.add_edge("delegate", END)
    return builder.compile()

# 测试
if __name__ == "__main__":
    print("=== 父子图通信测试 ===\n")

    graph = create_parent_graph()
    result = graph.invoke({
        "task": "process this data",
        "subtask_result": "",
        "final_result": ""
    })

    print(f"\n最终结果: {result['final_result']}")
```

### 关键点解析

1. **Command.PARENT 标记**：
   ```python
   Command(goto=Command.PARENT, update={...})
   ```
   [来源: reference/source_节点函数约定_02.md]

2. **命名空间处理**：框架自动处理父子图之间的命名空间转换

3. **状态传递**：子图的 `update` 会合并到父图状态中

---

## 场景4：多路径条件跳转

### 业务场景

实现一个订单处理流程，根据多个条件组合决定处理路径。

### 完整代码

```python
"""
场景4：多路径条件跳转
演示复杂的条件判断和多路径跳转
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 定义状态
class OrderState(TypedDict):
    order_id: str
    amount: float
    is_vip: bool
    has_coupon: bool
    payment_method: str
    discount: float
    final_amount: float
    status: str

# 订单分析节点
def analyze_order(state: OrderState) -> Command[Literal["vip_fast", "coupon_check", "normal", "reject"]]:
    """
    分析订单，根据多个条件决定处理路径
    """
    amount = state["amount"]
    is_vip = state["is_vip"]
    has_coupon = state["has_coupon"]

    print(f"📦 分析订单: 金额={amount}, VIP={is_vip}, 优惠券={has_coupon}")

    # 多条件判断
    if amount < 0:
        return Command(goto="reject", update={"status": "invalid"})
    elif is_vip and amount > 1000:
        return Command(goto="vip_fast", update={"status": "vip_processing"})
    elif has_coupon:
        return Command(goto="coupon_check", update={"status": "checking_coupon"})
    else:
        return Command(goto="normal", update={"status": "normal_processing"})

# VIP 快速通道
def vip_fast_track(state: OrderState) -> dict:
    """VIP 快速处理"""
    discount = state["amount"] * 0.2  # 8折
    final_amount = state["amount"] - discount
    print(f"⭐ VIP快速通道: 优惠{discount}, 实付{final_amount}")
    return {
        "discount": discount,
        "final_amount": final_amount,
        "status": "completed"
    }

# 优惠券检查
def check_coupon(state: OrderState) -> dict:
    """检查并应用优惠券"""
    discount = min(state["amount"] * 0.1, 50)  # 最多优惠50
    final_amount = state["amount"] - discount
    print(f"🎫 优惠券应用: 优惠{discount}, 实付{final_amount}")
    return {
        "discount": discount,
        "final_amount": final_amount,
        "status": "completed"
    }

# 普通处理
def normal_process(state: OrderState) -> dict:
    """普通订单处理"""
    final_amount = state["amount"]
    print(f"📋 普通处理: 实付{final_amount}")
    return {
        "discount": 0,
        "final_amount": final_amount,
        "status": "completed"
    }

# 拒绝订单
def reject_order(state: OrderState) -> dict:
    """拒绝无效订单"""
    print(f"❌ 订单拒绝: 无效金额")
    return {
        "discount": 0,
        "final_amount": 0,
        "status": "rejected"
    }

# 构建图
def create_order_graph():
    """创建订单处理图"""
    builder = StateGraph(OrderState)

    # 添加节点
    builder.add_node("analyze", analyze_order)
    builder.add_node("vip_fast", vip_fast_track)
    builder.add_node("coupon_check", check_coupon)
    builder.add_node("normal", normal_process)
    builder.add_node("reject", reject_order)

    # 添加边
    builder.add_edge(START, "analyze")
    builder.add_edge("vip_fast", END)
    builder.add_edge("coupon_check", END)
    builder.add_edge("normal", END)
    builder.add_edge("reject", END)

    return builder.compile()

# 测试
if __name__ == "__main__":
    graph = create_order_graph()

    test_orders = [
        {
            "order_id": "001",
            "amount": 1500,
            "is_vip": True,
            "has_coupon": False,
            "payment_method": "credit",
            "discount": 0,
            "final_amount": 0,
            "status": "pending"
        },
        {
            "order_id": "002",
            "amount": 500,
            "is_vip": False,
            "has_coupon": True,
            "payment_method": "alipay",
            "discount": 0,
            "final_amount": 0,
            "status": "pending"
        },
        {
            "order_id": "003",
            "amount": 300,
            "is_vip": False,
            "has_coupon": False,
            "payment_method": "wechat",
            "discount": 0,
            "final_amount": 0,
            "status": "pending"
        }
    ]

    for order in test_orders:
        print(f"\n=== 处理订单 {order['order_id']} ===")
        result = graph.invoke(order)
        print(f"状态: {result['status']}, 实付: {result['final_amount']}\n")
```

### 关键点解析

1. **多条件判断**：在节点内部使用 if-elif-else 处理复杂逻辑

2. **类型安全的多目标**：`Literal` 枚举所有可能的跳转目标

3. **状态更新时机**：在 Command 中更新状态，确保跳转前状态已更新

---

## 场景5：Command 与重试机制结合

### 业务场景

实现一个 API 调用节点，失败时通过 Command 跳转到重试或降级节点。

### 完整代码

```python
"""
场景5：Command 与重试机制结合
演示 Command 对象与错误处理、重试机制的结合使用
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy
import random

# 定义状态
class APIState(TypedDict):
    request: str
    response: str
    attempts: int
    error: str
    fallback_used: bool

# API 调用节点 - 可能失败
def call_api(state: APIState) -> Command[Literal["success", "retry", "fallback"]]:
    """
    调用外部 API，根据结果决定下一步
    """
    attempts = state["attempts"] + 1
    print(f"🌐 API调用 (第{attempts}次)")

    # 模拟 API 调用（30% 成功率）
    if random.random() < 0.3:
        print("✅ API调用成功")
        return Command(
            goto="success",
            update={
                "response": "API返回数据",
                "attempts": attempts
            }
        )
    elif attempts < 3:
        print(f"⚠️ API调用失败，准备重试")
        return Command(
            goto="retry",
            update={
                "error": "连接超时",
                "attempts": attempts
            }
        )
    else:
        print("❌ API调用失败，使用降级方案")
        return Command(
            goto="fallback",
            update={
                "error": "多次重试失败",
                "attempts": attempts
            }
        )

# 成功处理
def handle_success(state: APIState) -> dict:
    """处理成功响应"""
    print(f"📊 处理成功响应: {state['response']}")
    return {"fallback_used": False}

# 重试节点
def handle_retry(state: APIState) -> Command[Literal["call_api"]]:
    """
    重试逻辑，等待后重新调用 API
    """
    print(f"🔄 等待后重试...")
    # 实际应用中这里会有延迟
    return Command(goto="call_api")  # 跳回 API 调用节点

# 降级处理
def handle_fallback(state: APIState) -> dict:
    """使用降级方案"""
    print(f"🛡️ 使用缓存数据作为降级方案")
    return {
        "response": "缓存数据",
        "fallback_used": True
    }

# 构建图
def create_api_graph():
    """创建 API 调用图"""
    builder = StateGraph(APIState)

    # 添加节点（为 call_api 配置重试策略）
    builder.add_node(
        "call_api",
        call_api,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=1.0,
            backoff_factor=2.0,
            retry_on=Exception
        )
    )
    builder.add_node("success", handle_success)
    builder.add_node("retry", handle_retry)
    builder.add_node("fallback", handle_fallback)

    # 添加边
    builder.add_edge(START, "call_api")
    builder.add_edge("success", END)
    # retry 节点会跳回 call_api
    builder.add_edge("fallback", END)

    return builder.compile()

# 测试
if __name__ == "__main__":
    print("=== API调用与重试测试 ===\n")

    graph = create_api_graph()
    result = graph.invoke({
        "request": "获取用户数据",
        "response": "",
        "attempts": 0,
        "error": "",
        "fallback_used": False
    })

    print(f"\n最终结果:")
    print(f"  响应: {result['response']}")
    print(f"  尝试次数: {result['attempts']}")
    print(f"  使用降级: {result['fallback_used']}")
```

[来源: reference/source_节点函数约定_02.md]

### 关键点解析

1. **Command 与 RetryPolicy 结合**：节点级别的重试 + Command 控制的业务重试

2. **循环跳转**：retry 节点可以跳回 call_api 节点形成循环

3. **状态累积**：每次重试都更新 attempts 计数器

---

## 最佳实践总结

### 1. Command 使用原则

```python
# ✅ 推荐：明确的类型注解
def node(state: State) -> Command[Literal["next", "end"]]:
    return Command(goto="next", update={"key": "value"})

# ❌ 避免：无类型注解
def node(state: State):
    return Command(goto="next")
```

### 2. 错误处理

```python
# ✅ 推荐：验证目标节点存在
def node(state: State) -> Command[Literal["valid_node"]]:
    # 类型系统会确保 "valid_node" 存在
    return Command(goto="valid_node")

# ❌ 避免：硬编码字符串
def node(state: State):
    return Command(goto="maybe_invalid")  # 可能运行时错误
```

[来源: reference/search_节点函数约定_01.md]

### 3. 状态更新时机

```python
# ✅ 推荐：在 Command 中更新状态
return Command(
    goto="next",
    update={"processed": True}  # 跳转前更新
)

# ⚠️ 注意：分开更新可能导致状态不一致
state["processed"] = True  # 可能在跳转前丢失
return Command(goto="next")
```

### 4. 父图通信

```python
# ✅ 推荐：使用 Command.PARENT
return Command(
    goto=Command.PARENT,
    update={"result": "from_subgraph"}
)

# ❌ 避免：手动构造命名空间
return Command(goto="../parent_node")  # 不推荐
```

---

## 常见问题

### Q1: Command 和条件边有什么区别？

**A**:
- **Command**：节点内部动态决定跳转，更灵活
- **条件边**：图结构静态定义，适合固定路由

```python
# Command 方式（动态）
def node(state):
    if complex_logic(state):
        return Command(goto="path_a")
    return Command(goto="path_b")

# 条件边方式（静态）
def router(state):
    return "path_a" if condition else "path_b"

builder.add_conditional_edges("node", router)
```

### Q2: Command 可以跳转到多个节点吗？

**A**: 单个 Command 只能跳转到一个节点，但可以返回 Command 列表：

```python
def node(state) -> list[Command]:
    return [
        Command(goto="node_a", update={"a": 1}),
        Command(goto="node_b", update={"b": 2})
    ]
```

### Q3: Command.PARENT 在非子图中使用会怎样？

**A**: 会抛出异常。Command.PARENT 只能在子图节点中使用。

---

## 性能优化建议

1. **避免深层嵌套**：过多的 Command 跳转会影响性能
2. **合并状态更新**：在 Command 中一次性更新多个字段
3. **使用类型注解**：帮助框架优化执行路径
4. **缓存子图实例**：避免重复创建子图

---

## 总结

Command 对象是 LangGraph 中控制流程的核心机制：

1. **基础跳转**：`Command(goto="target")` 实现节点间跳转
2. **状态更新**：`Command(goto="target", update={...})` 同时更新状态
3. **动态路由**：根据运行时条件决定跳转目标
4. **父图通信**：`Command.PARENT` 实现子图向父图通信
5. **类型安全**：使用 `Literal` 类型确保跳转目标有效

掌握 Command 对象的使用，是构建复杂工作流的关键。

---

**文档版本**: v1.0
**最后更新**: 2026-02-25
**相关文档**:
- 03_核心概念_2_状态更新机制.md
- 03_核心概念_4_错误处理机制.md
