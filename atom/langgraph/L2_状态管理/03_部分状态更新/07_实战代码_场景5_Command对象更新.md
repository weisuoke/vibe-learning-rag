# 实战代码：场景5 - Command对象更新

> 使用Command对象同时实现状态更新和流程控制，展示动态路由和复杂决策流程

---

## 概述

本文档提供3个完整的实战场景，展示如何在LangGraph中使用Command对象进行状态更新和流程控制。每个场景都是完整可运行的代码，展示了不同的业务应用。

**核心价值：**
- 动态路由：根据执行结果动态决定下一步
- 流程控制：实现循环、跳转、提前退出
- 代码集中：状态更新和流程控制在同一返回值中

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 场景1：审批工作流系统

这个场景展示了一个多级审批工作流，根据审批结果动态路由到不同的节点。

```python
"""
场景1：审批工作流系统
演示：使用Command对象实现多级审批流程
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ===== 1. 定义状态 =====
class ApprovalState(TypedDict):
    """审批状态"""
    request_id: str
    request_type: Literal["leave", "expense", "purchase"]
    amount: float
    requester: str
    status: Literal["pending", "approved", "rejected", "escalated"]
    approver: str
    comments: str
    approval_level: int

# ===== 2. 节点函数 =====
def initial_review(state: ApprovalState) -> Command:
    """初审"""
    print(f"\n=== 初审申请 {state['request_id']} ===")
    print(f"申请类型: {state['request_type']}")
    print(f"金额: ${state['amount']}")
    
    # 根据金额决定审批流程
    if state['amount'] < 100:
        # 小额：直接批准
        print("→ 金额较小，直接批准")
        return Command(
            goto=END,
            update={
                "status": "approved",
                "approver": "system_auto",
                "comments": "Auto-approved (amount < $100)",
                "approval_level": 1
            }
        )
    elif state['amount'] < 1000:
        # 中额：需要经理审批
        print("→ 需要经理审批")
        return Command(
            goto="manager_review",
            update={
                "status": "pending",
                "comments": "Forwarded to manager",
                "approval_level": 1
            }
        )
    else:
        # 大额：需要总监审批
        print("→ 金额较大，需要总监审批")
        return Command(
            goto="director_review",
            update={
                "status": "escalated",
                "comments": "Escalated to director",
                "approval_level": 1
            }
        )

def manager_review(state: ApprovalState) -> Command:
    """经理审批"""
    print(f"\n=== 经理审批 ===")
    print(f"审批级别: {state['approval_level']}")
    
    # 模拟审批决策（实际应用中可能基于更复杂的逻辑）
    if state['amount'] < 500:
        # 批准
        print("→ 经理批准")
        return Command(
            goto=END,
            update={
                "status": "approved",
                "approver": "manager",
                "comments": "Approved by manager",
                "approval_level": 2
            }
        )
    else:
        # 升级到总监
        print("→ 金额超出权限，升级到总监")
        return Command(
            goto="director_review",
            update={
                "status": "escalated",
                "comments": "Escalated to director by manager",
                "approval_level": 2
            }
        )

def director_review(state: ApprovalState) -> Command:
    """总监审批"""
    print(f"\n=== 总监审批 ===")
    print(f"审批级别: {state['approval_level']}")
    
    # 模拟审批决策
    if state['amount'] < 5000:
        # 批准
        print("→ 总监批准")
        return Command(
            goto=END,
            update={
                "status": "approved",
                "approver": "director",
                "comments": "Approved by director",
                "approval_level": 3
            }
        )
    else:
        # 拒绝（超出预算）
        print("→ 金额超出预算，拒绝")
        return Command(
            goto=END,
            update={
                "status": "rejected",
                "approver": "director",
                "comments": "Rejected: exceeds budget limit",
                "approval_level": 3
            }
        )

# ===== 3. 构建图 =====
def build_approval_workflow():
    """构建审批工作流图"""
    graph = StateGraph(ApprovalState)
    
    # 添加节点
    graph.add_node("initial", initial_review)
    graph.add_node("manager_review", manager_review)
    graph.add_node("director_review", director_review)
    
    # 只需要添加起始边，其他路由由Command对象控制
    graph.add_edge(START, "initial")
    
    return graph.compile()

# ===== 4. 运行示例 =====
if __name__ == "__main__":
    app = build_approval_workflow()
    
    # 测试不同金额的申请
    test_cases = [
        {"amount": 50, "desc": "小额申请"},
        {"amount": 300, "desc": "中额申请"},
        {"amount": 800, "desc": "需要升级的申请"},
        {"amount": 6000, "desc": "超预算申请"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print("\n" + "=" * 60)
        print(f"测试案例 {i}: {case['desc']} (${case['amount']})")
        print("=" * 60)
        
        initial_state = {
            "request_id": f"REQ-{i:03d}",
            "request_type": "expense",
            "amount": case['amount'],
            "requester": "john.doe",
            "status": "pending",
            "approver": "",
            "comments": "",
            "approval_level": 0,
        }
        
        result = app.invoke(initial_state)
        
        print("\n" + "-" * 60)
        print("最终结果:")
        print(f"  状态: {result['status']}")
        print(f"  审批人: {result['approver']}")
        print(f"  审批级别: {result['approval_level']}")
        print(f"  备注: {result['comments']}")
        print("-" * 60)
```

**运行输出示例：**
```
============================================================
测试案例 1: 小额申请 ($50)
============================================================

=== 初审申请 REQ-001 ===
申请类型: expense
金额: $50
→ 金额较小，直接批准

------------------------------------------------------------
最终结果:
  状态: approved
  审批人: system_auto
  审批级别: 1
  备注: Auto-approved (amount < $100)
------------------------------------------------------------

============================================================
测试案例 2: 中额申请 ($300)
============================================================

=== 初审申请 REQ-002 ===
申请类型: expense
金额: $300
→ 需要经理审批

=== 经理审批 ===
审批级别: 1
→ 经理批准

------------------------------------------------------------
最终结果:
  状态: approved
  审批人: manager
  审批级别: 2
  备注: Approved by manager
------------------------------------------------------------
```

**关键特性：**
1. **多级路由**：根据金额自动路由到不同审批级别
2. **动态升级**：经理可以将申请升级到总监
3. **提前退出**：小额申请直接批准，无需多级审批
4. **状态追踪**：记录审批级别和审批人

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 场景2：任务重试与容错系统

这个场景展示了一个智能重试系统，支持指数退避和最大重试次数限制。

```python
"""
场景2：任务重试与容错系统
演示：使用Command对象实现智能重试逻辑
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import random
import time

# ===== 1. 定义状态 =====
class TaskState(TypedDict):
    """任务状态"""
    task_id: str
    task_name: str
    attempts: int
    max_attempts: int
    success: bool
    error: str
    result: str
    retry_delay: float
    total_time: float

# ===== 2. 节点函数 =====
def execute_task(state: TaskState) -> Command:
    """执行任务（模拟可能失败的操作）"""
    attempts = state["attempts"] + 1
    start_time = time.time()
    
    print(f"\n尝试 #{attempts}: 执行任务 '{state['task_name']}'")
    
    # 模拟任务执行（70%成功率）
    success = random.random() > 0.3
    
    elapsed = time.time() - start_time
    total_time = state.get("total_time", 0) + elapsed
    
    if success:
        print(f"✓ 任务成功！(耗时: {elapsed:.2f}s)")
        return Command(
            goto=END,
            update={
                "attempts": attempts,
                "success": True,
                "result": f"Task '{state['task_name']}' completed successfully",
                "total_time": total_time
            }
        )
    else:
        print(f"✗ 任务失败")
        
        if attempts < state["max_attempts"]:
            # 计算指数退避延迟
            retry_delay = min(2 ** attempts, 60)  # 最多60秒
            print(f"→ 准备重试 (剩余 {state['max_attempts'] - attempts} 次)")
            print(f"→ 等待 {retry_delay}s 后重试...")
            
            # 模拟延迟（实际应用中可能使用真实的延迟）
            # time.sleep(retry_delay)
            
            return Command(
                goto="execute_task",  # 回到自己
                update={
                    "attempts": attempts,
                    "success": False,
                    "error": f"Attempt {attempts} failed",
                    "retry_delay": retry_delay,
                    "total_time": total_time
                }
            )
        else:
            print(f"✗ 达到最大重试次数，任务失败")
            return Command(
                goto=END,
                update={
                    "attempts": attempts,
                    "success": False,
                    "error": f"Failed after {attempts} attempts",
                    "total_time": total_time
                }
            )

# ===== 3. 构建图 =====
def build_retry_system():
    """构建重试系统图"""
    graph = StateGraph(TaskState)
    
    graph.add_node("execute_task", execute_task)
    graph.add_edge(START, "execute_task")
    
    return graph.compile()

# ===== 4. 运行示例 =====
if __name__ == "__main__":
    app = build_retry_system()
    
    # 测试任务
    print("=" * 60)
    print("任务重试系统测试")
    print("=" * 60)
    
    initial_state = {
        "task_id": "TASK-001",
        "task_name": "Process data batch",
        "attempts": 0,
        "max_attempts": 5,
        "success": False,
        "error": "",
        "result": "",
        "retry_delay": 0,
        "total_time": 0,
    }
    
    result = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("最终结果:")
    print(f"  任务ID: {result['task_id']}")
    print(f"  任务名称: {result['task_name']}")
    print(f"  尝试次数: {result['attempts']}")
    print(f"  成功: {result['success']}")
    print(f"  总耗时: {result['total_time']:.2f}s")
    if result['success']:
        print(f"  结果: {result['result']}")
    else:
        print(f"  错误: {result['error']}")
    print("=" * 60)
```

**关键特性：**
1. **循环重试**：节点可以跳转到自己实现循环
2. **指数退避**：重试延迟随尝试次数指数增长
3. **最大次数限制**：防止无限循环
4. **状态追踪**：记录每次尝试的结果和总耗时

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 场景3：智能客服路由系统

这个场景展示了一个智能客服系统，根据客户查询内容动态路由到不同的处理部门。

```python
"""
场景3：智能客服路由系统
演示：使用Command对象实现智能查询分类和路由
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ===== 1. 定义状态 =====
class CustomerState(TypedDict):
    """客户服务状态"""
    query_id: str
    query: str
    category: Literal["sales", "support", "complaint", "billing", "unknown"]
    priority: Literal["high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative"]
    handled: bool
    response: str
    handler: str

# ===== 2. 节点函数 =====
def classify_query(state: CustomerState) -> Command:
    """分类客户查询"""
    print(f"\n=== 分析查询 ===")
    print(f"查询内容: {state['query']}")
    
    query = state["query"].lower()
    
    # 简单的关键词分类（实际应用中可能使用NLP模型）
    if any(word in query for word in ["buy", "price", "purchase", "product"]):
        category = "sales"
        priority = "high"
        sentiment = "positive"
        next_node = "handle_sales"
    elif any(word in query for word in ["problem", "error", "not working", "broken"]):
        category = "support"
        priority = "high"
        sentiment = "negative"
        next_node = "handle_support"
    elif any(word in query for word in ["complaint", "refund", "angry", "terrible"]):
        category = "complaint"
        priority = "high"
        sentiment = "negative"
        next_node = "handle_complaint"
    elif any(word in query for word in ["bill", "invoice", "payment", "charge"]):
        category = "billing"
        priority = "medium"
        sentiment = "neutral"
        next_node = "handle_billing"
    else:
        category = "unknown"
        priority = "low"
        sentiment = "neutral"
        next_node = "handle_unknown"
    
    print(f"分类结果: {category}")
    print(f"优先级: {priority}")
    print(f"情感: {sentiment}")
    print(f"→ 路由到: {next_node}")
    
    return Command(
        goto=next_node,
        update={
            "category": category,
            "priority": priority,
            "sentiment": sentiment
        }
    )

def handle_sales(state: CustomerState) -> Command:
    """处理销售查询"""
    print("\n=== 销售部门处理 ===")
    
    response = f"感谢您对我们产品的兴趣！关于 '{state['query']}'，我们的销售团队会在1小时内联系您。"
    
    return Command(
        goto=END,
        update={
            "handled": True,
            "response": response,
            "handler": "sales_team"
        }
    )

def handle_support(state: CustomerState) -> Command:
    """处理技术支持"""
    print("\n=== 技术支持处理 ===")
    
    response = f"我们已收到您的技术问题：'{state['query']}'。技术团队正在分析，预计24小时内回复。"
    
    return Command(
        goto=END,
        update={
            "handled": True,
            "response": response,
            "handler": "support_team"
        }
    )

def handle_complaint(state: CustomerState) -> Command:
    """处理投诉"""
    print("\n=== 投诉处理（高优先级）===")
    
    response = f"非常抱歉给您带来不便。关于 '{state['query']}'，我们会立即升级处理，1小时内回复。"
    
    return Command(
        goto=END,
        update={
            "handled": True,
            "response": response,
            "handler": "complaint_team"
        }
    )

def handle_billing(state: CustomerState) -> Command:
    """处理账单问题"""
    print("\n=== 账单部门处理 ===")
    
    response = f"关于您的账单问题：'{state['query']}'，我们的财务团队会在2个工作日内核实并回复。"
    
    return Command(
        goto=END,
        update={
            "handled": True,
            "response": response,
            "handler": "billing_team"
        }
    )

def handle_unknown(state: CustomerState) -> Command:
    """处理未知类型"""
    print("\n=== 转人工客服 ===")
    
    response = f"感谢您的咨询。您的问题 '{state['query']}' 已转接人工客服，请稍候。"
    
    return Command(
        goto=END,
        update={
            "handled": True,
            "response": response,
            "handler": "human_agent"
        }
    )

# ===== 3. 构建图 =====
def build_customer_service_system():
    """构建客服路由系统图"""
    graph = StateGraph(CustomerState)
    
    # 添加节点
    graph.add_node("classify", classify_query)
    graph.add_node("handle_sales", handle_sales)
    graph.add_node("handle_support", handle_support)
    graph.add_node("handle_complaint", handle_complaint)
    graph.add_node("handle_billing", handle_billing)
    graph.add_node("handle_unknown", handle_unknown)
    
    # 只需要添加起始边，其他路由由Command对象控制
    graph.add_edge(START, "classify")
    
    return graph.compile()

# ===== 4. 运行示例 =====
if __name__ == "__main__":
    app = build_customer_service_system()
    
    # 测试不同类型的查询
    test_queries = [
        "I want to buy your premium product",
        "My app is not working properly, getting errors",
        "I want a refund, this is terrible service!",
        "Why was I charged twice on my bill?",
        "What's your company address?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 60)
        print(f"查询 {i}")
        print("=" * 60)
        
        result = app.invoke({
            "query_id": f"Q-{i:03d}",
            "query": query,
            "category": "unknown",
            "priority": "low",
            "sentiment": "neutral",
            "handled": False,
            "response": "",
            "handler": "",
        })
        
        print("\n" + "-" * 60)
        print("处理结果:")
        print(f"  类别: {result['category']}")
        print(f"  优先级: {result['priority']}")
        print(f"  情感: {result['sentiment']}")
        print(f"  处理人: {result['handler']}")
        print(f"  响应: {result['response']}")
        print("-" * 60)
```

**关键特性：**
1. **智能分类**：基于关键词自动分类查询
2. **动态路由**：根据分类结果路由到不同处理节点
3. **优先级管理**：自动设置优先级和情感标签
4. **灵活扩展**：易于添加新的查询类别和处理节点

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 最佳实践总结

### 1. 明确的流程控制

```python
# ✅ 好的实践：明确的goto目标
def node(state: State) -> Command:
    if condition:
        return Command(goto="specific_node", update={...})
    else:
        return Command(goto=END, update={...})

# ❌ 不好的实践：模糊的流程
def node(state: State) -> Command:
    return Command(update={...})  # 缺少goto
```

### 2. 避免无限循环

```python
# ✅ 好的实践：有退出条件
def node(state: State) -> Command:
    if state["attempts"] >= state["max_attempts"]:
        return Command(goto=END, update={...})
    return Command(goto="node", update={...})

# ❌ 不好的实践：无限循环
def node(state: State) -> Command:
    return Command(goto="node", update={...})
```

### 3. 状态更新与流程控制分离

```python
# ✅ 好的实践：清晰的逻辑
def node(state: State) -> Command:
    # 1. 处理业务逻辑
    result = process_data(state)
    
    # 2. 决定下一步
    if result.success:
        next_node = "success_handler"
        update = {"status": "success"}
    else:
        next_node = "error_handler"
        update = {"status": "error"}
    
    # 3. 返回Command
    return Command(goto=next_node, update=update)
```

### 4. 避免混用Command和条件边

```python
# ✅ 清晰：只用Command
def node(state: State) -> Command:
    return Command(goto="next", update={...})

# ❌ 混乱：同时使用Command和条件边
def node(state: State) -> Command:
    return Command(goto="next", update={...})

graph.add_conditional_edges("node", router, {...})  # 不会生效
```

**[来源: reference/search_部分状态更新_01.md:98-125]**

---

## 常见陷阱

### 陷阱1：忘记添加目标节点

```python
# ❌ 错误：goto指向不存在的节点
def node(state: State) -> Command:
    return Command(goto="nonexistent_node", update={...})

# ✅ 正确：确保目标节点存在
graph.add_node("target_node", target_node_func)
```

### 陷阱2：循环没有退出条件

```python
# ❌ 错误：无限循环
def node(state: State) -> Command:
    return Command(goto="node", update={...})

# ✅ 正确：有退出条件
def node(state: State) -> Command:
    if state["attempts"] >= max_attempts:
        return Command(goto=END, update={...})
    return Command(goto="node", update={...})
```

### 陷阱3：忽略状态一致性

```python
# ❌ 错误：状态字段不一致
def node(state: State) -> Command:
    return Command(
        goto="next",
        update={"staus": "done"}  # 拼写错误
    )

# ✅ 正确：使用正确的字段名
def node(state: State) -> Command:
    return Command(
        goto="next",
        update={"status": "done"}
    )
```

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 总结

### 核心要点

1. **双重功能**：Command对象同时实现状态更新和流程控制
2. **动态路由**：节点可以根据执行结果动态决定下一步
3. **灵活性高**：支持循环、跳转、提前退出等复杂逻辑
4. **代码集中**：状态更新和流程控制在同一个返回值中

### 使用场景

- 需要动态路由的工作流
- 实现循环重试逻辑
- 条件跳转和提前退出
- 复杂的决策流程

### 与其他机制的对比

| 特性 | 字典返回 | Pydantic模型 | Command对象 |
|------|---------|------------|-------------|
| 状态更新 | ✅ | ✅ | ✅ |
| 类型安全 | ❌ | ✅ | ✅ |
| 流程控制 | ❌ | ❌ | ✅ |
| 动态路由 | ❌ | ❌ | ✅ |
| 复杂度 | 低 | 中 | 高 |
| 适用场景 | 简单更新 | 类型安全更新 | 复杂流程控制 |

**[来源: reference/source_部分状态更新_01.md:28-45]**

---

## 参考资料

- [reference/source_部分状态更新_01.md:28-45] - Command对象更新示例
- [reference/context7_langgraph_01.md] - LangGraph官方文档
- [reference/search_部分状态更新_01.md] - 社区最佳实践

---

**版本：** v1.0
**创建时间：** 2026-02-26
**维护者：** Claude Code
