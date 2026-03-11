---
type: context7_documentation
library: langgraph
version: latest / 1.0.8-docset
fetched_at: 2026-03-07
knowledge_point: 09_超时控制
context7_query: workflow timeout control step_timeout graph execution timeout runtime context timeout
---

# Context7 文档：LangGraph 官方资料中的“超时控制”相关信息

## 文档来源
- 库名称：LangGraph
- 版本：latest / 1.0.8-docset
- Context7 Library ID：`/websites/langchain_oss_python_langgraph`
- 交叉校验 Library ID：`/langchain-ai/langgraph/1.0.8`

## 关键信息提取

### 1. 官方公开文档更强调“可中断的长运行工作流”，而不是直接讲 `step_timeout`

Context7 返回的官方资料主要集中在：
- `interrupts`
- `use-functional-api`
- durable execution / checkpoint / resume

代表性来源：
- `https://docs.langchain.com/oss/python/langgraph/interrupts`
- `https://docs.langchain.com/oss/python/langgraph/use-functional-api`

官方示例强调：

```python
from langgraph.types import interrupt

def approval_node(state: State):
    approved = interrupt("Do you approve this action?")
    return {"approved": approved}
```

这说明 LangGraph 对“人类等待”“长时间审批”“跨会话恢复”的官方首选方案是：
**中断 + checkpoint + resume**，而不是一味把 timeout 拉长。

### 2. `interrupt` / `Command(resume=...)` 是“长等待”的官方思路

Context7 给出的完整示例反复展示：

```python
graph.invoke(inputs, config=config)
graph.invoke(Command(resume=...), config=config)
```

对 timeout 设计的启示是：

- 如果等待的是人或外部异步流程，优先“暂停工作流”；
- 如果等待的是网络 / I/O 响应，优先节点内设置 deadline；
- 只有对当前 step 的总体等待时间，才交给 `step_timeout`。

### 3. 官方文档中没有把 `step_timeout` 作为高频教学 API 展开

本轮 Context7 查询没有返回官方教程页对 `step_timeout` 的系统讲解。

这意味着：

1. `step_timeout` 更接近“运行时能力 / 源码级能力”；
2. 实战中必须结合源码和测试来理解其真实语义；
3. 不应仅凭教程记忆去推断 timeout 行为。

### 4. Functional API / Interrupt 文档提供了 timeout 的替代设计路线

当你面对以下场景时，官方更推荐：

- 审批、人工确认 → `interrupt`
- 跨会话恢复 → checkpointer
- 长运行 agent / workflow → durable execution + resume

而不是：

- 把单次 node 执行无限拉长；
- 让前端一直挂着等待一个超长请求；
- 依赖单一的全局 timeout 解决所有问题。

## 结论

从官方文档视角看，“超时控制”的核心不是“把 timeout 调大”，而是：

1. 用 `interrupt` 处理人类等待；
2. 用 durable execution 处理长生命周期流程；
3. 用源码级 `step_timeout` 约束单个 orchestration step；
4. 把真正的 I/O 超时放进节点内部。

也就是说：**LangGraph 官方路线更偏“设计正确的等待方式”，而不是“给一个万能 timeout 参数”。**

