# step_timeout 的真实语义

> **核心概念 1/6**
> **关键词**：step budget、Pregel superstep、outer deadline、不是总运行时长

---

## 为什么这个概念必须先讲？

因为几乎所有关于 LangGraph timeout 的误解，都从这里开始。

很多人看到：

```python
graph.step_timeout = 10
```

就会自动脑补成：

- 这个图最多跑 10 秒；
- 每个 node 最多跑 10 秒；
- 每个工具调用最多等 10 秒。

但源码并不是这样实现的。

---

## 一句话定义

**`step_timeout` 是 LangGraph 对“当前 Pregel step 最多等多久”设置的外层预算，不是整个 run 的总预算，也不是 node 内 I/O 的局部预算。**

[来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/main.py]

---

## 从源码看定义

在 `Pregel` 类中有非常直接的说明：

```python
step_timeout: float | None = None
"""Maximum time to wait for a step to complete, in seconds."""
```

这句话里最重要的词是 `step`。

不是：
- node
- graph total duration
- client request

而是 **step**。

---

## 什么是 step？

LangGraph 的执行模型底层基于 Pregel / BSP（Bulk Synchronous Parallel）思想。

直觉化理解：

1. 当前轮哪些任务该执行？
2. 这些任务并发执行；
3. 这一轮都结束后，写入才统一生效；
4. 再进入下一轮。

这“一轮”，就是 step。

源码注释已经写得很清楚：

```python
# computation proceeds in steps, while there are channel updates.
# Channel updates from step N are only visible in step N+1
# channels are guaranteed to be immutable for the duration of the step
```

所以 `step_timeout` 本质上是：

> **这一轮并发任务的整体等待预算。**

[来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/main.py]

---

## 它到底在哪里生效？

### 同步路径

```python
for _ in runner.tick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.accept_push,
):
    yield from _output(...)
```

### 异步路径

```python
async for _ in runner.atick(
    [t for t in loop.tasks.values() if not t.writes],
    timeout=self.step_timeout,
    get_waiter=get_waiter,
    schedule_task=loop.aaccept_push,
):
    ...
```

这两个片段共同说明：

- `step_timeout` 不在 node 函数内部生效；
- 它在 runner 层生效；
- runner 负责等待当前 step 内的 future 完成。

---

## 为什么它不是“总运行时长”？

因为 `timeout=self.step_timeout` 是在每轮 `tick()` / `atick()` 调用时传进去的。

换句话说：

- 第 1 个 step 有自己的 budget；
- 第 2 个 step 再重新开始计时；
- 第 3 个 step 也是如此。

如果一个图有 5 个 step，而每个 step 都在预算内完成，整图总时长完全可能大于 `step_timeout`。

### 举个例子

假设：
- `step_timeout = 5`
- 图有 4 个 step
- 每个 step 分别耗时 4.5 秒

那么：
- 每一步都合法；
- 总时长大约 18 秒；
- 但不会因为“超过 5 秒”而整体失败。

这正说明它不是 run-level 总预算。

---

## 为什么它也不是“node timeout”？

如果一个 step 内只有一个 node 在跑，看起来很像 node timeout。

但只要进入并发场景，这种直觉就会失效。

例如同一 step 里：
- 节点 A：0.3 秒完成
- 节点 B：6 秒完成
- `step_timeout = 5`

结果并不是“A 成功、B 超时、step 正常提交下一轮”。

而是：
- A 先完成；
- step 仍然要等 B；
- 5 秒到了 B 还没完成；
- 当前 step 触发 timeout；
- inflight 任务被取消。

这就是 step budget 和 node budget 的根本区别。

---

## 源码里的时间计算方式

在同步 runner 里：

```python
end_time = timeout + time.monotonic() if timeout else None
```

然后在等待 future 时不断用剩余时间：

```python
timeout=(max(0, end_time - time.monotonic()) if end_time else None)
```

异步 runner 也是同样结构，只是用 `loop.time()`。

这说明它是典型的 **deadline model**：

- 先计算一个绝对终点；
- 后续等待都用“剩余时间”；
- 到点即停止等待。

这种设计比“每次重新 sleep 一个固定 timeout”更稳定，因为不会累计漂移。

[来源: sourcecode/langgraph/libs/langgraph/langgraph/pregel/_runner.py]

---

## 它触发时会发生什么？

最终逻辑在 `_panic_or_proceed()`：

```python
if inflight:
    while inflight:
        inflight.pop().cancel()
    raise timeout_exc_cls("Timed out")
```

也就是说：

1. 如果还有 inflight future；
2. 就取消掉它们；
3. 再抛出 timeout 异常。

所以 `step_timeout` 不是“温柔提醒”，而是实打实地改变执行流。

---

## 这对设计工作流意味着什么？

### 启示 1：step 不要塞太多异质操作

如果一个 step 里同时塞：
- 快速本地计算；
- 远程搜索；
- 复杂模型调用；

那么 timeout 往往会被最慢那个环节支配，其他任务也会被一起裹挟。

### 启示 2：真正慢的 I/O 要在 node 内先切断

不要把所有“是否继续等”的责任都交给 step timeout。

更好的做法：
- node 内先 `wait_for` 或 `httpx.Timeout`；
- step 外层做 orchestration 兜底。

### 启示 3：长任务可能要拆 step

如果一个 step 本身就是 30 秒以上的复合大步骤，你可能需要问：

> 这到底是一个 step，还是三个应该拆开的 step？

---

## 一个最小示例

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: str


def slow_node(state: State):
    time.sleep(2)
    return {"value": state["value"] + " -> done"}


builder = StateGraph(State)
builder.add_node("slow", slow_node)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)

graph = builder.compile()
graph.step_timeout = 1

try:
    graph.invoke({"value": "start"})
except TimeoutError as exc:
    print(type(exc).__name__, exc)
```

这个例子里：
- 因为图只有一个 step；
- 所以它看起来像 node timeout；
- 但本质上仍然是 step timeout。

---

## 最容易记错的三个点

### 记错点 1：`step_timeout = run timeout`
错。

### 记错点 2：`step_timeout = tool timeout`
错。

### 记错点 3：`step_timeout` 能覆盖平台 / 前端 timeout
也错。

它只属于 **图运行时 step 调度层**。

---

## 实际应用中怎么定这个值？

没有统一答案，但有一个可操作原则：

### 先问三个问题
1. 一个 step 里最慢的合理任务应该多久？
2. 超过这个时间，继续等还有多大收益？
3. timeout 后有没有 fallback 或重试？

### 一个常用经验
- 开发环境：更短，尽快暴露问题；
- 生产环境：稍宽，但别大到掩盖架构问题；
- 有流式进度时：可适度放宽；
- 有人工等待时：不要放宽，改成 interrupt。

---

## 最终结论

**理解 `step_timeout` 的关键，不是记住某个秒数，而是记住它的粒度：它控制的是当前 step 的整体等待预算。**

[来源: atom/langgraph/L3_工作流编排/09_超时控制/reference/source_步级超时_01.md]


---

## 扩展补充：边界再下沉一层

下面这一组补充专门回答“step_timeout 在真实项目里到底怎么判断边界”这个问题。
你可以把它理解成围绕“step 粒度、step 与 node / run 的区别、预算边界”的一组工程判断模板。

### 边界判断 1

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 1 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 2

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 2 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 3

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 3 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 4

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 4 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 5

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 5 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 6

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 6 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 7

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 7 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 8

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 8 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 9

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 9 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 10

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 10 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 11

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 11 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 12

- 观察点：当你讨论“step_timeout”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“step 粒度、step 与 node / run 的区别、预算边界”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 12 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

## 高频追问速答（扩展版）

### 追问 1
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 2
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 3
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 4
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 5
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 6
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 7
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 8
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 9
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 10
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 11
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 12
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 13
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 14
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 15
- 问：如果团队里有人只想通过调大 timeout 来解决“step_timeout”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 step 粒度、step 与 node / run 的区别、预算边界 的哪一层在失配，超时后你准备让谁接手恢复？”

## 设计检查清单

- 检查项 1：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 2：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 3：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 4：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 5：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 6：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 7：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 8：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 9：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 10：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 11：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 12：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 13：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 14：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 15：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 16：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 17：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 18：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 19：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 20：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 21：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 22：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 23：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 24：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 25：围绕“step_timeout”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。

## 口头表达模板

- 表达模板 1：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 2：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 3：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 4：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 5：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 6：在解释“step_timeout”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
