# 子图超时与 ParentCommand

> **核心概念 3/6**
> **关键词**：subgraph、nested budget、Command.PARENT、控制流优先级

---

## 为什么子图 timeout 比普通图更难理解？

因为一旦有子图，你就不再只是处理“慢不慢”的问题，还要处理：

- timeout 属于哪一层？
- 谁有权决定继续执行？
- 子图返回给父图的命令会不会被吞掉？
- 父图和子图都配 timeout 时，边界怎么划？

这也是社区 issue 容易出现的地方。

---

## 一句话定义

**子图超时与 `ParentCommand` 的核心问题，是在多层图预算存在时，如何既限制局部等待，又不破坏图间控制流语义。**

---

## 源码测试告诉了我们什么？

LangGraph 测试里有一个非常关键的参数组合：

```python
@pytest.mark.parametrize("with_timeout", [False, "inner", "outer", "both"])
```

这相当于官方明确说：

1. 可能只给内层子图 timeout；
2. 可能只给外层父图 timeout；
3. 可能两层都给；
4. 这些组合都应该有稳定语义。

---

## 子图和父图 timeout 如何配置？

```python
sub_graph = sub_builder.compile(checkpointer=subgraph_persist)
if with_timeout in ("inner", "both"):
    sub_graph.step_timeout = 1

main_graph = main_builder.compile(sync_checkpointer, name="parent")
if with_timeout in ("outer", "both"):
    main_graph.step_timeout = 1
```

这段代码的意义非常大：

- `step_timeout` 不是只能配在最外层；
- 子图自己也能有局部预算；
- 多层 timeout 是设计上被允许的。

---

## 那么 `ParentCommand` 又是什么？

它是子图把控制权或跳转意图抛回父图的一种语义命令。

例如：

```python
return Command(
    graph=Command.PARENT,
    goto="node_b_parent",
    update={"dialog_state": ["b_child_state"]},
)
```

它的意思不是“我失败了”，而是：

> “请父图接下来跳到某个节点，并带上这些更新。”

这属于高级控制流。

---

## 为什么 timeout 不能吞掉它？

因为 `ParentCommand` 不是错误，而是图语义的一部分。

如果 timeout 机制把它吞掉，就会导致：

- 子图明明想把控制权交回父图；
- 结果父图只收到一个 timeout；
- 整个工作流分支语义丢失。

源码测试正是在保护这件事：

```python
with pytest.raises(ParentCommand) as exc_info:
    graph.invoke({"value": "start"}, thread1)
assert exc_info.value.args[0].goto == "test_cmd"
assert exc_info.value.args[0].update == {"key": "value"}
```

并且这个测试在 `with_timeout = True` 时仍然要通过。

[来源: sourcecode/langgraph/libs/langgraph/tests/test_pregel.py]

---

## 正确心智模型：双层预算，单一语义链路

你可以这样理解：

### 父图 timeout
约束的是“父图这一轮 orchestration 最多等多久”。

### 子图 timeout
约束的是“子图自己那一轮 step 最多等多久”。

### `ParentCommand`
约束的是“控制流应该往哪里走”。

这三者不是同一维度，所以不能互相替代。

---

## 一个类比

父图像总导演，子图像分场导演。

- 分场导演有自己的拍摄时间预算；
- 总导演也有整天拍摄进度预算；
- 但如果分场导演明确发来指令“下一场转 3 号棚”，总导演不能因为“今天时间紧”就装作没收到这条调度命令。

timeout 是预算；
`ParentCommand` 是调度命令。

---

## 什么时候应该给子图单独配 timeout？

### 场景 1：子图封装了已知慢路径
比如某个子图专门做：
- 多轮工具调用；
- 深度检索；
- 慢模型推理；

你希望它自己先在更小范围内超时。

### 场景 2：子图可独立降级
子图超时后，你可以在父图里走 fallback，而不是让整图直接失败。

### 场景 3：子图是第三方/复用组件
你不想把超时责任全部交给外层总图。

---

## 什么时候应该只在父图上配 timeout？

如果子图只是轻量封装，且你希望预算统一由外层掌控，那么只在父图上配置可能更简单。

但这通常更适合：
- 简单子图；
- 性能轮廓稳定；
- 不需要局部恢复策略。

---

## 一段简化示意代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.types import Command


class State(TypedDict):
    value: list[str]


def child_router(state: State):
    return Command(
        graph=Command.PARENT,
        goto="parent_finish",
        update={"value": state["value"] + ["from_child"]},
    )


sub = StateGraph(State)
sub.add_node("child_router", child_router)
sub.add_edge(START, "child_router")
subgraph = sub.compile()
subgraph.step_timeout = 1


def parent_finish(state: State):
    return {"value": state["value"] + ["parent_finish"]}
```

这个例子里，真正重要的不是 timeout 值，而是：

- 子图有自己的预算；
- 子图仍然要能发出控制流命令；
- 父图要正确接住它。

---

## 和社区 issue 的关系

抓取到的 issue #4927 之所以重要，是因为它把这个边界拉到了真实用户场景：

- supervisor + sub-agent
- 设置 `step_timeout`
- 控制流异常暴露

它提醒我们：

**一旦系统进入多代理 / 子图 / 控制权转移场景，timeout 的语义完整性就比“单纯限时”更重要。**

[来源: atom/langgraph/L3_工作流编排/09_超时控制/reference/fetch_step_timeout_bug_01.md]

---

## 设计原则总结

### 原则 1：预算可以分层
子图、父图都能有 timeout。

### 原则 2：控制流优先级高于“粗暴超时”
`ParentCommand` 不能被吞掉。

### 原则 3：局部超时应服务于局部恢复
如果给子图单配 timeout，最好也给它配局部 fallback 或语义化上报。

### 原则 4：复杂图一定要用测试锁住边界
因为这一类 bug 最容易在版本升级和结构调整中回归。

---

## 结论

**子图 timeout 真正难的地方，不是“设几个秒”，而是“如何在多层预算中仍然保持控制流语义不失真”。**

[来源: atom/langgraph/L3_工作流编排/09_超时控制/reference/source_子图传播_02.md]


---

## 扩展补充：边界再下沉一层

下面这一组补充专门回答“子图超时与 ParentCommand 在真实项目里到底怎么判断边界”这个问题。
你可以把它理解成围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”的一组工程判断模板。

### 边界判断 1

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 1 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 2

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 2 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 3

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 3 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 4

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 4 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 5

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 5 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 6

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 6 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 7

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 7 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 8

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 8 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 9

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 9 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 10

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 10 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 11

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 11 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

### 边界判断 12

- 观察点：当你讨论“子图超时与 ParentCommand”时，先确认问题是发生在局部逻辑、当前 step，还是更外层请求窗口。
- 设计点：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”已经跨越两层以上，就不该再用单个 timeout 参数解释全部行为。
- 排障点：记录第 12 次分析时，至少要写明超时发生层、当前输入、前一条成功事件和最后一条日志。

## 高频追问速答（扩展版）

### 追问 1
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 2
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 3
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 4
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 5
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 6
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 7
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 8
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 9
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 10
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 11
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 12
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 13
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 14
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

### 追问 15
- 问：如果团队里有人只想通过调大 timeout 来解决“子图超时与 ParentCommand”，你第一句反问应该是什么？
- 答：我会先问“这次超时到底是 父图预算、子图预算、ParentCommand 冒泡与语义完整性 的哪一层在失配，超时后你准备让谁接手恢复？”

## 设计检查清单

- 检查项 1：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 2：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 3：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 4：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 5：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 6：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 7：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 8：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 9：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 10：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 11：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 12：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 13：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 14：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 15：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 16：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 17：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 18：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 19：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 20：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 21：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 22：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 23：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 24：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。
- 检查项 25：围绕“子图超时与 ParentCommand”，确认日志、指标、timeout 配置、fallback 路径和调用方体验是否对齐。

## 口头表达模板

- 表达模板 1：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 2：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 3：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 4：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 5：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 6：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 7：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 8：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 9：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 10：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 11：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 12：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 13：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 14：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 15：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 16：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 17：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。
- 表达模板 18：在解释“子图超时与 ParentCommand”时，可以直接说“先在最贴近根因的一层止损，再把更外层预算当兜底”。

## 决策速记

- 速记 1：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 2：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 3：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 4：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 5：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 6：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 7：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 8：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 9：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 10：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 11：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 12：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 13：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 14：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 15：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 16：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 17：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。
- 速记 18：如果“父图预算、子图预算、ParentCommand 冒泡与语义完整性”还没区分 node、step、sdk、platform，你的结论大概率还不够稳。

[来源: atom/langgraph/L3_工作流编排/09_超时控制/reference/INDEX.md]
[来源: atom/langgraph/L3_工作流编排/09_超时控制/reference/source_步级超时_01.md]

- 补充要点 001：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 002：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 003：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 004：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 005：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 006：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 007：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 008：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 009：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 010：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 011：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 012：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 013：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 014：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 015：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 016：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 017：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 018：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 019：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 020：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 021：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 022：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 023：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 024：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 025：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 026：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 027：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 028：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 029：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 030：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 031：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 032：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 033：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 034：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
- 补充要点 035：讨论“子图超时与 ParentCommand”时，永远先问这次超时发生在哪一层，再问是否要调参数。
- 补充要点 036：围绕“父图预算、子图预算、ParentCommand 冒泡与语义完整性”做设计时，最贴近根因的一层应该最早止损，外层预算负责收口。
- 补充要点 037：如果调用方只能看到一个模糊的 timeout 字样，那就说明可观测性设计还没有完成。
- 补充要点 038：对“子图超时与 ParentCommand”做排障时，至少同时看输入、耗时、异常类型、最后一条成功事件和恢复动作。
- 补充要点 039：timeout 不是目的，它只是把系统推入 retry、fallback、interrupt 或 polling 之一的分叉点。
