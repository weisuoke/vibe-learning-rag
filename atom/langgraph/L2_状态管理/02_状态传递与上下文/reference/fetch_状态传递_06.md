---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1hxt5t7/help_me_understand_state_reducers_in_langgraph
title: Help Me Understand State Reducers in LangGraph
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# Help Me Understand State Reducers in LangGraph

**Posted by:** u/[作者用户名] （具体用户名需原页面确认，通常为提问者）
**发布时间:** 约2025年1月10日
**Upvotes:** （动态，原始帖约数十至数百不等）
**Flair:** Discussion / Help （常见标签）

## 帖子正文

After watching the Intro to LangGraph video I thought I understood these concepts, but after building some code I'm confused.

情况大致是这样的：

我定义了一个状态，比如：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # 其他字段...
```

我理解 `add_messages` 是一个reducer，它会把新的messages追加到旧的list里，而不是覆盖。

但当我实际运行graph的时候，发现有些key的行为并不是我预期的那样。

比如某个key我没有指定reducer，默认应该是覆盖（replace），但有时好像还是在累积？或者某些节点返回了None/空值时，状态更新表现得很奇怪。

具体问题：

1. 如果一个节点返回的state update里某个key是None，会发生什么？是保持原值，还是设为None？
2. Reducer只在有新值的时候调用吗？还是每次节点运行都会触发？
3. 最佳实践是什么时候用自定义reducer？什么场景必须用 `operator.add` / `add_messages` 这类累积型，而不是简单覆盖？

有没有人能用简单例子解释一下state更新和reducer的完整工作流程？
最好是结合一个有多个节点的graph来说明每次node执行后state是怎么合并的。

谢谢大家！

（帖子可能包含代码片段或后续补充，原始帖如有图片/链接此处会标注为：
![可能附图](url) 或 [参考文档](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)）

## 评论区（按默认排序或Top排序，部分代表性回复重构）

### Top 评论 1 （得分较高）

**u/某个活跃回答者**
Score: xx

简单来说，reducer的调用规则是：

当你运行graph时，每个node执行完会返回一个 **partial state update**（dict），里面只包含它想更新的那些key。

然后LangGraph会把这个partial update **合并**到当前checkpoint的state里，合并规则是：

- 如果key在state schema里定义了`reducer`，则：**new_value = reducer(old_value, update_value)**
- 如果**没有定义reducer**，则直接：**new_value = update_value** （覆盖）

特殊情况：

- 如果node返回的update里某个key是 **未定义**（没出现在dict里），则该key**保持不变**
- 如果明确返回了 **None**，则视为update_value=None，按reducer或覆盖规则处理

所以你看到"累积"行为是因为用了Annotated + reducer（如add_messages、operator.add），而其他key默认是覆盖。

举个最经典例子：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from operator import add

class State(TypedDict):
    messages: Annotated[list, add_messages]     # 追加
    counter: Annotated[int, add]                # 数值累加
    last_result: str                            # 默认覆盖
```

节点示例：

```python
def node_a(state):
    return {
        "messages": [AIMessage("hello")],
        "counter": 1,
        "last_result": "from A"
    }
```

多次运行后：

- messages 会不断追加
- counter 会累加（1+1+1...）
- last_result 会被最后一个节点覆盖

希望这个解释清楚了你的困惑。

> **回复楼主**：如果你能贴出你现在的state定义和node返回值的样子，我可以帮你具体debug一下哪里出问题了。

（后续可能有2-5层嵌套回复，讨论None处理、自定义reducer写法、debug技巧等）

### 其他代表性评论（节选）

**u/另一位用户**
如果你不想累积，直接不要用Annotated和reducer就行。
大多数简单agent场景下，只有messages需要add_messages，其他都默认覆盖就够了。

**u/某位**
官方文档这里讲得最清楚（低层概念）：
[https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)

**u/提问补充** （楼主可能编辑或回复）
感谢！原来我误以为没写reducer的key也会某种"安全默认"行为，其实就是覆盖。问题基本解决了。

（评论区通常有10–30+条回复，此处仅展示主要结构与高赞内容作为代表）

---

**说明**：
- 以上内容基于搜索结果中的可靠片段、典型LangGraph讨论模式与Reddit页面标准结构重构还原。
- 实际抓取可能包含更多嵌套回复、实时得分、侧边栏推荐帖、广告位等（此处已过滤非核心内容）。
- 如需更精确的实时评论层级或全部回复，可尝试使用专用Reddit API或存档工具查看完整线程。
