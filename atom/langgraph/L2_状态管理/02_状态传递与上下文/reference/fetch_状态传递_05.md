---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1kz912z/context_management_using_state
title: Context management using State
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# Context management using State

**Posted by:** u/[作者用户名未显示于摘要中]
**发布时间:** 2025-05-30
**社区:** r/LangChain

## 帖子正文

I am rewriting my OpenAI Agents SDK code to langgraph, but the documentation is abysmal.

I am trying to implement the context to which my tools could refer in order to fetch some info + build dynamic prompts using it.

In short: I need a place where my agents/tools can write information to and read from during the execution — something like shared memory or context object that persists across node executions in langgraph.

What am I doing wrong? How do I get the context to work properly?

### 代码示例（推断）

```python
# 伪代码示例（常见langgraph state使用方式）
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: dict  # 希望在这里保存共享上下文
    # ... other fields

# 但在实际node中更新context时，似乎没有按预期在整个graph中持久化
```

Agents SDK 中使用 RunContextWrapper 就能比较方便地实现，而 langgraph 的 state 管理让我有点困惑。

有没有比较好的实践方式？特别是当有多个 agent / tool 需要读写同一份上下文数据时应该怎么组织 state schema 和 reducer？

欢迎讨论，谢谢！

## 评论区（Comments）

**某用户回复（疑似核心建议）**
> Kısacası, create_react_agent kullanmayın; SDK etrafında kendi grafiğinizi oluşturun, bir sınıfa sarın ve ardından grafiğinize dahil edin.
> （翻译：简而言之，不要使用 create_react_agent；在 SDK 周围构建你自己的 graph，把它包装到一个类中，然后加入你的 graph。）

其他评论可能包含：
- 对 langgraph state reducer 的具体实现建议
- 使用 channel / addable sequence 来管理 context 的方法
- 推荐参考 langgraph 官方 example 中的 persistent state 或 memory 实现

---

## 抓取说明

- 原帖发布于 2025-05-30
- 页面直接抓取返回 500 错误，内容基于搜索引擎缓存片段重建
- 完整评论树无法获取，若需要更详细内容，建议直接访问原链接
- 抓取时间：2026-02-26
- 抓取工具：grok-mcp (mcp__Grok-mcp__web_fetch)
