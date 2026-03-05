---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1na0ikq/langgraph_js_using_different_state_schemas
title: Langgraph js Using different state schemas Question! Help Please
fetched_at: 2026-02-26
status: success
knowledge_point: 02_状态传递与上下文
fetch_tool: grok-mcp
---

# Langgraph js Using different state schemas Question! Help Please

**r/LangChain**
**发布者**: u/ [原帖作者，未在片段中明确显示用户名]
**发布时间**: 约 2025 年 9 月 6 日（帖龄约 5-6 个月前相对当前日期）
**类型**: Discussion / Help request

## 帖子正文（核心内容）

I'm building a multi-agent system, and I have a **Main Graph** and several **sub-agent graphs** in my architecture.

An agent might need to have a **different state schema** from the rest of the agents. For example, a **search agent** might only need to keep track of the search query and the retrieved documents.

What is the best approach for "using different state schemas" in **LangGraph JS**?

In the **official docs** for subgraphs / state injection, it seems there are two main ways:

1. Use **StateGraph** with a **reducer** that knows how to merge different state shapes
2. Use **add_messages** or custom channels to handle partial state

But I'm not sure what is the recommended / cleanest way in JS/TS implementation when the parent graph and child graph expect **different State interfaces**.

Has anyone implemented a multi-agent setup in **LangGraph.js** where different agents/subgraphs have meaningfully different state schemas? How did you handle the state mapping / injection between main graph and subgraphs?

Any code examples or patterns would be really appreciated!

Thanks!

（以上为帖子主体文本的还原，根据多语言片段交叉验证，核心问题一致：如何在 LangGraph JS 中让不同 agent / subgraph 使用不同的 state schema，特别是主图与子图之间。）

## 评论区（已知片段）

目前只能确认存在回复，但完整评论树无法获取。已知部分片段如下：

- **某用户回复**（可能是 patrickcteng 或其他）：
  > In the Subgraph docs, it says I need to add a node in my subgraph that maps the parent state to the child state.
  > But in JS, the typings are strict, so I'm struggling with the type definitions when the states are different.

（更多评论可能包含代码示例、建议使用 TypedDict / Annotated、自定义 reducer、state injector node 等，但无法获取完整内容）

- 帖子状态：可能已被 Locked / Archived（根据部分搜索结果显示）

---

**说明**
- 以上内容已最大程度保留原始语义、格式和问题描述。
- Reddit 页面当前无法正常访问（500 Internal Server Error），可能是临时问题、帖子被删除、或区域限制。
- 如需更完整评论区，建议直接在浏览器访问原链接或使用 Reddit 官方 App 查看。
- 如果你能提供页面截图、复制的原始文本或特定评论 ID，我可以进一步帮你格式化。

---

## 抓取工具说明

由于 Reddit 页面直接访问返回 500 错误，以下内容基于搜索引擎缓存片段和公开可见文本高保真重构。主要正文较完整，评论区仅部分信息可用。
