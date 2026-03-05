---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1f6kl5z/whats_more_important_observability_or_evaluations/
title: What's more important? Observability or Evaluations?
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 可观测性集成
---

# r/LangChain

## What's more important? Observability or Evaluations?

**提交者**: u/kiiwiip
**分数**: 4 points (100% upvoted)
**提交时间**: 01 Sep 2024

I am wondering what's more important when you are building apps using LLMs? I have realized having a good observability lets me understand what's going on and generally eye ball and understand how well my app is doing or the model is generating responses.

I am able to optimize and iterate based on this. Which brings to my question as to whether evals are really needed? Or is it more relevant for more complicated workflows? What are your thoughts?

## 评论 (6 comments)

### 评论 1
**作者**: u/[unknown]
**分数**: 4-6 points
**时间**: 1 year ago

You need evals to track how changes impact overall metrics, while observability lets you investigate particular incidents. They are complementary.

Heres an example:

You have a SQL agent that is failing to answer some query. You add a column selection tool to reduce noise in the query building prompt. Now the agent answers correctly!

But how do you know if that change makes other queries fail? Maybe it's choosing the wrong columns in some cases and failing where it was right before.

Evals would tell you, "Your change reduced precision by 10% but increased accuracy by 20!"

Observability can then tell you, "This is one case where your agent made a mistake. This is what tools it used and what the final response was."

Based on that analysis, you can decide which changes to keep with confidence that they will make your rag/agent better overall.

> **回复** (1 point, 1 year ago):
> Great thanks for explaining!

### 评论 2
**作者**: [deleted]
**分数**: 1-3 points
**时间**: 1 year ago

One lets you see where your bottlenecks or weak links are, one lets you measure the impact of change.

> **回复** (0-2 points, 1 year ago):
> What does observability mean to you?

>> **回复** (0-2 points, 1 year ago):
>> ability to gather insights from all parts of my stack - request flows, request parameters, responses, status, latency etc.

### 评论 3
**作者**: u/[unknown]
**分数**: 1-3 points
**时间**: 1 year ago

Personally, I think both observability and evaluations are important when building apps using LLMs. Observability helps me to understand the overall performance of my app and make necessary adjustments to optimize its performance. On the other hand, evaluations provide a more detailed and precise analysis of the app's performance, especially for more complicated workflows. In my experience, a combination of both has been the most effective approach. Observability gives me a general understanding of how my app is doing, while evaluations provide specific insights and help me fine-tune my app for better results. So, in my opinion, it's important to have a balance of both observability and evaluations to ensure the success of your app.
