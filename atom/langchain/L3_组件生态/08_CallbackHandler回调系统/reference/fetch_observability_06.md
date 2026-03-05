---
type: fetched_content
source: https://www.reddit.com/r/LangChain/comments/1neh5sw/what_are_the_best_open_source_llm_observability/
title: What are the best open source LLM observability platforms/packages?
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 可观测性集成
---

# What are the best open source LLM observability platforms/packages? : LangChain

**submitted 5 months ago by** (author not extracted)

Looking to instrument all aspects of LLMs - costs, token usage, function calling, metadata, full text search, etc

all 27 comments

## Comments

### [deleted] 12 points 5 months ago

Langfuse.

### [deleted] 0 points 4 months ago

connect LangFuse with AnannasAI LLM provider in Prod

### [deleted] 1 point 5 months ago

What security reasons are you referring to? Genuinely curious because I'm starting to research self hosting langfuse but not super familiar with the tradeoffs

### [deleted] 0 points 4 months ago

Most companies need the enterprise version because of security and compliance requirements, things like SSO, audit logs, RBAC, and vendor security certifications. The open-source version is great for testing, but large orgs usually can't use it in production without these features.

### [deleted] 3 points 5 months ago

Check out Project Monocle from Linux Foundation. It's built on Otel and has native support for LangChain and most agent frameworks, model inference providers and vector databases.

It's on GitHub -

### [deleted] 1 point 5 months ago

This just created traces for my python app in no time and near zero code experience. Wow!!

### [deleted] 3 points 5 months ago

Check out (maintainer here).

It is based on opentelemetry, which means 1) a battle tested SDK 2) compatible with many auto-instrumentation SDKs (openinference, pydantic ai, openllmetry..) and most frameworks.

It comes with full-text search, and end-to-end integration with prompt management and evals.

### [deleted] 1 point 5 months ago

mlflow is fantastic and has a Langchain connector

### [deleted] 1 point 5 months ago

Arize phoenix

### [deleted] 0 points 5 months ago

Emit a metric and plot on Grafana directly ? Why do you want to instrument metadata and full text search ?

### [deleted] 0 points 5 months ago

Yea I never got why Prometheus/Grafana isn't the obvious answer. Battle treated at the largest of scales and quickest of speeds. There is no other answer lol

### [deleted] 0 points 5 months ago

You can try You'll definitely love this

### [deleted] 0 points 5 months ago

Transparently get agent and LLM traces integrated into any W3C compatible sink for tracing:

### [deleted] 0 points 4 months ago

You can actually do some quick testing to see which one you love best with it's like but for LLM traces. It supports all the major off the shelf observability platforms including custom otel collectors. It allows you to just put your credentials in for any of the LLM observability tools and then it will automatically send your trace data there. It just takes one line of code in your LLM app. Here is the github sdk

### [deleted] 0 points 4 months ago

Here's a concise list of **open-source LLM observability platforms/packages** that cover usage, costs, metadata, and query/search capabilities:

# 1. LangSmith (open-source parts)
* Provides logging, metadata tracking, and evaluation pipelines.
* Supports multi-agent and function-call observability.

# 2. LangChain + LlamaIndex
* Track prompts, responses, token usage.
* Index outputs for semantic/full-text search.
* Highly customizable for multi-step workflows.

# 3. OpenTelemetry + Vector Databases
* Instrument LLM calls, metadata, and custom metrics.
* Combine with Milvus, FAISS, or Pinecone for full-text/semantic search.

# 4. Arize AI (open-source SDK components)
* Focused on embeddings, drift detection, and performance tracking.
* Good for token-level monitoring and evaluation of LLM outputs.

# 5. LangFuse
* Open-source SDK for LLM logging, structured metadata, and function calls.
* Supports aggregation, search, and monitoring at scale.

**TL;DR:** For **full LLM observability** , the most flexible open-source stack combines:

* **Instrumentation:** OpenTelemetry or LangFuse
* **Prompt/response indexing:** LangChain + LlamaIndex
* **Storage/search:** Vector DB (FAISS/Milvus)
* **Monitoring/metrics:** Arize SDK components or custom dashboards

### [deleted] 1 point 3 months ago

Hey ! Have you tested Helicone? Would be intrigued to hear what you liked or not about it.

I lead DevRel there, so want to make sure we're building something that serves your needs!

Lots of what you've mentioned there for others is already included in our platform:

- tracking prompts, responses, token usage, latency, token-level monitoring, tool/function calling, agentic sessions, etc

- custom properties for filtering, aggregation, and evaluation of llm outputs

- caching & rate limiting

- prompt management & versioning tool

- fully open sourced

plus, the integration is done through our AI gateway, so you get all the benefits of an AI gateway by default - automatic fallback when providers are down, uptime & rate limit aware routing, passthrough billing with a single API key, etc..

Anyway, would love to hear if you've tested it and if there's anything we can improve 🙏

### [deleted] 0 points 2 months ago

I haven't tested Helicone extensively yet, but everything you outlined sounds very promising, especially the token-level monitoring, agentic session tracking, and prompt versioning. Those are exactly the kinds of observability and control features that make debugging multi-agent workflows much smoother.

I'm curious how it compares in practice to tools like CoAgent or LangSmith, particularly around end-to-end evaluation and handling drift in production. Once I have some time to test it out, I'll share more detailed feedback. Appreciate the transparency and open-source approach, it definitely makes adoption less risky.

### [deleted] 0 points 3 months ago

if I wanted an AI response, I wouldnt have come to reddit

### [deleted] 0 points 3 months ago

you might wanna check out , theyve built a pretty solid tool for LLM observability, covering usage, costs, and metadata tracking in a clean way.

### [deleted] 0 points 3 months ago

stop spamming everywhere. for everyone anannas is heavily spamming reddit all over the place. Certainly not a trustworthy option!

### [deleted] 0 points 19 days ago

you should check !

### [deleted] 0 points 5 days ago

### [deleted] 0 points 5 days ago

Imo langwatch is the best llm obserability tool

### [deleted] -2 points 5 months ago

litellm?

### [deleted] -3 points 5 months ago

Just add logs, what else do you need? is more work search, decide and implement those fancy frameworks than code something yourself.
