---
type: fetched_content
source: https://github.com/agentreplay/agentreplay
title: GitHub - agentreplay/agentreplay: Local-First Desktop Observability & AI Memory for Your Agents and Coding Tools
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 可观测性集成
---

# agentreplay/agentreplay

**Local-First Desktop Observability & AI Memory for Your Agents and Coding Tools**

[3 Branches](https://github.com/agentreplay/agentreplay/branches) [5 Tags](https://github.com/agentreplay/agentreplay/tags)

**License**: AGPL-3.0
**Primary Language**: Rust (62.3%)
**Stars**: 4
**Forks**: 1
**Watching**: 1

### Navigation
- **Code** (current)
- Issues
- Pull requests
- Actions
- Projects
- Security
- Insights

---

## Folders and files

| Name | Last commit message | Last commit date |
|------|---------------------|------------------|
| [.azure-pipelines](https://github.com/agentreplay/agentreplay/tree/main/.azure-pipelines) | Taur updated release check version bump to 0.2.2 | last week Feb 17, 2026 |
| [.github](https://github.com/agentreplay/agentreplay/tree/main/.github) | Taur updated release check version bump to 0.2.2 | last week Feb 17, 2026 |
| [.vite/deps](https://github.com/agentreplay/agentreplay/tree/main/.vite/deps) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-cli](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-cli) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-core](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-core) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-evals](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-evals) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-experiments](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-experiments) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-index](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-index) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-memory](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-memory) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-observability](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-observability) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-plugins](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-plugins) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-prompts](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-prompts) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-query](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-query) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-server](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-server) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-skill-tester](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-skill-tester) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-storage](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-storage) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [agentreplay-tauri](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-tauri) | Taur updated release check version bump to 0.2.2 | last week Feb 17, 2026 |
| [agentreplay-telemetry](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-telemetry) | Updated terms and bug related python sdk. | 2 weeks ago Feb 8, 2026 |
| [agentreplay-ui](https://github.com/agentreplay/agentreplay/tree/main/agentreplay-ui) | Taur updated release check version bump to 0.2.2 | last week Feb 17, 2026 |
| [examples](https://github.com/agentreplay/agentreplay/tree/main/examples) | Updated code. | 3 weeks ago Feb 1, 2026 |
| [learn-doc](https://github.com/agentreplay/agentreplay/tree/main/learn-doc) | Added skill tester and performance improvements | last week Feb 17, 2026 |
| [mcp_test](https://github.com/agentreplay/agentreplay/tree/main/mcp_test) | Improved the performance. | 2 weeks ago Feb 11, 2026 |
| [sdks](https://github.com/agentreplay/agentreplay/tree/main/sdks) | Taur updated release check version bump to 0.2.2 | last week Feb 17, 2026 |

---

# Agent Replay

### Local-First Desktop Observability & AI Memory for Your Agents and Coding Tools.

**No Docker. No servers. No cloud. Just run.**

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Performance](#-performance) • [Contributing](#-contributing)

---

## Why Agent Replay?

| What We're NOT | What We ARE |
|----------------|-------------|
| ~~Docker containers~~ | **Native desktop app** - double-click and run |
| ~~Web servers to run~~ | **Everything built-in** - storage, UI, APIs |
| ~~Cloud accounts required~~ | **Unlimited local memory** - use your full disk |
| ~~Complex infrastructure~~ | **Works with Claude Code, Cursor, Windsurf, Cline** |
| ~~Memory limits~~ | **Zero configuration** - start tracing in seconds |
| ~~Monthly subscriptions~~ | **100% offline capable** - your data stays local |

> **Built for AI coding agents**
> Agent Replay gives your tools like Claude Code, Cursor, and VS Code agents persistent memory and full observability without cloud dependencies.

---

## Quick Start

### Download Agent Replay Desktop

| Platform | Download | Architecture |
|----------|----------|--------------|
| **macOS** | [Agent Replay.dmg](https://github.com/agentreplay/agentreplay/releases/latest/download/Agent.Replay.Alpha_aarch64.dmg) | Apple Silicon (M1/M2/M3/M4) |
| **macOS** | [Agent Replay.dmg](https://github.com/agentreplay/agentreplay/releases/latest/download/Agent.Replay.Alpha_x64.dmg) | Intel |
| **Linux** | [Agent Replay.AppImage](https://github.com/agentreplay/agentreplay/releases/latest/download/Agent.Replay.Alpha_amd64.AppImage) | x86_64 |
| **Linux** | [Agent Replay.deb](https://github.com/agentreplay/agentreplay/releases/latest/download/Agent.Replay.Alpha_amd64.deb) | x86_64 (Debian/Ubuntu) |
| **Windows** | [Agent Replay Setup.exe](https://github.com/agentreplay/agentreplay/releases/latest/download/Agent.Replay.Alpha_x64-setup.exe) | x86_64 |

### Or Build from Source

```bash
# Clone and run (that's it!)
git clone https://github.com/agentreplay/agentreplay.git
cd agentreplay
./run-tauri.sh
```

**That's it.** No Docker. No `docker-compose up`. No environment variables. No database setup. Just a native app with everything inside.

---

## Overview

Agent Replay is a **local-first desktop application** that gives your AI agents and coding tools:

- **Unlimited persistent memory** - stored on your machine, not in the cloud
- **Full observability** - see every decision, tool call, and reasoning step
- **Instant performance** - native desktop app, not a browser tab
- **Complete privacy** - your conversations and code never leave your machine

### Works With Your Favorite AI Tools

| Tool | Integration | Status |
|------|-------------|--------|
| **Claude Code** | [Native Plugin](https://github.com/agentreplay/agentreplay-claude-plugin) | Ready |
| **Cursor** | MCP + Extension | Ready |
| **Windsurf** | MCP server | Ready |
| **Cline** | MCP server | Ready |
| **VS Code + Copilot** | Extension | Ready |
| **Custom Agents** | Python/JS/Rust SDK | Ready |

### Claude Code Plugin (Recommended)

```bash
# Add the marketplace
/plugin marketplace add agentreplay/agentreplay-claude-plugin

# Install the plugin
/plugin install agentreplay
```

**Available Commands:**
- `/agentreplay:dashboard` - Open dashboard
- `/agentreplay:status` - Check server health
- `/agentreplay:remember [query]` - Search memories
- `/agentreplay:traces [count]` - List recent traces

### Powered by SochDB - Everything Built-In

| Feature | Benefit |
|---------|---------|
| **Embedded database** | No external services needed |
| **LSM-tree storage** | Write-optimized for trace ingestion |
| **Columnar storage** | 80% less I/O for analytics |
| **Vector indexes** | Semantic search over your agent memory |
| **ACID transactions** | Crash-safe, no data loss |

---

## Features

### AI Agent Memory (RAG Built-In)
- Semantic memory storage
- Instant retrieval
- Session continuity
- Cross-session learning
- No token limits
- HNSW/Vamana indexes

### Full Observability
- Every tool call traced
- Reasoning chains visualized
- Token usage tracked
- Cost analytics
- OTLP ingestion

### Multi-Provider LLM Support
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini Pro, Gemini Ultra)
- DeepSeek
- Ollama (local models)
- Mistral
- Cohere
- Custom endpoints

### Model Comparison Engine
- Parallel execution
- Independent streaming
- Cost comparison
- Latency tracking

### Evaluation Framework (20+ Evaluators)

**Quality Evaluators:**
- Correctness
- Relevance
- Coherence
- Fluency
- Groundedness

**Safety Evaluators:**
- Toxicity detection
- Bias detection
- PII detection
- Prompt injection detection

**Performance Evaluators:**
- Latency
- Token efficiency
- Cost per query

**RAG-Specific:**
- Context relevance
- Answer relevance
- Faithfulness
- Context recall

**Advanced Features:**
- Custom evaluators
- Batch evaluation
- A/B testing
- Regression detection

### Prompt Registry & Versioning
- Automatic versioning
- Template variables
- Semantic versioning
- Traffic splitting
- Rollback support
- Performance tracking

### Plugin System
- Install from directory/file
- Dev mode with hot-reload
- SDK for Python/Rust/JS
- Event hooks
- Custom UI components

### Analytics & Dashboards
- Time-series metrics
- True percentiles (DDSketch)
- Unique counts (HyperLogLog)
- Custom dashboards
- Export to CSV/JSON

### Backup & Restore
- One-click backup
- Export/Import ZIP
- Merge mode
- Incremental backups

### Core Capabilities

**High-Performance Ingestion**
- 10,000 spans/min on laptop
- Async batching
- Zero-copy serialization

**Native Causal Graph Support**
- Parent-child relationships
- Span context propagation
- Distributed tracing

**Comprehensive Evaluation Framework**
- 20+ built-in evaluators
- Custom evaluator support
- Batch evaluation
- A/B testing

**Cost Intelligence**
- Per-request cost tracking
- Provider comparison
- Budget alerts
- Cost optimization suggestions

**A/B Testing**
- Traffic splitting
- Statistical significance
- Automatic winner selection

**Powerful Query Engine**
- SQL-like syntax
- Vector similarity search
- Full-text search
- Time-range queries

### Desktop-First Architecture
- Double-click to run
- Everything embedded
- Cross-platform (Windows/macOS/Linux)
- System tray
- Embedded HTTP server + OTLP
- Auto-updates

### SDKs & Integrations

**Python SDK:**
- LangChain
- LlamaIndex
- AutoGen
- CrewAI
- Haystack
- DSPy
- Semantic Kernel
- OpenAI SDK
- Anthropic SDK
- Raw Python

**JavaScript/TypeScript SDK:**
- LangChain.js
- Vercel AI SDK
- OpenAI SDK
- Anthropic SDK

**Rust SDK:**
- Native Rust support
- Zero-cost abstractions

**Go SDK:**
- Coming soon

### Enterprise Features

**Compliance Reporting:**
- Audit logs
- Data lineage
- GDPR compliance

**Advanced Analytics:**
- Custom metrics
- Anomaly detection
- Predictive analytics

**Security & Governance:**
- Role-based access
- Data encryption
- Secure storage

---

**Star us on GitHub if you find Agent Replay useful!**

Made with ❤️ by the Agent Replay team
