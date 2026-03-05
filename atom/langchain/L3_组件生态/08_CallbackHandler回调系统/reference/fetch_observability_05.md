---
type: fetched_content
source: https://github.com/FareedKhan-dev/production-grade-agentic-system
title: GitHub - FareedKhan-dev/production-grade-agentic-system - Core 7 layers of production grade agentic system
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 可观测性集成
---

# GitHub - FareedKhan-dev/production-grade-agentic-system

**Core 7 layers of production grade agentic system**

[1 Branch](https://github.com/FareedKhan-dev/production-grade-agentic-system/branches)
[0 Tags](https://github.com/FareedKhan-dev/production-grade-agentic-system/tags)

**Go to file** • **Code**

## Folders and files

| Name | Last commit | Message | Commit |
|------|-------------|---------|--------|
| [evals](https://github.com/FareedKhan-dev/production-grade-agentic-system/tree/master/evals) | 2 months ago | Project | d083ecfe |
| [grafana/dashboards](https://github.com/FareedKhan-dev/production-grade-agentic-system/tree/master/grafana/dashboards) | 2 months ago | Project | d083ecfe |
| [prometheus](https://github.com/FareedKhan-dev/production-grade-agentic-system/tree/master/prometheus) | 2 months ago | Project | d083ecfe |
| [scripts](https://github.com/FareedKhan-dev/production-grade-agentic-system/tree/master/scripts) | 2 months ago | Project | d083ecfe |
| [src](https://github.com/FareedKhan-dev/production-grade-agentic-system/tree/master/src) | 2 months ago | Project | d083ecfe |
| [.env.example](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/.env.example) | 2 months ago | Project | d083ecfe |
| [.python-version](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/.python-version) | 2 months ago | Project | d083ecfe |
| [Dockerfile](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/Dockerfile) | 2 months ago | Project | d083ecfe |
| [LICENSE](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/LICENSE) | 2 months ago | Project | d083ecfe |
| [Makefile](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/Makefile) | 2 months ago | Project | d083ecfe |
| [README.md](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/README.md) | 2 months ago | Guide | 20def05 |
| [SECURITY.md](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/SECURITY.md) | 2 months ago | Project | d083ecfe |
| [docker-compose.yml](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/docker-compose.yml) | 2 months ago | Project | d083ecfe |
| [pyproject.toml](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/pyproject.toml) | 2 months ago | Project | d083ecfe |
| [schema.sql](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/schema.sql) | 2 months ago | Project | d083ecfe |
| [uv.lock](https://github.com/FareedKhan-dev/production-grade-agentic-system/blob/master/uv.lock) | 2 months ago | Project | d083ecfe |

**View all files**

## Repository files navigation

# Production-Grade Agentic AI System

Modern **agentic AI systems**, whether running in **development, staging, or production**, are built as a **set of well-defined architectural layers** rather than a single service. Each layer is responsible for a specific concern such as **agent orchestration, memory management, security controls, scalability, and fault handling**. A production-grade agentic system typically combines these layers to ensure agents remain reliable, observable, and safe under real-world workloads.

![Production Grade Agentic System](https://miro.medium.com/v2/resize:fit:2560/1*GB6tXauVBaHVGDE4L_FkYg.png)
*Production Grade Agentic System (Created by Fareed Khan)*

There are **two key aspects** that must be continuously monitored in an agentic system.

1. The first is **agent behavior**, which includes reasoning accuracy, tool usage correctness, memory consistency, safety boundaries, and context handling across multiple turns and agents.
2. The second is **system reliability and performance**, covering latency, availability, throughput, cost efficiency, failure recovery, and dependency health across the entire architecture.

Both are important for operating **multi-agent systems** reliably at scale.

In this blog, we will build all the core architectural layers needed to deploy a production-ready agentic system, **so teams can confidently deploy AI agents in their own infrastructure or for their clients.**

You can clone the repo:

```bash
git clone https://github.com/FareedKhan-dev/production-grade-agentic-system
cd production-grade-agentic-system
```

## Table of Content

- [Creating Modular Codebase](#ab61)
  - [Managing Dependencies](#d7f1)
  - [Setting Environment Configuration](#dfa0)
  - [Containerization Strategy](#66e6)
- [Building Data Persistence Layer](#c31d)
  - [Structured Modeling](#49d1)
  - [Entity Definition](#da20)
  - [Data Transfer Objects (DTOs)](#0bdf)
- [Security & Safeguards Layer](#1942)
  - [Rate Limiting Feature](#1649)
  - [Sanitization Check Logic](#ed53)
  - [Context Management](#2115)
- [The Service Layer for AI Agents](#9ef9)
  - [Connection Pooling](#c497)
  - [LLM Unavailability Handling](#2fe7)
  - [Circuit Breaking](#0d26)
- [Multi-Agentic Architecture](#2767)
  - [Long-Term Memory Integration](#097b)
  - [Tool Calling Feature](#6f9a)
- [Building The API Gateway](#458a)
  - [Auth Endpoints](#8a02)
  - [Real-Time Streaming](#0d8e)
- [Observability & Operational Testing](#86b1)
  - [Creating Metrics to Evaluate](#0055)
  - [Middleware Based Testing](#9c23)
  - [Streaming Endpoints Interaction](#47b1)
  - [Context Management Using Async](#5e3d)
  - [DevOps Automation](#1b72)
- [Evaluation Framework](#ff63)
  - [LLM-as-a-Judge](#9e4a)
  - [Automated Grading](#e936)
- [Architecture Stress Testing](#f484)
  - [Simulating our Traffic](#8f52)
  - [Performance Analysis](#0703)

## Creating Modular Codebase

Normally, Python projects start small and gradually become messy as they grow. When building production-grade systems, developers typically adopt a **Modular Architecture** approach.

This means separating different components of the application into distinct modules. By doing so, it becomes easier to maintain, test, and update individual parts without impacting the entire system.

Let's create a structured directory layout for our AI system:

```bash
├── app/                         # Main Application Source Code
│   ├── api/                     # API Route Handlers
│   │   └── v1/                  # Versioned API (v1 endpoints)
│   ├── core/                    # Core Application Config & Logic
│   │   ├── langgraph/           # AI Agent / LangGraph Logic
│   │   │   └── tools/           # Agent Tools (search, actions, etc.)
│   │   └── prompts/             # AI System & Agent Prompts
│   ├── models/                  # Database Models (SQLModel)
│   ├── schemas/                 # Data Validation Schemas (Pydantic)
│   ├── services/                # Business Logic Layer
│   └── utils/                   # Shared Helper Utilities
├── evals/                       # AI Evaluation Framework
│   └── metrics/                 # Evaluation Metrics & Criteria
│       └── prompts/             # LLM-as-a-Judge Prompt Definitions
├── grafana/                     # Grafana Observability Configuration
│   └── dashboards/              # Grafana Dashboards
│       └── json/                # Dashboard JSON Definitions
├── prometheus/                  # Prometheus Monitoring Configuration
├── scripts/                     # DevOps & Local Automation Scripts
│   └── rules/                   # Project Rules for Cursor
└── .github/                     # GitHub Configuration
    └── workflows/               # GitHub Actions CI/CD Workflows
```

**This directory structure might seem complex to you at first but we are following a generic best-practice pattern** that is used in many agentic systems or even in pure software engineering. Each folder has a specific purpose.

## Managing Dependencies

The very first step in building a production-grade AI system is to create a dependency management strategy.

**Repository metadata:**
- **Topics**: production, langchain, langgraph, agentic-ai
- **License**: MIT
- **Stars**: 536
- **Forks**: 125
- **Watchers**: 2
- **Languages**: Python 84.9%, Shell 9.9%, Makefile 4.4%, Dockerfile 0.8%

**Note**: This is a comprehensive GitHub repository showcasing production-grade agentic AI system architecture with 7 core layers including observability, evaluation, and operational testing. The full README contains extensive code examples, architecture diagrams, and implementation details for building scalable multi-agent systems.
