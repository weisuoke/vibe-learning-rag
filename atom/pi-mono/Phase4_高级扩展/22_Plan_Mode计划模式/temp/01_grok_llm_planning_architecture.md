# LLM-based Planning Systems Architecture (2025-2026)

**Fetched:** 2026-02-21

## Key Research Findings

### 1. Brain-inspired Agentic Architecture (Nature, 2025)
**Source:** https://www.nature.com/articles/s41467-025-63804-5

提出模块化代理规划器MAP架构，通过模拟大脑前额叶功能的专用LLM模块交互进行规划，包括错误监控、动作提议、状态预测等，提升复杂任务规划能力。

**Key Concepts:**
- Modular Agent Planner (MAP) architecture
- Brain prefrontal cortex simulation
- Error monitoring modules
- Action proposal systems
- State prediction mechanisms

### 2. System Architecture for Agentic LLMs (Berkeley, 2025)
**Source:** https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-5.pdf

伯克利博士论文探讨代理型LLM系统架构，涵盖模拟环境、训练算法、工具使用和系统保障，重点介绍LEAP和TEMPERA等高级反思与规划框架。

**Key Frameworks:**
- LEAP: Advanced reflection framework
- TEMPERA: Planning framework
- Simulation environments
- Training algorithms
- Tool usage patterns
- System guarantees

### 3. TodoEvolve: Learning to Architect Agent Planning Systems (2026)
**Source:** https://arxiv.org/pdf/2602.07839

2026年论文提出TodoEvolve方法，通过学习自动构建LLM代理规划系统架构，分析现有规划系统多样性并提升代理规划能力。

**Key Insights:**
- Automated architecture construction
- Planning system diversity analysis
- Learning-based approach to planning

### 4. Comprehensive Review of LLM Planning (2025)
**Source:** https://arxiv.org/pdf/2505.19683

全面综述LLM用于规划的研究现状，包括方法论、应用和未来方向，系统整理LLM-based planning领域的进展。

### 5. AI Agent Architecture Guide (Redis, 2026)
**Source:** https://redis.io/en/blog/ai-agent-architecture

2026年AI代理架构指南，详述生产级代理系统组件，包括记忆系统、推理引擎、多代理协调和规划执行模式。

**Production Components:**
- Memory systems
- Reasoning engines
- Multi-agent coordination
- Planning-execution patterns

### 6. Plan-and-Act Framework (ICML 2025)
**Source:** https://icml.cc/virtual/2025/poster/43522

提出Plan-and-Act框架，将高级规划与低级执行分离，通过合成数据增强规划生成，适用于长时域任务的LLM代理。

**Key Pattern:**
- Separation of high-level planning and low-level execution
- Synthetic data for planning enhancement
- Long-horizon task support

### 7. Best Open Source LLMs for Planning (2026)
**Source:** https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Planning-Tasks

评测2026年最佳开源LLM用于规划任务，重点介绍DeepSeek-R1、Qwen3系列等模型在任务分解、多步推理和工具编排方面的架构优势。

**Top Models:**
- DeepSeek-R1
- Qwen3 series
- Task decomposition capabilities
- Multi-step reasoning
- Tool orchestration

## Architectural Patterns

### Pattern 1: Modular Planning
- Separate specialized modules for different planning aspects
- Error monitoring, action proposal, state prediction
- Brain-inspired architecture

### Pattern 2: Plan-then-Execute
- Clear separation between planning and execution phases
- Planning generates high-level strategy
- Execution handles low-level implementation

### Pattern 3: Reflection-Enhanced Planning
- LEAP and TEMPERA frameworks
- Continuous reflection on planning quality
- Adaptive planning based on feedback

### Pattern 4: Learning-Based Architecture
- TodoEvolve approach
- Automated architecture construction
- Diversity-aware planning systems

## Relevance to Pi-mono

Pi-mono's philosophy aligns with these architectural patterns:
- **Modular design**: Extensions as specialized planning modules
- **Separation of concerns**: Planning vs. execution separation
- **Observability**: File-based planning for transparency
- **Flexibility**: User-controlled planning strategies
