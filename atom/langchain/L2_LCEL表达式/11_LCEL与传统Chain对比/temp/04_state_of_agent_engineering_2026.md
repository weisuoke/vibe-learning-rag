# State of Agent Engineering 2026

**Source**: https://www.langchain.com/state-of-agent-engineering
**Fetched**: 2026-02-21
**Survey Period**: November 18 - December 2, 2025
**Respondents**: 1,340 professionals

## Key Findings

### Production Momentum
- **57% have deployed agents in production** (up from 51% in previous year)
- **30.4% actively developing** with clear deployment plans
- Large enterprises (10k+ employees): **67% in production**
- Small organizations (<100): **50% in production**

### Quality is Production Killer
- **32% cite quality as top barrier** (accuracy, relevance, consistency)
- Latency is second challenge (20%)
- **Cost concerns have decreased** compared to previous years

### Observability is Basic Requirement
- **89% have implemented observability** for agents
- **62% have detailed tracing** (inspect individual steps and tool calls)
- **94% of production deployments** have observability
- **71.5% of production deployments** have complete tracing

### Multi-Model Usage is Norm
- **Over 75% use multiple models** in production or development
- OpenAI GPT models dominate (67%+ adoption)
- **33% deploy open source models** internally
- **57% don't perform model fine-tuning** (rely on base models + prompt engineering + RAG)

## Leading Use Cases

1. **Customer Service** (26.5%) - Most common agent use case
2. **Research & Data Analysis** (24.4%)
3. **Internal Workflow Automation** (18%)

**Enterprise (10k+ employees) priorities**:
1. Internal productivity (26.8%)
2. Customer service (24.7%)
3. Research & data analysis (22.2%)

## Biggest Barriers to Production

1. **Quality** (33%) - Accuracy, relevance, consistency, tone, policy compliance
2. **Latency** (20%) - Response time critical for customer-facing use cases
3. **Security** (24.9% for enterprises 2k+) - Becomes second concern for large organizations

**Enterprise-specific challenges**:
- Agent-generated hallucinations
- Output consistency
- Context engineering at scale

## Evaluation Practices

### Offline Evaluation
- **52.4% run offline evaluations** on test sets
- Higher adoption (77.2%) among production deployments

### Online Evaluation
- **37.3% overall adoption**
- **44.8% among production deployments**

### Evaluation Methods
- **Human review** (59.8%) - Critical for nuanced/high-risk situations
- **LLM-as-judge** (53.3%) - Increasingly used for scaling quality assessment
- **Nearly 25% combine both** offline and online evaluation

## Daily Agent Usage

### Top Categories
1. **Coding agents dominate**: Claude Code, Cursor, GitHub Copilot, Amazon Q, Windsurf, Antigravity
2. **Research agents**: ChatGPT, Claude, Gemini, Perplexity for exploring domains, summarizing documents
3. **Custom agents**: Built with LangChain and LangGraph for QA testing, knowledge base search, SQL, customer support

## Model Landscape

### OpenAI Dominance with Diversity
- **67%+ use OpenAI GPT models**
- **75%+ use multiple models** (routing based on complexity, cost, latency)
- **33% invest in open source** model infrastructure

### Fine-tuning Status
- **57% don't perform fine-tuning**
- Rely on base models + prompt engineering + RAG
- Fine-tuning reserved for high-impact or specialized use cases

## Survey Demographics

**Top 5 Industries**:
- Tech (63%)
- Financial Services (10%)
- Healthcare (6%)
- Education (4%)
- Consumer Goods (3%)

**Company Size**:
- <100 people (49%)
- 100-500 (18%)
- 500-2000 (15%)
- 2000-10,000 (9%)
- 10,000+ (9%)
