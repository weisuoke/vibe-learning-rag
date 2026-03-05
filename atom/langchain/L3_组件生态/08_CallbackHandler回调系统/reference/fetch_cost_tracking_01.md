---
type: fetched_content
source: https://github.com/FareedKhan-dev/agentic-rag
title: GitHub - FareedKhan-dev/agentic-rag: Agentic RAG to achieve human like reasoning
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
fetch_tool: Grok-mcp web-fetch
knowledge_point_tag: 成本追踪和性能监控
---

# GitHub - FareedKhan-dev/agentic-rag

**Agentic RAG to achieve human like reasoning**

## Building the Enhanced Agentic RAG Pipeline

Standard RAG systems find and summarize facts, but they don't really think. **Agentic RAG goes further** it reads, checks, connects, and reasons, making it feel less like a search tool and more like an expert. The improved workflow adds steps that mimic how humans solve problems. The goal is not just to answer, but to truly understand the question.

### Performance Evaluation (Speed & Cost)

```python
import time
from langchain_core.callbacks.base import BaseCallbackHandler

class TokenCostCallback(BaseCallbackHandler):
    """Callback to track token usage across LLM calls and estimate cost."""

    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.prompt_cost_per_1m = 5.00
        self.completion_cost_per_1m = 15.00

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get('token_usage', {})
        self.total_prompt_tokens += usage.get('prompt_tokens', 0)
        self.total_completion_tokens += usage.get('completion_tokens', 0)

    def get_summary(self):
        prompt_cost = (self.total_prompt_tokens / 1_000_000) * self.prompt_cost_per_1m
        completion_cost = (self.total_completion_tokens / 1_000_000) * self.completion_cost_per_1m
        total_cost = prompt_cost + completion_cost
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": total_cost
        }
```

### Example Output

```
--- Performance Evaluation ---
End-to-End Latency: 24.31 seconds

Cost Summary:
{
  "total_prompt_tokens": 11234,
  "total_completion_tokens": 1489,
  "estimated_cost_usd": 0.078505
}
```

## Key Features for CallbackHandler

This repository demonstrates several important aspects of callback handling in LangChain:

1. **Token Usage Tracking**: Custom callback handler to monitor token consumption
2. **Cost Estimation**: Real-time cost calculation based on token usage
3. **Performance Monitoring**: Latency tracking for end-to-end operations
4. **LLM Call Interception**: Using `on_llm_end` to capture response metadata

## Repository Structure

- **code.ipynb**: Main implementation notebook
- **requirements.txt**: Python dependencies
- **README.md**: Project documentation

## Topics

- openai
- ai-agents
- rag
- llm
- agentic-rag

## Resources

- [Medium Article](https://medium.com/@fareedkhandev/687e1fd79f61)
- [GitHub Repository](https://github.com/FareedKhan-dev/agentic-rag)

## License

MIT license

## Stats

- 190 stars
- 65 forks
- Jupyter Notebook 100.0%

---

**Note**: This is a comprehensive example of building an advanced agentic RAG pipeline with built-in cost tracking and performance monitoring using LangChain's callback system. The full implementation includes multiple phases covering knowledge base construction, specialist agents, reasoning engine, evaluation, and stress testing.

For complete code examples and detailed implementation, please visit the original repository at: https://github.com/FareedKhan-dev/agentic-rag
