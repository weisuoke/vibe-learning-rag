---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: PromptTemplate高级用法
context7_query: PromptTemplate template composition combine
---

# Context7 文档：LangChain PromptTemplate Composition

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/

## 关键信息提取

### 1. WatsonxLLM 链式组合

**来源**：https://docs.langchain.com/oss/javascript/integrations/llms/ibm

```javascript
import { PromptTemplate } from "@langchain/core/prompts"

const prompt = PromptTemplate.fromTemplate("How to say {input} in {output_language}:\n")

const chain = prompt.pipe(instance);
await chain.invoke(
  {
    output_language: "German",
    input: "I love programming.",
  }
)
```

**关键点**：
- 使用 `pipe` 方法进行链式组合
- 支持多个变量

### 2. OpenAI LLM 链式组合

**来源**：https://docs.langchain.com/oss/python/integrations/llms/openai

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)
```

**关键点**：
- 使用 `|` 操作符进行链式组合
- 展示了 LCEL 表达式的基础用法

### 3. Chaining 概念

**来源**：https://docs.langchain.com/oss/javascript/integrations/llms/ibm

**关键点**：
- LangChain 支持使用 pipe 操作符进行链式组合
- 可以将 PromptTemplate 与模型实例组合
- 支持定义参数化的 prompt
- 支持复杂的多步操作

### 4. OpenAI 集成 Chaining

**来源**：https://docs.langchain.com/oss/python/integrations/llms/openai

**关键点**：
- LangChain 支持将 OpenAI 模型与其他组件（如 PromptTemplate）进行链式组合
- 使用 pipe 操作符构建复杂工作流
- 在发送到模型之前通过 prompt template 处理输入

### 5. ChatPromptTemplates

**来源**：https://docs.langchain.com/oss/javascript/integrations/chat/openai

**关键点**：
- ChatPromptTemplates 用于格式化消息数组
- 由模板数组组成
- 调用时构造多个不同角色的消息（system、user 等）
- 每个消息可以有自己的变量进行格式化
- 支持复杂的多消息 prompt 结构

## 缺失的高级特性

从 Context7 返回的结果来看，主要是链式组合（Chaining）的基础用法，缺少以下高级特性：

1. **模板组合（Template Composition）**：
   - 没有找到使用 `+` 操作符组合模板的文档
   - 没有找到组合时变量合并规则的文档
   - 没有找到组合时的注意事项

2. **模板继承（Template Inheritance）**：
   - 没有找到模板继承的文档
   - 没有找到如何扩展现有模板的文档

3. **高级组合模式**：
   - 没有找到复杂组合场景的实践案例
   - 没有找到组合时的性能优化建议

## 下一步行动

需要使用 Grok-mcp 搜索社区资料和实践案例，特别是：
- LangChain PromptTemplate + operator 实践案例
- LangChain PromptTemplate composition patterns
- LangChain PromptTemplate advanced composition 2025-2026
