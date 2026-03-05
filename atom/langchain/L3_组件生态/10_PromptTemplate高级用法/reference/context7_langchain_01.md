---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: PromptTemplate高级用法
context7_query: PromptTemplate advanced usage partial variables template composition
---

# Context7 文档：LangChain PromptTemplate

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/

## 关键信息提取

### 1. 基础 PromptTemplate 创建

**来源**：https://docs.langchain.com/oss/python/integrations/llms/google_vertex_ai

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is the meaning of {thing}?")
```

**关键点**：
- 使用 `from_template` 方法创建模板
- 使用 `{variable}` 语法定义变量占位符

### 2. ChatPromptTemplate 动态变量

**来源**：https://docs.langchain.com/oss/python/integrations/chat/perplexity

```python
from langchain_core.prompts import ChatPromptTemplate

chat = ChatPerplexity(temperature=0, model="sonar")
prompt = ChatPromptTemplate.from_messages([("human", "Tell me a joke about {topic}")])
chain = prompt | chat
response = chain.invoke({"topic": "cats"})
```

**关键点**：
- `ChatPromptTemplate` 用于对话场景
- 使用 `from_messages` 创建消息模板
- 支持动态变量传递

### 3. PromptTemplate 与 LLM 链式组合

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
- 支持多个变量
- 通过 `invoke` 方法传递变量值

### 4. TypeScript 中的 PromptTemplate

**来源**：https://docs.langchain.com/oss/javascript/integrations/chat/openai

```typescript
import { ChatPromptTemplate } from "@langchain/core/prompts";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  ["user", "Tell me a joke about {topic}"]
]);

await promptTemplate.invoke({ topic: "cats" });
```

**关键点**：
- TypeScript 版本的 API 与 Python 类似
- 使用 `fromMessages` 创建消息模板

### 5. WatsonxLLM 链式组合

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
- JavaScript 中使用 `pipe` 方法进行链式组合
- API 设计与 Python 版本保持一致

## 需要进一步查询的内容

从 Context7 返回的结果来看，主要是基础用法示例，缺少以下高级特性的详细文档：

1. **Partial Variables（部分变量）**：
   - 如何预填充部分变量
   - `partial` 方法的使用
   - 部分变量的应用场景

2. **Template Composition（模板组合）**：
   - 使用 `+` 操作符组合模板
   - 组合时的变量合并规则
   - 组合时的注意事项

3. **Template Inheritance（模板继承）**：
   - 是否支持模板继承
   - 如何扩展现有模板
   - 继承的最佳实践

4. **高级模板格式**：
   - Jinja2 模板的高级用法
   - Mustache 模板的嵌套变量
   - 不同格式的性能对比

5. **实际应用场景**：
   - 在 RAG 系统中的应用
   - 在 Agent 系统中的应用
   - 多语言模板管理

## 下一步行动

需要使用 Context7 进行更具体的查询：
- 查询 "LangChain PromptTemplate partial"
- 查询 "LangChain PromptTemplate composition"
- 查询 "LangChain PromptTemplate jinja2"

同时需要使用 Grok-mcp 搜索社区资料和实践案例。
