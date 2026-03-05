---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: PromptTemplate高级用法
context7_query: PromptTemplate partial variables
---

# Context7 文档：LangChain PromptTemplate Partial Variables

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/

## 关键信息提取

### 1. TypeScript PromptTemplate 创建

**来源**：https://docs.langchain.com/oss/javascript/integrations/chat/anthropic

```typescript
import { PromptTemplate } from "@langchain/core/prompts";

const promptTemplate = PromptTemplate.fromTemplate(
  "Tell me a joke about {topic}"
);

await promptTemplate.invoke({ topic: "cats" });
```

**关键点**：
- 使用 `fromTemplate` 方法创建模板
- 使用 `invoke` 方法传递变量

### 2. GoogleDriveLoader 自定义搜索模板

**来源**：https://docs.langchain.com/oss/python/integrations/document_loaders/google_drive

```python
from langchain_core.prompts.prompt import PromptTemplate

loader = GoogleDriveLoader(
    folder_id=folder_id,
    recursive=False,
    template=PromptTemplate(
        input_variables=["query", "query_name"],
        template="fullText contains '{query}' and name contains '{query_name}' and trashed=false",
    ),  # Default template to use
    query="machine learning",
    query_name="ML",
    num_results=2,  # Maximum number of file to load
)
for doc in loader.load():
    print("---")
    print(doc.page_content.strip()[:60] + "...")
```

**关键点**：
- 显式指定 `input_variables`
- 使用多个变量的模板
- 在实际应用中使用 PromptTemplate

### 3. 模板变量添加

**来源**：https://docs.langchain.com/langsmith/create-a-prompt

**关键点**：
- 使用 `{variable_name}` 添加变量（f-string 格式，单花括号）
- 使用 `{{variable_name}}` 添加变量（mustache 格式，双花括号）
- 可以高亮文本并转换为变量

### 4. String PromptTemplates

**来源**：https://docs.langchain.com/oss/javascript/integrations/chat/openai

**关键点**：
- String PromptTemplates 用于格式化单个字符串
- 适用于简单输入场景
- 提供变量占位符，可在运行时填充
- 适合基于用户输入或上下文参数生成变化的 prompt

## 缺失的高级特性

从 Context7 返回的结果来看，仍然缺少以下高级特性的详细文档：

1. **Partial Variables（部分变量）**：
   - 没有找到 `partial` 方法的使用示例
   - 没有找到预填充部分变量的文档
   - 没有找到部分变量的应用场景

2. **高级用法**：
   - 没有找到模板组合的详细文档
   - 没有找到模板继承的文档
   - 没有找到不同模板格式的对比

## 下一步行动

需要使用 Grok-mcp 搜索社区资料和实践案例，特别是：
- LangChain PromptTemplate partial variables 实践案例
- LangChain PromptTemplate composition 实践案例
- LangChain PromptTemplate advanced usage 2025-2026
