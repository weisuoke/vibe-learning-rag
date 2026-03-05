---
type: search_result
search_query: LangChain PromptTemplate partial variables advanced usage 2025 2026
search_engine: grok-mcp
searched_at: 2026-02-26
knowledge_point: PromptTemplate高级用法
---

# 搜索结果：LangChain PromptTemplate Partial Variables

## 搜索摘要

搜索关键词：LangChain PromptTemplate partial variables advanced usage 2025 2026
搜索平台：GitHub, Reddit, Twitter
搜索结果数：9 个

## 相关链接

1. **[LangChain PromptTemplate官方文档 - partial_variables](https://reference.langchain.com/v0.3/python/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html)**
   - LangChain核心PromptTemplate类文档，详细说明partial_variables参数用于预填充模板变量，支持字符串和函数形式，避免每次调用重复传入固定值

2. **[How to work with partial Prompt Templates](https://lagnchain.readthedocs.io/en/latest/modules/prompts/prompt_templates/examples/partial.html)**
   - LangChain旧版文档示例，展示partial_variables的使用，包括字符串固定值和函数动态值（如当前日期）的两种方式

3. **[Using partial_variables - LangChain OpenTutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/02-prompt/01-prompttemplate)**
   - 教程详细解释partial_variables在PromptTemplate中的高级应用，特别适合共享常见变量如日期时间等场景（2025更新）

4. **[Mastering PromptTemplates in LangChain](https://bhavikjikadara.medium.com/mastering-prompttemplates-in-langchain-74f679c467ec)**
   - Medium文章介绍Partial PromptTemplate高级用法，支持部分填充变量并延迟完成，常用于语言翻译等动态场景

5. **[Partial values in LangChainGo](https://tmc.github.io/langchaingo/docs/modules/model_io/prompts/prompt_templates/partial_values)**
   - LangChain Go版本文档，展示partial formatting支持字符串和函数两种方式，实现模板部分预绑定

6. **[LangChain Prompt Templates Complete Guide](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples)**
   - 全面指南解释使用partial()方法创建模板层次结构，减少冗余并优化开发流程（2026更新）

7. **[partial_variables not working with ChatPromptTemplate Issue](https://github.com/langchain-ai/langchain/issues/17560)**
   - GitHub issue讨论ChatPromptTemplate中partial_variables兼容性问题及历史版本行为差异

8. **[Dynamic Prompts with LangChain Templates](https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates--71d0c244)**
   - 文章探讨LangChain模板的高级变量注入，包括复杂数据结构和动态partial使用（2025更新）

9. **[Prompt Templates & Output Parsers Tutorial](https://langchain-tutorials.com/lessons/langchain-essentials/lesson-6)**
   - 教程展示Partial Templates高级模式，用于预填充常量变量如日期、系统设置，提升模板复用性

## 关键信息提取

### 1. Partial Variables 核心概念

**来源**：LangChain 官方文档

**关键点**：
- `partial_variables` 参数用于预填充模板变量
- 支持两种形式：
  - **字符串固定值**：预设常量
  - **函数动态值**：运行时计算（如当前日期）
- 避免每次调用重复传入固定值

### 2. 使用场景

**来源**：LangChain OpenTutorial (2025更新)

**适用场景**：
- 共享常见变量（日期、时间、系统设置）
- 语言翻译等动态场景
- 模板层次结构创建
- 减少冗余并优化开发流程

### 3. 实现方式

**来源**：LangChain 旧版文档

**两种方式**：
1. **字符串固定值**：
   ```python
   prompt = PromptTemplate(
       template="...",
       partial_variables={"constant": "fixed_value"}
   )
   ```

2. **函数动态值**：
   ```python
   from datetime import datetime

   def get_current_date():
       return datetime.now().strftime("%Y-%m-%d")

   prompt = PromptTemplate(
       template="...",
       partial_variables={"date": get_current_date}
   )
   ```

### 4. 高级用法

**来源**：Medium 文章

**关键特性**：
- 部分填充变量并延迟完成
- 支持复杂数据结构
- 动态 partial 使用

### 5. 兼容性问题

**来源**：GitHub Issue #17560

**注意事项**：
- ChatPromptTemplate 中 partial_variables 存在兼容性问题
- 历史版本行为差异
- 需要注意版本兼容性

### 6. Go 版本实现

**来源**：LangChainGo 文档

**关键点**：
- Go 版本也支持 partial formatting
- 支持字符串和函数两种方式
- 实现模板部分预绑定

## 需要进一步抓取的链接

根据排除规则，以下链接需要抓取：

**高优先级**（2025-2026 年资料）：
1. https://langchain-opentutorial.gitbook.io/langchain-opentutorial/02-prompt/01-prompttemplate
2. https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples
3. https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates--71d0c244

**中优先级**（社区讨论）：
4. https://bhavikjikadara.medium.com/mastering-prompttemplates-in-langchain-74f679c467ec
5. https://github.com/langchain-ai/langchain/issues/17560
6. https://langchain-tutorials.com/lessons/langchain-essentials/lesson-6

**低优先级**（旧版文档）：
7. https://lagnchain.readthedocs.io/en/latest/modules/prompts/prompt_templates/examples/partial.html

**排除**（官方文档，已通过 Context7 获取）：
- ~~https://reference.langchain.com/v0.3/python/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html~~

**排除**（Go 版本，不在 Python 范围内）：
- ~~https://tmc.github.io/langchaingo/docs/modules/model_io/prompts/prompt_templates/partial_values~~
