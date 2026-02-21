# 核心概念 01: Markdown 模板

## 概述

Markdown 模板是 Pi-mono Prompt Templates 系统的基础格式。理解 Markdown 模板的设计和使用是掌握整个模板系统的关键。

---

## 为什么选择 Markdown？

### 问题：模板需要什么格式？

在设计模板系统时，格式选择至关重要。我们需要一种格式能够：

1. **人类可读** - 易于编写和理解
2. **机器可解析** - 易于程序处理
3. **富文本支持** - 支持代码块、列表、标题等
4. **开发者熟悉** - 无需学习新格式
5. **工具支持** - 所有编辑器都支持

### 候选格式对比

| 格式 | 人类可读 | 机器可解析 | 富文本 | 开发者熟悉 | 工具支持 |
|------|---------|-----------|--------|-----------|---------|
| **纯文本** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **JSON** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **YAML** | ⚠️ | ✅ | ❌ | ⚠️ | ✅ |
| **XML** | ❌ | ✅ | ❌ | ⚠️ | ✅ |
| **Markdown** | ✅ | ✅ | ✅ | ✅ | ✅ |

**结论：** Markdown 是最优选择。

---

## Markdown 模板的结构

### 基本结构

```markdown
---
description: Template description
---
Template content here...
```

**两个部分：**

1. **YAML Frontmatter**（可选）- 元数据
2. **Markdown Body**（必需）- 模板内容

---

### YAML Frontmatter

**作用：** 存储模板的元数据。

**常用字段：**

```yaml
---
description: Template description (required for good practice)
author: team@example.com (optional)
version: 1.0.0 (optional)
tags: [tag1, tag2] (optional)
created: 2026-02-20 (optional)
updated: 2026-02-20 (optional)
---
```

**示例：**

```markdown
---
description: Review GitHub PRs with structured analysis
author: team@example.com
version: 1.0.0
tags: [code-review, pr, github]
created: 2026-01-01
updated: 2026-02-20
---
You are an expert code reviewer...
```

---

### Markdown Body

**作用：** 模板的实际内容，会直接发送给 LLM。

**支持的 Markdown 特性：**

1. **标题**
```markdown
# H1 标题
## H2 标题
### H3 标题
```

2. **列表**
```markdown
- 无序列表项 1
- 无序列表项 2

1. 有序列表项 1
2. 有序列表项 2
```

3. **代码块**
````markdown
```typescript
const foo = 'bar';
```
````

4. **引用**
```markdown
> 这是一个引用
```

5. **粗体和斜体**
```markdown
**粗体文本**
*斜体文本*
```

6. **链接**
```markdown
[链接文本](https://example.com)
```

---

## Markdown 模板的最佳实践

### 1. 使用清晰的结构

**❌ 糟糕的结构：**
```markdown
---
description: Review code
---
Review this code and tell me if it's good or bad.
```

**✅ 好的结构：**
```markdown
---
description: Structured code review with quality analysis
---
# Code Review

You are an expert code reviewer with 10+ years of experience.

## Task
Review the following code: $1

## Focus Areas
1. **Code Quality**: Readability, maintainability, best practices
2. **Potential Issues**: Bugs, edge cases, security concerns
3. **Performance**: Efficiency, optimization opportunities

## Output Format
Provide your review in the following structure:

### Summary
[Brief overview of the code]

### Issues Found
[List of issues with severity levels]

### Recommendations
[Specific, actionable suggestions]
```

---

### 2. 包含完整的上下文

**❌ 缺少上下文：**
```markdown
---
description: Generate tests
---
Generate tests for $1
```

**✅ 包含完整上下文：**
```markdown
---
description: Generate comprehensive unit tests
---
# Test Generation

You are a testing expert specializing in TypeScript and Jest.

## Context
- Project uses TypeScript
- Testing framework: Jest
- Coverage target: 80%+
- Follow AAA pattern (Arrange, Act, Assert)

## Task
Generate comprehensive unit tests for: $1

## Requirements
1. Test all public methods
2. Include edge cases
3. Test error scenarios
4. Use descriptive test names
5. Add comments for complex test logic

## Output Format
```typescript
describe('ComponentName', () => {
  describe('methodName', () => {
    it('should handle normal case', () => {
      // Arrange
      // Act
      // Assert
    });

    it('should handle edge case', () => {
      // ...
    });
  });
});
```
```

---

### 3. 使用 Markdown 格式增强可读性

**示例：使用标题组织内容**

```markdown
---
description: API documentation generator
---
# API Documentation Generator

## Role
You are a technical writer specializing in API documentation.

## Input
API endpoint: $1

## Analysis Steps
1. Identify HTTP method and path
2. Extract parameters
3. Document request/response format
4. Add usage examples

## Output Template
### Endpoint: [METHOD] [PATH]

**Description**: [Brief description]

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ... | ... | ... | ... |

**Request Example**:
```json
{
  "key": "value"
}
```

**Response Example**:
```json
{
  "result": "success"
}
```
```

---

### 4. 使用代码块展示示例

**示例：包含代码示例**

```markdown
---
description: Refactoring suggestions
---
# Code Refactoring

## Task
Analyze and suggest refactorings for: $1

## Refactoring Patterns

### 1. Extract Method
**Before**:
```typescript
function processData(data: any[]) {
  // 50 lines of complex logic
}
```

**After**:
```typescript
function processData(data: any[]) {
  const validated = validateData(data);
  const transformed = transformData(validated);
  return saveData(transformed);
}
```

### 2. Replace Conditional with Polymorphism
[More examples...]

## Your Analysis
[Provide specific refactoring suggestions for the given code]
```

---

## Markdown 模板的高级技巧

### 1. 使用表格组织信息

```markdown
---
description: Technology comparison
---
# Technology Comparison

Compare the following technologies: $@

## Comparison Matrix

| Feature | Tech 1 | Tech 2 | Tech 3 |
|---------|--------|--------|--------|
| Performance | | | |
| Scalability | | | |
| Learning Curve | | | |
| Community Support | | | |
| Cost | | | |

## Detailed Analysis
[Provide detailed comparison based on the matrix]
```

---

### 2. 使用检查清单

```markdown
---
description: Pre-deployment checklist
---
# Pre-Deployment Checklist

Review the following for deployment: $1

## Checklist

### Code Quality
- [ ] All tests passing
- [ ] No linting errors
- [ ] Code reviewed and approved
- [ ] Documentation updated

### Security
- [ ] No hardcoded secrets
- [ ] Dependencies updated
- [ ] Security scan passed
- [ ] Access controls verified

### Performance
- [ ] Load testing completed
- [ ] Database queries optimized
- [ ] Caching configured
- [ ] CDN setup verified

### Monitoring
- [ ] Logging configured
- [ ] Alerts set up
- [ ] Dashboards created
- [ ] Runbooks updated

## Review Results
[Provide detailed review for each checklist item]
```

---

### 3. 使用引用强调重要信息

```markdown
---
description: Security audit
---
# Security Audit

Audit the following code for security issues: $1

> **CRITICAL**: Focus on OWASP Top 10 vulnerabilities

## Security Checklist

1. **Injection Attacks**
   > Check for SQL injection, command injection, XSS

2. **Authentication**
   > Verify secure authentication mechanisms

3. **Sensitive Data**
   > Ensure proper encryption and storage

## Audit Results
[Provide detailed security analysis]
```

---

## Markdown 模板与 LLM 的交互

### 理解：模板内容直接发送给 LLM

**重要：** Markdown 模板的内容会**原样发送**给 LLM，不会有额外处理。

**示例流程：**

```
1. 用户调用
/review src/utils.ts

2. Pi 加载模板
~/.pi/agent/prompts/review.md

3. Pi 替换变量
$1 → src/utils.ts

4. Pi 发送给 LLM
[替换后的完整 Markdown 内容]

5. LLM 处理
理解 Markdown 格式
生成结构化响应
```

---

### Markdown 格式对 LLM 的影响

**1. 标题帮助 LLM 理解结构**

```markdown
# Main Task
## Subtask 1
## Subtask 2
```

LLM 会理解这是一个主任务和两个子任务。

**2. 列表帮助 LLM 组织输出**

```markdown
Provide your analysis in the following format:
1. Summary
2. Issues
3. Recommendations
```

LLM 会按照这个格式组织输出。

**3. 代码块帮助 LLM 识别代码**

````markdown
```typescript
// LLM 知道这是 TypeScript 代码
const foo = 'bar';
```
````

**4. 表格帮助 LLM 结构化数据**

```markdown
| Feature | Rating |
|---------|--------|
| Performance | 8/10 |
```

LLM 会生成类似格式的表格。

---

## 实际案例分析

### 案例 1: 代码审查模板

**需求：** 创建一个结构化的代码审查模板。

**实现：**

```markdown
---
description: Comprehensive code review with structured analysis
author: team@example.com
version: 2.0.0
tags: [code-review, quality, best-practices]
---
# Code Review

You are an expert code reviewer with 10+ years of experience in TypeScript and modern web development.

## Context
- Project: TypeScript/React application
- Testing: Jest + React Testing Library
- Linting: ESLint + Prettier
- Target: Production-ready code

## Task
Review the following code: $1

## Review Criteria

### 1. Code Quality (Weight: 40%)
- **Readability**: Is the code easy to understand?
- **Maintainability**: Is the code easy to modify?
- **Best Practices**: Does it follow TypeScript/React conventions?

### 2. Correctness (Weight: 30%)
- **Logic**: Is the logic correct?
- **Edge Cases**: Are edge cases handled?
- **Error Handling**: Are errors properly handled?

### 3. Performance (Weight: 15%)
- **Efficiency**: Are there performance bottlenecks?
- **Optimization**: Can it be optimized?

### 4. Security (Weight: 15%)
- **Vulnerabilities**: Are there security issues?
- **Input Validation**: Is user input validated?

## Output Format

### Summary
[One paragraph overview of the code]

### Strengths
- [List 2-3 things done well]

### Issues
| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| High | [Description] | Line X | [Fix] |
| Medium | [Description] | Line Y | [Fix] |

### Recommendations
1. **[Category]**: [Specific, actionable recommendation]
2. **[Category]**: [Specific, actionable recommendation]

### Overall Rating
- Code Quality: X/10
- Correctness: X/10
- Performance: X/10
- Security: X/10
- **Overall**: X/10

### Next Steps
[Prioritized list of actions to take]
```

**使用：**
```bash
/review src/components/UserProfile.tsx
```

**效果：**
- 结构化的输出
- 清晰的评分
- 可操作的建议

---

### 案例 2: 文档生成模板

**需求：** 为函数生成完整的文档。

**实现：**

```markdown
---
description: Generate comprehensive JSDoc documentation
author: team@example.com
version: 1.0.0
tags: [documentation, jsdoc, typescript]
---
# JSDoc Generator

You are a technical writer specializing in TypeScript documentation.

## Task
Generate comprehensive JSDoc documentation for: $1

## Documentation Standards

### Function Documentation Template
```typescript
/**
 * [Brief one-line description]
 *
 * [Detailed description explaining what the function does,
 * why it exists, and any important context]
 *
 * @param {Type} paramName - [Parameter description]
 * @param {Type} paramName - [Parameter description]
 * @returns {Type} [Return value description]
 *
 * @throws {ErrorType} [When this error is thrown]
 *
 * @example
 * ```typescript
 * // [Example usage]
 * const result = functionName(arg1, arg2);
 * ```
 *
 * @see [Related function or documentation]
 * @since 1.0.0
 */
```

### Class Documentation Template
```typescript
/**
 * [Brief one-line description]
 *
 * [Detailed description of the class purpose,
 * responsibilities, and usage patterns]
 *
 * @class
 * @example
 * ```typescript
 * const instance = new ClassName(options);
 * instance.method();
 * ```
 */
```

## Requirements
1. Use clear, concise language
2. Include practical examples
3. Document edge cases
4. Explain complex logic
5. Link to related documentation

## Output
[Generate the JSDoc documentation following the templates above]
```

**使用：**
```bash
/jsdoc src/utils/validation.ts
```

---

### 案例 3: 测试生成模板

**需求：** 生成全面的单元测试。

**实现：**

```markdown
---
description: Generate comprehensive unit tests with Jest
author: team@example.com
version: 1.0.0
tags: [testing, jest, typescript]
---
# Unit Test Generator

You are a testing expert specializing in Jest and TypeScript.

## Context
- Framework: Jest
- Language: TypeScript
- Pattern: AAA (Arrange, Act, Assert)
- Coverage Target: 80%+

## Task
Generate comprehensive unit tests for: $1

## Test Structure

### File Organization
```typescript
// [filename].test.ts
import { functionName } from './filename';

describe('ComponentName', () => {
  describe('methodName', () => {
    it('should handle normal case', () => {
      // Arrange
      const input = ...;
      const expected = ...;

      // Act
      const result = methodName(input);

      // Assert
      expect(result).toBe(expected);
    });
  });
});
```

## Test Categories

### 1. Happy Path Tests
- Test normal, expected usage
- Verify correct output for valid input

### 2. Edge Case Tests
- Empty input
- Null/undefined
- Boundary values
- Large datasets

### 3. Error Handling Tests
- Invalid input
- Expected exceptions
- Error messages

### 4. Integration Tests (if applicable)
- Component interactions
- Side effects
- State changes

## Test Naming Convention
```
should [expected behavior] when [condition]

Examples:
- should return true when input is valid
- should throw error when input is null
- should update state when button is clicked
```

## Requirements
1. Test all public methods
2. Achieve 80%+ coverage
3. Use descriptive test names
4. Add comments for complex logic
5. Mock external dependencies
6. Test async operations properly

## Output
[Generate the complete test file following the structure above]
```

**使用：**
```bash
/test src/services/auth.ts
```

---

## Markdown 模板的常见错误

### 错误 1: Frontmatter 格式错误

**❌ 错误：**
```markdown
--
description: Wrong delimiter
--
Content...
```

**✅ 正确：**
```markdown
---
description: Correct delimiter
---
Content...
```

---

### 错误 2: 缺少必要的上下文

**❌ 错误：**
```markdown
---
description: Review code
---
Review $1
```

**✅ 正确：**
```markdown
---
description: Comprehensive code review
---
You are an expert code reviewer.

Review the following code: $1

Focus on:
- Code quality
- Potential bugs
- Best practices

Provide specific, actionable feedback.
```

---

### 错误 3: 过度使用 Markdown 格式

**❌ 错误：**
```markdown
---
description: Over-formatted template
---
# **IMPORTANT** Review

> **NOTE**: This is a code review

## **TASK**
- **Review** the **code**: $1
- **Focus** on **quality**

### **OUTPUT**
**Provide** a **detailed** **review**
```

**✅ 正确：**
```markdown
---
description: Clean code review template
---
# Code Review

Review the following code: $1

Focus on:
- Code quality
- Potential bugs
- Best practices

Provide a detailed review.
```

---

## 2025-2026 年的最佳实践

根据 2025-2026 年的实际应用，以下是推荐的最佳实践：

### 1. 使用 Markdown 作为标准格式

**来源：** [GitHub: Awesome AI System Prompts](https://github.com/dontriskit/awesome-ai-system-prompts)

**实践：**
- 所有提示都以 Markdown 格式存储
- 使用 YAML frontmatter 存储元数据
- 通过 Git 进行版本控制

### 2. 结构化提示设计

**来源：** [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

**实践：**
- 明确角色定义
- 提供完整上下文
- 清晰的任务描述
- 结构化的输出格式

### 3. 模板文档化

**来源：** [Reddit: LLM真实系统提示Markdown结构化](https://www.reddit.com/r/PromptEngineering/comments/1pseks9/for_people_building_real_systems_with_llms_how_do/)

**实践：**
- 在模板中包含使用说明
- 提供示例
- 说明期望的输出格式

---

## 总结

**Markdown 模板的核心价值：**

1. **人类可读** - 易于编写和理解
2. **机器可解析** - 易于程序处理
3. **富文本支持** - 支持代码块、列表、标题等
4. **开发者熟悉** - 无需学习新格式
5. **工具支持** - 所有编辑器都支持

**记住：**
- 使用清晰的结构
- 包含完整的上下文
- 利用 Markdown 格式增强可读性
- 模板内容会直接发送给 LLM

---

**版本：** v1.0
**最后更新：** 2026-02-20
**维护者：** Claude Code
