# 核心概念 01：Skills 标准与 SKILL.md 格式

## Agent Skills 标准演进历史

### 2023 年：萌芽期

**背景：**
- ChatGPT Plugins 发布（2023 年 3 月）
- LangChain 推出 Tools 概念
- AutoGPT 引入 Plugins 系统

**问题：**
- 每个工具/框架都有自己的格式
- 无法跨工具复用
- 缺乏统一标准

**示例格式（各不相同）：**

```python
# LangChain Tool
from langchain.tools import Tool

tool = Tool(
    name="code-review",
    description="Review code for issues",
    func=lambda x: review_code(x)
)

# AutoGPT Plugin
class CodeReviewPlugin:
    def __init__(self):
        self.name = "code-review"
        self.description = "Review code"

    def execute(self, code):
        return review_code(code)
```

---

### 2024 年：标准化尝试

**背景：**
- Claude Code (Pi-mono) 发布（2024 年初）
- Anthropic 开始探索 Agent 能力标准化
- 社区呼吁统一格式

**Pi-mono 的创新：**
```markdown
<!-- SKILL.md 格式诞生 -->
---
name: code-review
description: Review code for quality and security
---

You are an expert code reviewer...
```

**关键特性：**
1. **Markdown 格式** - 人类可读，易于编辑
2. **YAML frontmatter** - 结构化元数据
3. **文件系统加载** - 零配置，零依赖
4. **命令式调用** - `/skill:name` 简洁直观

---

### 2025 年：开放标准形成

**背景：**
- Anthropic 正式推动 Agent Skills 开放标准
- Vercel、Google、OpenAI 等公司采用
- 社区形成共识

**Agent Skills 标准 v1.0（2025 年 6 月）：**

**核心规范：**
1. **文件命名**：`SKILL.md`（大小写敏感）
2. **目录结构**：`skill-name/SKILL.md`
3. **元数据格式**：YAML frontmatter
4. **必需字段**：`name` + `description`
5. **可选字段**：`disable-model-invocation`

**官方文档：**
- [Agent Skills 标准](https://agentskills.io/)
- [Anthropic Skills 仓库](https://github.com/anthropics/skills)

**行业采用：**
- **Anthropic Claude Code** - 完全兼容
- **Vercel AI SDK** - 支持 SKILL.md 格式
- **Google Gemini** - 实验性支持
- **OpenAI GPTs** - 计划支持

---

### 2026 年：生态系统繁荣

**背景：**
- Skills 成为 AI 软件开发的新单元
- 技能市场和社区蓬勃发展
- 跨工具兼容性成为标准

**生态系统规模（2026 年 2 月）：**

| 平台/项目 | 技能数量 | 特点 |
|-----------|----------|------|
| [anthropics/skills](https://github.com/anthropics/skills) | 50+ | 官方技能库 |
| [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) | 100+ | AI 编码助手技能 |
| [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) | 300+ | 社区技能集合 |
| [obra/superpowers](https://github.com/obra/superpowers) | 20+ | 可组合技能框架 |

**行业影响：**
- [Skills are the New Unit of AI Software Development](https://www.accessnewswire.com/newsroom/en/computers-technology-and-internet/skills-are-the-new-unit-of-ai-software-development-1138920)
- [Anthropic Makes Agent Skills an Open Standard](https://medium.com/lab7ai-insights/anthropic-makes-agent-skills-an-open-standard-for-modular-ai-agent-capabilities-781c697b4d3b)
- [Vercel Releases Agent Skills](https://www.marktechpost.com/2026/01/18/vercel-releases-agent-skills)

---

## SKILL.md 格式规范详解

### 文件结构

```markdown
---
[YAML Frontmatter - 元数据]
---

[Markdown Body - 技能逻辑]
```

**完整示例：**

```markdown
---
name: code-review
description: Review code for quality, security, and performance issues
disable-model-invocation: false
---

You are an expert code reviewer with 10+ years of experience in software development.

## Your Expertise

- Security: OWASP Top 10, secure coding practices
- Performance: Algorithm complexity, optimization techniques
- Maintainability: SOLID principles, design patterns
- Testing: Unit testing, integration testing, TDD

## Review Process

1. **Understand the Context**
   - Read the code carefully
   - Understand the intent and requirements
   - Consider the broader system architecture

2. **Security Analysis**
   - Check for SQL injection vulnerabilities
   - Look for XSS attack vectors
   - Verify authentication and authorization
   - Review data validation and sanitization

3. **Performance Review**
   - Analyze algorithm complexity (Big O)
   - Identify potential bottlenecks
   - Check for unnecessary computations
   - Review database query efficiency

4. **Code Quality Assessment**
   - Evaluate readability and maintainability
   - Check adherence to coding standards
   - Assess error handling
   - Review test coverage

## Output Format

Provide your review in the following structure:

### ✅ Strengths
- List positive aspects of the code

### ⚠️ Issues
- **Critical**: Security vulnerabilities, major bugs
- **Important**: Performance issues, design flaws
- **Minor**: Style issues, minor improvements

### 💡 Suggestions
- Specific, actionable recommendations
- Code examples where helpful
- Links to relevant documentation

## Example

```typescript
// ❌ Vulnerable code
app.get('/user', (req, res) => {
  const userId = req.query.id;
  db.query(`SELECT * FROM users WHERE id = ${userId}`);
});

// ✅ Secure code
app.get('/user', (req, res) => {
  const userId = req.query.id;
  db.query('SELECT * FROM users WHERE id = ?', [userId]);
});
```

Be constructive, specific, and helpful in your feedback.
```

---

### YAML Frontmatter 字段说明

#### 必需字段

**1. `name` - 技能名称**

```yaml
name: code-review
```

**验证规则（来自 skills.ts:91-115）：**

```typescript
function validateName(name: string, parentDirName: string): string[] {
  const errors: string[] = [];

  // 1. 必须与父目录名匹配
  if (name !== parentDirName) {
    errors.push(`name "${name}" does not match parent directory "${parentDirName}"`);
  }

  // 2. 最多 64 字符
  if (name.length > MAX_NAME_LENGTH) {  // MAX_NAME_LENGTH = 64
    errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
  }

  // 3. 只能包含小写字母、数字、连字符
  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push(`name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)`);
  }

  // 4. 不能以连字符开头或结尾
  if (name.startsWith("-") || name.endsWith("-")) {
    errors.push(`name must not start or end with a hyphen`);
  }

  // 5. 不能包含连续连字符
  if (name.includes("--")) {
    errors.push(`name must not contain consecutive hyphens`);
  }

  return errors;
}
```

**示例：**

```yaml
# ✅ 正确
name: code-review
name: test-generation
name: debug-analyzer
name: security-audit-2024

# ❌ 错误
name: Code-Review          # 包含大写字母
name: code_review          # 包含下划线
name: -code-review         # 以连字符开头
name: code-review-         # 以连字符结尾
name: code--review         # 包含连续连字符
name: code review          # 包含空格
```

**2. `description` - 技能描述**

```yaml
description: Review code for quality, security, and performance issues
```

**验证规则（来自 skills.ts:120-130）：**

```typescript
function validateDescription(description: string | undefined): string[] {
  const errors: string[] = [];

  // 1. 必需字段
  if (!description || description.trim() === "") {
    errors.push("description is required");
  }

  // 2. 最多 1024 字符
  if (description.length > MAX_DESCRIPTION_LENGTH) {  // MAX_DESCRIPTION_LENGTH = 1024
    errors.push(`description exceeds ${MAX_DESCRIPTION_LENGTH} characters`);
  }

  return errors;
}
```

**最佳实践：**

```yaml
# ✅ 好的描述：简洁、清晰、具体
description: Review code for quality, security, and performance issues

# ✅ 好的描述：说明适用场景
description: Generate unit tests for TypeScript functions using Jest

# ✅ 好的描述：突出核心价值
description: Debug issues systematically using structured problem-solving approach

# ❌ 不好的描述：过于简单
description: Review code

# ❌ 不好的描述：过于冗长
description: This skill provides comprehensive code review capabilities including but not limited to security analysis, performance optimization, code quality assessment, maintainability evaluation, test coverage analysis, and best practices recommendations for various programming languages and frameworks...
```

#### 可选字段

**3. `disable-model-invocation` - 禁用模型调用**

```yaml
disable-model-invocation: false  # 默认值
```

**用途：**
- `false`（默认）：技能会出现在 System Prompt 中，LLM 可以自动调用
- `true`：技能不会出现在 System Prompt 中，只能通过 `/skill:name` 显式调用

**使用场景：**

```yaml
# 场景 1：需要显式调用的技能（避免 LLM 误触发）
---
name: dangerous-operation
description: Perform dangerous system operations
disable-model-invocation: true  # 必须显式调用
---

# 场景 2：内部技能（被其他技能调用，不直接暴露给 LLM）
---
name: internal-helper
description: Internal helper skill for other skills
disable-model-invocation: true
---

# 场景 3：普通技能（LLM 可以自动调用）
---
name: code-review
description: Review code for issues
disable-model-invocation: false  # 或省略此字段
---
```

**实现机制（来自 skills.ts:290-316）：**

```typescript
export function formatSkillsForPrompt(skills: Skill[]): string {
  // 过滤掉 disableModelInvocation=true 的技能
  const visibleSkills = skills.filter((s) => !s.disableModelInvocation);

  if (visibleSkills.length === 0) {
    return "";
  }

  const lines = [
    "\n\nThe following skills provide specialized instructions for specific tasks.",
    "Use the read tool to load a skill's file when the task matches its description.",
    "",
    "<available_skills>",
  ];

  for (const skill of visibleSkills) {
    lines.push("  <skill>");
    lines.push(`    <name>${escapeXml(skill.name)}</name>`);
    lines.push(`    <description>${escapeXml(skill.description)}</description>`);
    lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
    lines.push("  </skill>");
  }

  lines.push("</available_skills>");

  return lines.join("\n");
}
```

**4. 自定义字段**

```yaml
---
name: code-review
description: Review code
# 可以添加自定义字段（但 Pi-mono 不会使用）
version: 1.0.0
author: Your Name
tags: [code, review, security]
---
```

**接口定义（来自 skills.ts:66-71）：**

```typescript
export interface SkillFrontmatter {
  name?: string;
  description?: string;
  "disable-model-invocation"?: boolean;
  [key: string]: unknown;  // 允许自定义字段
}
```

**注意：**
- 自定义字段会被保留，但不会被 Pi-mono 使用
- 可以用于文档、版本管理等目的
- 不会影响技能的加载和执行

---

### Markdown 正文结构

#### 基本原则

1. **清晰的角色定义**
   ```markdown
   You are an expert code reviewer with 10+ years of experience.
   ```

2. **明确的能力说明**
   ```markdown
   ## Your Expertise
   - Security: OWASP Top 10
   - Performance: Algorithm optimization
   - Maintainability: SOLID principles
   ```

3. **结构化的流程**
   ```markdown
   ## Review Process
   1. Understand the context
   2. Analyze security
   3. Review performance
   4. Assess code quality
   ```

4. **具体的输出格式**
   ```markdown
   ## Output Format
   ### ✅ Strengths
   ### ⚠️ Issues
   ### 💡 Suggestions
   ```

5. **实用的示例**
   ```markdown
   ## Example
   ```typescript
   // ❌ Bad
   // ✅ Good
   ```
   ```

#### 推荐结构模板

```markdown
---
name: skill-name
description: Brief description
---

# 1. 角色定义
You are [role] with [expertise].

# 2. 能力说明
## Your Expertise
- Area 1: Details
- Area 2: Details

# 3. 工作流程
## Process
1. Step 1: Description
2. Step 2: Description
3. Step 3: Description

# 4. 输出格式
## Output Format
[Specify the expected output structure]

# 5. 示例
## Example
[Provide concrete examples]

# 6. 注意事项
## Important Notes
- Note 1
- Note 2
```

#### 高级技巧

**1. 使用条件逻辑**

```markdown
If the code is in TypeScript:
- Check type annotations
- Verify interface definitions

If the code is in Python:
- Check type hints
- Verify docstrings
```

**2. 使用检查清单**

```markdown
## Security Checklist
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF tokens
- [ ] Authentication checks
```

**3. 使用表格**

```markdown
## Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| Critical | Security vulnerabilities | Fix immediately |
| High | Major bugs | Fix before release |
| Medium | Performance issues | Fix soon |
| Low | Style issues | Fix when convenient |
```

**4. 使用代码块**

```markdown
## Example: Secure Input Validation

```typescript
// ❌ Vulnerable
function processInput(input: string) {
  eval(input);  // Never do this!
}

// ✅ Secure
function processInput(input: string) {
  if (!/^[a-zA-Z0-9]+$/.test(input)) {
    throw new Error('Invalid input');
  }
  return input;
}
```
```

---

## Pi-mono 的 SKILL.md 加载机制

### 加载流程

**完整流程（来自 skills.ts:146-280）：**

```typescript
// 1. 入口函数
export function loadSkillsFromDir(options: LoadSkillsFromDirOptions): LoadSkillsResult {
  const { dir, source } = options;
  return loadSkillsFromDirInternal(dir, source, true);
}

// 2. 递归扫描目录
function loadSkillsFromDirInternal(
  dir: string,
  source: string,
  includeRootFiles: boolean,
  ignoreMatcher?: IgnoreMatcher,
  rootDir?: string,
): LoadSkillsResult {
  const skills: Skill[] = [];
  const diagnostics: ResourceDiagnostic[] = [];

  if (!existsSync(dir)) {
    return { skills, diagnostics };
  }

  const root = rootDir ?? dir;
  const ig = ignoreMatcher ?? ignore();
  addIgnoreRules(ig, dir, root);  // 读取 .gitignore 等

  const entries = readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    // 跳过隐藏文件
    if (entry.name.startsWith(".")) {
      continue;
    }

    // 跳过 node_modules
    if (entry.name === "node_modules") {
      continue;
    }

    const fullPath = join(dir, entry.name);

    // 处理符号链接
    let isDirectory = entry.isDirectory();
    let isFile = entry.isFile();
    if (entry.isSymbolicLink()) {
      const stats = statSync(fullPath);
      isDirectory = stats.isDirectory();
      isFile = stats.isFile();
    }

    // 检查 ignore 规则
    const relPath = toPosixPath(relative(root, fullPath));
    const ignorePath = isDirectory ? `${relPath}/` : relPath;
    if (ig.ignores(ignorePath)) {
      continue;
    }

    // 递归扫描子目录
    if (isDirectory) {
      const subResult = loadSkillsFromDirInternal(fullPath, source, false, ig, root);
      skills.push(...subResult.skills);
      diagnostics.push(...subResult.diagnostics);
      continue;
    }

    if (!isFile) {
      continue;
    }

    // 加载技能文件
    const isRootMd = includeRootFiles && entry.name.endsWith(".md");
    const isSkillMd = !includeRootFiles && entry.name === "SKILL.md";
    if (!isRootMd && !isSkillMd) {
      continue;
    }

    const result = loadSkillFromFile(fullPath, source);
    if (result.skill) {
      skills.push(result.skill);
    }
    diagnostics.push(...result.diagnostics);
  }

  return { skills, diagnostics };
}

// 3. 加载单个技能文件
function loadSkillFromFile(
  filePath: string,
  source: string,
): { skill: Skill | null; diagnostics: ResourceDiagnostic[] } {
  const diagnostics: ResourceDiagnostic[] = [];

  try {
    // 读取文件内容
    const rawContent = readFileSync(filePath, "utf-8");

    // 解析 frontmatter
    const { frontmatter } = parseFrontmatter<SkillFrontmatter>(rawContent);

    const skillDir = dirname(filePath);
    const parentDirName = basename(skillDir);

    // 验证 description
    const descErrors = validateDescription(frontmatter.description);
    for (const error of descErrors) {
      diagnostics.push({ type: "warning", message: error, path: filePath });
    }

    // 使用 frontmatter 中的 name，或回退到目录名
    const name = frontmatter.name || parentDirName;

    // 验证 name
    const nameErrors = validateName(name, parentDirName);
    for (const error of nameErrors) {
      diagnostics.push({ type: "warning", message: error, path: filePath });
    }

    // 即使有警告也加载技能（除非 description 完全缺失）
    if (!frontmatter.description || frontmatter.description.trim() === "") {
      return { skill: null, diagnostics };
    }

    return {
      skill: {
        name,
        description: frontmatter.description,
        filePath,
        baseDir: skillDir,
        source,
        disableModelInvocation: frontmatter["disable-model-invocation"] === true,
      },
      diagnostics,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "failed to parse skill file";
    diagnostics.push({ type: "warning", message, path: filePath });
    return { skill: null, diagnostics };
  }
}
```

### 关键特性

**1. 递归扫描**
- 支持任意深度的子目录
- 自动发现所有 SKILL.md 文件

**2. Ignore 规则支持**
- 读取 `.gitignore`、`.ignore`、`.fdignore`
- 自动跳过 `node_modules`
- 自动跳过隐藏文件（`.` 开头）

**3. 符号链接支持**
- 自动跟随符号链接
- 处理断开的符号链接

**4. 错误处理**
- 验证失败时生成诊断信息
- 即使有警告也尝试加载技能
- 只有 description 完全缺失时才拒绝加载

**5. 多源支持**
- 全局技能：`~/.pi/agent/skills/`
- 项目技能：`./.pi/skills/`
- 自定义路径：通过配置指定

---

## 2025-2026 Skills 生态系统

### 官方技能库

**1. [anthropics/skills](https://github.com/anthropics/skills)**
- 50+ 官方技能
- 涵盖代码审查、测试生成、调试等
- 高质量参考实现

**2. [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)**
- 100+ AI 编码助手技能
- 专注于前端开发
- Next.js、React、TypeScript 等

**3. [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)**
- 300+ 社区技能
- Claude Code 兼容
- 涵盖各种领域

**4. [obra/superpowers](https://github.com/obra/superpowers)**
- 20+ 可组合技能
- 强调技能组合
- 工作流导向

### 技能市场

**1. Vercel Agent Skills 市场**
- 官方技能市场
- 一键安装
- 社区评分和评论

**2. Anthropic Skills Hub**
- Anthropic 官方技能中心
- 精选高质量技能
- 定期更新

**3. GitHub Skills Registry**
- 基于 GitHub 的技能注册表
- 开源社区驱动
- 自由分发

### 行业采用

**采用 SKILL.md 标准的公司和项目：**

| 公司/项目 | 采用情况 | 技能数量 | 特点 |
|-----------|----------|----------|------|
| Anthropic | 完全采用 | 50+ | 标准制定者 |
| Vercel | 完全采用 | 100+ | 前端生态 |
| Google | 实验性支持 | 未公开 | Gemini 集成 |
| OpenAI | 计划支持 | 未公开 | GPTs 集成 |
| Pi-mono | 完全采用 | 内置 | 开源实现 |

### 未来趋势

**1. Skills 成为 AI 软件开发的新单元**
- 引用：[Skills are the New Unit](https://www.accessnewswire.com/newsroom/en/computers-technology-and-internet/skills-are-the-new-unit-of-ai-software-development-1138920)
- 传统：函数 → 类 → 模块 → 包
- AI 时代：Skills → Agent → System

**2. 跨工具兼容性**
- SKILL.md 成为事实标准
- 一次编写，到处运行
- 技能市场互联互通

**3. 技能组合模式**
- Sequential（顺序执行）
- Parallel（并行执行）
- Conditional（条件执行）
- 引用：arXiv paper "Agent Skills for LLMs" (91.6% accuracy)

**4. 企业级技能管理**
- 私有技能仓库
- 权限和访问控制
- 版本管理和审计

---

## 总结

**SKILL.md 格式的核心价值：**

1. **标准化** - 开放标准，跨工具兼容
2. **简单性** - Markdown + YAML，易于编写
3. **可读性** - 人类可读，易于维护
4. **可扩展** - 支持自定义字段
5. **生态系统** - 丰富的技能库和市场

**Pi-mono 的实现特点：**

1. **严格验证** - 确保技能格式正确
2. **递归扫描** - 支持任意目录结构
3. **Ignore 支持** - 尊重 .gitignore 规则
4. **错误处理** - 友好的诊断信息
5. **多源加载** - 全局 + 项目级别

**2025-2026 生态系统：**

1. **官方支持** - Anthropic、Vercel、Google
2. **社区繁荣** - 300+ 开源技能
3. **市场形成** - 技能市场和注册表
4. **标准成熟** - SKILL.md 成为事实标准

---

**版本：** v1.0
**最后更新：** 2026-02-20
**维护者：** Claude Code
