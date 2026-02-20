# 核心概念：SYSTEM.md 系统提示词

> 深入理解 SYSTEM.md 和 APPEND_SYSTEM.md 的区别、使用场景和最佳实践

---

## 一、SYSTEM.md 是什么？

**定义：** SYSTEM.md 和 APPEND_SYSTEM.md 是用于定制 Pi Agent 系统提示词的配置文件，控制 AI 的工作风格、行为准则和性格特征。

**核心特点：**
- 📝 Markdown 格式
- 🎭 定制 AI 的"人设"
- 🔄 启动时自动加载
- 🔥 支持热重载（/reload）
- ⚠️ 两种模式：替换 vs 追加

**类比：** SYSTEM.md 就像给 AI 写的"工作守则"和"性格设定"。

---

## 二、两种模式的区别

### 2.1 SYSTEM.md：完全替换

**行为：** 完全替换 Pi Agent 的默认系统提示词。

```markdown
# .pi/SYSTEM.md
你是一个 TypeScript 专家，专注于函数式编程。

# 结果：Pi Agent 的系统提示词
你是一个 TypeScript 专家，专注于函数式编程。
（默认的所有提示词都被替换了）
```

**风险：** 丢失 Pi Agent 的所有默认能力！

```markdown
# ❌ 使用 SYSTEM.md 后的问题
- 不知道如何使用工具（Read、Write、Bash 等）
- 不知道如何读写文件
- 不知道如何执行命令
- 只知道你写的内容
```

### 2.2 APPEND_SYSTEM.md：追加

**行为：** 在默认系统提示词后追加自定义内容。

```markdown
# .pi/APPEND_SYSTEM.md
你必须使用函数式编程风格。

# 结果：Pi Agent 的系统提示词
[默认的系统提示词...]
你必须使用函数式编程风格。
（追加到默认提示词后面）
```

**优势：** 保留 Pi Agent 的所有默认能力 + 添加自定义规则。

### 2.3 对比表

| 特性 | SYSTEM.md | APPEND_SYSTEM.md |
|------|-----------|------------------|
| **行为** | 完全替换默认提示词 | 追加到默认提示词 |
| **保留默认能力** | ❌ 否 | ✅ 是 |
| **使用难度** | 高（需要重写完整提示词） | 低（只需添加规则） |
| **风险** | 高（可能破坏基础功能） | 低（安全） |
| **推荐度** | 1%（高级用户） | 99%（推荐） |

**来源：** 官方文档 README.md 的 SYSTEM.md vs APPEND_SYSTEM.md 说明

---

## 三、文件位置和优先级

### 3.1 两个层级

```bash
# 全局配置（所有项目生效）
~/.pi/agent/SYSTEM.md
~/.pi/agent/APPEND_SYSTEM.md

# 项目配置（当前项目生效，覆盖全局）
/project/.pi/SYSTEM.md
/project/.pi/APPEND_SYSTEM.md
```

### 3.2 优先级规则

**项目级覆盖全局级**

```markdown
# 全局：~/.pi/agent/APPEND_SYSTEM.md
你是一个简洁的开发助手。

# 项目：.pi/APPEND_SYSTEM.md
你必须使用函数式编程。

# 最终效果
[默认系统提示词...]
你必须使用函数式编程。
（项目级覆盖全局级）
```

### 3.3 不能同时使用 SYSTEM.md 和 APPEND_SYSTEM.md

```bash
# ❌ 错误：同时存在两个文件
.pi/SYSTEM.md
.pi/APPEND_SYSTEM.md

# ✅ 正确：只使用一个
.pi/APPEND_SYSTEM.md
```

**规则：** 如果同时存在，SYSTEM.md 优先（完全替换）。

---

## 四、使用场景

### 4.1 APPEND_SYSTEM.md 的使用场景（推荐）

**场景 1：定制代码风格**

```markdown
# .pi/APPEND_SYSTEM.md

# 代码风格
- 使用函数式编程风格
- 优先使用 const 和箭头函数
- 避免使用 any 类型
- 所有函数必须有类型注解

# 命名规范
- 组件：PascalCase
- 函数：camelCase
- 常量：UPPER_SNAKE_CASE
```

**场景 2：添加安全约束**

```markdown
# .pi/APPEND_SYSTEM.md

# 安全规则
- 不要修改 src/legacy/ 目录下的代码
- 不要删除测试文件
- 不要修改 package.json 的依赖版本
- 不要提交 .env 文件到 Git
```

**场景 3：定制工作方式**

```markdown
# .pi/APPEND_SYSTEM.md

# 工作方式
- 代码优先，少说多做
- 每次修改后自动运行测试
- 提交前自动运行 lint
- 使用 Conventional Commits 格式
```

**场景 4：领域特定知识**

```markdown
# .pi/APPEND_SYSTEM.md

# 项目背景
这是一个金融交易系统，处理敏感的用户资金数据。

# 特殊要求
- 所有金额计算使用 Decimal 类型，不使用 float
- 所有 API 调用必须有重试机制
- 所有数据库操作必须在事务中执行
- 所有错误必须记录到日志系统
```

### 4.2 SYSTEM.md 的使用场景（高级）

**只在以下情况使用 SYSTEM.md：**

1. **完全自定义 AI 行为**
   - 你清楚默认提示词的所有内容
   - 你有能力重新编写完整的系统提示词
   - 你需要从零定义 AI 的所有能力

2. **特殊用途的 Agent**
   - 只做特定任务（如代码审查、文档生成）
   - 不需要通用的开发能力

**示例：代码审查专用 Agent**

```markdown
# .pi/SYSTEM.md

你是一个代码审查专家，专注于发现代码中的问题。

# 审查重点
1. 代码风格是否符合规范
2. 是否有潜在的 bug
3. 是否有性能问题
4. 是否有安全漏洞
5. 是否有可读性问题

# 输出格式
对于每个问题，输出：
- 问题位置（文件名:行号）
- 问题类型（风格/bug/性能/安全/可读性）
- 问题描述
- 建议修复方案

# 工作流程
1. 读取代码文件
2. 逐行分析
3. 输出问题列表
4. 提供修复建议
```

---

## 五、内容设计指南

### 5.1 推荐结构

```markdown
# .pi/APPEND_SYSTEM.md

# 角色定位
你是一个专注于 TypeScript 开发的 AI 助手。

# 代码风格
- 使用函数式编程
- 优先使用 const
- 避免使用 any

# 工作方式
- 代码优先，少说多做
- 每次修改后运行测试

# 安全规则
- 不要修改 src/legacy/
- 不要删除测试文件

# 禁止行为
- 不要使用 eval()
- 不要使用 with 语句
- 不要使用 var 声明变量
```

### 5.2 内容设计原则

**1. 简洁明了**

```markdown
# ❌ 过于冗长
你应该始终遵循最佳实践，编写高质量的代码，
确保代码的可读性、可维护性和可扩展性...

# ✅ 简洁清晰
- 代码简洁易读
- 遵循 SOLID 原则
```

**2. 可操作性**

```markdown
# ❌ 模糊描述
代码要写得好。

# ✅ 具体指令
- 函数长度不超过 50 行
- 使用有意义的变量名
- 添加必要的注释
```

**3. 优先级明确**

```markdown
# ✅ 使用优先级
# 必须遵守
- 不要修改 src/legacy/

# 推荐遵守
- 优先使用函数式编程

# 可选
- 添加 JSDoc 注释
```

---

## 六、与 AGENTS.md 的关系

### 6.1 职责分工

| 文件 | 职责 | 内容 |
|------|------|------|
| **AGENTS.md** | 项目知识 | 技术栈、目录结构、开发规范、常用命令 |
| **SYSTEM.md** | AI 行为 | 工作风格、代码风格、安全规则、禁止行为 |

### 6.2 协同工作

```markdown
# AGENTS.md（项目知识）
# 技术栈
- TypeScript + React
- Vite + pnpm

# 目录结构
- src/components/ - React 组件
- src/utils/ - 工具函数

# 常用命令
- pnpm dev - 启动开发服务器
- pnpm test - 运行测试

---

# APPEND_SYSTEM.md（AI 行为）
# 代码风格
- 使用函数式编程
- 优先使用 const

# 工作方式
- 代码优先，少说多做
- 每次修改后运行测试
```

**加载顺序：**
1. 加载默认系统提示词
2. 加载 SYSTEM.md / APPEND_SYSTEM.md（定义 AI 行为）
3. 加载 AGENTS.md（注入项目知识）

**最终效果：**
```
默认系统提示词
+
APPEND_SYSTEM.md（AI 行为）
+
AGENTS.md（项目知识）
=
完整的 AI 上下文
```

### 6.3 内容划分建议

**放在 AGENTS.md：**
- ✅ 项目是什么（技术栈、目录结构）
- ✅ 如何开发（常用命令、工作流程）
- ✅ 团队约定（命名规范、提交规范）

**放在 APPEND_SYSTEM.md：**
- ✅ AI 的工作风格（简洁 vs 详细）
- ✅ 代码风格偏好（函数式 vs 面向对象）
- ✅ 安全约束（禁止操作）
- ✅ 行为准则（优先级、决策规则）

---

## 七、实战案例

### 案例 1：函数式编程风格

```markdown
# .pi/APPEND_SYSTEM.md

# 代码风格：函数式编程
- 优先使用纯函数（无副作用）
- 使用 const 声明所有变量
- 使用箭头函数
- 避免使用 class（除非必要）
- 使用 map、filter、reduce 代替循环

# 示例
```typescript
// ✅ 推荐
const double = (x: number) => x * 2;
const numbers = [1, 2, 3].map(double);

// ❌ 避免
function double(x) {
  return x * 2;
}
let numbers = [];
for (let i = 0; i < 3; i++) {
  numbers.push(double(i + 1));
}
```
```

### 案例 2：安全优先

```markdown
# .pi/APPEND_SYSTEM.md

# 安全规则（最高优先级）
1. 不要修改 src/legacy/ 目录（遗留代码，正在迁移）
2. 不要删除测试文件（即使看起来无用）
3. 不要修改 package.json 的依赖版本（需要团队讨论）
4. 不要提交 .env 文件到 Git（包含敏感信息）

# 数据安全
- 所有用户输入必须验证和清理
- 所有 SQL 查询使用参数化查询
- 所有 API 调用使用 HTTPS
- 所有密码使用 bcrypt 加密

# 错误处理
- 不要在错误信息中暴露敏感信息
- 所有错误必须记录到日志系统
- 用户看到的错误信息要友好且安全
```

### 案例 3：代码优先，少说多做

```markdown
# .pi/APPEND_SYSTEM.md

# 工作方式
- 代码优先，少说多做
- 不要解释为什么这样做（除非用户问）
- 不要提供多个方案（直接给最佳方案）
- 不要问"需要我继续吗？"（直接完成任务）

# 输出格式
- 直接输出代码，不要冗长的解释
- 只在关键地方添加注释
- 不要输出"让我来帮你..."之类的开场白

# 测试
- 每次修改代码后自动运行测试
- 测试失败时立即修复
- 不要问"需要我运行测试吗？"
```

### 案例 4：领域特定（金融系统）

```markdown
# .pi/APPEND_SYSTEM.md

# 项目背景
这是一个金融交易系统，处理用户的资金交易。

# 金额处理规则
- 所有金额使用 Decimal 类型，不使用 number
- 金额计算保留 2 位小数
- 金额显示使用千分位分隔符

# 交易安全
- 所有交易操作必须在数据库事务中执行
- 交易失败时必须回滚
- 所有交易必须记录审计日志

# API 调用
- 所有外部 API 调用必须有重试机制（最多 3 次）
- 所有 API 调用必须有超时设置（10 秒）
- 所有 API 错误必须记录到日志

# 代码示例
```typescript
// ✅ 正确的金额处理
import Decimal from 'decimal.js';

const amount = new Decimal('100.50');
const fee = new Decimal('0.05');
const total = amount.plus(fee); // 100.55

// ❌ 错误的金额处理
const amount = 100.50;
const fee = 0.05;
const total = amount + fee; // 可能有精度问题
```
```

### 案例 5：Mono-repo 项目

```markdown
# .pi/APPEND_SYSTEM.md

# Mono-repo 规则
这是一个 mono-repo 项目，包含多个子包。

# 依赖管理
- 共享依赖放在根目录的 package.json
- 子包特定依赖放在子包的 package.json
- 使用 pnpm workspace 管理依赖

# 代码共享
- 共享代码放在 packages/shared/
- 不要在子包之间直接引用代码
- 使用 workspace 协议引用子包

# 构建顺序
- 先构建 packages/shared/
- 再构建其他子包
- 使用 pnpm -r build 构建所有包
```

---

## 八、常见问题

### Q1: 应该用 SYSTEM.md 还是 APPEND_SYSTEM.md？

**A:** 99% 的情况用 APPEND_SYSTEM.md。

```markdown
# ✅ 推荐：APPEND_SYSTEM.md
- 保留 Pi Agent 的所有默认能力
- 只需添加项目特定规则
- 安全，不会破坏基础功能

# ❌ 不推荐：SYSTEM.md
- 需要重写完整的系统提示词
- 容易丢失默认能力
- 高风险，难以维护
```

### Q2: SYSTEM.md 和 AGENTS.md 有什么区别？

**A:** 职责不同。

```markdown
# AGENTS.md（项目知识）
- 项目是什么
- 如何开发
- 团队约定

# SYSTEM.md（AI 行为）
- AI 的工作风格
- 代码风格偏好
- 安全约束
```

### Q3: 可以同时使用 SYSTEM.md 和 APPEND_SYSTEM.md 吗？

**A:** 不可以。如果同时存在，SYSTEM.md 优先（完全替换）。

```bash
# ❌ 错误
.pi/SYSTEM.md
.pi/APPEND_SYSTEM.md

# ✅ 正确
.pi/APPEND_SYSTEM.md
```

### Q4: 如何知道 SYSTEM.md 是否生效？

**A:** 测试 AI 的行为是否符合预期。

```markdown
# 1. 创建 APPEND_SYSTEM.md
echo "你必须使用函数式编程" > .pi/APPEND_SYSTEM.md

# 2. 重新加载
pi
/reload

# 3. 测试：让 Pi 写一个函数
# Pi 应该使用函数式风格（箭头函数、const 等）
```

### Q5: SYSTEM.md 的内容有长度限制吗？

**A:** 没有硬性限制，但建议保持简洁（500 行以内）。

**原因：**
- 过长的内容会增加 token 消耗
- 降低 AI 的理解效率
- 难以维护

---

## 九、最佳实践

### 9.1 全局 vs 项目配置

**全局配置（~/.pi/agent/APPEND_SYSTEM.md）：**
- ✅ 个人工作风格（简洁 vs 详细）
- ✅ 通用代码风格（函数式 vs 面向对象）
- ✅ 个人偏好（少说多做）

**项目配置（.pi/APPEND_SYSTEM.md）：**
- ✅ 项目特定规则（安全约束）
- ✅ 领域特定知识（金融、医疗）
- ✅ 团队共识（代码风格）

### 9.2 Git 管理策略

```bash
# 项目结构
project/
├── .pi/
│   └── APPEND_SYSTEM.md   # 提交到 Git（团队共享）
└── .gitignore

# .gitignore（如果是个人配置）
.pi/APPEND_SYSTEM.md
```

**决策标准：**
- 团队共识 → 提交到 Git
- 个人偏好 → 不提交（.gitignore）

### 9.3 内容组织

```markdown
# ✅ 推荐：分层组织
# 必须遵守（最高优先级）
- 不要修改 src/legacy/

# 推荐遵守
- 使用函数式编程

# 可选
- 添加 JSDoc 注释

---

# ❌ 不推荐：平铺直叙
- 不要修改 src/legacy/
- 使用函数式编程
- 添加 JSDoc 注释
```

### 9.4 测试和验证

```bash
# 1. 创建配置
cat > .pi/APPEND_SYSTEM.md << 'EOF'
# 代码风格
- 使用函数式编程
- 优先使用 const
EOF

# 2. 重新加载
pi
/reload

# 3. 测试
# 让 Pi 写一个函数，检查是否使用函数式风格

# 4. 调整
# 如果不满意，修改配置并重新加载
```

---

## 十、高级技巧

### 10.1 使用 Markdown 特性

```markdown
# .pi/APPEND_SYSTEM.md

# 代码风格
- 使用函数式编程
- 优先使用 const

# 示例代码
```typescript
// ✅ 推荐
const add = (a: number, b: number) => a + b;

// ❌ 避免
function add(a, b) {
  return a + b;
}
```

# 注意事项
⚠️ 不要使用 any 类型
💡 优先使用类型推导
```

### 10.2 使用条件规则

```markdown
# .pi/APPEND_SYSTEM.md

# 代码风格（根据文件类型）
- .ts 文件：使用 TypeScript 严格模式
- .tsx 文件：使用函数式组件
- .test.ts 文件：使用 describe/it 结构
```

### 10.3 使用优先级

```markdown
# .pi/APPEND_SYSTEM.md

# 优先级 1：安全（必须遵守）
- 不要修改 src/legacy/
- 不要删除测试文件

# 优先级 2：代码风格（推荐遵守）
- 使用函数式编程
- 优先使用 const

# 优先级 3：可选
- 添加 JSDoc 注释
- 使用 Prettier 格式化
```

---

## 十一、总结

**SYSTEM.md 的核心要点：**

1. **两种模式** - SYSTEM.md（替换）vs APPEND_SYSTEM.md（追加）
2. **推荐使用** - 99% 的情况用 APPEND_SYSTEM.md
3. **职责分工** - AGENTS.md（项目知识）vs SYSTEM.md（AI 行为）
4. **内容设计** - 简洁明了、可操作、优先级明确
5. **最佳实践** - 全局个人偏好，项目团队共识

**记住：** APPEND_SYSTEM.md 是定制 Pi Agent 工作风格的最佳方式，保留默认能力的同时添加项目特定规则！

**参考资源：**
- 官方文档：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md
- SYSTEM.md 说明：README.md 的 "Customizing the system prompt" 章节
