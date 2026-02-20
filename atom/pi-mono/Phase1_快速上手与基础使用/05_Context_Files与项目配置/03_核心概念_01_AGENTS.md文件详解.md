# 核心概念：AGENTS.md 文件详解

> 深入理解 AGENTS.md 的加载机制、内容结构和最佳实践

---

## 一、AGENTS.md 是什么？

**定义：** AGENTS.md（或 CLAUDE.md）是 Pi Agent 的项目知识库文件，用 Markdown 格式描述项目规范、开发流程和团队约定。

**核心特点：**
- 📝 纯文本 Markdown 格式
- 🔄 启动时自动加载
- 📚 支持多层级累加
- 🔥 支持热重载（/reload）
- 🤝 可提交到 Git 团队共享

**类比：** AGENTS.md 就像项目的 README.md，但它是专门写给 AI 看的。

---

## 二、加载机制详解

### 2.1 搜索路径

Pi Agent 启动时，会从当前目录向上搜索所有 AGENTS.md 文件：

```bash
# 假设当前目录：/home/user/projects/my-app/src/components

# 搜索路径（从下往上）
1. /home/user/projects/my-app/src/components/AGENTS.md
2. /home/user/projects/my-app/src/AGENTS.md
3. /home/user/projects/my-app/AGENTS.md
4. /home/user/projects/AGENTS.md
5. /home/user/AGENTS.md
6. /home/AGENTS.md
7. ~/.pi/agent/AGENTS.md  # 全局配置
```

**关键点：** 所有找到的文件都会被加载，不是找到第一个就停止！

### 2.2 加载顺序（累加机制）

找到的文件按照"从全局到特定"的顺序合并：

```
加载顺序：
全局 AGENTS.md
↓ 拼接
父目录 AGENTS.md
↓ 拼接
当前目录 AGENTS.md
↓
最终内容 = 所有文件内容拼接
```

**实际案例：**

```markdown
# ~/.pi/agent/AGENTS.md（全局）
- 个人习惯：使用 Vim 快捷键
- 代码风格：简洁优先

# /projects/AGENTS.md（项目根）
- 项目类型：TypeScript + React
- 包管理器：pnpm

# /projects/my-app/AGENTS.md（子项目）
- 状态管理：Zustand
- 样式方案：Tailwind CSS

# Pi Agent 最终看到的内容（按顺序拼接）
- 个人习惯：使用 Vim 快捷键
- 代码风格：简洁优先
- 项目类型：TypeScript + React
- 包管理器：pnpm
- 状态管理：Zustand
- 样式方案：Tailwind CSS
```

### 2.3 与 Node.js 模块解析的对比

```typescript
// Node.js 模块解析（覆盖机制）
require('lodash')
// 找到第一个 node_modules/lodash 就停止
// 结果：只加载一个模块

// Pi Agent AGENTS.md（累加机制）
// 找到所有 AGENTS.md 并合并
// 结果：所有层级的内容都生效
```

**关键区别：**
- Node.js：覆盖（找到第一个就停止）
- Pi Agent：累加（找到所有并合并）

### 2.4 启动时的加载日志

```bash
$ pi

Loaded context files:
- ~/.pi/agent/AGENTS.md
- /projects/my-app/AGENTS.md

Ready to assist!
```

**来源：** 官方文档 README.md 的 context files loading mechanism

---

## 三、文件内容结构

### 3.1 推荐结构模板

```markdown
# 项目概述
简短描述项目是什么，解决什么问题。

# 技术栈
- 编程语言：TypeScript 5
- 前端框架：React 18
- 构建工具：Vite 5
- 包管理器：pnpm
- 状态管理：Zustand
- 样式方案：Tailwind CSS

# 目录结构
```
src/
├── components/     # React 组件
├── hooks/          # 自定义 Hooks
├── utils/          # 工具函数
├── api/            # API 调用
├── types/          # TypeScript 类型定义
└── styles/         # 全局样式
```

# 开发规范
- 代码风格：Prettier + ESLint
- 提交规范：Conventional Commits
- 分支策略：Git Flow
- 测试要求：单元测试覆盖率 > 80%

# 命名约定
- 组件：PascalCase（例如：UserProfile.tsx）
- 文件：kebab-case（例如：user-profile.utils.ts）
- 变量：camelCase（例如：userName）
- 常量：UPPER_SNAKE_CASE（例如：API_BASE_URL）

# 常用命令
- `pnpm dev` - 启动开发服务器（http://localhost:5173）
- `pnpm build` - 构建生产版本
- `pnpm test` - 运行测试
- `pnpm lint` - 代码检查
- `pnpm format` - 格式化代码

# 工作流程
1. 从 main 分支创建 feature 分支
2. 开发并提交代码（遵循 Conventional Commits）
3. 运行测试和 lint
4. 创建 Pull Request
5. Code Review 通过后合并

# 注意事项
- ⚠️ 不要修改 src/legacy/ 目录下的代码
- ⚠️ API 调用必须使用 src/api/client.ts
- ⚠️ 所有组件必须有 TypeScript 类型
- ⚠️ 不要提交 .env 文件到 Git

# 常见问题
Q: 如何添加新的 API 端点？
A: 在 src/api/endpoints/ 目录下创建新文件，使用 client.ts 的封装方法。

Q: 如何添加新的路由？
A: 在 src/routes.tsx 中添加路由配置。
```

### 3.2 内容设计原则

**1. 简洁明了**
```markdown
# ❌ 过于冗长
本项目是一个使用 TypeScript 和 React 构建的现代化 Web 应用程序，
它采用了最新的前端技术栈，包括但不限于 Vite 作为构建工具...

# ✅ 简洁清晰
TypeScript + React 项目，使用 Vite 构建。
```

**2. 可操作性**
```markdown
# ❌ 模糊描述
代码要写得好，要遵循最佳实践。

# ✅ 具体指令
- 使用 Prettier 格式化代码
- 遵循 ESLint 规则
- 函数长度不超过 50 行
```

**3. 分层组织**
```markdown
# ❌ 平铺直叙
- 使用 TypeScript
- 使用 React
- 使用 Vite
- 使用 pnpm
- 组件放在 src/components/
- 工具函数放在 src/utils/
...

# ✅ 分层组织
# 技术栈
- TypeScript + React + Vite

# 目录结构
- src/components/ - 组件
- src/utils/ - 工具函数
```

---

## 四、Mono-repo 的渐进式披露模式

### 4.1 什么是渐进式披露？

**定义：** 在 mono-repo 项目中，根目录定义通用规范，子包定义特定规范，避免重复。

**来源：** Twitter @badlogicgames 的 progressive disclosure 最佳实践

### 4.2 目录结构设计

```bash
/monorepo/
├── AGENTS.md                    # 通用规范（所有子包共享）
├── packages/
│   ├── frontend/
│   │   └── AGENTS.md            # 前端特定规范
│   ├── backend/
│   │   └── AGENTS.md            # 后端特定规范
│   └── shared/
│       └── AGENTS.md            # 共享库特定规范
```

### 4.3 内容设计示例

**根目录 AGENTS.md（通用规范）：**

```markdown
# Monorepo 项目规范

# 通用规范
- 代码风格：Prettier + ESLint
- 提交规范：Conventional Commits
- 分支策略：Git Flow
- 测试要求：覆盖率 > 80%

# 包管理
- 使用 pnpm workspace
- 共享依赖放在根目录
- 子包独立版本管理

# 常用命令
- `pnpm install` - 安装所有依赖
- `pnpm build` - 构建所有包
- `pnpm test` - 运行所有测试
```

**frontend/AGENTS.md（前端特定）：**

```markdown
# 前端包规范

# 技术栈
- React 18 + TypeScript
- Vite 5
- Tailwind CSS
- Zustand

# 目录结构
```
src/
├── components/     # React 组件
├── pages/          # 页面组件
├── hooks/          # 自定义 Hooks
└── api/            # API 调用
```

# 前端特定规范
- 组件必须使用 TypeScript
- 样式使用 Tailwind CSS
- 状态管理使用 Zustand
- API 调用使用 React Query

# 常用命令
- `pnpm dev` - 启动开发服务器
- `pnpm build` - 构建前端
```

**backend/AGENTS.md（后端特定）：**

```markdown
# 后端包规范

# 技术栈
- Node.js + TypeScript
- Express
- PostgreSQL
- Prisma ORM

# 目录结构
```
src/
├── routes/         # 路由定义
├── controllers/    # 控制器
├── services/       # 业务逻辑
├── models/         # 数据模型
└── middleware/     # 中间件
```

# 后端特定规范
- API 遵循 RESTful 设计
- 使用 Prisma 操作数据库
- 错误处理统一使用 middleware
- 所有 API 必须有单元测试

# 常用命令
- `pnpm dev` - 启动开发服务器
- `pnpm migrate` - 运行数据库迁移
```

### 4.4 渐进式披露的优势

**1. 避免重复**
```markdown
# ❌ 错误：子包重复根目录的内容
# frontend/AGENTS.md
- 代码风格：Prettier + ESLint  # 重复了根目录
- 提交规范：Conventional Commits  # 重复了根目录
- React 18 + TypeScript

# ✅ 正确：子包只包含特定内容
# frontend/AGENTS.md
- React 18 + TypeScript
- Tailwind CSS
- Zustand
```

**2. 清晰的职责分离**
- 根目录：团队通用规范
- 子包：技术栈特定规范

**3. 易于维护**
- 修改通用规范：只需修改根目录
- 修改特定规范：只需修改子包

---

## 五、团队协作最佳实践

### 5.1 Git 管理策略

```bash
# 项目结构
project/
├── AGENTS.md              # ✅ 提交到 Git（团队共享）
├── .gitignore
│   └── .pi/               # ❌ 忽略个人配置
└── .pi/
    └── settings.json      # 个人配置（不提交）

# .gitignore 内容
.pi/
```

### 5.2 团队协作流程

**1. 项目启动阶段**
```bash
# 1. 创建 AGENTS.md
cat > AGENTS.md << 'EOF'
# 项目规范
- TypeScript + React
- 使用 pnpm
- 遵循 ESLint
EOF

# 2. 提交到 Git
git add AGENTS.md
git commit -m "docs: add project conventions"
git push
```

**2. 团队成员 Onboarding**
```bash
# 1. 克隆项目
git clone <repo-url>
cd project

# 2. 启动 Pi（自动加载 AGENTS.md）
pi

# 3. Pi Agent 自动理解项目规范
```

**3. 规范更新流程**
```bash
# 1. 修改 AGENTS.md
echo "- 新规范：使用 React Query" >> AGENTS.md

# 2. 提交并推送
git add AGENTS.md
git commit -m "docs: add React Query convention"
git push

# 3. 团队成员拉取更新
git pull

# 4. 重新加载配置
pi
/reload
```

### 5.3 内容设计建议

**只包含团队共识：**
```markdown
# ✅ 团队共识（应该写在 AGENTS.md）
- 代码风格：Prettier + ESLint
- 提交规范：Conventional Commits
- 测试要求：覆盖率 > 80%

# ❌ 个人偏好（不应该写在 AGENTS.md）
- 使用 Vim 快捷键
- 喜欢简洁的代码风格
- 优先使用函数式编程
```

**个人偏好应该放在全局 AGENTS.md：**
```bash
# ~/.pi/agent/AGENTS.md（个人全局配置）
- 使用 Vim 快捷键
- 代码优先，少说多做
```

---

## 六、热重载机制

### 6.1 /reload 命令

```bash
# 在 Pi 中执行
/reload

# 效果：重新加载所有 Context Files
# - 重新搜索和加载 AGENTS.md
# - 重新加载 settings.json
# - 重新加载 SYSTEM.md / APPEND_SYSTEM.md
```

### 6.2 使用场景

**场景 1：调试配置**
```bash
# 1. 修改 AGENTS.md
echo "测试规则" >> AGENTS.md

# 2. 重新加载
/reload

# 3. 测试是否生效
# 问 Pi："项目有什么规范？"
```

**场景 2：快速迭代**
```bash
# 1. 尝试不同的规范描述
echo "- 使用函数式编程" >> AGENTS.md
/reload
# 测试效果

# 2. 如果不满意，修改
sed -i '' 's/函数式编程/面向对象编程/' AGENTS.md
/reload
# 再次测试
```

### 6.3 热重载 vs 重启

| 操作 | 速度 | 保留对话历史 | 使用场景 |
|------|------|--------------|----------|
| **/reload** | 快（秒级） | ✅ 是 | 调试配置、快速迭代 |
| **重启 Pi** | 慢（需要重新启动） | ❌ 否 | 更新 Pi 版本、清空历史 |

---

## 七、高级技巧

### 7.1 使用 Markdown 特性

**代码块：**
```markdown
# 常用命令
```bash
pnpm dev    # 启动开发服务器
pnpm build  # 构建生产版本
```
```

**表格：**
```markdown
# 目录结构
| 目录 | 说明 |
|------|------|
| src/components/ | React 组件 |
| src/utils/ | 工具函数 |
```

**列表：**
```markdown
# 开发规范
- 代码风格：Prettier + ESLint
- 提交规范：Conventional Commits
  - feat: 新功能
  - fix: 修复 bug
  - docs: 文档更新
```

### 7.2 使用注释和提示

```markdown
# 注意事项
⚠️ 不要修改 src/legacy/ 目录
💡 API 调用统一使用 src/api/client.ts
🔥 所有组件必须有 TypeScript 类型
```

### 7.3 使用链接

```markdown
# 相关文档
- [API 文档](./docs/api.md)
- [架构设计](./docs/architecture.md)
- [贡献指南](./CONTRIBUTING.md)
```

---

## 八、常见问题

### Q1: AGENTS.md 和 CLAUDE.md 有什么区别？

**A:** 没有区别，两者是等价的。Pi Agent 会同时搜索这两个文件名。

```bash
# 以下两种方式都可以
AGENTS.md
CLAUDE.md

# Pi Agent 会加载找到的所有文件
```

### Q2: 如何知道 AGENTS.md 是否被加载？

**A:** Pi 启动时会显示加载的文件：

```bash
$ pi

Loaded context files:
- ~/.pi/agent/AGENTS.md
- /projects/my-app/AGENTS.md

Ready to assist!
```

### Q3: AGENTS.md 的内容有长度限制吗？

**A:** 没有硬性限制，但建议保持简洁（1000 行以内）。过长的内容会：
- 增加 token 消耗
- 降低 AI 的理解效率
- 难以维护

**最佳实践：** 使用渐进式披露，将通用规范放在根目录，特定规范放在子目录。

### Q4: 可以在 AGENTS.md 中使用环境变量吗？

**A:** 不可以。AGENTS.md 是纯文本文件，不支持环境变量替换。

```markdown
# ❌ 不支持
API_BASE_URL: ${API_BASE_URL}

# ✅ 直接写明
API_BASE_URL: https://api.example.com
```

如果需要敏感信息，应该使用 .env 文件，并在 AGENTS.md 中说明：

```markdown
# 环境变量
项目使用 .env 文件管理敏感信息：
- API_BASE_URL - API 基础 URL
- API_KEY - API 密钥
```

---

## 九、实战案例

### 案例 1：个人项目

```markdown
# 个人博客项目

# 技术栈
- Next.js 14 + TypeScript
- Tailwind CSS
- MDX

# 目录结构
- app/ - Next.js App Router
- content/ - MDX 博客文章
- components/ - React 组件

# 开发规范
- 使用 TypeScript 严格模式
- 组件使用函数式组件
- 样式使用 Tailwind CSS

# 常用命令
- `pnpm dev` - 启动开发服务器
- `pnpm build` - 构建生产版本
```

### 案例 2：团队项目

```markdown
# 电商平台项目

# 项目概述
基于 TypeScript + React 的电商平台，支持商品浏览、购物车、订单管理。

# 技术栈
- 前端：React 18 + TypeScript + Vite
- 状态管理：Zustand
- 样式：Tailwind CSS
- API 调用：React Query
- 路由：React Router v6

# 目录结构
```
src/
├── features/       # 功能模块（按业务划分）
│   ├── products/   # 商品模块
│   ├── cart/       # 购物车模块
│   └── orders/     # 订单模块
├── components/     # 共享组件
├── hooks/          # 共享 Hooks
├── utils/          # 工具函数
└── api/            # API 调用
```

# 开发规范
- 代码风格：Prettier + ESLint
- 提交规范：Conventional Commits
- 分支策略：feature/* 分支开发，PR 合并到 main
- 测试要求：核心功能必须有单元测试

# 命名约定
- 组件：PascalCase（UserProfile.tsx）
- 文件：kebab-case（user-profile.utils.ts）
- 功能模块：kebab-case（products/, cart/）

# 工作流程
1. 从 main 创建 feature 分支
2. 开发并提交（遵循 Conventional Commits）
3. 运行 `pnpm test` 和 `pnpm lint`
4. 创建 PR，等待 Code Review
5. Review 通过后合并到 main

# 注意事项
- ⚠️ 不要直接修改 src/legacy/ 目录
- ⚠️ API 调用必须使用 src/api/client.ts
- ⚠️ 所有组件必须有 TypeScript 类型
- ⚠️ 敏感信息使用 .env 文件，不要提交到 Git

# 常用命令
- `pnpm dev` - 启动开发服务器（http://localhost:5173）
- `pnpm build` - 构建生产版本
- `pnpm test` - 运行测试
- `pnpm lint` - 代码检查
- `pnpm format` - 格式化代码
```

---

## 十、总结

**AGENTS.md 的核心要点：**

1. **加载机制** - 累加式加载，所有层级的文件都生效
2. **内容结构** - 项目概述、技术栈、目录结构、开发规范、常用命令
3. **渐进式披露** - 根目录通用规范 + 子目录特定规范
4. **团队协作** - 提交到 Git，团队共享规范
5. **热重载** - 使用 /reload 快速更新配置

**记住：** AGENTS.md 是 Pi Agent 理解项目的"说明书"，写得越清晰，AI 的工作效率越高！

**参考资源：**
- 官方文档：https://github.com/badlogic/pi-mono/blob/main/AGENTS.md
- 官方 README：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md
- Twitter 最佳实践：@badlogicgames 的 progressive disclosure 模式
