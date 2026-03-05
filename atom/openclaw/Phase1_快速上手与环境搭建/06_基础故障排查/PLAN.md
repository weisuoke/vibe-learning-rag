# OpenClaw「基础故障排查」知识点生成计划

**版本**: v1.0
**创建时间**: 2026-02-24
**目标目录**: `atom/openclaw/Phase1_快速上手与环境搭建/06_基础故障排查/`

---

## 一、任务概述

为 OpenClaw 项目生成「基础故障排查」知识点的完整文档体系，遵循原子化知识点规范。

**目标**：
- 生成完整的 10 个维度文档
- 覆盖 7 个核心概念
- 提供 6 个实战场景
- 基于 2025-2026 最新资料
- 所有代码使用 TypeScript
- 所有类比使用后端开发 + 日常生活

---

## 二、数据来源记录

### 2.1 源码分析（第一优先级）✅

已完成源码文件读取和分析，保存到 `reference/` 目录：

| 文件名 | 来源 | 内容摘要 |
|--------|------|----------|
| `source_troubleshooting_main.md` | `docs/help/troubleshooting.md` | 60秒诊断流程、决策树、7种故障场景 |
| `source_channels_troubleshooting.md` | `docs/channels/troubleshooting.md` | 通道级故障排查、8个通道的故障特征 |
| `source_gateway_troubleshooting.md` | `docs/gateway/troubleshooting.md` | 网关深度故障排查、7个深度场景 |
| `source_doctor_command.md` | `docs/cli/doctor.md` | doctor 命令文档、健康检查和修复 |
| `source_debugging_tools.md` | `docs/help/debugging.md` | 调试工具、开发模式、原始流日志 |
| `source_faq_summary.md` | `docs/help/faq.md` | FAQ 摘要、60秒诊断、环境要求 |
| `source_package_info.md` | `package.json` | 包信息、依赖、脚本、环境要求 |

**核心发现**：
- OpenClaw 提供完整的诊断工具链
- 60秒快速诊断流程是核心方法论
- doctor 命令是自动修复的关键工具
- 支持 8 个主要通道（WhatsApp, Telegram, Discord, Slack, iMessage, Signal, Matrix, LINE）
- 日志系统提供实时跟踪和分析

### 2.2 Context7 官方文档（第二优先级）✅

已完成 Context7 库查询，保存到 `reference/context7_libraries.md`：

| 库名 | Library ID | 基准分数 | 代码片段数 | 用途 |
|------|-----------|---------|-----------|------|
| pnpm | /pnpm/pnpm | 89.8 | 245 | 包管理器故障排查 |
| TypeScript | /microsoft/typescript | 62.2 | 22317 | TypeScript 调试技术 |
| Node.js | /nodejs/node | 92.2 | 18524 | Node.js 诊断工具 |

**后续查询计划**：
- pnpm 安装问题排查
- TypeScript 编译错误诊断
- Node.js 进程监控和健康检查

### 2.3 Grok-mcp 网络搜索（第三优先级）✅

已完成网络搜索，保存到 `reference/` 目录：

**搜索 1: OpenClaw troubleshooting**
- 8 个搜索结果
- 涵盖官方文档、社区经验、GitHub Issues、视频教程
- 关键发现：端口 18789 绑定问题、网关重启循环、Docker 部署超时

**搜索 2: openclaw doctor command**
- 6 个搜索结果
- 涵盖官方文档、安装指南、自托管指南
- 关键发现：doctor 命令的主要用途、常用选项、社区最佳实践

---

## 三、核心概念拆分（7个）

基于源码分析和搜索结果，识别出 7 个核心概念：

### 3.1 核心概念 1：openclaw doctor 诊断工具

**定义**：OpenClaw 的健康检查和自动修复工具

**关键点**：
- 健康检查功能
- 自动修复机制（--repair, --fix）
- 配置迁移和清理
- 配置备份
- 环境变量检查

**实战应用**：
- 安装后验证
- 升级后检查
- 配置错误修复
- 服务冲突解决

**数据来源**：
- `source_doctor_command.md`
- `search_doctor_command.md`
- `source_troubleshooting_main.md`

### 3.2 核心概念 2：日志系统与查看

**定义**：OpenClaw 的日志记录和实时跟踪系统

**关键点**：
- `openclaw logs --follow` 实时日志
- 日志文件位置（`/tmp/openclaw/openclaw-*.log`）
- 日志特征识别
- 原始流日志（--raw-stream）
- 日志级别和过滤

**实战应用**：
- 实时故障诊断
- 日志特征识别
- 错误消息解读
- 调试模式日志

**数据来源**：
- `source_debugging_tools.md`
- `source_troubleshooting_main.md`
- `source_gateway_troubleshooting.md`

### 3.3 核心概念 3：网关状态检查

**定义**：Gateway 运行时状态管理和监控

**关键点**：
- `openclaw gateway status` 命令
- Runtime 状态（running/stopped）
- RPC probe 检查
- 端口监听状态
- 配置一致性检查

**实战应用**：
- 网关启动验证
- 端口冲突诊断
- 配置不一致检测
- 服务状态监控

**数据来源**：
- `source_gateway_troubleshooting.md`
- `source_troubleshooting_main.md`
- `source_faq_summary.md`

### 3.4 核心概念 4：频道连接诊断

**定义**：多通道消息系统的连接和消息流诊断

**关键点**：
- `openclaw channels status --probe` 命令
- 8 个主要通道的故障特征
- 配对机制（pairing）
- 提及门控（mention gating）
- 权限和范围检查

**实战应用**：
- 通道连接验证
- 消息流诊断
- 配对问题解决
- 权限配置检查

**数据来源**：
- `source_channels_troubleshooting.md`
- `source_gateway_troubleshooting.md`
- `source_troubleshooting_main.md`

### 3.5 核心概念 5：配置验证

**定义**：OpenClaw 配置文件的验证和修复机制

**关键点**：
- 配置文件位置（`~/.openclaw/openclaw.json`）
- 配置验证规则
- 配置迁移
- 配置备份和恢复
- 环境变量覆盖

**实战应用**：
- 配置错误检测
- 配置迁移
- 配置备份
- 环境变量冲突解决

**数据来源**：
- `source_doctor_command.md`
- `source_gateway_troubleshooting.md`
- `source_faq_summary.md`

### 3.6 核心概念 6：常见错误类型

**定义**：OpenClaw 常见故障模式和错误消息

**关键点**：
- 端口 18789 绑定问题
- 网关重启循环
- 认证失败（unauthorized）
- 配对待定（pairing required）
- 权限缺失（permission required）
- 配置不匹配

**实战应用**：
- 错误消息识别
- 快速定位问题类型
- 选择正确的修复方法

**数据来源**：
- `source_gateway_troubleshooting.md`
- `source_channels_troubleshooting.md`
- `search_openclaw_troubleshooting.md`

### 3.7 核心概念 7：自动修复机制

**定义**：OpenClaw 的自动化故障修复系统

**关键点**：
- `openclaw doctor --repair` 自动修复
- `openclaw doctor --fix` 修复别名
- 配置清理和备份
- 权限修复
- 服务重启

**实战应用**：
- 自动化故障修复
- 配置清理
- 权限问题解决
- 服务恢复

**数据来源**：
- `source_doctor_command.md`
- `search_doctor_command.md`
- `source_gateway_troubleshooting.md`

---

## 四、实战场景拆分（6个）

### 4.1 场景 1：60秒快速诊断

**场景描述**：使用标准化诊断流程快速定位问题

**涉及命令**：
```bash
openclaw status
openclaw status --all
openclaw gateway probe
openclaw gateway status
openclaw doctor
openclaw channels status --probe
openclaw logs --follow
```

**预期输出**：
- 健康状态摘要
- 问题类型识别
- 修复建议

**数据来源**：
- `source_troubleshooting_main.md`
- `source_faq_summary.md`

### 4.2 场景 2：网关无法启动排查

**场景描述**：诊断和修复网关启动失败问题

**常见原因**：
- 端口 18789 被占用
- 配置错误（gateway.mode 未设置）
- 认证配置缺失
- 权限问题

**诊断步骤**：
1. 检查端口占用
2. 验证配置文件
3. 检查认证设置
4. 运行 doctor --repair

**数据来源**：
- `source_gateway_troubleshooting.md`
- `search_openclaw_troubleshooting.md`

### 4.3 场景 3：频道消息不流动排查

**场景描述**：诊断通道连接但消息不流动的问题

**常见原因**：
- 配对待定（pairing required）
- 提及门控（mention required）
- 权限缺失（missing_scope）
- 路由策略阻止

**诊断步骤**：
1. 检查通道状态
2. 验证配对状态
3. 检查提及要求
4. 验证权限配置

**数据来源**：
- `source_channels_troubleshooting.md`
- `source_gateway_troubleshooting.md`

### 4.4 场景 4：日志分析实战

**场景描述**：通过日志特征识别和解决问题

**日志特征示例**：
- `drop guild message (mention required` → 提及门控
- `pairing request` → 配对待定
- `unauthorized` → 认证失败
- `EADDRINUSE` → 端口冲突

**分析步骤**：
1. 实时日志跟踪
2. 识别日志特征
3. 定位问题类型
4. 应用修复方法

**数据来源**：
- `source_troubleshooting_main.md`
- `source_gateway_troubleshooting.md`
- `source_debugging_tools.md`

### 4.5 场景 5：配置验证与修复

**场景描述**：使用 doctor 命令验证和修复配置

**操作步骤**：
1. 运行 `openclaw doctor`
2. 查看健康检查报告
3. 运行 `openclaw doctor --repair`
4. 验证修复结果

**修复内容**：
- 配置迁移
- 无效键清理
- 环境变量冲突解决
- 权限修复

**数据来源**：
- `source_doctor_command.md`
- `search_doctor_command.md`

### 4.6 场景 6：生产环境故障排查

**场景描述**：综合诊断生产环境的复杂问题

**诊断流程**：
1. 60秒快速诊断
2. 深度探测（--deep）
3. 日志分析
4. 配置验证
5. 自动修复
6. 验证恢复

**涉及工具**：
- openclaw status --all
- openclaw doctor --deep
- openclaw logs --follow
- openclaw gateway status --json
- openclaw channels status --probe

**数据来源**：
- 所有 source 文件
- 所有 search 文件

---

## 五、文件清单（预计 20-25 个文件）

### 5.1 基础维度（第一部分）- 3 个文件

1. `00_概览.md` - 知识点概览和学习路径
2. `01_30字核心.md` - 一句话核心定义
3. `02_第一性原理.md` - 从根本问题出发的思考

### 5.2 核心概念（7 个文件）

4. `03_核心概念_1_openclaw_doctor诊断工具.md`
5. `03_核心概念_2_日志系统与查看.md`
6. `03_核心概念_3_网关状态检查.md`
7. `03_核心概念_4_频道连接诊断.md`
8. `03_核心概念_5_配置验证.md`
9. `03_核心概念_6_常见错误类型.md`
10. `03_核心概念_7_自动修复机制.md`

### 5.3 基础维度（第二部分）- 3 个文件

11. `04_最小可用.md` - 20%核心知识
12. `05_双重类比.md` - 后端开发 + 日常生活类比
13. `06_反直觉点.md` - 3个常见误区

### 5.4 实战代码（6 个场景）

14. `07_实战代码_场景1_60秒快速诊断.md`
15. `07_实战代码_场景2_网关无法启动排查.md`
16. `07_实战代码_场景3_频道消息不流动排查.md`
17. `07_实战代码_场景4_日志分析实战.md`
18. `07_实战代码_场景5_配置验证与修复.md`
19. `07_实战代码_场景6_生产环境故障排查.md`

### 5.5 基础维度（第三部分）- 3 个文件

20. `08_面试必问.md` - 1-2个高频面试问题
21. `09_化骨绵掌.md` - 10个2分钟知识卡片
22. `10_一句话总结.md` - 最终总结

### 5.6 参考资料目录

23. `reference/INDEX.md` - 资料索引

**总计**：22-23 个文件

---

## 六、生成规范

### 6.1 代码语言

**必须使用 TypeScript/JavaScript**：
- ✅ 所有代码示例使用 TypeScript
- ✅ 使用 Node.js 22+ 特性
- ✅ 使用 pnpm 作为包管理器
- ❌ 不使用 Python

### 6.2 类比系统

**必须使用后端开发 + 日常生活类比**：
- ✅ 后端开发类比（Nginx, Redis, API Gateway, 微服务等）
- ✅ 日常生活类比（邮局、图书馆、医院等）
- ❌ 不使用前端开发类比

### 6.3 文件长度

**严格控制 300-500 行**：
- 每个文件 300-500 行
- 超过 500 行立即拆分
- 核心概念文件可适当延长到 600 行

### 6.4 引用规范

**所有内容必须标注来源**：
- 源码引用：`**来源**: sourcecode/openclaw/docs/...`
- Context7 引用：`**来源**: Context7 - /nodejs/node`
- 搜索引用：`**来源**: Grok-mcp search - ...`

### 6.5 实战代码规范

**所有代码必须可运行**：
- 完整的 TypeScript 代码
- 包含必要的 import 语句
- 包含类型定义
- 包含预期输出示例
- 包含错误处理

---

## 七、生成进度跟踪

### 阶段一：数据收集 ✅

- [x] 源码分析（7个文件）
- [x] Context7 查询（3个库）
- [x] Grok-mcp 搜索（2次搜索）
- [x] 数据保存到 reference/
- [x] 生成 PLAN.md

### 阶段二：补充调研（待执行）

- [ ] 识别需要补充资料的部分
- [ ] Context7 深度查询
- [ ] Grok-mcp 补充搜索
- [ ] 生成 FETCH_TASK.json
- [ ] 更新 PLAN.md
- [ ] 生成 reference/INDEX.md

### 阶段三：文档生成（待执行）

**基础维度（第一部分）**：
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

**核心概念（7个）**：
- [ ] 03_核心概念_1_openclaw_doctor诊断工具.md
- [ ] 03_核心概念_2_日志系统与查看.md
- [ ] 03_核心概念_3_网关状态检查.md
- [ ] 03_核心概念_4_频道连接诊断.md
- [ ] 03_核心概念_5_配置验证.md
- [ ] 03_核心概念_6_常见错误类型.md
- [ ] 03_核心概念_7_自动修复机制.md

**基础维度（第二部分）**：
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

**实战代码（6个场景）**：
- [ ] 07_实战代码_场景1_60秒快速诊断.md
- [ ] 07_实战代码_场景2_网关无法启动排查.md
- [ ] 07_实战代码_场景3_频道消息不流动排查.md
- [ ] 07_实战代码_场景4_日志分析实战.md
- [ ] 07_实战代码_场景5_配置验证与修复.md
- [ ] 07_实战代码_场景6_生产环境故障排查.md

**基础维度（第三部分）**：
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

**参考资料**：
- [ ] reference/INDEX.md

---

## 八、质量标准

### 8.1 内容质量

- [ ] 所有 10 个维度完整
- [ ] 7 个核心概念详细讲解
- [ ] 6 个实战场景可运行代码
- [ ] 所有内容基于 2025-2026 最新资料
- [ ] 文档质量达到现有文档标准（90/100）

### 8.2 代码质量

- [ ] 所有代码使用 TypeScript
- [ ] 所有代码可直接运行
- [ ] 代码有详细注释
- [ ] 变量命名清晰有意义
- [ ] 输出结果清晰可读

### 8.3 类比质量

- [ ] 所有类比使用后端开发 + 日常生活
- [ ] 类比准确不误导
- [ ] 类比易于理解

### 8.4 引用质量

- [ ] 所有内容标注来源
- [ ] reference/ 目录包含完整资料
- [ ] 引用准确可追溯

---

## 九、风险与注意事项

### 9.1 技术风险

1. **代码语言**：必须使用 TypeScript，不是 Python
2. **类比系统**：必须使用后端开发类比，不是前端
3. **版本匹配**：OpenClaw 要求 Node.js >= 22，pnpm 10.23.0

### 9.2 内容风险

1. **文件长度**：严格控制 300-500 行，超过立即拆分
2. **数据来源**：优先源码，其次 Context7，最后网络
3. **引用规范**：所有内容必须标注来源

### 9.3 执行风险

1. **不使用 subagent**：逐个文件生成，避免并行问题
2. **用户确认**：在生成文档前需要用户确认拆解方案
3. **进度跟踪**：使用 PLAN.md 跟踪生成进度

---

## 十、下一步行动

### 10.1 立即行动（阶段一完成）

1. ✅ 向用户展示核心概念拆解方案
2. ⏳ 等待用户确认或反馈
3. ⏳ 根据反馈调整拆解方案

### 10.2 后续行动（阶段二）

1. 识别需要补充资料的部分
2. 执行补充调研
3. 生成 FETCH_TASK.json
4. 更新 PLAN.md

### 10.3 最终行动（阶段三）

1. 按顺序生成所有文档
2. 质量检查
3. 生成 reference/INDEX.md
4. 最终验证

---

**计划状态**：阶段一完成，等待用户确认
**下一步**：用户确认拆解方案后进入阶段二
**预计总工作量**：7-11 小时
