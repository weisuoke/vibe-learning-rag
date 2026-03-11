# 10_Onboarding Wizard 与首次配置 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_onboarding_01.md - Onboarding 系统完整架构分析（wizard.rs, main.rs, schema.rs）

### Context7 官方文档
- ✓ reference/context7_clap_01.md - Clap CLI 解析器官方文档（derive, subcommand, optional args）
- ✓ reference/context7_toml_01.md - TOML 序列化/反序列化官方文档（to_string_pretty, from_str）

### 网络搜索
- ✓ reference/search_onboarding_01.md - ZeroClaw onboarding wizard setup（GitHub 搜索）
- ✓ reference/search_onboarding_02.md - ZeroClaw 社区 onboarding 体验（Reddit 搜索）

### 待抓取链接
- 无需额外抓取（现有资料已充分覆盖）

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_CLI命令路由与三种模式.md - Quick/Interactive/Repair 模式分发 [来源: 源码 main.rs + Context7 clap]
- [ ] 03_核心概念_2_Provider选择系统.md - 50+ Provider Tier 分层菜单与认证差异 [来源: 源码 wizard.rs + 网络]
- [ ] 03_核心概念_3_模型发现与缓存.md - 实时模型获取、缓存机制、默认回退 [来源: 源码 wizard.rs]
- [ ] 03_核心概念_4_Config结构与TOML持久化.md - serde + toml 序列化、原子写入、加密管道 [来源: 源码 schema.rs + Context7 toml]
- [ ] 03_核心概念_5_工作区脚手架系统.md - 文件生成、路径解析、环境变量 [来源: 源码 wizard.rs]
- [ ] 03_核心概念_6_完整Wizard扩展配置.md - Memory/Channel/Tunnel/Hardware/Tools [来源: 源码 wizard.rs + 网络]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_QuickSetup一键启动.md - 命令行快速配置 + 验证 [来源: 源码 + 网络]
- [ ] 07_实战代码_场景2_手动编辑config_toml.md - 配置文件直接修改 + config 命令 [来源: 源码 + 网络]
- [ ] 07_实战代码_场景3_Provider切换实战.md - Ollama → OpenRouter 完整切换 [来源: 源码 + 网络]

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（无需额外抓取，现有资料充分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）
  - [ ] 批次 1：00_概览 + 01_30字核心 + 02_第一性原理
  - [ ] 批次 2：03_核心概念_1 + 03_核心概念_2
  - [ ] 批次 3：03_核心概念_3 + 03_核心概念_4
  - [ ] 批次 4：03_核心概念_5 + 03_核心概念_6
  - [ ] 批次 5：04_最小可用 + 05_双重类比 + 06_反直觉点
  - [ ] 批次 6：07_实战代码_场景1 + 07_实战代码_场景2
  - [ ] 批次 7：07_实战代码_场景3 + 08_面试必问
  - [ ] 批次 8：09_化骨绵掌 + 10_一句话总结
