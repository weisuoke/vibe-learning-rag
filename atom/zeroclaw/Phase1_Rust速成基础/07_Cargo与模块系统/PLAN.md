# 07_Cargo与模块系统 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_cargo_module_01.md - ZeroClaw Cargo.toml、lib.rs、workspace 结构、feature flags、模块组织模式完整分析

### Context7 官方文档
- ✓ reference/context7_cargo_01.md - Cargo 官方文档：Cargo.toml 结构、workspace、依赖管理、feature flags
- ✓ reference/context7_rust_module_01.md - Google Comprehensive Rust：模块系统、可见性、路径解析、re-export

### 网络搜索
- ✓ reference/search_cargo_module_01.md - Reddit 社区最佳实践：workspace、模块组织、feature flags、TypeScript 类比

### 待抓取链接
（无需额外抓取，现有资料已充分覆盖所有知识点）

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）
- [ ] 03_核心概念_1_Cargo_toml与包管理.md - Cargo.toml 结构、语义化版本、cargo 常用命令 [来源: Context7/源码/网络]
- [ ] 03_核心概念_2_依赖管理.md - dependencies/dev-dependencies/build-dependencies、版本规范、路径依赖 [来源: Context7/源码/网络]
- [ ] 03_核心概念_3_模块系统.md - mod/use/pub、文件映射、路径解析、可见性控制 [来源: Context7/源码/网络]
- [ ] 03_核心概念_4_Workspace.md - 虚拟 manifest、workspace.dependencies 继承、多 crate 组织 [来源: 源码/Context7/网络]
- [ ] 03_核心概念_5_Feature_Flags.md - 条件编译、可选依赖、feature 统一、#[cfg(feature)] [来源: 源码/Context7/网络]

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）
- [ ] 07_实战代码_场景1_从零创建项目.md - cargo new、添加依赖、组织模块、运行测试 [来源: Context7/网络]
- [ ] 07_实战代码_场景2_Workspace多crate项目.md - 仿 ZeroClaw workspace 结构 [来源: 源码/Context7]
- [ ] 07_实战代码_场景3_Feature_Flags实战.md - 可选功能的条件编译 [来源: 源码/Context7/网络]

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
- [x] 阶段二：补充调研（现有资料已充分覆盖）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）
  - [ ] 第一批：00_概览 + 01_30字核心 + 02_第一性原理
  - [ ] 第二批：03_核心概念_1 + 03_核心概念_2 + 03_核心概念_3
  - [ ] 第三批：03_核心概念_4 + 03_核心概念_5 + 04_最小可用
  - [ ] 第四批：05_双重类比 + 06_反直觉点 + 07_实战代码_场景1
  - [ ] 第五批：07_实战代码_场景2 + 07_实战代码_场景3 + 08_面试必问
  - [ ] 第六批：09_化骨绵掌 + 10_一句话总结
