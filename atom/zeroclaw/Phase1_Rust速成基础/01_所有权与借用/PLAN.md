# 01_所有权与借用 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_ownership_01.md - ZeroClaw 中的所有权与借用模式全面分析
  - 分析文件：delegate.rs, mod.rs, anthropic.rs, memory/mod.rs, loop_.rs, shell.rs, schema.rs, engine.rs, pdf_read.rs, memory_store.rs, cron_run.rs, policy.rs, lark.rs, drive.rs, traits.rs

### Context7 官方文档
- ✓ reference/context7_rust_book_01.md - The Rust Programming Language (2024 Edition)
  - 所有权三规则、移动语义、借用规则、生命周期注解、省略规则、'static

### 网络搜索
- ✓ reference/search_ownership_01.md - Rust ownership borrowing best practices 2025-2026
  - 社区最佳实践、Rust 2024 Edition 更新、pretzelhammer 十大误解、生产环境模式

### 待抓取链接
- [ ] https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md

## 文件清单

### 基础维度文件（第一部分）
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（6 个）
- [ ] 03_核心概念_1_所有权三规则与移动语义.md [来源: Rust Book + 源码 delegate.rs]
- [ ] 03_核心概念_2_不可变借用.md [来源: 源码 memory/mod.rs + Rust Book]
- [ ] 03_核心概念_3_可变借用.md [来源: 源码 config/schema.rs, sop/engine.rs]
- [ ] 03_核心概念_4_生命周期注解.md [来源: 源码 agent/loop_.rs + Rust Book]
- [ ] 03_核心概念_5_智能指针与所有权共享.md [来源: 源码 tools/mod.rs, delegate.rs]
- [ ] 03_核心概念_6_内部可变性.md [来源: 源码 security/policy.rs, drive.rs]

### 基础维度文件（第二部分）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（3 个场景，全部 Rust）
- [ ] 07_实战代码_场景1_所有权基础练习.md [来源: Rust Book + 源码]
- [ ] 07_实战代码_场景2_ZeroClaw工具注册表.md [来源: 源码 tools/mod.rs]
- [ ] 07_实战代码_场景3_异步共享状态管理.md [来源: 源码 drive.rs + Tokio 模式]

### 基础维度文件（第三部分）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [ ] 阶段二：补充调研（针对需要更多资料的部分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）
  - [ ] 批次 1：00_概览 + 01_30字核心
  - [ ] 批次 2：02_第一性原理 + 03_核心概念_1 + 03_核心概念_2
  - [ ] 批次 3：03_核心概念_3 + 03_核心概念_4 + 03_核心概念_5
  - [ ] 批次 4：03_核心概念_6 + 04_最小可用 + 05_双重类比
  - [ ] 批次 5：06_反直觉点 + 07_实战_场景1 + 07_实战_场景2
  - [ ] 批次 6：07_实战_场景3 + 08_面试必问 + 09_化骨绵掌
  - [ ] 批次 7：10_一句话总结
