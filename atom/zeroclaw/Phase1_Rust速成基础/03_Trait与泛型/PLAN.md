# 03_Trait与泛型 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_trait_generics_01.md - ZeroClaw 10个核心 Trait 定义 + 泛型模式分析

### Context7 官方文档
- ✓ reference/context7_rust_reference_01.md - Rust Reference: traits, generics, where clause

### 网络搜索
- ✓ reference/search_trait_generics_01.md - Rust trait generics best practices 2025-2026
- ✓ reference/search_async_trait_01.md - Rust async_trait Send Sync patterns

### 待抓取链接
- 无需额外抓取，现有资料已充分覆盖

## 知识点拆解

### 核心概念分析

基于 k.md 定义：
> trait 定义与实现、泛型约束（where）、关联类型
> 前端类比：TypeScript interface + generics
> ZeroClaw 场景：Provider/Tool/Memory/Channel 四大 Trait 定义

拆解为 **5 个核心概念**：

1. **Trait 定义与实现** - trait 声明、方法签名、默认实现、impl Trait for Type
2. **泛型函数与泛型结构体** - 类型参数 `<T>`、单态化、impl Trait 语法糖
3. **Trait 约束与 where 子句** - 泛型约束语法、多重约束、where 子句高级用法
4. **关联类型** - 关联类型 vs 泛型参数、Iterator::Item 模式
5. **超级 Trait 与 Trait 组合** - Send + Sync、Trait 继承、#[async_trait]

拆解为 **3 个实战场景**：

1. **实现 ZeroClaw Tool Trait** - 最简单的核心 Trait，5 个方法
2. **泛型工厂函数** - create_memory_with_builders 模式，where 子句实战
3. **构建可插拔架构** - 用 Trait + 泛型实现 Provider 注册与切换

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（5 个）
- [x] 03_核心概念_1_Trait定义与实现.md [来源: 源码+Context7]
- [x] 03_核心概念_2_泛型函数与泛型结构体.md [来源: 源码+Context7]
- [x] 03_核心概念_3_Trait约束与where子句.md [来源: Context7+网络]
- [x] 03_核心概念_4_关联类型.md [来源: Context7+网络]
- [x] 03_核心概念_5_超级Trait与Trait组合.md [来源: 源码+网络]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（3 个）
- [x] 07_实战代码_场景1_实现Tool_Trait.md [来源: 源码]
- [x] 07_实战代码_场景2_泛型工厂函数.md [来源: 源码]
- [x] 07_实战代码_场景3_可插拔架构.md [来源: 源码+网络]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 生成批次计划（使用 subagent）

### 批次 1（3 个 subagent × 2 文件）
- Agent A: 00_概览.md + 01_30字核心.md
- Agent B: 02_第一性原理.md + 03_核心概念_1_Trait定义与实现.md
- Agent C: 03_核心概念_2_泛型函数与泛型结构体.md + 03_核心概念_3_Trait约束与where子句.md

### 批次 2（3 个 subagent × 2 文件）
- Agent D: 03_核心概念_4_关联类型.md + 03_核心概念_5_超级Trait与Trait组合.md
- Agent E: 04_最小可用.md + 05_双重类比.md
- Agent F: 06_反直觉点.md + 07_实战代码_场景1_实现Tool_Trait.md

### 批次 3（3 个 subagent × 2 文件）
- Agent G: 07_实战代码_场景2_泛型工厂函数.md + 07_实战代码_场景3_可插拔架构.md
- Agent H: 08_面试必问.md + 09_化骨绵掌.md
- Agent I: 10_一句话总结.md + PLAN.md 最终更新

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析（基于源码探索）
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研（现有资料充分，无需额外抓取）
- [x] 阶段三：文档生成
  - [x] 批次 1: 00_概览 + 01_30字核心 + 02_第一性原理 + 核心概念1 + 核心概念2 + 核心概念3
  - [x] 批次 2: 核心概念4 + 核心概念5 + 04_最小可用 + 05_双重类比 + 06_反直觉点 + 实战1
  - [x] 批次 3: 实战2 + 实战3 + 08_面试必问 + 09_化骨绵掌 + 10_一句话总结
