# 05_动态分发与 Trait Object - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_dynamic_dispatch_01.md - ZeroClaw 10个核心 Trait 动态分发模式、Box vs Arc 选择、ArcDelegatingTool 桥接、Observer as_any() 模式

### Context7 官方文档
- ✓ reference/context7_rust_reference_01.md - Rust Reference: dyn compatible methods, dyn incompatible traits, trait object vtable, fat pointer

### 网络搜索
- ✓ reference/search_dynamic_dispatch_01.md - Rust trait object dynamic dispatch best practices 2025-2026, trait upcasting (1.86), dynosaur async, performance benchmarks

### 待抓取链接
- 无需额外抓取，现有资料已充分覆盖

## 知识点拆解

### 核心概念分析

基于 k.md 定义：
> Box<dyn Trait>、Arc<dyn Trait>、对象安全、静态 vs 动态分发
> 前端类比：依赖注入容器、多态接口
> ZeroClaw 场景：`Box<dyn Provider>`、`Arc<dyn Memory>`、`Vec<Box<dyn Tool>>`

拆解为 **5 个核心概念**：

1. **Trait Object 与 dyn 关键字** - 胖指针结构（data + vtable）、类型擦除、dyn Trait 语法
2. **Box<dyn Trait> 与堆分配** - 单一所有权的 trait object、工厂模式返回、异构集合
3. **Arc<dyn Trait> 与共享所有权** - 线程安全共享、Arc vs Box 选择、ArcDelegatingTool 桥接模式
4. **对象安全（dyn 兼容性）** - 5条规则、async trait 与 #[async_trait]、where Self: Sized 技巧
5. **静态分发 vs 动态分发** - 泛型单态化 vs vtable、性能对比、何时选哪个

拆解为 **4 个实战场景**（用户要求增加）：

1. **ZeroClaw 可插拔架构** - Box<dyn Provider>、Vec<Box<dyn Tool>>、config.toml 驱动运行时选择
2. **异构集合与工厂模式** - create_provider/create_memory 工厂函数、match 分支构建
3. **Trait Object 高级模式** - as_any() 下行转换、ArcDelegatingTool 桥接、trait upcasting
4. **enum_dispatch 对比** - 封闭集合优化、enum vs dyn 性能对比、何时用 enum 替代 dyn

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（5 个）
- [x] 03_核心概念_1_Trait_Object与dyn关键字.md [来源: Context7+网络]
- [x] 03_核心概念_2_Box_dyn_Trait与堆分配.md [来源: 源码+Context7]
- [x] 03_核心概念_3_Arc_dyn_Trait与共享所有权.md [来源: 源码+网络]
- [x] 03_核心概念_4_对象安全与dyn兼容性.md [来源: Context7+网络]
- [x] 03_核心概念_5_静态分发vs动态分发.md [来源: 网络+源码]

### 基础维度文件（续）
- [x] 04_最小可用.md
- [x] 05_双重类比.md
- [x] 06_反直觉点.md

### 实战代码文件（4 个）
- [x] 07_实战代码_场景1_ZeroClaw可插拔架构.md [来源: 源码]
- [x] 07_实战代码_场景2_异构集合与工厂模式.md [来源: 源码]
- [x] 07_实战代码_场景3_Trait_Object高级模式.md [来源: 源码+网络]
- [x] 07_实战代码_场景4_enum_dispatch对比.md [来源: 网络]

### 基础维度文件（续）
- [x] 08_面试必问.md
- [x] 09_化骨绵掌.md
- [x] 10_一句话总结.md

## 生成批次计划（使用 subagent）

### 批次 1（3 个 subagent × 2 文件）
- Agent A: 00_概览.md + 01_30字核心.md
- Agent B: 02_第一性原理.md + 03_核心概念_1_Trait_Object与dyn关键字.md
- Agent C: 03_核心概念_2_Box_dyn_Trait与堆分配.md + 03_核心概念_3_Arc_dyn_Trait与共享所有权.md

### 批次 2（3 个 subagent × 2 文件）
- Agent D: 03_核心概念_4_对象安全与dyn兼容性.md + 03_核心概念_5_静态分发vs动态分发.md
- Agent E: 04_最小可用.md + 05_双重类比.md
- Agent F: 06_反直觉点.md + 07_实战代码_场景1_ZeroClaw可插拔架构.md

### 批次 3（3 个 subagent × 2 文件）
- Agent G: 07_实战代码_场景2_异构集合与工厂模式.md + 07_实战代码_场景3_Trait_Object高级模式.md
- Agent H: 07_实战代码_场景4_enum_dispatch对比.md + 08_面试必问.md
- Agent I: 09_化骨绵掌.md + 10_一句话总结.md

## 生成进度
- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案（用户要求增加实战场景，已调整）
  - [x] 1.4 Plan 最终确定
- [x] 阶段二：补充调研 — 无需额外抓取
- [x] 阶段三：文档生成（读取 reference/ 中的所有资料） ✓
  - [x] 批次 1：基础维度 + 核心概念 1-3 ✓
  - [x] 批次 2：核心概念 4-5 + 基础维度续 ✓
  - [x] 批次 3：实战代码续 + 基础维度续 ✓
