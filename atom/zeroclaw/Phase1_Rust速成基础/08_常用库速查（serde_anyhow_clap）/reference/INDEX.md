# 资料索引

生成时间：2026-03-11

## 概览
- 总文件数：5
- 源码分析：1 个
- Context7 文档：3 个
- 搜索结果：1 个
- 抓取内容：0 个

## 按知识点分类

### serde 序列化框架
#### 源码分析
- [source_常用库_01.md](source_常用库_01.md) - ZeroClaw Cargo.toml / main.rs / config/schema.rs 中的 serde 使用模式

#### Context7 文档
- [context7_serde_01.md](context7_serde_01.md) - serde 官方文档：derive、三层属性体系、skip/rename/default

### anyhow 错误处理
#### 源码分析
- [source_常用库_01.md](source_常用库_01.md) - ZeroClaw 中 anyhow Result/Context/bail 的使用模式

#### Context7 文档
- [context7_anyhow_01.md](context7_anyhow_01.md) - anyhow 官方文档：Result、Context、downcast、错误链

### clap CLI 解析
#### 源码分析
- [source_常用库_01.md](source_常用库_01.md) - ZeroClaw main.rs 中 Parser/Subcommand/ValueEnum 的使用

#### Context7 文档
- [context7_clap_01.md](context7_clap_01.md) - clap 官方文档：derive API 四件套（Parser/Subcommand/Args/ValueEnum）

### tracing 日志系统
#### 源码分析
- [source_常用库_01.md](source_常用库_01.md) - ZeroClaw 各模块中 tracing 的使用模式

### 社区最佳实践
#### 搜索结果
- [search_常用库_01.md](search_常用库_01.md) - Reddit r/rust 2025-2026 最佳实践讨论

## 按文件类型分类

### 源码分析（1 个）
1. [source_常用库_01.md](source_常用库_01.md) - ZeroClaw 源码分析（Cargo.toml / main.rs / lib.rs / config/schema.rs）

### Context7 文档（3 个）
1. [context7_serde_01.md](context7_serde_01.md) - serde 官方文档
2. [context7_anyhow_01.md](context7_anyhow_01.md) - anyhow 官方文档
3. [context7_clap_01.md](context7_clap_01.md) - clap 官方文档

### 搜索结果（1 个）
1. [search_常用库_01.md](search_常用库_01.md) - Rust serde/anyhow/clap 最佳实践 2025-2026

## 质量评估
- 高质量资料：5 个（全部为官方文档或源码分析）
- 中等质量资料：0 个
- 低质量资料：0 个

## 覆盖度分析
- serde 序列化：✓ 完全覆盖（源码 + Context7）
- anyhow 错误处理：✓ 完全覆盖（源码 + Context7）
- clap CLI 解析：✓ 完全覆盖（源码 + Context7）
- tracing 日志：✓ 基本覆盖（源码分析）
- 社区最佳实践：✓ 完全覆盖（Grok-mcp 搜索）
