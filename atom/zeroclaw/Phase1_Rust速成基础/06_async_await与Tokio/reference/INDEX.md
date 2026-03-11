# 资料索引

生成时间：2026-03-10

## 概览
- 总文件数：4
- 源码分析：1 个
- Context7 文档：1 个
- 搜索结果：2 个
- 抓取内容：0 个

## 按知识点分类

### async fn 与 Future 机制
#### 源码分析
- [source_async_tokio_01.md](source_async_tokio_01.md) - ZeroClaw async/await 全景（1939 async fn、124 #[async_trait]）

#### Context7 文档
- [context7_tokio_01.md](context7_tokio_01.md) - Tokio 官方文档 + async-trait 文档

#### 搜索结果
- [search_async_tokio_02.md](search_async_tokio_02.md) - Rust Future vs JS Promise 对比

### Tokio 运行时与最佳实践
#### 搜索结果
- [search_async_tokio_01.md](search_async_tokio_01.md) - 2025-2026 Reddit 最佳实践与常见陷阱

## 按文件类型分类

### 源码分析（1 个）
1. [source_async_tokio_01.md](source_async_tokio_01.md) - ZeroClaw async/await 与 Tokio 全景分析
   - 分析了 12 个核心文件
   - 关键统计：1939 async fn、124 #[async_trait]、14 tokio::spawn、7 tokio::select!

### Context7 文档（1 个）
1. [context7_tokio_01.md](context7_tokio_01.md) - Tokio + async-trait 官方文档
   - tokio::spawn、select!、mpsc、timeout 等 API 文档
   - async-trait 宏原理与使用方法

### 搜索结果（2 个）
1. [search_async_tokio_01.md](search_async_tokio_01.md) - Rust async/await + Tokio 最佳实践（2025-2026）
   - 7 个最佳实践 + 6 个常见陷阱
2. [search_async_tokio_02.md](search_async_tokio_02.md) - Rust Future vs JavaScript Promise 对比
   - 惰性 vs 立即执行、状态机编译、取消行为

## 质量评估
- 高质量资料：4 个（全部来源可靠）
- 中等质量资料：0 个
- 低质量资料：0 个

## 覆盖度分析
- async fn 与 Future 机制：✓ 完全覆盖（3 个资料）
- Tokio 运行时：✓ 完全覆盖（2 个资料）
- tokio::spawn：✓ 完全覆盖（2 个资料）
- tokio::select!：✓ 完全覆盖（2 个资料）
- mpsc 通道：✓ 完全覆盖（2 个资料）
- #[async_trait]：✓ 完全覆盖（2 个资料）
- 最佳实践与陷阱：✓ 完全覆盖（1 个专题资料）
- Rust vs JS 对比：✓ 完全覆盖（1 个专题资料）
