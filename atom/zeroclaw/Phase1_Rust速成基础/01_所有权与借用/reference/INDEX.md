# 资料索引

生成时间：2026-03-09

## 概览
- 总文件数：4
- 源码分析：1 个
- Context7 文档：1 个
- 搜索结果：1 个
- 抓取内容：1 个

## 按知识点分类

### 核心概念_1_所有权三规则与移动语义
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - DelegateTool 移动语义、into_iter() 消费

#### Context7 文档
- [context7_rust_book_01.md](context7_rust_book_01.md) - 所有权三规则、Move vs Copy、函数传参

#### 搜索结果
- [search_ownership_01.md](search_ownership_01.md) - 社区最佳实践：优先借用、构造函数返回所有权

### 核心概念_2_不可变借用
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - create_memory(&MemoryConfig, &Path) 不可变借用示例

#### Context7 文档
- [context7_rust_book_01.md](context7_rust_book_01.md) - 多个不可变引用、NLL

### 核心概念_3_可变借用
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - apply_env_overrides(&mut self)、SOP 引擎状态管理

#### Context7 文档
- [context7_rust_book_01.md](context7_rust_book_01.md) - 借用规则、可变与不可变冲突

### 核心概念_4_生命周期注解
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - find_tool<'a>、convert_tools<'a>、'static MutexGuard

#### Context7 文档
- [context7_rust_book_01.md](context7_rust_book_01.md) - 'a 语法、省略规则、结构体生命周期

#### 抓取内容
- [fetch_lifetime_misconceptions_01.md](fetch_lifetime_misconceptions_01.md) - pretzelhammer 10 大生命周期误解

### 核心概念_5_智能指针与所有权共享
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - Box<dyn Tool>、Arc<SecurityPolicy>、Arc<Vec<Arc<dyn Tool>>>

#### 搜索结果
- [search_ownership_01.md](search_ownership_01.md) - 智能指针使用指南（Box → Rc → Arc → RefCell → Cow）

### 核心概念_6_内部可变性
#### 源码分析
- [source_ownership_01.md](source_ownership_01.md) - Mutex<Vec<Instant>>、Arc<Mutex<Option<Instant>>>

#### 搜索结果
- [search_ownership_01.md](search_ownership_01.md) - 并发模式表（Arc<Mutex>、Arc<RwLock>、mpsc）

## 按文件类型分类

### 源码分析（1 个）
1. [source_ownership_01.md](source_ownership_01.md) - ZeroClaw 所有权与借用模式全面分析（15 个文件）

### Context7 文档（1 个）
1. [context7_rust_book_01.md](context7_rust_book_01.md) - The Rust Programming Language (2024 Edition)

### 搜索结果（1 个）
1. [search_ownership_01.md](search_ownership_01.md) - 2025-2026 社区最佳实践与常见误区

### 抓取内容（1 个）
1. [fetch_lifetime_misconceptions_01.md](fetch_lifetime_misconceptions_01.md) - pretzelhammer 十大生命周期误解

## 质量评估
- 高质量资料：4 个
- 中等质量资料：0 个
- 低质量资料：0 个

## 覆盖度分析
- 所有权三规则与移动语义：✓ 完全覆盖（3 个资料）
- 不可变借用：✓ 完全覆盖（2 个资料）
- 可变借用：✓ 完全覆盖（2 个资料）
- 生命周期注解：✓ 完全覆盖（3 个资料）
- 智能指针与所有权共享：✓ 完全覆盖（2 个资料）
- 内部可变性：✓ 完全覆盖（2 个资料）
