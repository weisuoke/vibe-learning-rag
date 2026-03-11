# 资料索引

生成时间：2026-03-07

## 概览
- 总文件数：7
- 源码分析：3 个
- Context7 文档：1 个
- 搜索结果：2 个
- 抓取内容：2 个（高优先级已完成）

## 按知识点分类

### 核心概念_1_RetryPolicy配置详解
- [source_重试机制_02.md](source_重试机制_02.md) - RetryPolicy NamedTuple 定义
- [context7_langgraph_01.md](context7_langgraph_01.md) - 官方文档用法示例

### 核心概念_2_异常匹配策略
- [source_重试机制_02.md](source_重试机制_02.md) - _should_retry_on 三种匹配方式
- [source_集成方式_03.md](source_集成方式_03.md) - 测试用例验证

### 核心概念_3_退避策略
- [source_重试机制_02.md](source_重试机制_02.md) - 退避计算公式
- [fetch_断路器指南_02.md](fetch_断路器指南_02.md) - 指数退避 + Jitter 生产实践

### 核心概念_4_LangGraph错误层次体系
- [source_错误体系_01.md](source_错误体系_01.md) - 完整异常层次结构

### 核心概念_5_默认重试行为
- [source_重试机制_02.md](source_重试机制_02.md) - default_retry_on 源码分析
- [fetch_断路器指南_02.md](fetch_断路器指南_02.md) - 可重试 vs 不可重试状态码

### 核心概念_6_降级与Fallback模式
- [search_生产模式_02.md](search_生产模式_02.md) - 社区最佳实践
- [fetch_高级错误处理_01.md](fetch_高级错误处理_01.md) - 多层级错误处理架构
- [fetch_断路器指南_02.md](fetch_断路器指南_02.md) - 断路器 + Fallback 链

### 实战场景
- [context7_langgraph_01.md](context7_langgraph_01.md) - 官方代码示例
- [source_集成方式_03.md](source_集成方式_03.md) - 测试用例参考
- [search_错误处理_01.md](search_错误处理_01.md) - 社区实践案例
- [search_生产模式_02.md](search_生产模式_02.md) - 生产级模式

## 按文件类型分类

### 源码分析（3 个）
1. [source_错误体系_01.md](source_错误体系_01.md) - errors.py 错误层次结构
2. [source_重试机制_02.md](source_重试机制_02.md) - _retry.py 重试核心逻辑
3. [source_集成方式_03.md](source_集成方式_03.md) - StateGraph 集成 + 测试

### Context7 文档（1 个）
1. [context7_langgraph_01.md](context7_langgraph_01.md) - 官方重试策略文档

### 搜索结果（2 个）
1. [search_错误处理_01.md](search_错误处理_01.md) - GitHub/Reddit 社区实践
2. [search_生产模式_02.md](search_生产模式_02.md) - 生产级容错模式

### 抓取内容（2 个）
1. [fetch_高级错误处理_01.md](fetch_高级错误处理_01.md) - SparkCo 高级错误处理
2. [fetch_断路器指南_02.md](fetch_断路器指南_02.md) - LLM 重试/回退/断路器指南

## 覆盖度分析
- RetryPolicy 配置：✓ 完全覆盖（3 个资料）
- 异常匹配策略：✓ 完全覆盖（2 个资料）
- 退避策略：✓ 完全覆盖（2 个资料）
- 错误层次体系：✓ 完全覆盖（1 个资料 + 源码）
- 默认重试行为：✓ 完全覆盖（2 个资料）
- 降级与 Fallback：✓ 完全覆盖（3 个资料）
- 实战场景：✓ 完全覆盖（4 个资料）
