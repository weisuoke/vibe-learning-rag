# 资料索引

生成时间：2026-03-07

## 概览
- 总文件数：9
- 源码分析：3 个
- Context7 文档：2 个
- 搜索结果：2 个
- 抓取内容：2 个

## 按知识点分类

### 核心概念_1_step_timeout的真实语义
- [source_步级超时_01.md](source_步级超时_01.md) - `Pregel.step_timeout` 定义、传递与抛错链路
- [context7_langgraph_01.md](context7_langgraph_01.md) - 官方文档侧的长等待替代方案

### 核心概念_2_超时异常与取消传播
- [source_步级超时_01.md](source_步级超时_01.md) - `_panic_or_proceed()` 取消 inflight 并抛错
- [source_流式平台_03.md](source_流式平台_03.md) - `asyncio.TimeoutError` 与任务取消

### 核心概念_3_子图超时与ParentCommand
- [source_子图传播_02.md](source_子图传播_02.md) - 子图 / 父图双层 timeout
- [fetch_step_timeout_bug_01.md](fetch_step_timeout_bug_01.md) - 社区 issue 暴露复杂控制流边界

### 核心概念_4_流式消费背压与astream超时
- [source_流式平台_03.md](source_流式平台_03.md) - `astream()` 背压触发 timeout 测试
- [fetch_长任务超时_02.md](fetch_长任务超时_02.md) - 长任务不发状态导致前端超时

### 核心概念_5_节点内I/O超时组合
- [context7_httpx_02.md](context7_httpx_02.md) - HTTPX connect/read/write/pool timeout
- [source_步级超时_01.md](source_步级超时_01.md) - 图外层 step budget 的边界

### 核心概念_6_SDK与平台运行超时
- [source_流式平台_03.md](source_流式平台_03.md) - SDK timeout、RunStatus timeout、APITimeoutError
- [search_生产部署超时_02.md](search_生产部署超时_02.md) - 平台部署 / 长运行任务 timeout 讨论

### 实战场景
- [context7_httpx_02.md](context7_httpx_02.md) - 节点内 HTTP timeout 设计
- [search_超时控制_01.md](search_超时控制_01.md) - 社区 guardrail 组合实践
- [search_生产部署超时_02.md](search_生产部署超时_02.md) - 异步 run + polling 生产模式

## 按文件类型分类

### 源码分析（3 个）
1. [source_步级超时_01.md](source_步级超时_01.md) - `step_timeout` 定义、调用链与抛错逻辑
2. [source_子图传播_02.md](source_子图传播_02.md) - 子图 timeout 与 `ParentCommand` 传播
3. [source_流式平台_03.md](source_流式平台_03.md) - 流式背压、SDK timeout、RunStatus timeout

### Context7 文档（2 个）
1. [context7_langgraph_01.md](context7_langgraph_01.md) - 官方 durable execution / interrupt 路线
2. [context7_httpx_02.md](context7_httpx_02.md) - HTTPX timeout 官方配置说明

### 搜索结果（2 个）
1. [search_超时控制_01.md](search_超时控制_01.md) - `step_timeout` 与 timeout best practices
2. [search_生产部署超时_02.md](search_生产部署超时_02.md) - 长任务与部署 timeout

### 抓取内容（2 个）
1. [fetch_step_timeout_bug_01.md](fetch_step_timeout_bug_01.md) - `step_timeout` + 多代理 issue
2. [fetch_长任务超时_02.md](fetch_长任务超时_02.md) - 长运行任务不发进度导致超时

## 质量评估
- 高质量资料：7 个
- 中等质量资料：2 个
- 低质量资料：0 个

## 覆盖度分析
- 步级超时语义：✓ 完全覆盖（3 个资料）
- 取消传播：✓ 完全覆盖（2 个资料）
- 子图 timeout：✓ 完全覆盖（2 个资料）
- 流式背压：✓ 完全覆盖（2 个资料）
- I/O timeout 组合：✓ 完全覆盖（2 个资料）
- SDK / 平台 timeout：✓ 完全覆盖（3 个资料）

