# 节点函数约定 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_节点函数约定_01.md - StateGraph 核心实现分析（state.py）
- ✓ reference/source_节点函数约定_02.md - 重试机制实现分析（_retry.py）

**关键发现**：
- 节点函数签名约定：State, Config, Runtime 参数
- 返回值类型：dict, Command, None, list
- 自动类型推断机制
- 异步节点自动检测
- RetryPolicy 配置和实现
- 指数退避算法

### Context7 官方文档
- ✓ reference/context7_langgraph_01.md - LangGraph 节点函数官方文档

**关键发现**：
- 节点函数基础定义和示例
- 部分状态更新机制
- 条件路由函数
- 错误处理模式
- RetryPolicy 配置示例
- Human-in-the-loop 模式

### 网络搜索
- ✓ reference/search_节点函数约定_01.md - GitHub 社区讨论和问题（2025-2026）

**关键发现**：
- 异步生成器节点问题
- 错误处理增强需求
- 流式错误处理问题
- 异步取消错误
- Command 错误提示改进
- @task 装饰器异常处理

### 待抓取链接（可选）

根据搜索结果，识别出以下社区讨论链接（按优先级排序）：

**High 优先级**（2025-2026 年的最新讨论）：
- https://github.com/langchain-ai/langgraph/issues/6170 - 节点鲁棒错误处理增强
- https://github.com/langchain-ai/langgraph/issues/6726 - ToolNode未捕获CancelledError
- https://github.com/langchain-ai/langgraph/issues/6397 - ToolNode编程式调用失败

**Medium 优先级**（技术讨论）：
- https://github.com/langchain-ai/langgraphjs/issues/1158 - 异步生成器节点未执行问题
- https://github.com/langchain-ai/langgraphjs/issues/1831 - 工具错误在streamEvents中被忽略
- https://github.com/langchain-ai/langgraph/issues/5556 - Command引用不存在节点错误不明显

**Low 优先级**（已解决或较旧的问题）：
- https://github.com/langchain-ai/langgraphjs/issues/812 - streamEvents try-catch问题
- https://github.com/langchain-ai/langgraph/issues/4294 - @task装饰器异常未捕获

**注意**：这些链接都是社区讨论，不是官方文档或源码仓库链接。根据当前收集的资料，已经足够生成高质量的文档。如果需要更深入的社区实践案例，可以在阶段二进行补充调研。

---

## 文件清单

### 基础维度文件
- [x] 00_概览.md
- [x] 01_30字核心.md
- [x] 02_第一性原理.md

### 核心概念文件（5个）

#### 1. 函数签名约定
- [x] 03_核心概念_1_函数签名约定.md
  - 参数类型（State, Config, Runtime）
  - 返回值类型（dict, Command, None）
  - 类型推断机制
  - **来源**：source_节点函数约定_01.md（state.py 源码分析）

#### 2. 状态更新机制
- [x] 03_核心概念_2_状态更新机制.md
  - 部分更新 vs 完整更新
  - Reducer 函数的作用
  - Command 对象的使用
  - **来源**：source_节点函数约定_01.md + context7_langgraph_01.md

#### 3. 异步节点实现
- [x] 03_核心概念_3_异步节点实现.md ✓
  - async/await 语法
  - 异步节点的执行机制
  - 与同步节点的区别
  - 异步生成器的问题
  - **来源**：source_节点函数约定_01.md + search_节点函数约定_01.md

#### 4. 错误处理机制
- [x] 03_核心概念_4_错误处理机制.md ✓
  - 异常捕获
  - RetryPolicy 配置
  - 错误传播
  - 流式错误处理
  - **来源**：source_节点函数约定_02.md + context7_langgraph_01.md + search_节点函数约定_01.md

#### 5. 节点配置选项
- [x] 03_核心概念_5_节点配置选项.md
  - retry_policy
  - cache_policy
  - metadata
  - defer
  - destinations
  - **来源**：source_节点函数约定_01.md

### 基础维度文件（续）
- [x] 04_最小可用.md ✓
- [x] 05_双重类比.md ✓
- [x] 06_反直觉点.md ✓

### 实战代码文件（5个）

#### 1. 基础节点函数实现
- [x] 07_实战代码_场景1_基础节点函数实现.md ✓
  - 简单的状态更新
  - 使用 Config 参数
  - 使用 Runtime 参数
  - **来源**：context7_langgraph_01.md

#### 2. 异步节点实现
- [x] 07_实战代码_场景2_异步节点实现.md ✓
  - async 函数定义
  - 异步 API 调用
  - 异步错误处理
  - **来源**：context7_langgraph_01.md + search_节点函数约定_01.md

#### 3. 错误处理与重试
- [x] 07_实战代码_场景3_错误处理与重试.md ✓
  - RetryPolicy 配置
  - 自定义重试条件
  - 错误恢复策略
  - 流式错误处理
  - **来源**：context7_langgraph_01.md + source_节点函数约定_02.md

#### 4. Command 对象使用
- [x] 07_实战代码_场景4_Command对象使用.md ✓
  - 控制流程跳转
  - 动态路由
  - 父图通信
  - Command 与状态更新结合
  - **来源**：source_节点函数约定_01.md + context7_langgraph_01.md + source_节点函数约定_02.md

#### 5. Human-in-the-loop
- [x] 07_实战代码_场景5_Human-in-the-loop.md ✓
  - interrupt 函数使用
  - 用户输入处理
  - 状态恢复
  - **来源**：context7_langgraph_01.md

### 基础维度文件（续）
- [x] 08_面试必问.md ✓
- [x] 09_化骨绵掌.md ✓
- [x] 10_一句话总结.md ✓

---

## 生成进度

### 阶段一：Plan 生成 ✅ 已完成
- [x] 1.1 Brainstorm 分析
- [x] 1.2 多源数据收集
  - [x] A. 源码分析（state.py, _call.py, _retry.py）
  - [x] B. Context7 官方文档查询
  - [x] C. Grok-mcp 网络搜索
  - [x] D. 数据整合
- [x] 1.3 用户确认拆解方案
- [x] 1.4 Plan 最终确定

### 阶段二：补充调研（跳过）
- [x] 2.1 识别需要补充资料的部分 - 当前资料已足够
- [ ] 2.2 执行补充调研
  - [ ] A. Context7 深度查询（如需要）
  - [ ] B. Grok-mcp 补充搜索（如需要）
  - [ ] C. 生成抓取任务文件（如需要）
- [ ] 2.3 等待抓取完成（如需要）
- [ ] 2.4 更新 PLAN.md

**当前评估**：基于已收集的资料（源码分析 + Context7 文档 + 网络搜索），已经足够生成高质量的文档。建议直接进入阶段三。如果在生成过程中发现需要更多资料，可以回到阶段二进行补充调研。

### 阶段三：文档生成 ✅ 已完成
- [x] 3.1 读取所有 reference/ 资料
- [x] 3.2 按顺序生成文档
  - [x] 基础维度文件（第一部分）✓
    - [x] 00_概览.md
    - [x] 01_30字核心.md
    - [x] 02_第一性原理.md
  - [x] 核心概念文件（5个）✓
    - [x] 03_核心概念_1_函数签名约定.md
    - [x] 03_核心概念_2_状态更新机制.md
    - [x] 03_核心概念_3_异步节点实现.md
    - [x] 03_核心概念_4_错误处理机制.md
    - [x] 03_核心概念_5_节点配置选项.md
  - [x] 基础维度文件（第二部分）✓
    - [x] 04_最小可用.md
    - [x] 05_双重类比.md
    - [x] 06_反直觉点.md
  - [x] 实战代码文件（5个）✓
    - [x] 07_实战代码_场景1_基础节点函数实现.md
    - [x] 07_实战代码_场景2_异步节点实现.md
    - [x] 07_实战代码_场景3_错误处理与重试.md
    - [x] 07_实战代码_场景4_Command对象使用.md
    - [x] 07_实战代码_场景5_Human-in-the-loop.md
  - [x] 基础维度文件（第三部分）✓
    - [x] 08_面试必问.md
    - [x] 09_化骨绵掌.md
    - [x] 10_一句话总结.md
- [x] 3.3 质量检查
- [x] 3.4 完成标记

---

## 技术要点总结

### 核心技术点（从数据来源中提取）

1. **函数签名约定**
   - 基础签名：`def node(state: State) -> dict`
   - 带 Config：`def node(state: State, config: RunnableConfig) -> dict`
   - 带 Runtime：`def node(state: State, runtime: Runtime[Context]) -> dict`
   - 自动类型推断：从第一个参数推断 input_schema

2. **返回值处理**
   - dict：部分状态更新
   - Command：控制流程跳转
   - None：不更新状态
   - list[Command | dict]：多个更新
   - 带 Annotated 的自定义类型

3. **异步节点**
   - 自动检测 `async` 函数
   - 同步函数包装成异步执行（run_in_executor）
   - 异步生成器（async function*）可能有问题

4. **错误处理**
   - RetryPolicy 配置：max_attempts, initial_interval, backoff_factor, max_interval, jitter, retry_on
   - 指数退避算法
   - 自定义重试条件（异常类型或 Callable）
   - 流式错误处理注意事项

5. **节点配置**
   - retry_policy：重试策略
   - cache_policy：缓存策略
   - metadata：元数据
   - defer：延迟执行
   - destinations：目标节点（用于 Command）

### 常见误区（从社区讨论中提取）

1. "异步生成器可以作为节点函数" → 可能有问题，推荐使用普通异步函数
2. "所有异常都会自动重试" → 需要配置 RetryPolicy
3. "CancelledError 会被正常捕获" → 需要特殊处理
4. "streamEvents 中的错误会自动传播" → 需要在循环内处理
5. "Command 指向无效节点会静默失败" → 会抛出 InvalidUpdateError

### 最佳实践（从官方文档和社区中提取）

1. **节点函数定义**
   - 使用类型注解
   - 返回部分状态更新
   - 避免使用异步生成器

2. **错误处理**
   - 配置 RetryPolicy
   - 使用自定义重试条件
   - 在节点内部捕获预期异常

3. **异步节点**
   - 使用 async/await
   - 处理 CancelledError
   - 注意流式错误处理

4. **Command 使用**
   - 验证目标节点存在
   - 使用类型注解（Literal）
   - 处理 ParentCommand

---

## 下一步操作

### 选项 A：直接进入阶段三（推荐）
基于已收集的资料，直接开始文档生成。

**优点**：
- 资料已经足够全面
- 可以快速完成文档生成
- 如果发现不足，可以随时补充

**执行命令**：
```bash
# 开始阶段三：文档生成
# 使用 subagent 批量生成文件
```

### 选项 B：先进行阶段二补充调研
如果需要更多社区实践案例，可以先抓取 GitHub issues。

**优点**：
- 获取更多社区实践案例
- 了解最新的问题和解决方案

**缺点**：
- 需要额外时间
- 当前资料已经足够

**执行命令**：
```bash
# 生成 FETCH_TASK.json
# 等待第三方工具抓取
# 然后进入阶段三
```

---

**建议**：直接进入阶段三，使用现有资料生成文档。如果在生成过程中发现需要更多资料，可以随时回到阶段二进行补充调研。

**生成时间估算**：
- 基础维度文件：10个文件
- 核心概念文件：5个文件
- 实战代码文件：5个文件
- **总计：20个文件**

**质量保证**：
- 每个文件 300-500 行
- 所有代码完整可运行
- 包含完整的引用来源
- 遵循 CLAUDE_LANGGRAPH.md 规范
