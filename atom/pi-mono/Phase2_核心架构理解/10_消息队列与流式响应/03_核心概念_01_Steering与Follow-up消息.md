# 核心概念 01：Steering 与 Follow-up 消息

> **核心价值**: Steering 和 Follow-up 消息是 AI Agent 实现灵活交互控制的关键机制，让用户可以中断、追问和引导 Agent 的行为。

---

## 概述

在 pi-mono 中，用户与 Agent 的交互不仅仅是简单的"问-答"模式。用户可以：
- **中断当前响应**（Steering message）- 按 Enter 键
- **追加消息等待完成**（Follow-up message）- 按 Alt+Enter 键

这两种消息机制让 Agent 交互更加灵活和自然，是 2025-2026 多轮对话 Agent 的标准设计模式。

---

## 1. Steering Message（中断消息）

### 1.1 定义

**Steering Message** 是一种特殊的用户消息，用于**立即中断**当前 Agent 的响应和工具调用。

**核心特征**：
- **立即生效**：不等待当前响应完成
- **取消剩余工具**：清空待执行的工具调用队列
- **最高优先级**：优先于所有其他消息类型

### 1.2 触发方式

在 pi coding agent 中，用户通过以下方式触发 Steering message：

```typescript
// 用户操作：按 Enter 键（不是 Alt+Enter）
// 系统行为：
// 1. 创建 SteeringMessage
// 2. 中断当前 LLM 流
// 3. 取消剩余工具调用
// 4. 立即处理新消息
```

**用户视角**：
- 正在输入消息
- 按 **Enter 键**（普通回车）
- Agent 立即停止当前输出
- 开始处理新消息

### 1.3 使用场景

#### 场景 1：纠正错误方向

```
用户: "帮我重构这个函数"
Agent: "好的，我会将函数拆分成多个小函数..."
用户: [按 Enter] "等等，我只是想优化性能，不要拆分"
```

**效果**：Agent 立即停止拆分，转而优化性能。

#### 场景 2：中断长时间操作

```
用户: "分析整个代码库的依赖关系"
Agent: "正在扫描文件... (1/1000)"
用户: [按 Enter] "停止，只分析 src/ 目录"
```

**效果**：Agent 立即停止扫描，只分析指定目录。

#### 场景 3：改变任务优先级

```
用户: "生成测试用例"
Agent: "正在生成单元测试..."
用户: [按 Enter] "先修复那个 bug，测试稍后再说"
```

**效果**：Agent 立即切换到修复 bug。

### 1.4 实现原理

```typescript
// 消息类型定义
interface SteeringMessage {
  type: 'steering';
  content: string;
  timestamp: number;
}

// Steering 消息处理流程
async function handleSteeringMessage(message: SteeringMessage) {
  // 1. 中断当前 LLM 流
  if (currentLLMStream) {
    currentLLMStream.abort();
  }

  // 2. 清空工具调用队列
  toolCallQueue.clear();

  // 3. 标记当前响应为已中断
  currentResponse.status = 'interrupted';

  // 4. 立即处理新消息
  await processMessage(message);
}
```

### 1.5 TypeScript/Node.js 类比

**类比 1：AbortController**

```typescript
// Steering message 类似于 AbortController
const controller = new AbortController();

// 开始一个长时间操作
fetch('/api/data', { signal: controller.signal })
  .then(response => response.json())
  .catch(err => {
    if (err.name === 'AbortError') {
      console.log('操作被中断');
    }
  });

// 用户触发 Steering message = controller.abort()
controller.abort();
```

**类比 2：Promise.race()**

```typescript
// Steering message 类似于 Promise.race() 中的快速路径
const result = await Promise.race([
  longRunningTask(),      // 当前 Agent 响应
  steeringMessage()       // Steering 消息（立即胜出）
]);
```

### 1.6 日常生活类比

**类比：打断对话**

想象你在和朋友聊天：
- 朋友正在讲一个长故事
- 你突然想起重要的事："等等，我想起来了..."
- 朋友立即停止讲故事，听你说

**Steering message = 打断对话，立即说新话题**

---

## 2. Follow-up Message（追问消息）

### 2.1 定义

**Follow-up Message** 是一种特殊的用户消息，用于**等待当前响应完成后**再追加新消息。

**核心特征**：
- **等待完成**：不中断当前响应
- **追加到队列**：加入消息队列等待处理
- **保持上下文**：基于当前响应的结果继续对话

### 2.2 触发方式

在 pi coding agent 中，用户通过以下方式触发 Follow-up message：

```typescript
// 用户操作：按 Alt+Enter 键
// 系统行为：
// 1. 创建 FollowUpMessage
// 2. 加入消息队列
// 3. 等待当前响应完成
// 4. 处理 Follow-up 消息
```

**用户视角**：
- 正在输入消息
- 按 **Alt+Enter 键**（组合键）
- Agent 继续完成当前输出
- 完成后自动处理新消息

### 2.3 使用场景

#### 场景 1：追加需求

```
用户: "创建一个用户登录功能"
Agent: "正在创建登录功能... [生成代码中]"
用户: [按 Alt+Enter] "记得添加密码强度验证"
```

**效果**：Agent 完成登录功能后，自动添加密码验证。

#### 场景 2：连续提问

```
用户: "解释这个函数的作用"
Agent: "这个函数用于... [详细解释中]"
用户: [按 Alt+Enter] "那它的性能如何？"
```

**效果**：Agent 解释完作用后，自动回答性能问题。

#### 场景 3：补充信息

```
用户: "帮我写一个 API 接口"
Agent: "正在设计 API... [生成中]"
用户: [按 Alt+Enter] "对了，需要支持分页"
```

**效果**：Agent 完成基础 API 后，自动添加分页支持。

### 2.4 实现原理

```typescript
// 消息类型定义
interface FollowUpMessage {
  type: 'follow-up';
  content: string;
  timestamp: number;
  waitFor: string; // 等待的响应 ID
}

// Follow-up 消息处理流程
async function handleFollowUpMessage(message: FollowUpMessage) {
  // 1. 加入消息队列
  messageQueue.enqueue(message);

  // 2. 等待当前响应完成
  await currentResponse.waitForCompletion();

  // 3. 处理 Follow-up 消息
  await processMessage(message);
}
```

### 2.5 TypeScript/Node.js 类比

**类比 1：Promise.then()**

```typescript
// Follow-up message 类似于 Promise.then()
currentTask()
  .then(() => followUpTask())  // 等待完成后执行
  .then(() => anotherFollowUp());
```

**类比 2：Event Queue**

```typescript
// Follow-up message 类似于事件队列中的下一个事件
eventQueue.push(currentEvent);
eventQueue.push(followUpEvent);  // 等待前一个事件完成

// 事件循环按顺序处理
while (eventQueue.length > 0) {
  const event = eventQueue.shift();
  await processEvent(event);
}
```

### 2.6 日常生活类比

**类比：等对方说完再补充**

想象你在和朋友聊天：
- 朋友正在讲一个故事
- 你想补充一句，但不想打断
- 你等朋友说完："对了，我想补充一下..."

**Follow-up message = 等对方说完，再补充一句**

---

## 3. Steering vs Follow-up 对比

### 3.1 核心区别

| 特性 | Steering Message | Follow-up Message |
|------|-----------------|-------------------|
| **触发方式** | Enter 键 | Alt+Enter 键 |
| **行为** | 立即中断 | 等待完成 |
| **优先级** | 最高（立即处理） | 普通（排队等待） |
| **工具调用** | 取消剩余工具 | 保留工具调用 |
| **上下文** | 新上下文 | 基于当前上下文 |
| **使用场景** | 纠正方向、中断操作 | 追加需求、连续提问 |

### 3.2 决策树

```
用户想发送消息
    ↓
需要立即中断当前响应？
    ↓
  是 → 按 Enter（Steering）
    ↓
  否 → 想等当前响应完成？
    ↓
      是 → 按 Alt+Enter（Follow-up）
      否 → 等待当前响应完成后再发送
```

### 3.3 代码对比

```typescript
// Steering Message：立即中断
async function steering(message: string) {
  // 1. 中断当前流
  currentStream?.abort();

  // 2. 清空队列
  queue.clear();

  // 3. 立即处理
  await process(message);
}

// Follow-up Message：等待完成
async function followUp(message: string) {
  // 1. 加入队列
  queue.enqueue(message);

  // 2. 等待当前任务
  await currentTask;

  // 3. 处理消息
  await process(message);
}
```

---

## 4. 2025-2026 最新实践

### 4.1 Anthropic 研究：代理自治性增强

> **2025-2026 最新实践**: 根据 Anthropic 2025-2026 研究，代理自治性增强，推荐使用实时 steering 和可观测性工具而非逐动作审批。

**核心观点**：
- **Steering > 逐动作审批**：让用户在需要时中断，而不是每步都确认
- **可观测性**：提供实时反馈，让用户了解 Agent 在做什么
- **自治性**：Agent 应该能够自主完成任务，用户只在必要时介入

**引用来源**：
- Anthropic Research - Measuring agent autonomy (2025-2026)
- https://www.anthropic.com/research/measuring-agent-autonomy

### 4.2 GitHub Copilot CLI：Follow-up 消息排队

> **2025-2026 最新实践**: GitHub Copilot CLI 支持 Follow-up 消息排队机制，允许用户在 Agent 执行过程中追加多个消息。

**核心特性**：
- **消息队列**：支持多个 Follow-up 消息排队
- **上下文保持**：每个 Follow-up 消息基于前一个响应的上下文
- **批处理优化**：合并相关的 Follow-up 消息

**引用来源**：
- GitHub Copilot CLI - Follow-up message queuing
- https://github.com/github/copilot-cli

### 4.3 Medium 2026 Playbook：多轮对话架构

> **2025-2026 最新实践**: 根据 Medium 2026 Playbook，Steering 和 Follow-up 消息机制已成为多轮对话 Agent 的标准设计模式。

**核心设计模式**：
1. **会话树结构**：每个 Steering 创建新分支
2. **消息队列**：Follow-up 消息排队等待
3. **优先级管理**：Steering > User > Tool results

**引用来源**：
- Medium - Building Multi-Turn Conversations with AI Agents: The 2026 Playbook
- https://medium.com/@ai-agents/multi-turn-conversations-2026

### 4.4 agentjido/jido：会话树 + Steering & Follow-up 队列

> **2025-2026 最新实践**: GitHub agentjido/jido #119 提出了会话树结构 + Steering & Follow-up 队列的完整实现方案。

**核心架构**：
```typescript
// 会话树结构
interface SessionTree {
  root: SessionNode;
  branches: Map<string, SessionNode[]>;
}

// Steering 创建新分支
function steering(message: string): SessionNode {
  const newBranch = createBranch(currentNode, message);
  sessionTree.branches.set(newBranch.id, [newBranch]);
  return newBranch;
}

// Follow-up 在当前分支追加
function followUp(message: string): SessionNode {
  const newNode = appendToCurrentBranch(message);
  return newNode;
}
```

**引用来源**：
- GitHub agentjido/jido #119 - Session Tree Structure + Agent Steering & Follow-Up Queues
- https://github.com/agentjido/jido/issues/119

---

## 5. 实际应用示例

### 5.1 pi-mono 中的实现

在 pi-mono 的 `pi-coding-agent` 中，Steering 和 Follow-up 消息的实现：

```typescript
// packages/pi-coding-agent/src/messages.ts

export type MessageType =
  | 'user'
  | 'assistant'
  | 'tool-call'
  | 'tool-result'
  | 'steering'      // Steering message
  | 'follow-up';    // Follow-up message

export interface Message {
  id: string;
  type: MessageType;
  content: string;
  timestamp: number;
  parentId?: string;  // 用于会话树结构
}

// Steering 消息处理
export async function handleSteering(
  message: Message,
  agent: Agent
): Promise<void> {
  // 1. 中断当前流
  agent.abortCurrentStream();

  // 2. 清空工具队列
  agent.clearToolQueue();

  // 3. 创建新分支
  const newBranch = agent.session.createBranch(message);

  // 4. 处理消息
  await agent.processMessage(message);
}

// Follow-up 消息处理
export async function handleFollowUp(
  message: Message,
  agent: Agent
): Promise<void> {
  // 1. 加入消息队列
  agent.messageQueue.enqueue(message);

  // 2. 等待当前响应完成
  await agent.waitForCurrentResponse();

  // 3. 处理消息
  await agent.processMessage(message);
}
```

### 5.2 实际使用示例

```typescript
// 示例：用户使用 Steering 和 Follow-up

// 场景 1：Steering 中断
用户: "重构这个文件"
Agent: "正在分析文件结构..."
用户: [Enter] "等等，只重构 UserService 类"
// → Steering message 立即中断，只重构指定类

// 场景 2：Follow-up 追加
用户: "添加用户注册功能"
Agent: "正在创建注册接口..."
用户: [Alt+Enter] "记得添加邮箱验证"
// → Follow-up message 等待注册功能完成后自动添加邮箱验证

// 场景 3：连续 Follow-up
用户: "创建 API 文档"
Agent: "正在生成文档..."
用户: [Alt+Enter] "添加示例代码"
用户: [Alt+Enter] "添加错误码说明"
// → 两个 Follow-up 消息排队，依次处理
```

---

## 6. 最佳实践

### 6.1 何时使用 Steering

✅ **推荐使用场景**：
- Agent 走错方向，需要立即纠正
- 长时间操作需要中断
- 任务优先级发生变化
- 发现更紧急的问题

❌ **不推荐使用场景**：
- 只是想追加一个小需求（用 Follow-up）
- Agent 即将完成当前任务（用 Follow-up）
- 只是想补充信息（用 Follow-up）

### 6.2 何时使用 Follow-up

✅ **推荐使用场景**：
- 追加需求或补充信息
- 连续提问
- 基于当前结果继续对话
- 不想中断当前流程

❌ **不推荐使用场景**：
- 需要立即中断（用 Steering）
- 当前方向完全错误（用 Steering）
- 任务优先级变化（用 Steering）

### 6.3 设计建议

**1. 提供清晰的视觉反馈**

```typescript
// 显示消息类型
if (message.type === 'steering') {
  console.log('🛑 中断当前响应');
} else if (message.type === 'follow-up') {
  console.log('⏳ 等待完成后处理');
}
```

**2. 实现优雅的中断**

```typescript
// 保存中断前的状态
async function gracefulSteering(message: Message) {
  // 1. 保存当前进度
  const progress = await saveProgress();

  // 2. 中断当前流
  await abortCurrentStream();

  // 3. 记录中断原因
  logInterruption(message, progress);

  // 4. 处理新消息
  await processMessage(message);
}
```

**3. 优化 Follow-up 队列**

```typescript
// 合并相关的 Follow-up 消息
function optimizeFollowUpQueue(queue: Message[]): Message[] {
  // 如果多个 Follow-up 消息相关，合并处理
  return mergeRelatedMessages(queue);
}
```

---

## 7. 总结

### 7.1 核心要点

1. **Steering Message**：立即中断，最高优先级，用于纠正方向
2. **Follow-up Message**：等待完成，排队处理，用于追加需求
3. **2025-2026 标准**：已成为多轮对话 Agent 的标准设计模式
4. **实现关键**：AbortController + 消息队列 + 会话树结构

### 7.2 学习检查

- [ ] 理解 Steering 和 Follow-up 的区别
- [ ] 知道何时使用 Steering，何时使用 Follow-up
- [ ] 了解 2025-2026 最新实践（Anthropic、GitHub Copilot CLI）
- [ ] 能够设计 Steering 和 Follow-up 的实现方案

### 7.3 下一步

- **03_核心概念_02_消息队列架构.md**：学习消息队列的完整设计
- **07_实战代码_02_Steering消息处理.md**：手写 Steering 消息处理逻辑
- **07_实战代码_03_Follow-up消息处理.md**：手写 Follow-up 消息处理逻辑

---

**版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: Claude Code
