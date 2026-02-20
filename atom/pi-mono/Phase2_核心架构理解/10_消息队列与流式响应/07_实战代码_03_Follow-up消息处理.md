# 实战代码 03：Follow-up 消息处理

> **目标**: 实现 Follow-up 消息的等待逻辑，包括等待当前响应完成、上下文构建。

---

## 完整代码实现

```typescript
/**
 * Follow-up 消息处理实战示例
 * 演示：如何实现 Follow-up 消息的等待和上下文构建
 */

// ===== 1. 类型定义 =====

import { Message, MessageType, MessagePriority, MessageStatus } from './message-queue';

/**
 * 响应状态
 */
export enum ResponseStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

/**
 * 响应接口
 */
export interface Response {
  id: string;
  messageId: string;
  content: string;
  status: ResponseStatus;
  startTime: number;
  endTime?: number;
}

// ===== 2. 响应管理器 =====

/**
 * 响应管理器
 * 跟踪当前响应的状态
 */
export class ResponseManager {
  private currentResponse: Response | null = null;
  private completionCallbacks: Map<string, (() => void)[]> = new Map();

  /**
   * 开始新响应
   */
  startResponse(messageId: string): Response {
    const response: Response = {
      id: `response-${Date.now()}`,
      messageId,
      content: '',
      status: ResponseStatus.IN_PROGRESS,
      startTime: Date.now()
    };

    this.currentResponse = response;
    console.log(`Started response ${response.id} for message ${messageId}`);

    return response;
  }

  /**
   * 更新响应内容
   */
  updateResponse(content: string): void {
    if (this.currentResponse) {
      this.currentResponse.content += content;
    }
  }

  /**
   * 完成响应
   */
  completeResponse(): void {
    if (this.currentResponse) {
      this.currentResponse.status = ResponseStatus.COMPLETED;
      this.currentResponse.endTime = Date.now();

      const duration = this.currentResponse.endTime - this.currentResponse.startTime;
      console.log(`Completed response ${this.currentResponse.id} (${duration}ms)`);

      // 触发完成回调
      this.triggerCompletionCallbacks(this.currentResponse.messageId);

      this.currentResponse = null;
    }
  }

  /**
   * 等待响应完成
   */
  async waitForCompletion(messageId: string): Promise<void> {
    // 如果当前响应不是等待的消息，立即返回
    if (!this.currentResponse || this.currentResponse.messageId !== messageId) {
      return;
    }

    // 如果已经完成，立即返回
    if (this.currentResponse.status === ResponseStatus.COMPLETED) {
      return;
    }

    // 等待完成
    return new Promise<void>((resolve) => {
      if (!this.completionCallbacks.has(messageId)) {
        this.completionCallbacks.set(messageId, []);
      }
      this.completionCallbacks.get(messageId)!.push(resolve);
    });
  }

  /**
   * 触发完成回调
   */
  private triggerCompletionCallbacks(messageId: string): void {
    const callbacks = this.completionCallbacks.get(messageId);
    if (callbacks) {
      callbacks.forEach(callback => callback());
      this.completionCallbacks.delete(messageId);
    }
  }

  /**
   * 获取当前响应
   */
  getCurrentResponse(): Response | null {
    return this.currentResponse;
  }

  /**
   * 检查是否有活动响应
   */
  hasActiveResponse(): boolean {
    return this.currentResponse !== null &&
           this.currentResponse.status === ResponseStatus.IN_PROGRESS;
  }
}

// ===== 3. Follow-up 消息处理器 =====

/**
 * Follow-up 消息处理器
 */
export class FollowUpMessageHandler {
  private responseManager: ResponseManager;
  private messageHistory: Message[] = [];

  constructor(responseManager: ResponseManager) {
    this.responseManager = responseManager;
  }

  /**
   * 处理 Follow-up 消息
   */
  async handleFollowUp(message: Message & { waitFor: string }): Promise<void> {
    console.log('\n=== Handling Follow-up Message ===');
    console.log(`Content: ${message.content}`);
    console.log(`Wait for: ${message.waitFor}`);

    // 1. 等待指定消息的响应完成
    console.log('Step 1: Waiting for previous response to complete...');
    await this.responseManager.waitForCompletion(message.waitFor);
    console.log('Previous response completed');

    // 2. 构建上下文
    console.log('Step 2: Building context...');
    const context = this.buildContext(message);
    console.log(`Context built with ${context.length} messages`);

    // 3. 处理 Follow-up 消息
    console.log('Step 3: Processing Follow-up message...');
    await this.processFollowUpMessage(message, context);

    console.log('=== Follow-up Message Handled ===\n');
  }

  /**
   * 构建上下文
   */
  private buildContext(message: Message): Message[] {
    // 获取历史消息
    const context: Message[] = [];

    // 添加相关的历史消息
    for (const historyMessage of this.messageHistory) {
      // 只包含用户消息和助手消息
      if (
        historyMessage.type === MessageType.USER ||
        historyMessage.type === MessageType.ASSISTANT
      ) {
        context.push(historyMessage);
      }
    }

    // 添加当前 Follow-up 消息
    context.push(message);

    return context;
  }

  /**
   * 处理 Follow-up 消息
   */
  private async processFollowUpMessage(
    message: Message,
    context: Message[]
  ): Promise<void> {
    console.log(`Processing Follow-up with context of ${context.length} messages`);

    // 模拟 LLM 调用（实际应用中会调用真实的 LLM）
    const response = this.responseManager.startResponse(message.id);

    // 模拟流式生成
    const responseText = `基于之前的对话，我理解你想要：${message.content}`;
    for (const char of responseText) {
      this.responseManager.updateResponse(char);
      process.stdout.write(char);
      await new Promise(resolve => setTimeout(resolve, 30));
    }

    console.log('');
    this.responseManager.completeResponse();

    // 添加到历史
    this.messageHistory.push(message);
  }

  /**
   * 添加消息到历史
   */
  addToHistory(message: Message): void {
    this.messageHistory.push(message);
  }

  /**
   * 获取消息历史
   */
  getHistory(): Message[] {
    return [...this.messageHistory];
  }
}

// ===== 4. 使用示例 =====

/**
 * 示例 1：基础 Follow-up
 */
async function example1_BasicFollowUp() {
  console.log('=== 示例 1：基础 Follow-up ===\n');

  const responseManager = new ResponseManager();
  const handler = new FollowUpMessageHandler(responseManager);

  // 第一条用户消息
  const userMessage: Message = {
    id: 'msg-001',
    type: MessageType.USER,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '创建一个用户登录功能',
    timestamp: Date.now()
  };

  handler.addToHistory(userMessage);

  // 模拟处理第一条消息
  console.log('Processing first message...');
  const response1 = responseManager.startResponse(userMessage.id);
  const text1 = '好的，我会创建用户登录功能...';
  for (const char of text1) {
    responseManager.updateResponse(char);
    process.stdout.write(char);
    await new Promise(resolve => setTimeout(resolve, 30));
  }
  console.log('');
  responseManager.completeResponse();

  // Follow-up 消息
  const followUpMessage: Message & { waitFor: string } = {
    id: 'msg-002',
    type: MessageType.FOLLOW_UP,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '记得添加密码强度验证',
    timestamp: Date.now(),
    waitFor: userMessage.id
  };

  // 处理 Follow-up
  await handler.handleFollowUp(followUpMessage);

  console.log('\n');
}

/**
 * 示例 2：多个 Follow-up
 */
async function example2_MultipleFollowUps() {
  console.log('=== 示例 2：多个 Follow-up ===\n');

  const responseManager = new ResponseManager();
  const handler = new FollowUpMessageHandler(responseManager);

  // 第一条消息
  const msg1: Message = {
    id: 'msg-003',
    type: MessageType.USER,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '写一个 API 接口',
    timestamp: Date.now()
  };

  handler.addToHistory(msg1);

  console.log('Processing first message...');
  const response1 = responseManager.startResponse(msg1.id);
  responseManager.updateResponse('创建 API 接口...');
  await new Promise(resolve => setTimeout(resolve, 500));
  responseManager.completeResponse();

  // 第一个 Follow-up
  const followUp1: Message & { waitFor: string } = {
    id: 'msg-004',
    type: MessageType.FOLLOW_UP,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '添加分页支持',
    timestamp: Date.now(),
    waitFor: msg1.id
  };

  await handler.handleFollowUp(followUp1);

  // 第二个 Follow-up
  const followUp2: Message & { waitFor: string } = {
    id: 'msg-005',
    type: MessageType.FOLLOW_UP,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '添加错误处理',
    timestamp: Date.now(),
    waitFor: followUp1.id
  };

  await handler.handleFollowUp(followUp2);

  console.log('\n');
}

/**
 * 示例 3：Follow-up 与上下文
 */
async function example3_FollowUpWithContext() {
  console.log('=== 示例 3：Follow-up 与上下文 ===\n');

  const responseManager = new ResponseManager();
  const handler = new FollowUpMessageHandler(responseManager);

  // 构建对话历史
  const messages: Message[] = [
    {
      id: 'msg-006',
      type: MessageType.USER,
      priority: MessagePriority.HIGH,
      status: MessageStatus.PENDING,
      content: '我想创建一个博客系统',
      timestamp: Date.now()
    },
    {
      id: 'msg-007',
      type: MessageType.ASSISTANT,
      priority: MessagePriority.NORMAL,
      status: MessageStatus.COMPLETED,
      content: '好的，我会帮你创建博客系统',
      timestamp: Date.now()
    },
    {
      id: 'msg-008',
      type: MessageType.USER,
      priority: MessagePriority.HIGH,
      status: MessageStatus.PENDING,
      content: '需要支持 Markdown',
      timestamp: Date.now()
    }
  ];

  messages.forEach(msg => handler.addToHistory(msg));

  // 处理最后一条消息
  console.log('Processing last message...');
  const response = responseManager.startResponse(messages[2].id);
  responseManager.updateResponse('添加 Markdown 支持...');
  await new Promise(resolve => setTimeout(resolve, 500));
  responseManager.completeResponse();

  // Follow-up 消息
  const followUp: Message & { waitFor: string } = {
    id: 'msg-009',
    type: MessageType.FOLLOW_UP,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '还需要代码高亮',
    timestamp: Date.now(),
    waitFor: messages[2].id
  };

  await handler.handleFollowUp(followUp);

  console.log('\nMessage history:');
  handler.getHistory().forEach(msg => {
    console.log(`- [${msg.type}] ${msg.content}`);
  });

  console.log('\n');
}

/**
 * 示例 4：Follow-up 等待机制
 */
async function example4_FollowUpWaitMechanism() {
  console.log('=== 示例 4：Follow-up 等待机制 ===\n');

  const responseManager = new ResponseManager();
  const handler = new FollowUpMessageHandler(responseManager);

  // 开始长时间响应
  const msg: Message = {
    id: 'msg-010',
    type: MessageType.USER,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '生成长文本',
    timestamp: Date.now()
  };

  handler.addToHistory(msg);

  console.log('Starting long response...');
  const response = responseManager.startResponse(msg.id);

  // 模拟长时间生成
  const longText = '这是一个很长的响应...'.repeat(10);
  const generateTask = (async () => {
    for (const char of longText) {
      responseManager.updateResponse(char);
      process.stdout.write(char);
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    console.log('');
    responseManager.completeResponse();
  })();

  // 在生成过程中添加 Follow-up
  await new Promise(resolve => setTimeout(resolve, 500));

  const followUp: Message & { waitFor: string } = {
    id: 'msg-011',
    type: MessageType.FOLLOW_UP,
    priority: MessagePriority.HIGH,
    status: MessageStatus.PENDING,
    content: '补充一些细节',
    timestamp: Date.now(),
    waitFor: msg.id
  };

  console.log('\n[Follow-up message queued, waiting for completion...]');

  // 等待生成完成
  await generateTask;

  // 处理 Follow-up
  await handler.handleFollowUp(followUp);

  console.log('\n');
}

// ===== 5. 运行所有示例 =====

async function runAllExamples() {
  await example1_BasicFollowUp();
  await example2_MultipleFollowUps();
  await example3_FollowUpWithContext();
  await example4_FollowUpWaitMechanism();
}

// 运行示例
runAllExamples().catch(console.error);
```

---

## 核心要点

1. **等待机制**：使用 Promise 等待当前响应完成
2. **上下文构建**：收集历史消息，构建完整上下文
3. **回调触发**：响应完成时触发所有等待的回调
4. **消息历史**：维护消息历史，支持多轮对话

---

## 关键实现细节

### 1. 等待响应完成

```typescript
async waitForCompletion(messageId: string): Promise<void> {
  return new Promise<void>((resolve) => {
    // 注册回调
    this.completionCallbacks.get(messageId)!.push(resolve);
  });
}

// 响应完成时触发
completeResponse(): void {
  this.triggerCompletionCallbacks(this.currentResponse.messageId);
}
```

### 2. 上下文构建

```typescript
buildContext(message: Message): Message[] {
  const context: Message[] = [];

  // 只包含用户消息和助手消息
  for (const msg of this.messageHistory) {
    if (msg.type === 'user' || msg.type === 'assistant') {
      context.push(msg);
    }
  }

  context.push(message);
  return context;
}
```

### 3. 消息历史管理

```typescript
class FollowUpMessageHandler {
  private messageHistory: Message[] = [];

  addToHistory(message: Message): void {
    this.messageHistory.push(message);
  }

  getHistory(): Message[] {
    return [...this.messageHistory];
  }
}
```

---

## 扩展练习

1. **添加超时机制**：Follow-up 等待超过一定时间自动失败
2. **添加取消机制**：允许用户取消等待中的 Follow-up
3. **添加上下文压缩**：当历史消息过多时，自动压缩上下文
4. **添加智能上下文选择**：只包含相关的历史消息

---

**版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: Claude Code
