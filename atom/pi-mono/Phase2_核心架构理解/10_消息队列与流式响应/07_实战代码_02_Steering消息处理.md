# 实战代码 02：Steering 消息处理

> **目标**: 实现 Steering 消息的中断逻辑，包括中断当前 LLM 流、清空工具队列。

---

## 完整代码实现

```typescript
/**
 * Steering 消息处理实战示例
 * 演示：如何实现 Steering 消息的中断逻辑
 */

// ===== 1. 类型定义 =====

import { Message, MessageType, MessagePriority, MessageStatus } from './message-queue';

/**
 * LLM 流控制器
 */
export class LLMStreamController {
  private abortController: AbortController | null = null;
  private isStreaming: boolean = false;

  /**
   * 开始流式生成
   */
  async startStream(prompt: string): Promise<AsyncGenerator<string>> {
    this.abortController = new AbortController();
    this.isStreaming = true;

    return this.generateStream(prompt, this.abortController.signal);
  }

  /**
   * 中断流式生成
   */
  abort(): void {
    if (this.abortController && this.isStreaming) {
      this.abortController.abort();
      this.isStreaming = false;
      console.log('LLM stream aborted');
    }
  }

  /**
   * 检查是否正在流式生成
   */
  isActive(): boolean {
    return this.isStreaming;
  }

  /**
   * 生成流式响应（模拟）
   */
  private async *generateStream(
    prompt: string,
    signal: AbortSignal
  ): AsyncGenerator<string> {
    const response = `这是对 "${prompt}" 的响应。我会逐字生成内容...`;

    for (let i = 0; i < response.length; i++) {
      // 检查是否被中断
      if (signal.aborted) {
        console.log('Stream generation interrupted');
        return;
      }

      yield response[i];
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
}

/**
 * 工具调用队列
 */
export class ToolCallQueue {
  private queue: ToolCall[] = [];

  /**
   * 添加工具调用
   */
  enqueue(toolCall: ToolCall): void {
    this.queue.push(toolCall);
    console.log(`Tool call enqueued: ${toolCall.name}`);
  }

  /**
   * 清空队列
   */
  clear(): void {
    const count = this.queue.length;
    this.queue = [];
    console.log(`Cleared ${count} tool calls from queue`);
  }

  /**
   * 获取队列大小
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * 获取所有工具调用
   */
  getAll(): ToolCall[] {
    return [...this.queue];
  }
}

/**
 * 工具调用接口
 */
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, any>;
}

// ===== 2. Steering 消息处理器 =====

/**
 * Steering 消息处理器
 */
export class SteeringMessageHandler {
  private llmController: LLMStreamController;
  private toolQueue: ToolCallQueue;
  private currentResponse: string = '';

  constructor(
    llmController: LLMStreamController,
    toolQueue: ToolCallQueue
  ) {
    this.llmController = llmController;
    this.toolQueue = toolQueue;
  }

  /**
   * 处理 Steering 消息
   */
  async handleSteering(message: Message): Promise<void> {
    console.log('\n=== Handling Steering Message ===');
    console.log(`Content: ${message.content}`);

    // 1. 中断当前 LLM 流
    if (this.llmController.isActive()) {
      console.log('Step 1: Aborting current LLM stream...');
      this.llmController.abort();
    } else {
      console.log('Step 1: No active LLM stream to abort');
    }

    // 2. 清空工具调用队列
    if (this.toolQueue.size() > 0) {
      console.log(`Step 2: Clearing tool queue (${this.toolQueue.size()} tools)...`);
      this.toolQueue.clear();
    } else {
      console.log('Step 2: Tool queue is already empty');
    }

    // 3. 保存当前响应状态
    if (this.currentResponse) {
      console.log(`Step 3: Saving interrupted response (${this.currentResponse.length} chars)`);
      this.saveInterruptedResponse(this.currentResponse);
      this.currentResponse = '';
    }

    // 4. 处理新消息
    console.log('Step 4: Processing new message...');
    await this.processNewMessage(message);

    console.log('=== Steering Message Handled ===\n');
  }

  /**
   * 保存被中断的响应
   */
  private saveInterruptedResponse(response: string): void {
    // 实际应用中，可以保存到 Session 或日志
    console.log(`Interrupted response saved: "${response.substring(0, 50)}..."`);
  }

  /**
   * 处理新消息
   */
  private async processNewMessage(message: Message): Promise<void> {
    console.log(`Processing new message: ${message.content}`);
    // 实际应用中，这里会调用 LLM 生成新响应
  }

  /**
   * 设置当前响应
   */
  setCurrentResponse(response: string): void {
    this.currentResponse = response;
  }
}

// ===== 3. 使用示例 =====

/**
 * 示例 1：基础 Steering 中断
 */
async function example1_BasicSteering() {
  console.log('=== 示例 1：基础 Steering 中断 ===\n');

  const llmController = new LLMStreamController();
  const toolQueue = new ToolCallQueue();
  const handler = new SteeringMessageHandler(llmController, toolQueue);

  // 开始流式生成
  console.log('Starting LLM stream...');
  const stream = await llmController.startStream('帮我重构这个函数');

  // 模拟接收部分响应
  let receivedChars = 0;
  const streamConsumer = (async () => {
    for await (const chunk of stream) {
      handler.setCurrentResponse(handler['currentResponse'] + chunk);
      process.stdout.write(chunk);
      receivedChars++;

      // 模拟用户在接收 20 个字符后按 Enter 键
      if (receivedChars === 20) {
        console.log('\n\n[User presses Enter]');
        break;
      }
    }
  })();

  // 等待接收部分响应
  await streamConsumer;

  // 处理 Steering 消息
  await handler.handleSteering({
    id: 'steering-001',
    type: MessageType.STEERING,
    priority: MessagePriority.CRITICAL,
    status: MessageStatus.PENDING,
    content: '等等，先修复那个 bug',
    timestamp: Date.now()
  });

  console.log('\n');
}

/**
 * 示例 2：Steering 中断 + 清空工具队列
 */
async function example2_SteeringWithToolQueue() {
  console.log('=== 示例 2：Steering 中断 + 清空工具队列 ===\n');

  const llmController = new LLMStreamController();
  const toolQueue = new ToolCallQueue();
  const handler = new SteeringMessageHandler(llmController, toolQueue);

  // 添加工具调用到队列
  toolQueue.enqueue({
    id: 'tool-001',
    name: 'readFile',
    args: { path: 'file1.txt' }
  });

  toolQueue.enqueue({
    id: 'tool-002',
    name: 'readFile',
    args: { path: 'file2.txt' }
  });

  toolQueue.enqueue({
    id: 'tool-003',
    name: 'writeFile',
    args: { path: 'output.txt', content: 'data' }
  });

  console.log(`Tool queue size before steering: ${toolQueue.size()}`);

  // 开始流式生成
  const stream = await llmController.startStream('处理这些文件');

  // 模拟接收部分响应
  const streamConsumer = (async () => {
    let count = 0;
    for await (const chunk of stream) {
      process.stdout.write(chunk);
      count++;
      if (count === 10) break;
    }
  })();

  await streamConsumer;

  console.log('\n\n[User presses Enter]');

  // 处理 Steering 消息
  await handler.handleSteering({
    id: 'steering-002',
    type: MessageType.STEERING,
    priority: MessagePriority.CRITICAL,
    status: MessageStatus.PENDING,
    content: '停止，不要处理这些文件',
    timestamp: Date.now()
  });

  console.log(`Tool queue size after steering: ${toolQueue.size()}`);
  console.log('\n');
}

/**
 * 示例 3：连续 Steering
 */
async function example3_ConsecutiveSteering() {
  console.log('=== 示例 3：连续 Steering ===\n');

  const llmController = new LLMStreamController();
  const toolQueue = new ToolCallQueue();
  const handler = new SteeringMessageHandler(llmController, toolQueue);

  // 第一次流式生成
  console.log('First stream:');
  let stream = await llmController.startStream('任务 A');

  let count = 0;
  for await (const chunk of stream) {
    process.stdout.write(chunk);
    count++;
    if (count === 10) break;
  }

  console.log('\n\n[User presses Enter - First steering]');

  // 第一次 Steering
  await handler.handleSteering({
    id: 'steering-003',
    type: MessageType.STEERING,
    priority: MessagePriority.CRITICAL,
    status: MessageStatus.PENDING,
    content: '改成任务 B',
    timestamp: Date.now()
  });

  // 第二次流式生成
  console.log('\nSecond stream:');
  stream = await llmController.startStream('任务 B');

  count = 0;
  for await (const chunk of stream) {
    process.stdout.write(chunk);
    count++;
    if (count === 10) break;
  }

  console.log('\n\n[User presses Enter - Second steering]');

  // 第二次 Steering
  await handler.handleSteering({
    id: 'steering-004',
    type: MessageType.STEERING,
    priority: MessagePriority.CRITICAL,
    status: MessageStatus.PENDING,
    content: '改成任务 C',
    timestamp: Date.now()
  });

  console.log('\n');
}

/**
 * 示例 4：Steering 与状态恢复
 */
async function example4_SteeringWithStateRecovery() {
  console.log('=== 示例 4：Steering 与状态恢复 ===\n');

  const llmController = new LLMStreamController();
  const toolQueue = new ToolCallQueue();
  const handler = new SteeringMessageHandler(llmController, toolQueue);

  // 开始流式生成
  const stream = await llmController.startStream('生成长文本');

  // 接收部分响应
  let receivedText = '';
  let count = 0;
  for await (const chunk of stream) {
    receivedText += chunk;
    process.stdout.write(chunk);
    count++;
    if (count === 15) break;
  }

  handler.setCurrentResponse(receivedText);

  console.log('\n\n[User presses Enter]');

  // Steering 中断
  await handler.handleSteering({
    id: 'steering-005',
    type: MessageType.STEERING,
    priority: MessagePriority.CRITICAL,
    status: MessageStatus.PENDING,
    content: '停止生成',
    timestamp: Date.now()
  });

  console.log(`Received ${receivedText.length} characters before interruption`);
  console.log('\n');
}

// ===== 4. 运行所有示例 =====

async function runAllExamples() {
  await example1_BasicSteering();
  await example2_SteeringWithToolQueue();
  await example3_ConsecutiveSteering();
  await example4_SteeringWithStateRecovery();
}

// 运行示例
runAllExamples().catch(console.error);
```

---

## 核心要点

1. **中断 LLM 流**：使用 AbortController 中断当前流式生成
2. **清空工具队列**：取消所有待执行的工具调用
3. **保存状态**：保存被中断的响应内容
4. **立即处理**：Steering 消息具有最高优先级，立即处理

---

## 关键实现细节

### 1. AbortController 使用

```typescript
// 创建 AbortController
const controller = new AbortController();

// 传递 signal 给异步操作
fetch('/api/data', { signal: controller.signal });

// 中断操作
controller.abort();
```

### 2. 流式生成检查

```typescript
async function* generateStream(signal: AbortSignal) {
  for (const chunk of data) {
    // 每次生成前检查是否被中断
    if (signal.aborted) {
      return; // 立即停止
    }
    yield chunk;
  }
}
```

### 3. 状态保存

```typescript
// 保存被中断的响应
interface InterruptedResponse {
  messageId: string;
  content: string;
  timestamp: number;
  reason: 'steering' | 'error';
}
```

---

## 扩展练习

1. **添加恢复机制**：允许用户恢复被中断的响应
2. **添加中断历史**：记录所有中断事件
3. **添加优雅中断**：等待当前工具执行完成后再中断
4. **添加中断通知**：通知用户中断成功

---

**版本**: v1.0
**最后更新**: 2026-02-19
**维护者**: Claude Code
