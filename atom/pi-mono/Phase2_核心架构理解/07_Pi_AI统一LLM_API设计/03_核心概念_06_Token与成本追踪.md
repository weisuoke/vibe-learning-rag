# 核心概念6：Token 与成本追踪

> 理解如何跨 Provider 统一追踪 Token 使用量和成本

---

## 概念定义

**Token 与成本追踪**是指统一记录和聚合不同 LLM Provider 的 Token 使用量，并根据各 Provider 的定价自动计算成本，为优化和预算管理提供数据支持。

**核心价值：**
- **成本透明**：实时了解每次调用的成本
- **预算控制**：避免超出预算
- **优化决策**：基于数据选择性价比最高的模型
- **跨 Provider 聚合**：统一追踪多个 Provider 的使用量

---

## 第一性原理

### 问题的本质

**核心问题：** 不同 Provider 的 Token 计费方式和定价不同，如何统一追踪？

**Token 计费差异：**

1. **OpenAI**
   - 输入 Token：$0.15 / 1M tokens (GPT-4o-mini)
   - 输出 Token：$0.60 / 1M tokens
   - 分别计费

2. **Anthropic**
   - 输入 Token：$3.00 / 1M tokens (Claude Opus 4)
   - 输出 Token：$15.00 / 1M tokens
   - 分别计费

3. **Google**
   - 输入 Token：$0.075 / 1M tokens (Gemini 2.0 Flash)
   - 输出 Token：$0.30 / 1M tokens
   - 分别计费

4. **Ollama**
   - 本地运行，无 API 成本
   - 但有硬件成本（GPU、电费）

### 设计原则

**1. 统一格式**
- 所有 Provider 返回相同的 Token 统计格式
- 包含输入、输出、总计

**2. 自动计算**
- 根据 Provider 定价自动计算成本
- 支持自定义定价

**3. 实时追踪**
- 每次调用返回 Token 统计
- 支持批量聚合

**4. 历史记录**
- 保存历史使用记录
- 支持导出和分析

---

## 核心实现

### 1. Token 使用统计类型

```typescript
/**
 * Token 使用统计
 */
interface TokenUsage {
  /**
   * 输入 Token 数
   */
  inputTokens: number;

  /**
   * 输出 Token 数
   */
  outputTokens: number;

  /**
   * 总 Token 数
   */
  totalTokens: number;

  /**
   * 成本（美元）
   */
  cost?: number;

  /**
   * 缓存命中的 Token 数（如果支持）
   */
  cachedTokens?: number;
}

/**
 * 详细的使用记录
 */
interface UsageRecord {
  /**
   * 记录 ID
   */
  id: string;

  /**
   * 时间戳
   */
  timestamp: number;

  /**
   * Provider 名称
   */
  provider: string;

  /**
   * 模型名称
   */
  model: string;

  /**
   * Token 使用统计
   */
  usage: TokenUsage;

  /**
   * 请求元数据
   */
  metadata?: {
    userId?: string;
    sessionId?: string;
    tags?: string[];
    [key: string]: any;
  };
}
```

### 2. Provider 定价配置

```typescript
/**
 * Provider 定价配置
 */
interface ProviderPricing {
  /**
   * Provider 名称
   */
  provider: string;

  /**
   * 模型定价
   */
  models: Record<string, ModelPricing>;
}

/**
 * 模型定价
 */
interface ModelPricing {
  /**
   * 输入 Token 价格（美元 / 1M tokens）
   */
  inputPrice: number;

  /**
   * 输出 Token 价格（美元 / 1M tokens）
   */
  outputPrice: number;

  /**
   * 缓存 Token 价格（如果支持）
   */
  cachedPrice?: number;
}

/**
 * 2026 年主流模型定价
 */
const PRICING_2026: ProviderPricing[] = [
  {
    provider: 'openai',
    models: {
      'gpt-4o': {
        inputPrice: 2.50,
        outputPrice: 10.00
      },
      'gpt-4o-mini': {
        inputPrice: 0.15,
        outputPrice: 0.60
      },
      'gpt-4-turbo': {
        inputPrice: 10.00,
        outputPrice: 30.00
      }
    }
  },
  {
    provider: 'anthropic',
    models: {
      'claude-opus-4': {
        inputPrice: 3.00,
        outputPrice: 15.00,
        cachedPrice: 0.30  // 缓存命中价格
      },
      'claude-sonnet-4': {
        inputPrice: 0.60,
        outputPrice: 3.00,
        cachedPrice: 0.06
      },
      'claude-haiku-4': {
        inputPrice: 0.10,
        outputPrice: 0.50,
        cachedPrice: 0.01
      }
    }
  },
  {
    provider: 'google',
    models: {
      'gemini-2.0-flash': {
        inputPrice: 0.075,
        outputPrice: 0.30
      },
      'gemini-1.5-pro': {
        inputPrice: 1.25,
        outputPrice: 5.00
      }
    }
  },
  {
    provider: 'ollama',
    models: {
      '*': {  // 所有本地模型
        inputPrice: 0,
        outputPrice: 0
      }
    }
  }
];
```

### 3. 成本计算器

```typescript
/**
 * 成本计算器
 */
class CostCalculator {
  private pricing: Map<string, Map<string, ModelPricing>>;

  constructor(pricingConfig: ProviderPricing[]) {
    this.pricing = new Map();

    // 构建定价索引
    for (const config of pricingConfig) {
      const models = new Map<string, ModelPricing>();
      for (const [model, pricing] of Object.entries(config.models)) {
        models.set(model, pricing);
      }
      this.pricing.set(config.provider, models);
    }
  }

  /**
   * 计算成本
   */
  calculate(
    provider: string,
    model: string,
    usage: Omit<TokenUsage, 'cost'>
  ): number {
    // 1. 获取定价
    const providerPricing = this.pricing.get(provider);
    if (!providerPricing) {
      console.warn(`No pricing found for provider: ${provider}`);
      return 0;
    }

    // 2. 获取模型定价（支持通配符）
    let modelPricing = providerPricing.get(model);
    if (!modelPricing) {
      modelPricing = providerPricing.get('*');  // 尝试通配符
    }
    if (!modelPricing) {
      console.warn(`No pricing found for model: ${provider}/${model}`);
      return 0;
    }

    // 3. 计算成本
    const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputPrice;
    const outputCost = (usage.outputTokens / 1_000_000) * modelPricing.outputPrice;

    // 4. 缓存成本（如果有）
    let cachedCost = 0;
    if (usage.cachedTokens && modelPricing.cachedPrice) {
      cachedCost = (usage.cachedTokens / 1_000_000) * modelPricing.cachedPrice;
    }

    return inputCost + outputCost + cachedCost;
  }

  /**
   * 批量计算成本
   */
  calculateBatch(records: Array<{
    provider: string;
    model: string;
    usage: Omit<TokenUsage, 'cost'>;
  }>): number {
    return records.reduce((total, record) => {
      return total + this.calculate(record.provider, record.model, record.usage);
    }, 0);
  }

  /**
   * 估算成本（基于输入长度）
   */
  estimate(
    provider: string,
    model: string,
    inputLength: number,
    estimatedOutputLength: number
  ): number {
    // 粗略估算：1 token ≈ 4 字符
    const estimatedInputTokens = Math.ceil(inputLength / 4);
    const estimatedOutputTokens = Math.ceil(estimatedOutputLength / 4);

    return this.calculate(provider, model, {
      inputTokens: estimatedInputTokens,
      outputTokens: estimatedOutputTokens,
      totalTokens: estimatedInputTokens + estimatedOutputTokens
    });
  }
}
```

### 4. 使用量追踪器

```typescript
/**
 * 使用量追踪器
 */
class UsageTracker {
  private records: UsageRecord[] = [];
  private calculator: CostCalculator;

  constructor(pricingConfig: ProviderPricing[]) {
    this.calculator = new CostCalculator(pricingConfig);
  }

  /**
   * 记录使用量
   */
  track(
    provider: string,
    model: string,
    usage: Omit<TokenUsage, 'cost'>,
    metadata?: UsageRecord['metadata']
  ): UsageRecord {
    // 1. 计算成本
    const cost = this.calculator.calculate(provider, model, usage);

    // 2. 创建记录
    const record: UsageRecord = {
      id: this.generateId(),
      timestamp: Date.now(),
      provider,
      model,
      usage: {
        ...usage,
        cost
      },
      metadata
    };

    // 3. 保存记录
    this.records.push(record);

    return record;
  }

  /**
   * 获取总使用量
   */
  getTotalUsage(): TokenUsage {
    return this.records.reduce(
      (total, record) => ({
        inputTokens: total.inputTokens + record.usage.inputTokens,
        outputTokens: total.outputTokens + record.usage.outputTokens,
        totalTokens: total.totalTokens + record.usage.totalTokens,
        cost: (total.cost || 0) + (record.usage.cost || 0)
      }),
      {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        cost: 0
      }
    );
  }

  /**
   * 按 Provider 聚合
   */
  getUsageByProvider(): Record<string, TokenUsage> {
    const result: Record<string, TokenUsage> = {};

    for (const record of this.records) {
      if (!result[record.provider]) {
        result[record.provider] = {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
          cost: 0
        };
      }

      const usage = result[record.provider];
      usage.inputTokens += record.usage.inputTokens;
      usage.outputTokens += record.usage.outputTokens;
      usage.totalTokens += record.usage.totalTokens;
      usage.cost = (usage.cost || 0) + (record.usage.cost || 0);
    }

    return result;
  }

  /**
   * 按模型聚合
   */
  getUsageByModel(): Record<string, TokenUsage> {
    const result: Record<string, TokenUsage> = {};

    for (const record of this.records) {
      const key = `${record.provider}/${record.model}`;
      if (!result[key]) {
        result[key] = {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
          cost: 0
        };
      }

      const usage = result[key];
      usage.inputTokens += record.usage.inputTokens;
      usage.outputTokens += record.usage.outputTokens;
      usage.totalTokens += record.usage.totalTokens;
      usage.cost = (usage.cost || 0) + (record.usage.cost || 0);
    }

    return result;
  }

  /**
   * 按时间范围查询
   */
  getUsageByTimeRange(startTime: number, endTime: number): UsageRecord[] {
    return this.records.filter(
      record => record.timestamp >= startTime && record.timestamp <= endTime
    );
  }

  /**
   * 导出记录
   */
  export(): UsageRecord[] {
    return [...this.records];
  }

  /**
   * 清空记录
   */
  clear(): void {
    this.records = [];
  }

  /**
   * 生成唯一 ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

### 5. 集成到 Provider Adapter

```typescript
/**
 * 带追踪的 Provider Adapter
 */
class TrackedProviderAdapter implements ProviderAdapter {
  private adapter: ProviderAdapter;
  private tracker: UsageTracker;
  private provider: string;
  private model: string;

  constructor(
    adapter: ProviderAdapter,
    tracker: UsageTracker,
    provider: string,
    model: string
  ) {
    this.adapter = adapter;
    this.tracker = tracker;
    this.provider = provider;
    this.model = model;
  }

  async complete(context: Context): Promise<Message> {
    // 1. 调用底层 Adapter
    const message = await this.adapter.complete(context);

    // 2. 追踪使用量
    if (message.usage) {
      this.tracker.track(this.provider, this.model, message.usage);
    }

    return message;
  }

  async *stream(context: Context): AsyncGenerator<StreamEvent> {
    let usage: TokenUsage | undefined;

    // 1. 流式调用
    for await (const event of this.adapter.stream(context)) {
      if (event.type === 'end') {
        usage = event.usage;
      }
      yield event;
    }

    // 2. 追踪使用量
    if (usage) {
      this.tracker.track(this.provider, this.model, usage);
    }
  }

  get name(): string {
    return this.adapter.name;
  }

  get capabilities() {
    return this.adapter.capabilities;
  }
}
```

---

## 在 AI Agent 中的应用

### 场景1：实时成本监控

```typescript
/**
 * 实时成本监控
 */
async function monitorCost(
  userMessage: string
): Promise<void> {
  const tracker = new UsageTracker(PRICING_2026);

  // 创建带追踪的 Adapter
  const baseAdapter = new OpenAIAdapter(process.env.OPENAI_API_KEY!);
  const adapter = new TrackedProviderAdapter(
    baseAdapter,
    tracker,
    'openai',
    'gpt-4o-mini'
  );

  // 调用
  const context: Context = {
    messages: [{ role: 'user', content: userMessage }]
  };

  const message = await adapter.complete(context);

  // 显示成本
  const totalUsage = tracker.getTotalUsage();
  console.log(`\n📊 Usage:`);
  console.log(`  Input tokens: ${totalUsage.inputTokens}`);
  console.log(`  Output tokens: ${totalUsage.outputTokens}`);
  console.log(`  Total tokens: ${totalUsage.totalTokens}`);
  console.log(`  Cost: $${totalUsage.cost?.toFixed(6)}`);
}

// 使用示例
await monitorCost('Explain quantum computing in detail');
```

### 场景2：预算控制

```typescript
/**
 * 预算控制
 * 超出预算时自动切换到便宜的模型
 */
class BudgetController {
  private tracker: UsageTracker;
  private dailyBudget: number;

  constructor(tracker: UsageTracker, dailyBudget: number) {
    this.tracker = tracker;
    this.dailyBudget = dailyBudget;
  }

  /**
   * 检查是否超出预算
   */
  isOverBudget(): boolean {
    const today = new Date().setHours(0, 0, 0, 0);
    const tomorrow = today + 24 * 60 * 60 * 1000;

    const todayRecords = this.tracker.getUsageByTimeRange(today, tomorrow);
    const todayCost = todayRecords.reduce(
      (sum, record) => sum + (record.usage.cost || 0),
      0
    );

    return todayCost >= this.dailyBudget;
  }

  /**
   * 选择模型（基于预算）
   */
  selectModel(): { provider: string; model: string } {
    if (this.isOverBudget()) {
      // 超出预算，使用便宜的模型
      return {
        provider: 'openai',
        model: 'gpt-4o-mini'
      };
    } else {
      // 预算充足，使用强大的模型
      return {
        provider: 'anthropic',
        model: 'claude-opus-4'
      };
    }
  }
}

// 使用示例
const tracker = new UsageTracker(PRICING_2026);
const controller = new BudgetController(tracker, 10.0);  // $10/天

const { provider, model } = controller.selectModel();
console.log(`Selected model: ${provider}/${model}`);
```

### 场景3：成本优化分析

```typescript
/**
 * 成本优化分析
 * 分析哪个模型性价比最高
 */
async function analyzeCostEfficiency(
  testPrompt: string
): Promise<void> {
  const tracker = new UsageTracker(PRICING_2026);

  // 测试不同模型
  const models = [
    { provider: 'openai', model: 'gpt-4o-mini' },
    { provider: 'anthropic', model: 'claude-haiku-4' },
    { provider: 'google', model: 'gemini-2.0-flash' }
  ];

  console.log('Testing models...\n');

  for (const { provider, model } of models) {
    const adapter = getAdapter(provider, model);
    const trackedAdapter = new TrackedProviderAdapter(
      adapter,
      tracker,
      provider,
      model
    );

    const start = Date.now();
    const message = await trackedAdapter.complete({
      messages: [{ role: 'user', content: testPrompt }]
    });
    const latency = Date.now() - start;

    const usage = message.usage!;
    console.log(`${provider}/${model}:`);
    console.log(`  Latency: ${latency}ms`);
    console.log(`  Tokens: ${usage.totalTokens}`);
    console.log(`  Cost: $${usage.cost?.toFixed(6)}`);
    console.log(`  Cost per 1K tokens: $${((usage.cost! / usage.totalTokens) * 1000).toFixed(6)}`);
    console.log();
  }

  // 显示总成本
  const totalUsage = tracker.getTotalUsage();
  console.log(`Total cost: $${totalUsage.cost?.toFixed(6)}`);
}

// 使用示例
await analyzeCostEfficiency('Explain the concept of recursion');
```

### 场景4：使用报告生成

```typescript
/**
 * 使用报告生成
 * 生成详细的使用报告
 */
function generateUsageReport(tracker: UsageTracker): string {
  const totalUsage = tracker.getTotalUsage();
  const byProvider = tracker.getUsageByProvider();
  const byModel = tracker.getUsageByModel();

  let report = '# LLM Usage Report\n\n';

  // 总览
  report += '## Summary\n\n';
  report += `- Total tokens: ${totalUsage.totalTokens.toLocaleString()}\n`;
  report += `- Input tokens: ${totalUsage.inputTokens.toLocaleString()}\n`;
  report += `- Output tokens: ${totalUsage.outputTokens.toLocaleString()}\n`;
  report += `- Total cost: $${totalUsage.cost?.toFixed(4)}\n\n`;

  // 按 Provider
  report += '## By Provider\n\n';
  report += '| Provider | Tokens | Cost |\n';
  report += '|----------|--------|------|\n';
  for (const [provider, usage] of Object.entries(byProvider)) {
    report += `| ${provider} | ${usage.totalTokens.toLocaleString()} | $${usage.cost?.toFixed(4)} |\n`;
  }
  report += '\n';

  // 按模型
  report += '## By Model\n\n';
  report += '| Model | Tokens | Cost |\n';
  report += '|-------|--------|------|\n';
  for (const [model, usage] of Object.entries(byModel)) {
    report += `| ${model} | ${usage.totalTokens.toLocaleString()} | $${usage.cost?.toFixed(4)} |\n`;
  }

  return report;
}

// 使用示例
const tracker = new UsageTracker(PRICING_2026);
// ... 执行多次调用 ...
const report = generateUsageReport(tracker);
console.log(report);
```

### 场景5：成本预测

```typescript
/**
 * 成本预测
 * 基于历史数据预测未来成本
 */
class CostPredictor {
  private tracker: UsageTracker;

  constructor(tracker: UsageTracker) {
    this.tracker = tracker;
  }

  /**
   * 预测每日成本
   */
  predictDailyCost(): number {
    const now = Date.now();
    const oneDayAgo = now - 24 * 60 * 60 * 1000;

    const recentRecords = this.tracker.getUsageByTimeRange(oneDayAgo, now);
    const recentCost = recentRecords.reduce(
      (sum, record) => sum + (record.usage.cost || 0),
      0
    );

    return recentCost;
  }

  /**
   * 预测月度成本
   */
  predictMonthlyCost(): number {
    return this.predictDailyCost() * 30;
  }

  /**
   * 预测特定任务的成本
   */
  predictTaskCost(
    provider: string,
    model: string,
    estimatedInputLength: number,
    estimatedOutputLength: number
  ): number {
    const calculator = new CostCalculator(PRICING_2026);
    return calculator.estimate(
      provider,
      model,
      estimatedInputLength,
      estimatedOutputLength
    );
  }
}

// 使用示例
const tracker = new UsageTracker(PRICING_2026);
const predictor = new CostPredictor(tracker);

console.log(`Predicted daily cost: $${predictor.predictDailyCost().toFixed(2)}`);
console.log(`Predicted monthly cost: $${predictor.predictMonthlyCost().toFixed(2)}`);

const taskCost = predictor.predictTaskCost(
  'anthropic',
  'claude-opus-4',
  1000,  // 1000 字符输入
  2000   // 2000 字符输出
);
console.log(`Estimated task cost: $${taskCost.toFixed(6)}`);
```

---

## 设计权衡

### 优点

1. **成本透明**
   - 实时了解每次调用的成本
   - 避免意外超支

2. **优化决策**
   - 基于数据选择模型
   - 平衡性能和成本

3. **预算控制**
   - 自动切换模型
   - 防止超出预算

4. **跨 Provider 聚合**
   - 统一追踪多个 Provider
   - 便于对比分析

### 缺点

1. **定价变化**
   - Provider 定价经常变化
   - 需要及时更新配置

2. **估算不准**
   - 流式响应可能不返回 usage
   - 需要估算或后续查询

3. **存储开销**
   - 保存历史记录占用存储
   - 需要定期清理

---

## 实际案例（2025-2026）

### 案例1：LangSmith 的成本追踪

**背景：** LangSmith 提供详细的 LLM 使用追踪和成本分析

**功能：**
- 实时成本监控
- 按项目/用户聚合
- 成本预警
- 优化建议

**来源：** [LangSmith Docs](https://docs.smith.langchain.com/) (2026-02-10)

---

### 案例2：Helicone 的 LLM 可观测性

**背景：** Helicone 是专门的 LLM 可观测性平台

**功能：**
- Token 使用追踪
- 成本分析
- 性能监控
- 缓存优化

**来源：** [Helicone](https://www.helicone.ai/) (2026-01-15)

---

## 学习检查清单

完成本概念学习后，你应该能够：

- [ ] 理解 Token 计费的差异
- [ ] 能够实现成本计算器
- [ ] 能够实现使用量追踪器
- [ ] 能够集成到 Provider Adapter
- [ ] 能够实现实时成本监控
- [ ] 能够实现预算控制
- [ ] 能够生成使用报告
- [ ] 能够预测未来成本
- [ ] 理解设计权衡

---

## 参考资源

### 官方定价
- [OpenAI Pricing](https://openai.com/pricing) - OpenAI 定价
- [Anthropic Pricing](https://www.anthropic.com/pricing) - Anthropic 定价
- [Google AI Pricing](https://ai.google.dev/pricing) - Google 定价

### 相关工具
- [LangSmith](https://docs.smith.langchain.com/) - LLM 追踪平台
- [Helicone](https://www.helicone.ai/) - LLM 可观测性

---

**版本：** v1.0
**最后更新：** 2026-02-19
