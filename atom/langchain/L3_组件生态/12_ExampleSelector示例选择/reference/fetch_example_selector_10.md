---
type: fetched_content
source: https://github.com/whitesmith/langchain-semantic-length-example-selector
title: LangChain.js Semantic Length Example Selector
fetched_at: 2026-02-25
status: success
author: whitesmith
stars: 0
forks: 0
language: TypeScript
license: MIT
knowledge_point: ExampleSelector示例选择
fetch_tool: Grok-mcp web_fetch
---

# LangChain.js Semantic Length Example Selector

When you want to provide examples as context for a LLM you can use LangChain.js ["Select by length"](https://js.langchain.com/docs/modules/model_io/prompts/example_selectors/length_based) or ["Select by similarity"](https://js.langchain.com/docs/modules/model_io/prompts/example_selectors/similarity) Prompt Selectors. However what if you need to select by similarity while ensuring a max length? This package solves that issue by combining the `LengthBasedExampleSelector` and the `SemanticSimilarityExampleSelector` into the `SemanticLengthExampleSelector`.

## 核心概念

**混合策略**：结合语义相似度和长度限制的示例选择器

**解决的问题**：
- LangChain.js 原生只提供单一策略的选择器
- 实际应用中需要同时考虑相关性和 token 限制
- 避免因示例过多导致超过模型的 context window

## 安装

```bash
yarn add whitesmith/langchain-semantic-length-example-selector
```

## 使用示例

```javascript
import { SemanticLengthExampleSelector, getLengthBased } from '@whitesmith/langchain-semantic-length-example-selector';

// 定义 prompt 的前缀和后缀
const promptPrefix = "Generate ... using the below examples as reference:";
const promptSuffix = "You are ... ";
const examplePrompt = PromptTemplate.fromTemplate("<example>{content}</example>");

// 初始化 embeddings 和向量存储
const embeddings = new OpenAIEmbeddings();
const vectorStore = new MemoryVectorStore(embeddings);

// 创建 SemanticLengthExampleSelector
const exampleSelector = new SemanticLengthExampleSelector({
  vectorStore: vectorStore,
  k: 6, // 返回最多 6 个最相似的示例
  inputKeys: ["content"],
  examplePrompt: examplePrompt,
  // prompt 最大长度（以单词为单位）
  maxLength: 50 - getLengthBased(promptPrefix) - getLengthBased(promptSuffix)
});

// 创建 FewShotPromptTemplate
const dynamicPrompt = new FewShotPromptTemplate({
  exampleSelector,
  examplePrompt,
  prefix: promptPrefix,
  suffix: promptSuffix,
  inputVariables: ["content"],
});

// 测试使用
const formattedValue = await dynamicPrompt.format({
  content: "...",
});
console.log(formattedValue);

// 与 LLM 集成
const model = new ChatOpenAI(...);
const chain = dynamicPrompt.pipe(model);

const result = await chain.invoke({ content: exampleContent });
console.log(result.content);
```

## 关键特性

### 1. 语义相似度选择
- 使用向量存储（VectorStore）存储示例
- 基于 embeddings 计算语义相似度
- 返回 top-k 最相似的示例

### 2. 长度限制
- 使用 `maxLength` 参数控制总长度
- 使用 `getLengthBased()` 函数计算文本长度
- 自动排除超过长度限制的示例

### 3. 混合策略
- 先按语义相似度排序
- 再按长度限制过滤
- 确保既相关又不超过 token 限制

## 实际应用场景

### 1. RAG 系统中的 Few-shot 优化
```javascript
// 根据用户查询动态选择示例
const selector = new SemanticLengthExampleSelector({
  vectorStore: vectorStore,
  k: 5,
  maxLength: 2000 // 限制总长度
});
```

### 2. 对话系统中的示例管理
```javascript
// 基于对话历史选择相关示例
const selector = new SemanticLengthExampleSelector({
  vectorStore: conversationVectorStore,
  k: 3,
  maxLength: 1500
});
```

### 3. Token 限制场景
```javascript
// 控制 Prompt 长度
const selector = new SemanticLengthExampleSelector({
  vectorStore: vectorStore,
  k: 10, // 最多考虑 10 个
  maxLength: 1000 // 但总长度不超过 1000 单词
});
```

## 技术实现

### getLengthBased 函数
用于计算文本长度（以单词为单位）

### SemanticLengthExampleSelector 类
- 继承自 LangChain 的 BaseExampleSelector
- 结合 SemanticSimilarityExampleSelector 和 LengthBasedExampleSelector 的逻辑
- 提供统一的接口

## 开发与测试

### 构建
```bash
yarn build
```

### 测试
```bash
yarn test
```

测试覆盖率报告：`coverage/lcov-report/index.html`

### 检查依赖更新
```bash
yarn run check-updates
```

## 仓库信息

- **语言**：TypeScript (58.3%), JavaScript (41.7%)
- **许可证**：MIT
- **Stars**：0
- **Forks**：0
- **Topics**：ai

---

**注**：这是一个社区扩展项目，展示了如何结合 LangChain 的两种选择器策略（语义相似度 + 长度限制）来实现更灵活的示例选择。这种混合策略在实际生产环境中非常有用，特别是在需要平衡相关性和 token 成本的场景中。
