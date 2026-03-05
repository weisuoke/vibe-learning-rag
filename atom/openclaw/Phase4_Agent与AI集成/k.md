# Phase 4: Agent 与 AI 集成

**目标：** 深入理解 Agent 系统和 AI 模型集成

**学习时长：** 第 9-10 周

**知识点数：** 10 个

---

## 知识点列表

### 31. Pi-agent-core 深入
- Agent 运行时架构
- 工具调用机制
- 状态管理
- 消息处理

### 32. Pi-ai 统一 API
- 统一 LLM API 设计
- Provider 抽象
- 请求/响应格式
- 错误处理

### 33. Model 配置与切换
- models.json 配置
- Model 选择策略
- Model failover
- 源码位置：`src/config/`

### 34. Provider 管理（Anthropic, OpenAI, etc.）
- Anthropic Provider
- OpenAI Provider
- Bedrock Provider
- 自定义 Provider
- 源码位置：`src/provider-web.ts`

### 35. Prompt 工程
- System Prompt 设计
- Prompt Templates
- Context 管理
- Few-shot learning

### 36. 工具调用机制
- Tool Schema 定义
- Tool 注册
- Tool 执行
- Tool 结果处理

### 37. 流式响应处理
- SSE（Server-Sent Events）
- 流式解析
- 增量更新
- 用户体验优化

### 38. Context 管理
- Context Window 限制
- Context 压缩
- Context 优先级
- Memory 管理

### 39. Memory 系统
- 短期记忆
- 长期记忆
- 向量存储（LanceDB 扩展）
- 记忆检索

### 40. Agent 模式（RPC, Interactive, etc.）
- RPC 模式
- Interactive 模式
- Thinking 级别
- 模式切换

---

**验证标准：**
- ✅ 配置多个 Model Provider
- ✅ 理解 Agent 运行时
- ✅ 测试工具调用
- ✅ 理解流式响应
