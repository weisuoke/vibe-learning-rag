# Phase 5: 工具与内存系统 - 知识点列表

> 目标：理解 Tool 调度和 Memory 混合检索引擎
> 学习时长：第 10-11 周
> 前置要求：Phase 1-4 完成

---

## 知识点列表

### 37. Tool Trait 详解
- Tool trait：name/description/schema/execute、JSON Schema 参数定义
- 前端类比：React Hook 接口规范（固定签名、可组合）
- ZeroClaw 场景：tools/mod.rs — 44 个内置工具的统一接口

### 38. ToolDispatcher 与工具调度
- 工具调用解析（Native/XML）、路由匹配、执行、结果格式化
- 前端类比：Redux middleware dispatch chain
- ZeroClaw 场景：Agent 循环中工具调用的完整处理流程

### 39. Shell 工具实现
- 命令执行、安全控制（allowlist/blocklist）、输出捕获、超时处理
- 前端类比：Node.js child_process.exec
- ZeroClaw 场景：tools/shell.rs — 最核心也最危险的工具

### 40. 文件操作工具
- 文件读写、路径安全（workspace scoping）、符号链接防护
- 前端类比：fs.readFile / fs.writeFile
- ZeroClaw 场景：tools/file_read.rs + file_write.rs

### 41. HTTP 与 Browser 工具
- HTTP 请求工具、Playwright 浏览器自动化、页面抓取
- 前端类比：fetch + Puppeteer
- ZeroClaw 场景：tools/http_request.rs + tools/browser.rs

### 42. 自定义 Tool 开发
- 实现 Tool trait → 定义 JSON Schema → 注册 → 测试
- 前端类比：写一个自定义 React Hook
- ZeroClaw 场景：参考 examples/custom_tool.rs，开发一个实用工具

### 43. Memory Trait 详解
- Memory trait：save/recall/search、embedding 集成、上下文构建
- 前端类比：localStorage + 搜索引擎
- ZeroClaw 场景：memory/mod.rs — 持久化记忆的统一接口

### 44. SQLite 混合检索引擎
- 向量余弦相似度 + FTS5 BM25 全文搜索、权重融合（0.7/0.3）
- 前端类比：Algolia 全文搜索 + 向量搜索混合
- ZeroClaw 场景：SQLite BLOB 存储向量、LRU 缓存、零外部依赖

### 45. Markdown Memory 后端
- Markdown 文件持久化、标题分块、人类可读格式
- 前端类比：文件系统缓存 / flat-file CMS
- ZeroClaw 场景：memory/markdown/ — 适合轻量部署的内存后端

### 46. Memory 上下文管理与裁剪
- 历史消息裁剪策略、Token 计数、上下文窗口管理、优先级排序
- 前端类比：虚拟列表按需加载（只渲染可见内容）
- ZeroClaw 场景：Agent 如何决定给 LLM 发送哪些上下文
