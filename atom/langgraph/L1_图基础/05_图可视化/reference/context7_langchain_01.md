---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: 05_图可视化
context7_query: graph visualization draw_mermaid get_graph
---

# Context7 文档: LangChain 图可视化

## 文档来源
- 库名称: LangChain
- 版本: latest (2026-02-17)
- 官方文档链接: https://docs.langchain.com
- Context7 库 ID: /websites/langchain

## 关键信息提取

### 1. Python 图可视化基础

**生成 Mermaid 语法**

来源: https://docs.langchain.com/oss/python/langgraph/use-graph-api

```python
print(app.get_graph().draw_mermaid())
```

**功能**: 将 LangChain 图对象转换为 Mermaid 兼容的字符串,可用于渲染流程图。

### 2. Python 图可视化进阶

**在 Jupyter 中可视化**

来源: https://docs.langchain.com/oss/python/langgraph/use-graph-api

```python
from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))
```

**功能**:
- 编译 LangGraph builder
- 显示图结构为 Mermaid PNG 图像
- 展示所有配置的节点和边
- 需要 IPython display 工具

### 3. xray 参数使用

**深度可视化子图**

来源: https://docs.langchain.com/langsmith/evaluate-complex-agent

```python
display(Image(qa_graph.get_graph(xray=True).draw_mermaid_png()))
```

**功能**:
- `xray=True` 参数用于获取图的内部表示
- 递归展开子图结构
- 生成 Mermaid 图表为 PNG 图像
- 在交互环境中渲染图像
- 提供代理流程的可视化概览

### 4. TypeScript 图可视化

**生成 Mermaid PNG (TypeScript)**

来源: https://docs.langchain.com/oss/javascript/langgraph/use-graph-api

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

**功能**:
- 获取可绘制的图表示
- 生成 Mermaid PNG 图像
- 转换为 Uint8Array 缓冲区
- 写入文件
- 支持图结构和节点连接的可视化检查

**生成 Mermaid 语法 (TypeScript)**

来源: https://docs.langchain.com/oss/javascript/langgraph/use-graph-api

```typescript
const drawableGraph = await app.getGraphAsync();
console.log(drawableGraph.drawMermaid());
```

**功能**:
- 获取编译后的图
- 转换为 Mermaid 图表语法用于可视化
- 使用 `getGraphAsync()` 获取可绘制的图对象
- 使用 `drawMermaid()` 生成图表表示为字符串

## 核心 API 总结

### Python API

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `get_graph()` | 获取可绘制的图对象 | Graph 对象 |
| `get_graph(xray=True)` | 递归获取包含子图的图对象 | Graph 对象 |
| `draw_mermaid()` | 生成 Mermaid 文本语法 | 字符串 |
| `draw_mermaid_png()` | 生成 Mermaid PNG 图像 | 字节数据 |

### TypeScript API

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `getGraphAsync()` | 异步获取可绘制的图对象 | Promise<Graph> |
| `drawMermaid()` | 生成 Mermaid 文本语法 | 字符串 |
| `drawMermaidPng()` | 生成 Mermaid PNG 图像 | Promise<ArrayBuffer> |

## 使用场景

### 1. 开发调试
- 在开发过程中可视化图结构
- 验证节点和边的连接关系
- 检查条件路由的逻辑

### 2. 文档生成
- 为项目文档生成图表
- 导出为 PNG 图像嵌入文档
- 生成 Mermaid 语法用于 Markdown

### 3. 交互式探索
- 在 Jupyter Notebook 中交互式查看图结构
- 使用 xray 参数深入探索子图
- 快速理解复杂的代理流程

### 4. 团队协作
- 分享图结构给团队成员
- 讨论图设计和优化
- 代码审查时可视化变更

## 依赖要求

### Python
- IPython (用于 Jupyter 显示)
- 可选依赖 (用于 PNG 生成)

### TypeScript
- Node.js fs 模块
- 支持 async/await

## 最佳实践

1. **开发阶段**: 使用 `draw_mermaid()` 快速查看文本表示
2. **文档阶段**: 使用 `draw_mermaid_png()` 生成高质量图像
3. **调试阶段**: 使用 `xray=True` 深入查看子图结构
4. **生产环境**: 避免在生产代码中生成可视化 (性能考虑)

## 注意事项

1. PNG 生成需要额外依赖,可能在某些环境中失败
2. 大型图可能生成较大的图像文件
3. xray 参数会递归展开所有子图,可能导致图表过于复杂
4. Mermaid 语法有长度限制,超大图可能无法正确渲染
