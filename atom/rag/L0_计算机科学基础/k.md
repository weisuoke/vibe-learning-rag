# L0_计算机科学基础 - 知识点列表

> 数据结构在 AI Agent 中的应用 - 50% 经典原理 + 50% AI 实战

---

## 学习目标

掌握数据结构的核心原理，并理解它们在 AI Agent、RAG 系统、知识图谱中的实际应用。

**核心理念：**
- **原理与应用平衡**：50% 经典数据结构原理 + 50% AI Agent 应用
- **实战导向**：每个知识点都联系 2026 年最新 AI 技术
- **代码可运行**：所有示例使用 Python，可直接运行
- **初学者友好**：双重类比（前端 + 日常生活 + AI 场景）

---

## 模块1: 基础线性结构 (5个)

### 01. Array与List在Agent状态管理中的应用
- **前置依赖**: 无
- **核心**: 连续存储、随机访问O(1)、动态扩容
- **应用**: 对话历史、状态序列、批量Embedding
- **学习时长**: 1-1.5小时

### 02. Hash_Table与快速检索系统
- **前置依赖**: 01_Array与List
- **核心**: 哈希函数、冲突解决、负载因子
- **应用**: Token映射、缓存系统、去重检测
- **学习时长**: 1-1.5小时

### 03. Stack与Agent调用链追踪
- **前置依赖**: 01_Array与List
- **核心**: LIFO、push/pop、递归与栈
- **应用**: 调用链追踪、Prompt嵌套、回溯搜索
- **学习时长**: 1小时

### 04. Queue与任务调度系统
- **前置依赖**: 01_Array与List
- **核心**: FIFO、循环队列、优先级队列
- **应用**: 异步任务、批处理、流式生成
- **学习时长**: 1小时

### 05. Deque与滑动窗口技术
- **前置依赖**: 03_Stack, 04_Queue
- **核心**: 双端队列、两端O(1)操作
- **应用**: 滑动窗口Context、Token限制处理
- **学习时长**: 1小时

---

## 模块2: 树形结构 (4个)

### 06. Binary_Tree与决策树直觉
- **前置依赖**: 01_Array与List
- **核心**: 二叉树定义、遍历、高度与平衡
- **应用**: 决策流程、分类器、Agent行为树
- **学习时长**: 1.5小时

### 07. BST与有序数据检索
- **前置依赖**: 06_Binary_Tree
- **核心**: BST性质、查找/插入/删除O(log n)
- **应用**: 有序索引、范围查询、时间戳检索
- **学习时长**: 1.5小时

### 08. Heap与优先级任务管理
- **前置依赖**: 06_Binary_Tree
- **核心**: 堆性质、堆化、Top-K问题
- **应用**: 优先级队列、Beam Search、资源调度
- **学习时长**: 1.5小时

### 09. Trie与前缀匹配系统
- **前置依赖**: 06_Binary_Tree
- **核心**: 字典树、前缀查找、空间换时间
- **应用**: 自动补全、Token前缀树、实体识别
- **学习时长**: 1.5小时

---

## 模块3: 图结构 (3个)

### 10. Graph表示与知识图谱基础
- **前置依赖**: 02_Hash_Table
- **核心**: 邻接矩阵/表、有向/无向图
- **应用**: 知识图谱、实体关系、Agent拓扑
- **学习时长**: 1.5小时

### 11. BFS_DFS与路径搜索
- **前置依赖**: 10_Graph表示
- **核心**: 广度/深度优先搜索、遍历策略
- **应用**: 知识遍历、推理路径、关系发现
- **学习时长**: 1.5小时

### 12. 最短路径与推理链优化
- **前置依赖**: 11_BFS_DFS
- **核心**: Dijkstra、A*搜索、路径权重
- **应用**: 最优推理路径、多跳问答、Agent规划
- **学习时长**: 1.5小时

---

## 模块4: AI特定数据结构 (4个)

### 13. Vector_Store原理与实现(HNSW_IVF)
- **前置依赖**: 02_Hash_Table, 08_Heap
- **核心**: 向量空间、ANN、HNSW图、IVF倒排
- **应用**: Embedding检索、语义搜索、RAG向量库
- **学习时长**: 2小时

### 14. Knowledge_Graph实战设计
- **前置依赖**: 10_Graph表示, 11_BFS_DFS
- **核心**: 三元组(SPO)、图数据库、Cypher/SPARQL
- **应用**: 知识存储、关系推理、时序知识图谱(Graphiti)
- **学习时长**: 2小时

### 15. State_Machine与Agent状态管理
- **前置依赖**: 03_Stack, 10_Graph表示
- **核心**: 状态机模型、状态转移、FSA
- **应用**: Agent状态流转、LangGraph状态图、工作流
- **学习时长**: 2小时

### 16. Memory_System架构设计
- **前置依赖**: 02_Hash_Table, 13_Vector_Store, 14_Knowledge_Graph
- **核心**: 多级缓存、LRU/LFU、持久化
- **应用**: 短期/长期记忆、跨会话记忆(LangGraph Memory)
- **学习时长**: 2小时

---

## 学习路径建议

### 快速入门路径（8小时）
适合已有编程基础，想快速了解 AI Agent 数据结构应用的学习者。

```
01_Array与List → 02_Hash_Table → 03_Stack → 04_Queue
  ↓
10_Graph表示 → 13_Vector_Store → 15_State_Machine → 16_Memory_System
```

### 完整学习路径（16-24小时）
适合想系统掌握数据结构并深入理解 AI 应用的学习者。

```
模块1: 基础线性结构 (5-7小时)
  01 → 02 → 03 → 04 → 05

模块2: 树形结构 (6小时)
  06 → 07 → 08 → 09

模块3: 图结构 (4.5小时)
  10 → 11 → 12

模块4: AI特定结构 (8小时)
  13 → 14 → 15 → 16
```

### 实战项目驱动路径
边学边做，通过项目巩固知识。

**初级项目：Token映射与缓存系统（1周）**
- 学习：01, 02
- 实现：Token到ID映射、Prompt缓存、去重检测

**中级项目：对话式Agent状态管理（2-3周）**
- 学习：03, 04, 05, 15
- 实现：调用链追踪、任务调度、滑动窗口Context、状态流转

**高级项目：混合检索RAG系统（4-6周）**
- 学习：13, 14, 16
- 实现：向量检索、知识图谱、短期/长期记忆、混合检索

---

## 学习检查清单

### 模块1完成标准
- [ ] 理解连续存储与随机访问的区别
- [ ] 能用Python实现基本的Hash Table
- [ ] 理解Stack在调用链追踪中的应用
- [ ] 能设计简单的任务队列系统
- [ ] 掌握滑动窗口技术处理Token限制

### 模块2完成标准
- [ ] 能手写二叉树的三种遍历
- [ ] 理解BST的查找效率
- [ ] 能用Heap实现Top-K算法
- [ ] 理解Trie在Token前缀树中的应用

### 模块3完成标准
- [ ] 能用邻接表表示知识图谱
- [ ] 理解BFS/DFS的应用场景差异
- [ ] 能实现Dijkstra最短路径算法

### 模块4完成标准
- [ ] 理解HNSW向量检索原理
- [ ] 能设计简单的知识图谱Schema
- [ ] 理解LangGraph状态机模型
- [ ] 能设计多级记忆系统架构

---

## 2026年最新技术引用

本学习路径紧密结合 2026 年最新 AI Agent 技术：

### GitHub项目
- **Graphiti**: 时序知识图谱 (github.com/getzep/graphiti) - 14_Knowledge_Graph
- **LangGraph Memory**: PostgreSQL+pgvector (github.com/FareedKhan-dev/langgraph-long-memory) - 16_Memory_System
- **Hindsight**: 仿生记忆系统 (github.com/vectorize-io/hindsight) - 16_Memory_System
- **TrustGraph**: 混合向量-图检索 (github.com/trustgraph-ai/.github) - 13_Vector_Store, 14_Knowledge_Graph

### 技术趋势
- JIT Symbolic Memory Architecture - 16_Memory_System
- LangGraph stateful agents - 15_State_Machine
- Long-term memory across threads - 16_Memory_System
- Hybrid vector-graph retrieval - 13_Vector_Store, 14_Knowledge_Graph

---

## 双重类比速查表

| 数据结构 | 前端类比 | 日常生活类比 | AI Agent场景 |
|---------|---------|---------|-------------|
| Array/List | 组件列表 | 排队的人 | 对话历史 |
| Hash Table | Map/Object | 字典 | Token映射 |
| Stack | 调用栈 | 叠盘子 | 调用链追踪 |
| Queue | 事件队列 | 排队买票 | 异步任务 |
| Deque | 历史记录 | 滑动门 | 滑动窗口Context |
| Binary Tree | DOM树 | 家族树 | 决策树 |
| BST | 有序索引 | 图书馆分类 | 有序检索 |
| Heap | 优先级队列 | VIP通道 | Beam Search |
| Trie | 路由匹配 | 字典目录 | Token前缀树 |
| Graph | 依赖图 | 地铁线路图 | 知识图谱 |
| BFS/DFS | 树遍历 | 走迷宫 | 知识遍历 |
| Shortest Path | 路由最优 | 导航 | 最优推理链 |
| Vector Store | 搜索引擎 | 相似书推荐 | Embedding检索 |
| Knowledge Graph | GraphQL | 百科全书 | 知识存储 |
| State Machine | 路由状态 | 红绿灯 | Agent状态流转 |
| Memory System | 浏览器缓存 | 人的记忆 | 短期/长期记忆 |

---

## 学习建议

1. **按模块顺序学习**：模块1是基础，必须先掌握
2. **理论与实践结合**：每学完一个知识点，立即运行代码示例
3. **联系实际应用**：思考每个数据结构在你的项目中如何应用
4. **做笔记和总结**：用自己的话总结每个知识点的核心
5. **完成实战项目**：通过项目巩固所学知识

---

**总学习时长**: 16-24小时
**知识点数量**: 16个
**代码示例**: 16个完整可运行的Python示例
**实战项目**: 3个递进式项目

**开始学习**: 从 `01_Array与List在Agent状态管理中的应用` 开始！
