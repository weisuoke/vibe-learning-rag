# 核心概念 02：树形结构与 id/parentId 设计

> 深入理解如何用 id/parentId 在扁平的 JSONL 文件中构建树形结构

---

## 树形结构定义

### 什么是树？

**树（Tree）** 是一种数据结构，由节点（Node）和边（Edge）组成：

```
        root
       /    \
      A      B
     / \      \
    C   D      E
```

**关键特性：**
- 有一个根节点（root）
- 每个节点可以有多个子节点
- 每个节点（除了根节点）有且仅有一个父节点
- 没有环路（不能从一个节点回到自己）

### 为什么会话需要树形结构？

**会话的分支特性：**

```
用户: "Hello"
  ├─ AI: "Hi there!"
  │   ├─ 用户: "How are you?"
  │   │   └─ AI: "I'm good"
  │   └─ 用户: "What's your name?"  ← 分支
  │       └─ AI: "I'm Claude"
  └─ AI: "Hey!"  ← 另一个分支
```

**实际场景：**
1. **探索不同路径**：用户想尝试不同的问题
2. **回溯历史**：回到之前的某个状态，重新开始
3. **对比结果**：同时保留多个对话路径

---

## id/parentId 设计原理

### 核心思想

**用两个字段表示父子关系：**

```typescript
interface TreeNode {
  id: string;           // 当前节点的唯一标识
  parentId: string | null; // 父节点的 ID（根节点为 null）
  // ... 其他数据
}
```

**示例：**

```jsonl
{"id": "root", "parentId": null, "content": "Session start"}
{"id": "msg1", "parentId": "root", "content": "Hello"}
{"id": "msg2", "parentId": "msg1", "content": "Hi"}
{"id": "msg3", "parentId": "msg2", "content": "How are you?"}
{"id": "msg4", "parentId": "msg2", "content": "What's your name?"}
```

**树形结构：**

```
root
└── msg1 ("Hello")
    └── msg2 ("Hi")
        ├── msg3 ("How are you?")
        └── msg4 ("What's your name?")
```

### 为什么这个设计优秀？

**1. 极简**
- 只需要两个字段：id 和 parentId
- 不需要复杂的数据结构

**2. 扁平存储**
- 所有节点在同一个文件中
- 不需要嵌套的 JSON 结构

**3. 易于追加**
- 新节点只需要知道父节点的 ID
- 不需要修改现有节点

**4. 灵活分支**
- 任何节点都可以有多个子节点
- 支持任意复杂的树形结构

---

## ID 生成机制

### 8-char Hex ID

**Pi-mono 使用 8 个字符的十六进制 ID：**

```typescript
import { randomBytes } from 'crypto';

function generateId(): string {
  return randomBytes(4).toString('hex');
}

// 示例输出
generateId(); // "a3f2c8d1"
generateId(); // "7b9e4f2a"
generateId(); // "c1d8e5f3"
```

**为什么是 8 个字符？**

- **4 bytes = 32 bits = 2^32 = 4,294,967,296 种可能**
- 对于单个会话（通常 < 10,000 条消息），碰撞概率极低
- 短小精悍，易于阅读和调试

### 碰撞检测

**虽然碰撞概率极低，但仍需检测：**

```typescript
class IdGenerator {
  private usedIds = new Set<string>();

  generateId(): string {
    let id: string;
    let attempts = 0;
    const maxAttempts = 10;

    do {
      id = randomBytes(4).toString('hex');
      attempts++;

      if (attempts >= maxAttempts) {
        throw new Error('Failed to generate unique ID after 10 attempts');
      }
    } while (this.usedIds.has(id));

    this.usedIds.add(id);
    return id;
  }

  reset(): void {
    this.usedIds.clear();
  }
}
```

**使用示例：**

```typescript
const generator = new IdGenerator();

const id1 = generator.generateId(); // "a3f2c8d1"
const id2 = generator.generateId(); // "7b9e4f2a"
const id3 = generator.generateId(); // "c1d8e5f3"

// 保证唯一性
console.log(id1 !== id2 && id2 !== id3); // true
```

---

## 树的构建算法

### 从扁平列表到树

**问题：** 如何从 JSONL 文件（扁平列表）构建树形结构？

**输入：** 扁平的条目列表

```typescript
const entries = [
  { id: "root", parentId: null, content: "Start" },
  { id: "msg1", parentId: "root", content: "Hello" },
  { id: "msg2", parentId: "msg1", content: "Hi" },
  { id: "msg3", parentId: "msg2", content: "How are you?" },
  { id: "msg4", parentId: "msg2", content: "What's your name?" }
];
```

**输出：** 树形结构

```typescript
interface TreeNode {
  entry: SessionEntry;
  children: TreeNode[];
}

const tree: TreeNode = {
  entry: { id: "root", parentId: null, content: "Start" },
  children: [
    {
      entry: { id: "msg1", parentId: "root", content: "Hello" },
      children: [
        {
          entry: { id: "msg2", parentId: "msg1", content: "Hi" },
          children: [
            {
              entry: { id: "msg3", parentId: "msg2", content: "How are you?" },
              children: []
            },
            {
              entry: { id: "msg4", parentId: "msg2", content: "What's your name?" },
              children: []
            }
          ]
        }
      ]
    }
  ]
};
```

### 构建算法（两步法）

**步骤 1：建立 ID 到条目的映射**

```typescript
const entryMap = new Map<string, SessionEntry>();
for (const entry of entries) {
  entryMap.set(entry.id, entry);
}
```

**步骤 2：建立父子关系**

```typescript
const childrenMap = new Map<string, SessionEntry[]>();

for (const entry of entries) {
  if (entry.parentId !== null) {
    if (!childrenMap.has(entry.parentId)) {
      childrenMap.set(entry.parentId, []);
    }
    childrenMap.get(entry.parentId)!.push(entry);
  }
}
```

**步骤 3：递归构建树**

```typescript
function buildTree(entryId: string): TreeNode {
  const entry = entryMap.get(entryId)!;
  const children = childrenMap.get(entryId) || [];

  return {
    entry,
    children: children.map(child => buildTree(child.id))
  };
}

// 从根节点开始构建
const tree = buildTree("root");
```

### 完整实现

```typescript
interface SessionEntry {
  id: string;
  parentId: string | null;
  type: string;
  content?: string;
  [key: string]: any;
}

interface TreeNode {
  entry: SessionEntry;
  children: TreeNode[];
}

class TreeBuilder {
  private entryMap = new Map<string, SessionEntry>();
  private childrenMap = new Map<string, SessionEntry[]>();

  constructor(private entries: SessionEntry[]) {
    this.buildMaps();
  }

  private buildMaps(): void {
    // 建立 ID 映射
    for (const entry of this.entries) {
      this.entryMap.set(entry.id, entry);
    }

    // 建立父子关系
    for (const entry of this.entries) {
      if (entry.parentId !== null) {
        if (!this.childrenMap.has(entry.parentId)) {
          this.childrenMap.set(entry.parentId, []);
        }
        this.childrenMap.get(entry.parentId)!.push(entry);
      }
    }
  }

  buildTree(rootId: string): TreeNode {
    const entry = this.entryMap.get(rootId);
    if (!entry) {
      throw new Error(`Entry not found: ${rootId}`);
    }

    const children = this.childrenMap.get(rootId) || [];

    return {
      entry,
      children: children.map(child => this.buildTree(child.id))
    };
  }

  getRoot(): SessionEntry | undefined {
    return this.entries.find(e => e.parentId === null);
  }

  buildFullTree(): TreeNode | null {
    const root = this.getRoot();
    if (!root) return null;

    return this.buildTree(root.id);
  }
}
```

**使用示例：**

```typescript
const entries = [
  { id: "root", parentId: null, type: "session", content: "Start" },
  { id: "msg1", parentId: "root", type: "user", content: "Hello" },
  { id: "msg2", parentId: "msg1", type: "assistant", content: "Hi" },
  { id: "msg3", parentId: "msg2", type: "user", content: "How are you?" },
  { id: "msg4", parentId: "msg2", type: "user", content: "What's your name?" }
];

const builder = new TreeBuilder(entries);
const tree = builder.buildFullTree();

console.log(JSON.stringify(tree, null, 2));
```

---

## 树的遍历

### 从叶子到根（getBranch）

**问题：** 给定一个节点 ID，如何获取从该节点到根节点的路径？

**用途：** 构建会话上下文时，需要从当前消息回溯到会话开始

**算法：**

```typescript
class TreeTraverser {
  private entryMap = new Map<string, SessionEntry>();

  constructor(entries: SessionEntry[]) {
    for (const entry of entries) {
      this.entryMap.set(entry.id, entry);
    }
  }

  // 从叶子到根
  getBranch(fromId: string): SessionEntry[] {
    const branch: SessionEntry[] = [];
    let currentId: string | null = fromId;

    while (currentId !== null) {
      const entry = this.entryMap.get(currentId);
      if (!entry) {
        throw new Error(`Entry not found: ${currentId}`);
      }

      branch.push(entry);
      currentId = entry.parentId;
    }

    // 反转，使其从根到叶子
    return branch.reverse();
  }
}
```

**使用示例：**

```typescript
const entries = [
  { id: "root", parentId: null, content: "Start" },
  { id: "msg1", parentId: "root", content: "Hello" },
  { id: "msg2", parentId: "msg1", content: "Hi" },
  { id: "msg3", parentId: "msg2", content: "How are you?" }
];

const traverser = new TreeTraverser(entries);
const branch = traverser.getBranch("msg3");

console.log(branch.map(e => e.content));
// ["Start", "Hello", "Hi", "How are you?"]
```

**时间复杂度：** O(h)，其中 h 是树的高度（通常很小）

### 完整树构建（getTree）

**问题：** 如何构建完整的树形结构？

**算法：** 使用前面的 TreeBuilder

```typescript
const builder = new TreeBuilder(entries);
const tree = builder.buildFullTree();
```

**时间复杂度：** O(n)，其中 n 是节点数量

### 树的可视化

**将树转换为可读的字符串：**

```typescript
function visualizeTree(node: TreeNode, indent = 0): string {
  const prefix = '  '.repeat(indent);
  let result = `${prefix}├─ ${node.entry.id}: ${node.entry.content}\n`;

  for (const child of node.children) {
    result += visualizeTree(child, indent + 1);
  }

  return result;
}

// 使用
const tree = builder.buildFullTree();
if (tree) {
  console.log(visualizeTree(tree));
}
```

**输出：**

```
├─ root: Start
  ├─ msg1: Hello
    ├─ msg2: Hi
      ├─ msg3: How are you?
      ├─ msg4: What's your name?
```

---

## 手写实现：完整的树管理器

```typescript
import { randomBytes } from 'crypto';

interface SessionEntry {
  id: string;
  parentId: string | null;
  type: string;
  timestamp: string;
  [key: string]: any;
}

interface TreeNode {
  entry: SessionEntry;
  children: TreeNode[];
}

class SessionTreeManager {
  private entries: SessionEntry[] = [];
  private entryMap = new Map<string, SessionEntry>();
  private childrenMap = new Map<string, SessionEntry[]>();
  private usedIds = new Set<string>();

  // 生成唯一 ID
  private generateId(): string {
    let id: string;
    do {
      id = randomBytes(4).toString('hex');
    } while (this.usedIds.has(id));

    this.usedIds.add(id);
    return id;
  }

  // 添加条目
  addEntry(entry: Omit<SessionEntry, 'id' | 'timestamp'>): string {
    const id = this.generateId();
    const fullEntry: SessionEntry = {
      ...entry,
      id,
      timestamp: new Date().toISOString()
    };

    this.entries.push(fullEntry);
    this.entryMap.set(id, fullEntry);

    // 更新父子关系
    if (fullEntry.parentId !== null) {
      if (!this.childrenMap.has(fullEntry.parentId)) {
        this.childrenMap.set(fullEntry.parentId, []);
      }
      this.childrenMap.get(fullEntry.parentId)!.push(fullEntry);
    }

    return id;
  }

  // 获取条目
  getEntry(id: string): SessionEntry | undefined {
    return this.entryMap.get(id);
  }

  // 获取子节点
  getChildren(id: string): SessionEntry[] {
    return this.childrenMap.get(id) || [];
  }

  // 从叶子到根
  getBranch(fromId: string): SessionEntry[] {
    const branch: SessionEntry[] = [];
    let currentId: string | null = fromId;

    while (currentId !== null) {
      const entry = this.entryMap.get(currentId);
      if (!entry) {
        throw new Error(`Entry not found: ${currentId}`);
      }

      branch.push(entry);
      currentId = entry.parentId;
    }

    return branch.reverse();
  }

  // 构建子树
  buildSubtree(rootId: string): TreeNode {
    const entry = this.entryMap.get(rootId);
    if (!entry) {
      throw new Error(`Entry not found: ${rootId}`);
    }

    const children = this.childrenMap.get(rootId) || [];

    return {
      entry,
      children: children.map(child => this.buildSubtree(child.id))
    };
  }

  // 构建完整树
  buildFullTree(): TreeNode | null {
    const root = this.entries.find(e => e.parentId === null);
    if (!root) return null;

    return this.buildSubtree(root.id);
  }

  // 获取所有叶子节点
  getLeaves(): SessionEntry[] {
    const leaves: SessionEntry[] = [];

    for (const entry of this.entries) {
      const children = this.childrenMap.get(entry.id);
      if (!children || children.length === 0) {
        leaves.push(entry);
      }
    }

    return leaves;
  }

  // 可视化树
  visualizeTree(node: TreeNode, indent = 0): string {
    const prefix = '  '.repeat(indent);
    let result = `${prefix}├─ ${node.entry.id} (${node.entry.type})\n`;

    for (const child of node.children) {
      result += this.visualizeTree(child, indent + 1);
    }

    return result;
  }

  // 获取所有条目
  getAllEntries(): SessionEntry[] {
    return [...this.entries];
  }
}
```

### 使用示例

```typescript
const manager = new SessionTreeManager();

// 创建会话
const rootId = manager.addEntry({
  type: 'session',
  parentId: null,
  cwd: '/project'
});

// 添加消息
const msg1Id = manager.addEntry({
  type: 'user',
  parentId: rootId,
  content: 'Hello'
});

const msg2Id = manager.addEntry({
  type: 'assistant',
  parentId: msg1Id,
  content: 'Hi there!'
});

const msg3Id = manager.addEntry({
  type: 'user',
  parentId: msg2Id,
  content: 'How are you?'
});

// 创建分支
const msg4Id = manager.addEntry({
  type: 'user',
  parentId: msg2Id,
  content: 'What\'s your name?'
});

// 获取分支
const branch = manager.getBranch(msg3Id);
console.log('Branch:', branch.map(e => e.content || e.type));
// ["session", "Hello", "Hi there!", "How are you?"]

// 构建完整树
const tree = manager.buildFullTree();
if (tree) {
  console.log(manager.visualizeTree(tree));
}

// 获取叶子节点
const leaves = manager.getLeaves();
console.log('Leaves:', leaves.map(e => e.id));
// [msg3Id, msg4Id]
```

---

## 2025-2026 模式

### DAG vs Tree

**DAG（有向无环图）vs Tree（树）：**

| 特性 | Tree | DAG |
|------|------|-----|
| **父节点数量** | 每个节点最多 1 个父节点 | 每个节点可以有多个父节点 |
| **结构** | 严格的层次结构 | 更灵活的图结构 |
| **用途** | 简单的分支场景 | 复杂的依赖关系 |

**Pi-mono 使用 Tree 而非 DAG：**

```
Tree（Pi-mono）:
    root
    └── msg1
        └── msg2
            ├── msg3
            └── msg4

DAG（更复杂）:
    root
    └── msg1
        └── msg2
            ├── msg3
            │   └── msg5
            └── msg4
                └── msg5  ← 多个父节点
```

**为什么选择 Tree？**
- 更简单，易于理解
- 对于会话场景，Tree 已经足够
- 避免复杂的合并逻辑

**参考：**
- [Reddit - Git append-only DAG](https://www.reddit.com/r/RedditEng)

### Append-tree 库

**append-tree 是一个专门用于在 append-only log 上构建树的库：**

```typescript
// append-tree 的核心思想
interface LogEntry {
  id: string;
  parentId: string | null;
  data: any;
}

// 从 log 构建树
function buildTreeFromLog(log: LogEntry[]): TreeNode {
  // 类似 Pi-mono 的实现
  // ...
}
```

**参考：**
- [GitHub - mafintosh/append-tree](https://github.com/mafintosh/append-tree)

### 2025-2026 最佳实践

**1. 使用 Map 而非数组查找**

```typescript
// ❌ 低效
function findEntry(id: string): SessionEntry | undefined {
  return entries.find(e => e.id === id);
}

// ✅ 高效
const entryMap = new Map(entries.map(e => [e.id, e]));
function findEntry(id: string): SessionEntry | undefined {
  return entryMap.get(id);
}
```

**2. 懒加载子树**

```typescript
class LazyTreeNode {
  private _children: LazyTreeNode[] | null = null;

  constructor(
    public entry: SessionEntry,
    private childrenMap: Map<string, SessionEntry[]>
  ) {}

  get children(): LazyTreeNode[] {
    if (this._children === null) {
      const childEntries = this.childrenMap.get(this.entry.id) || [];
      this._children = childEntries.map(e => new LazyTreeNode(e, this.childrenMap));
    }
    return this._children;
  }
}
```

**3. 缓存常用查询**

```typescript
class CachedTreeManager extends SessionTreeManager {
  private branchCache = new Map<string, SessionEntry[]>();

  getBranch(fromId: string): SessionEntry[] {
    if (this.branchCache.has(fromId)) {
      return this.branchCache.get(fromId)!;
    }

    const branch = super.getBranch(fromId);
    this.branchCache.set(fromId, branch);
    return branch;
  }

  clearCache(): void {
    this.branchCache.clear();
  }
}
```

---

## 在 Pi-mono 中的应用

### SessionManager 的树操作

**文件位置：** `sourcecode/pi-mono/packages/coding-agent/src/core/session-manager.ts`

**核心方法：**

```typescript
// 获取分支（从叶子到根）
getBranch(fromId: string): SessionEntry[] {
  const branch: SessionEntry[] = [];
  let currentId: string | null = fromId;

  while (currentId !== null) {
    const entry = this.entryMap.get(currentId);
    if (!entry) break;

    branch.push(entry);
    currentId = entry.parentId;
  }

  return branch.reverse();
}

// 构建完整树
getTree(): SessionTreeNode {
  const root = this.entries.find(e => e.parentId === null);
  if (!root) {
    throw new Error('No root entry found');
  }

  return this.buildSubtree(root.id);
}

// 递归构建子树
private buildSubtree(entryId: string): SessionTreeNode {
  const entry = this.entryMap.get(entryId)!;
  const children = this.childrenMap.get(entryId) || [];

  return {
    entry,
    children: children.map(child => this.buildSubtree(child.id))
  };
}
```

**实际使用：**

```typescript
// 构建会话上下文
buildSessionContext(): Message[] {
  const branch = this.getBranch(this.leafId);
  const messages: Message[] = [];

  for (const entry of branch) {
    if (entry.type === 'user' || entry.type === 'assistant') {
      messages.push({
        role: entry.type,
        content: entry.content
      });
    }
  }

  return messages;
}
```

---

## 关键要点总结

1. **id/parentId 设计**：用两个字段表示树形结构
2. **8-char hex ID**：短小精悍，碰撞概率极低
3. **两步构建法**：先建立映射，再递归构建
4. **getBranch**：从叶子到根，O(h) 复杂度
5. **getTree**：构建完整树，O(n) 复杂度
6. **Tree vs DAG**：Pi-mono 选择更简单的 Tree
7. **性能优化**：使用 Map、懒加载、缓存

---

**下一步**：理解分支管理与导航 → `03_核心概念_03_分支管理与导航.md`
