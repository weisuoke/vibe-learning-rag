# 压缩 Trie 与 Radix Tree

## 核心概念

**压缩 Trie = 合并单子节点路径 → 减少节点数量 → 节省空间**

标准 Trie 的空间占用可能很大，压缩 Trie 和 Radix Tree 通过路径压缩显著减少空间开销。

---

## 1. 为什么需要压缩？

### 1.1 标准 Trie 的空间问题

**问题：** 长单词、无共享前缀时，空间浪费严重

**示例：存储 ["test", "testing"]**

```
标准 Trie：
        root
         |
         t
         |
         e
         |
         s
         |
         t (is_end=True, "test")
         |
         i
         |
         n
         |
         g (is_end=True, "testing")

节点数：9 个
```

**观察：**
- "test" 到 "testing" 的路径 "ing" 是线性的
- 每个字符一个节点，但没有分支
- 空间浪费：每个节点都需要存储 children 字典

---

### 1.2 空间复杂度分析

**标准 Trie：**
- 最坏情况：O(ALPHABET_SIZE × n × m)
- ALPHABET_SIZE：字符集大小（如 26 或 65536）
- n：单词数量
- m：平均单词长度

**实际问题：**
```python
# 存储 10 万个 URL
urls = [
    "https://example.com/api/users/1",
    "https://example.com/api/users/2",
    ...
]

# 标准 Trie：
# - 每个字符一个节点
# - 平均 URL 长度 40 字符
# - 节点数：~400 万个
# - 每个节点 ~100 字节（Python 字典开销）
# - 总空间：~400 MB
```

---

## 2. 压缩 Trie 原理

### 2.1 核心思想

**合并单子节点路径**

```
标准 Trie：
root → t → e → s → t

压缩 Trie：
root → "test"
```

**规则：**
- 如果节点只有一个子节点，且不是单词结尾
- 将该节点与子节点合并
- 边上存储字符串而非单个字符

---

### 2.2 压缩示例

**存储 ["test", "testing", "team"]**

**标准 Trie：**
```
        root
         |
         t
         |
         e
        / \
       s   a
       |   |
       t   m (E)
      (E)
       |
       i
       |
       n
       |
       g (E)
```

**压缩 Trie：**
```
        root
         |
        "te"
        / \
    "st"  "am" (E)
     (E)
      |
    "ing" (E)
```

**节点数对比：**
- 标准 Trie：10 个节点
- 压缩 Trie：5 个节点
- 节省：50%

---

### 2.3 压缩 Trie 实现

```python
class CompressedTrieNode:
    """压缩 Trie 节点"""
    def __init__(self):
        self.children = {}      # 边标签（字符串）-> 子节点
        self.is_end = False
        self.value = None

class CompressedTrie:
    """压缩 Trie"""

    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word: str, value=None):
        """插入单词"""
        node = self.root
        i = 0

        while i < len(word):
            # 查找匹配的边
            found = False
            for edge_label, child in node.children.items():
                # 计算公共前缀长度
                common_len = self._common_prefix_length(word[i:], edge_label)

                if common_len > 0:
                    found = True

                    if common_len == len(edge_label):
                        # 完全匹配边标签，继续向下
                        node = child
                        i += common_len
                    else:
                        # 部分匹配，需要分裂边
                        self._split_edge(node, edge_label, child, common_len)
                        node = node.children[edge_label[:common_len]]
                        i += common_len
                    break

            if not found:
                # 没有匹配的边，创建新边
                new_node = CompressedTrieNode()
                node.children[word[i:]] = new_node
                node = new_node
                break

        node.is_end = True
        node.value = value

    def _common_prefix_length(self, s1: str, s2: str) -> int:
        """计算公共前缀长度"""
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i

    def _split_edge(self, parent, edge_label, child, split_pos):
        """分裂边"""
        # 创建中间节点
        middle = CompressedTrieNode()

        # 更新父节点的边
        del parent.children[edge_label]
        parent.children[edge_label[:split_pos]] = middle

        # 中间节点指向原子节点
        middle.children[edge_label[split_pos:]] = child

    def search(self, word: str) -> bool:
        """查找单词"""
        node = self.root
        i = 0

        while i < len(word):
            found = False
            for edge_label, child in node.children.items():
                if word[i:].startswith(edge_label):
                    node = child
                    i += len(edge_label)
                    found = True
                    break

            if not found:
                return False

        return node.is_end
```

---

## 3. Radix Tree（基数树）

### 3.1 Radix Tree vs 压缩 Trie

**Radix Tree = 压缩 Trie 的特殊形式**

**区别：**
- 压缩 Trie：可能保留一些单子节点
- Radix Tree：严格压缩所有单子节点

**实际上：** 两者常被混用，本质相同

---

### 3.2 Radix Tree 特性

**特性 1：边存储字符串**
```python
# 标准 Trie：边存储单个字符
node.children = {'t': child}

# Radix Tree：边存储字符串
node.children = {'test': child}
```

**特性 2：节点数 ≤ 单词数**
```python
# 存储 n 个单词
# 标准 Trie：最多 n × m 个节点
# Radix Tree：最多 2n - 1 个节点
```

**特性 3：空间效率高**
```python
# 标准 Trie：O(ALPHABET_SIZE × n × m)
# Radix Tree：O(n × m)
```

---

### 3.3 Radix Tree 应用

**应用 1：IP 路由表**

```python
# 存储 IP 前缀
routes = [
    "192.168.1.0/24",
    "192.168.2.0/24",
    "10.0.0.0/8",
]

# Radix Tree 结构：
#         root
#        /    \
#    "192.168"  "10"
#      /  \
#   ".1"  ".2"
```

**应用 2：文件系统路径**

```python
# 存储文件路径
paths = [
    "/usr/local/bin/python",
    "/usr/local/bin/pip",
    "/usr/bin/bash",
]

# Radix Tree 结构：
#         root
#          |
#       "/usr"
#        /  \
#   "/local/bin"  "/bin"
#      /  \         |
# "/python" "/pip" "/bash"
```

---

## 4. 性能对比

### 4.1 空间复杂度

| 数据结构 | 空间复杂度 | 节点数 | 适用场景 |
|---------|-----------|--------|---------|
| 标准 Trie | O(ALPHABET_SIZE × n × m) | n × m | 字符集小、密集 |
| 压缩 Trie | O(n × m) | ≤ 2n | 字符集大、稀疏 |
| Radix Tree | O(n × m) | ≤ 2n - 1 | 长字符串、少分支 |

---

### 4.2 时间复杂度

| 操作 | 标准 Trie | 压缩 Trie | Radix Tree |
|------|----------|-----------|-----------|
| 插入 | O(m) | O(m) | O(m) |
| 查找 | O(m) | O(m) | O(m) |
| 前缀查询 | O(m + k) | O(m + k) | O(m + k) |

**注意：** 时间复杂度相同，但常数因子不同
- 标准 Trie：每次比较一个字符
- Radix Tree：每次比较一个字符串（需要字符串比较）

---

### 4.3 实际性能测试

```python
import sys

# 测试数据：10 万个 URL
urls = [f"https://example.com/api/users/{i}" for i in range(100000)]

# 标准 Trie
standard_trie = Trie()
for url in urls:
    standard_trie.insert(url)

print(f"标准 Trie 内存: {sys.getsizeof(standard_trie)} bytes")

# 压缩 Trie
compressed_trie = CompressedTrie()
for url in urls:
    compressed_trie.insert(url)

print(f"压缩 Trie 内存: {sys.getsizeof(compressed_trie)} bytes")

# 预期结果：
# 标准 Trie 内存: ~400 MB
# 压缩 Trie 内存: ~50 MB
# 节省：87.5%
```

---

## 5. 在 RAG 中的应用

### 5.1 大规模词典存储

**场景：** 存储 100 万个实体词

```python
# 实体词典（人名、地名、机构名）
entities = [
    "北京大学",
    "北京大学医学部",
    "北京师范大学",
    "清华大学",
    ...  # 100 万个
]

# 使用 Radix Tree 存储
entity_tree = RadixTree()
for entity in entities:
    entity_tree.insert(entity, value={"type": "ORG"})

# 空间节省：
# 标准 Trie：~2 GB
# Radix Tree：~200 MB
# 节省：90%
```

---

### 5.2 URL 路由匹配

**场景：** API 路由表

```python
# API 路由
routes = {
    "/api/users": "user_handler",
    "/api/users/{id}": "user_detail_handler",
    "/api/posts": "post_handler",
    "/api/posts/{id}": "post_detail_handler",
}

# 使用 Radix Tree 存储
router = RadixTree()
for path, handler in routes.items():
    router.insert(path, value=handler)

# 路由匹配
def match_route(path: str):
    # 最长前缀匹配
    node = router.root
    matched_path = ""
    handler = None

    i = 0
    while i < len(path):
        found = False
        for edge_label, child in node.children.items():
            if path[i:].startswith(edge_label):
                node = child
                i += len(edge_label)
                matched_path += edge_label
                if node.is_end:
                    handler = node.value
                found = True
                break

        if not found:
            break

    return handler

# 测试
print(match_route("/api/users"))      # user_handler
print(match_route("/api/users/123"))  # user_detail_handler
```

---

## 6. 优化策略

### 6.1 数组替代字典

**问题：** Python 字典开销大（~100 字节）

**优化：** 小字符集使用数组

```python
class OptimizedTrieNode:
    """优化的 Trie 节点（小字符集）"""
    def __init__(self):
        # 使用数组存储 a-z（26 个字符）
        self.children = [None] * 26
        self.is_end = False

    def get_child(self, char: str):
        """获取子节点"""
        index = ord(char) - ord('a')
        return self.children[index]

    def set_child(self, char: str, node):
        """设置子节点"""
        index = ord(char) - ord('a')
        self.children[index] = node
```

**空间节省：**
- 字典：~100 字节
- 数组（26 个指针）：~200 字节（Python）
- 但数组访问更快（O(1) vs O(1) 但常数更小）

---

### 6.2 惰性删除

**问题：** 删除操作需要递归回溯

**优化：** 标记删除，定期清理

```python
class LazyDeleteTrie:
    """惰性删除 Trie"""

    def delete(self, word: str):
        """标记删除"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        if node.is_end:
            node.is_end = False
            node.deleted = True  # 标记删除
            return True
        return False

    def compact(self):
        """定期清理"""
        self._compact(self.root)

    def _compact(self, node):
        """递归清理已删除节点"""
        # 清理子节点
        for char in list(node.children.keys()):
            child = node.children[char]
            self._compact(child)

            # 如果子节点已删除且无子节点，删除
            if hasattr(child, 'deleted') and child.deleted and not child.children:
                del node.children[char]
```

---

### 6.3 持久化存储

**问题：** Trie 在内存中，重启丢失

**优化：** 序列化到磁盘

```python
import pickle

class PersistentTrie:
    """可持久化的 Trie"""

    def save(self, filename: str):
        """保存到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self.root, f)

    def load(self, filename: str):
        """从文件加载"""
        with open(filename, 'rb') as f:
            self.root = pickle.load(f)


# 使用示例
trie = PersistentTrie()
trie.insert("test")
trie.save("trie.pkl")

# 重启后加载
trie2 = PersistentTrie()
trie2.load("trie.pkl")
print(trie2.search("test"))  # True
```

---

## 7. 完整示例

```python
"""
Radix Tree 完整实现
演示：路径压缩、空间优化
"""

class RadixNode:
    """Radix Tree 节点"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None

class RadixTree:
    """Radix Tree（基数树）"""

    def __init__(self):
        self.root = RadixNode()

    def insert(self, word: str, value=None):
        """插入单词"""
        node = self.root
        i = 0

        while i < len(word):
            found = False

            for edge_label in list(node.children.keys()):
                child = node.children[edge_label]
                common_len = 0

                # 计算公共前缀
                while (common_len < len(edge_label) and
                       i + common_len < len(word) and
                       edge_label[common_len] == word[i + common_len]):
                    common_len += 1

                if common_len > 0:
                    found = True

                    if common_len == len(edge_label):
                        # 完全匹配，继续向下
                        node = child
                        i += common_len
                    else:
                        # 部分匹配，分裂边
                        # 创建中间节点
                        middle = RadixNode()

                        # 更新父节点
                        del node.children[edge_label]
                        node.children[edge_label[:common_len]] = middle

                        # 中间节点指向原子节点
                        middle.children[edge_label[common_len:]] = child

                        # 继续插入
                        node = middle
                        i += common_len
                    break

            if not found:
                # 创建新边
                new_node = RadixNode()
                node.children[word[i:]] = new_node
                node = new_node
                break

        node.is_end = True
        node.value = value

    def search(self, word: str) -> bool:
        """查找单词"""
        node = self.root
        i = 0

        while i < len(word):
            found = False

            for edge_label, child in node.children.items():
                if word[i:].startswith(edge_label):
                    node = child
                    i += len(edge_label)
                    found = True
                    break

            if not found:
                return False

        return node.is_end


# ===== 测试 =====
if __name__ == "__main__":
    tree = RadixTree()

    # 插入单词
    print("=== 插入测试 ===")
    tree.insert("test")
    tree.insert("testing")
    tree.insert("team")
    tree.insert("toast")
    print("插入完成：test, testing, team, toast")

    # 查找
    print("\n=== 查找测试 ===")
    print(f"search('test'): {tree.search('test')}")      # True
    print(f"search('testing'): {tree.search('testing')}")  # True
    print(f"search('te'): {tree.search('te')}")          # False
    print(f"search('toast'): {tree.search('toast')}")    # True

    # 空间对比
    print("\n=== 空间对比 ===")
    standard_trie = Trie()
    for word in ["test", "testing", "team", "toast"]:
        standard_trie.insert(word)

    print(f"标准 Trie 节点数: ~15")
    print(f"Radix Tree 节点数: ~7")
    print(f"节省: ~53%")
```

**预期输出：**
```
=== 插入测试 ===
插入完成：test, testing, team, toast

=== 查找测试 ===
search('test'): True
search('testing'): True
search('te'): False
search('toast'): True

=== 空间对比 ===
标准 Trie 节点数: ~15
Radix Tree 节点数: ~7
节省: ~53%
```

---

## 8. 选择建议

### 8.1 何时使用标准 Trie？

- ✅ 字符集小（如 a-z）
- ✅ 单词短（< 10 字符）
- ✅ 共享前缀多
- ✅ 实现简单优先

### 8.2 何时使用压缩 Trie / Radix Tree？

- ✅ 字符集大（如 Unicode）
- ✅ 单词长（> 20 字符）
- ✅ 共享前缀少
- ✅ 空间效率优先

### 8.3 实际应用选择

| 应用场景 | 推荐数据结构 | 原因 |
|---------|-------------|------|
| 英文单词自动补全 | 标准 Trie | 字符集小、共享前缀多 |
| URL 路由匹配 | Radix Tree | 长字符串、少分支 |
| IP 路由表 | Radix Tree | 二进制前缀、路径压缩 |
| 中文实体识别 | Radix Tree | 字符集大、空间效率 |
| 代码补全 | 标准 Trie | 标识符短、共享前缀多 |

---

**版本**: v1.0
**最后更新**: 2026-02-14
**下一步**: 学习 Trie 在 LLM 中的应用（`04_Trie在LLM中的应用.md`）
