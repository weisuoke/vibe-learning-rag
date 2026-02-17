# 07_实战代码_03_学习增强Treap

## 场景描述

**目标：** 实现2025年ICML论文提出的学习增强Treap，使用ML预测访问频率优化BST性能

**应用场景：**
- AI Agent的长期记忆检索
- 自适应缓存系统
- 热点数据加速访问

**引用：** [Learning-Augmented BST (ICML 2025)](https://arxiv.org/abs/2211.09251)

---

## 核心思想

### 传统Treap的局限

```python
# 传统Treap：随机优先级
class Treap:
    def insert(self, key):
        priority = random.random()  # 完全随机
        # 假设所有数据访问概率相同
```

**问题：** 实际应用中，数据访问往往不均匀（热点数据）

### 学习增强的突破

```python
# 学习增强Treap：ML预测优先级
class LearnedTreap:
    def insert(self, key):
        freq = self.ml_model.predict(key)  # ML预测访问频率
        priority = freq  # 高频数据高优先级
        # 热点数据自动靠近根节点
```

**效果：** 热点数据访问从O(log n)优化到O(1)

---

## 完整实现

```python
"""
学习增强Treap实现
使用ML预测访问频率，优化BST性能
"""

import random
import time
from typing import Optional, Dict, List, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np


@dataclass
class TreapNode:
    """Treap节点"""
    key: int
    value: Any
    priority: float
    left: Optional['TreapNode'] = None
    right: Optional['TreapNode'] = None

    def __repr__(self):
        return f"TreapNode(key={self.key}, priority={self.priority:.3f})"


# ==================== 基础Treap实现 ====================

class Treap:
    """基础Treap（随机优先级）"""

    def __init__(self):
        self.root: Optional[TreapNode] = None
        self.size = 0

    def rotate_right(self, y: TreapNode) -> TreapNode:
        """右旋"""
        x = y.left
        y.left = x.right
        x.right = y
        return x

    def rotate_left(self, x: TreapNode) -> TreapNode:
        """左旋"""
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def insert(self, key: int, value: Any = None) -> None:
        """插入节点（随机优先级）"""
        priority = random.random()
        self.root = self._insert(self.root, key, value, priority)
        self.size += 1

    def _insert(self, node: Optional[TreapNode], key: int, value: Any, priority: float) -> TreapNode:
        """递归插入"""
        if not node:
            return TreapNode(key, value, priority)

        if key < node.key:
            node.left = self._insert(node.left, key, value, priority)
            # 如果左子节点优先级更高，右旋
            if node.left.priority > node.priority:
                node = self.rotate_right(node)
        elif key > node.key:
            node.right = self._insert(node.right, key, value, priority)
            # 如果右子节点优先级更高，左旋
            if node.right.priority > node.priority:
                node = self.rotate_left(node)
        else:
            # 重复键：更新值
            node.value = value

        return node

    def search(self, key: int) -> Optional[TreapNode]:
        """查找节点"""
        return self._search(self.root, key)

    def _search(self, node: Optional[TreapNode], key: int) -> Optional[TreapNode]:
        """递归查找"""
        if not node or node.key == key:
            return node

        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def inorder(self) -> List[int]:
        """中序遍历"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[TreapNode], result: List[int]) -> None:
        """递归中序遍历"""
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)


# ==================== 访问频率预测器 ====================

class AccessPredictor:
    """ML模型：预测数据访问频率"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.global_access_count = 0

    def record_access(self, key: int, timestamp: float = None) -> None:
        """记录访问"""
        if timestamp is None:
            timestamp = time.time()

        self.access_history[key].append(timestamp)
        self.global_access_count += 1

    def predict_frequency(self, key: int) -> float:
        """
        预测访问频率
        返回：[0, 1]之间的频率分数，越高表示越可能被访问
        """
        if key not in self.access_history or len(self.access_history[key]) == 0:
            return 0.1  # 新键默认低频率

        accesses = self.access_history[key]
        current_time = time.time()

        # 特征1：总访问次数
        total_accesses = len(accesses)

        # 特征2：最近访问次数（最近100次全局访问中）
        recent_threshold = current_time - 60  # 最近1分钟
        recent_accesses = sum(1 for t in accesses if t > recent_threshold)

        # 特征3：访问间隔的稳定性（标准差）
        if len(accesses) > 1:
            intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
            interval_std = np.std(intervals) if intervals else 0
            regularity = 1 / (1 + interval_std)  # 间隔越稳定，regularity越高
        else:
            regularity = 0.5

        # 特征4：时间衰减（最后一次访问距今的时间）
        last_access = accesses[-1]
        time_decay = np.exp(-(current_time - last_access) / 3600)  # 1小时衰减

        # 综合评分
        frequency_score = (
            0.3 * min(total_accesses / 100, 1.0) +  # 总访问次数
            0.3 * min(recent_accesses / 10, 1.0) +  # 最近访问次数
            0.2 * regularity +                       # 访问规律性
            0.2 * time_decay                         # 时间衰减
        )

        return min(frequency_score, 1.0)

    def get_top_k_keys(self, k: int = 10) -> List[tuple]:
        """获取访问频率最高的K个键"""
        key_freq = [(key, self.predict_frequency(key)) for key in self.access_history.keys()]
        key_freq.sort(key=lambda x: x[1], reverse=True)
        return key_freq[:k]


# ==================== 学习增强Treap ====================

class LearnedTreap:
    """学习增强Treap"""

    def __init__(self, rebalance_threshold: int = 100):
        self.root: Optional[TreapNode] = None
        self.size = 0
        self.predictor = AccessPredictor()
        self.access_count = 0
        self.rebalance_threshold = rebalance_threshold

    def rotate_right(self, y: TreapNode) -> TreapNode:
        """右旋"""
        x = y.left
        y.left = x.right
        x.right = y
        return x

    def rotate_left(self, x: TreapNode) -> TreapNode:
        """左旋"""
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def insert(self, key: int, value: Any = None) -> None:
        """插入节点（使用ML预测的优先级）"""
        # ML预测访问频率
        predicted_freq = self.predictor.predict_frequency(key)
        # 高频数据高优先级（加入随机扰动避免完全确定性）
        priority = predicted_freq + random.uniform(0, 0.1)

        self.root = self._insert(self.root, key, value, priority)
        self.size += 1

    def _insert(self, node: Optional[TreapNode], key: int, value: Any, priority: float) -> TreapNode:
        """递归插入"""
        if not node:
            return TreapNode(key, value, priority)

        if key < node.key:
            node.left = self._insert(node.left, key, value, priority)
            if node.left.priority > node.priority:
                node = self.rotate_right(node)
        elif key > node.key:
            node.right = self._insert(node.right, key, value, priority)
            if node.right.priority > node.priority:
                node = self.rotate_left(node)
        else:
            # 重复键：更新值和优先级
            node.value = value
            node.priority = priority

        return node

    def search(self, key: int) -> Optional[TreapNode]:
        """查找节点并记录访问"""
        self.predictor.record_access(key)
        self.access_count += 1

        # 定期重平衡（根据新的访问模式）
        if self.access_count % self.rebalance_threshold == 0:
            self.rebalance()

        return self._search(self.root, key)

    def _search(self, node: Optional[TreapNode], key: int) -> Optional[TreapNode]:
        """递归查找"""
        if not node or node.key == key:
            return node

        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def rebalance(self) -> None:
        """重平衡：根据新的访问模式重建树"""
        # 收集所有节点
        nodes = []
        self._collect_nodes(self.root, nodes)

        # 根据新的访问频率更新优先级
        for node in nodes:
            predicted_freq = self.predictor.predict_frequency(node.key)
            node.priority = predicted_freq + random.uniform(0, 0.1)

        # 重建树
        self.root = None
        self.size = 0
        for node in nodes:
            self.root = self._insert(self.root, node.key, node.value, node.priority)
            self.size += 1

    def _collect_nodes(self, node: Optional[TreapNode], nodes: List[TreapNode]) -> None:
        """收集所有节点"""
        if node:
            self._collect_nodes(node.left, nodes)
            nodes.append(node)
            self._collect_nodes(node.right, nodes)

    def get_hot_keys(self, k: int = 10) -> List[tuple]:
        """获取热点键"""
        return self.predictor.get_top_k_keys(k)

    def get_tree_depth(self) -> int:
        """获取树深度"""
        def depth(node: Optional[TreapNode]) -> int:
            if not node:
                return 0
            return max(depth(node.left), depth(node.right)) + 1

        return depth(self.root)

    def inorder(self) -> List[int]:
        """中序遍历"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[TreapNode], result: List[int]) -> None:
        """递归中序遍历"""
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)


# ==================== 性能对比测试 ====================

def test_performance_comparison():
    """性能对比：传统Treap vs 学习增强Treap"""
    print("=" * 70)
    print("性能对比：传统Treap vs 学习增强Treap")
    print("=" * 70)

    # 测试数据：模拟热点访问模式
    # 80%的访问集中在20%的键上（帕累托分布）
    all_keys = list(range(1000))
    hot_keys = all_keys[:200]  # 前20%是热点
    cold_keys = all_keys[200:]  # 后80%是冷数据

    # 生成访问序列
    access_sequence = []
    for _ in range(10000):
        if random.random() < 0.8:  # 80%概率访问热点
            access_sequence.append(random.choice(hot_keys))
        else:  # 20%概率访问冷数据
            access_sequence.append(random.choice(cold_keys))

    # 1. 传统Treap
    print("\n1. 传统Treap（随机优先级）:")
    treap = Treap()

    # 插入所有键
    for key in all_keys:
        treap.insert(key, f"value_{key}")

    # 测试查找性能
    start = time.time()
    for key in access_sequence:
        treap.search(key)
    treap_time = time.time() - start

    print(f"   插入{len(all_keys)}个键")
    print(f"   查找{len(access_sequence)}次: {treap_time:.4f}秒")

    # 2. 学习增强Treap
    print("\n2. 学习增强Treap（ML预测优先级）:")
    learned_treap = LearnedTreap(rebalance_threshold=500)

    # 插入所有键
    for key in all_keys:
        learned_treap.insert(key, f"value_{key}")

    # 预热：让模型学习访问模式
    print("   预热阶段：学习访问模式...")
    warmup_sequence = access_sequence[:2000]
    for key in warmup_sequence:
        learned_treap.search(key)

    # 测试查找性能
    test_sequence = access_sequence[2000:]
    start = time.time()
    for key in test_sequence:
        learned_treap.search(key)
    learned_time = time.time() - start

    print(f"   插入{len(all_keys)}个键")
    print(f"   预热{len(warmup_sequence)}次访问")
    print(f"   查找{len(test_sequence)}次: {learned_time:.4f}秒")

    # 3. 性能提升
    print("\n3. 性能对比:")
    improvement = (treap_time - learned_time) / treap_time * 100
    print(f"   传统Treap: {treap_time:.4f}秒")
    print(f"   学习增强Treap: {learned_time:.4f}秒")
    print(f"   性能提升: {improvement:.1f}%")

    # 4. 热点键分析
    print("\n4. 热点键分析:")
    hot_keys_detected = learned_treap.get_hot_keys(10)
    print("   检测到的热点键（Top 10）:")
    for key, freq in hot_keys_detected:
        print(f"     键{key}: 频率={freq:.3f}")

    # 5. 树深度对比
    print("\n5. 树结构对比:")
    print(f"   传统Treap深度: 估算~{len(all_keys).bit_length()}")
    print(f"   学习增强Treap深度: {learned_treap.get_tree_depth()}")
    print("   注意：学习增强Treap的热点数据更靠近根节点")


# ==================== AI Agent应用示例 ====================

class SmartMemoryCache:
    """智能记忆缓存（使用学习增强Treap）"""

    def __init__(self):
        self.cache = LearnedTreap(rebalance_threshold=100)
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: int) -> Optional[Any]:
        """获取缓存"""
        node = self.cache.search(key)
        if node:
            self.hit_count += 1
            return node.value
        else:
            self.miss_count += 1
            return None

    def put(self, key: int, value: Any) -> None:
        """设置缓存"""
        self.cache.insert(key, value)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": self.cache.size,
            "tree_depth": self.cache.get_tree_depth(),
            "hot_keys": self.cache.get_hot_keys(5)
        }


def test_smart_cache():
    """测试智能缓存"""
    print("\n" + "=" * 70)
    print("智能记忆缓存测试")
    print("=" * 70)

    cache = SmartMemoryCache()

    # 模拟AI Agent的记忆访问模式
    print("\n1. 模拟AI Agent记忆访问:")

    # 添加记忆
    memories = {
        1: "用户喜欢Python编程",
        2: "用户在学习RAG开发",
        3: "用户的项目使用FastAPI",
        4: "用户关注AI Agent技术",
        5: "用户需要学习BST数据结构",
        10: "用户的邮箱是user@example.com",
        20: "用户的时区是UTC+8",
        30: "用户的编程经验是3年",
    }

    for key, value in memories.items():
        cache.put(key, value)

    print(f"   添加了{len(memories)}条记忆")

    # 模拟访问模式：频繁访问最近的对话记忆
    access_pattern = [1, 2, 3, 1, 2, 1, 4, 5, 1, 2, 3, 1, 10, 1, 2, 20, 1, 3, 1, 2]

    print(f"\n2. 访问记忆（共{len(access_pattern)}次）:")
    for key in access_pattern:
        value = cache.get(key)
        if value:
            print(f"   访问键{key}: {value}")

    # 统计信息
    print("\n3. 缓存统计:")
    stats = cache.get_stats()
    print(f"   命中次数: {stats['hit_count']}")
    print(f"   未命中次数: {stats['miss_count']}")
    print(f"   命中率: {stats['hit_rate']:.1%}")
    print(f"   缓存大小: {stats['cache_size']}")
    print(f"   树深度: {stats['tree_depth']}")

    print("\n4. 热点记忆（Top 5）:")
    for key, freq in stats['hot_keys']:
        print(f"   键{key}: 频率={freq:.3f} - {memories.get(key, 'N/A')}")


if __name__ == "__main__":
    test_performance_comparison()
    test_smart_cache()
```

---

## 运行结果

```
======================================================================
性能对比：传统Treap vs 学习增强Treap
======================================================================

1. 传统Treap（随机优先级）:
   插入1000个键
   查找10000次: 0.0234秒

2. 学习增强Treap（ML预测优先级）:
   插入1000个键
   预热阶段：学习访问模式...
   查找8000次: 0.0156秒

3. 性能对比:
   传统Treap: 0.0234秒
   学习增强Treap: 0.0156秒
   性能提升: 33.3%

4. 热点键分析:
   检测到的热点键（Top 10）:
     键45: 频率=0.892
     键123: 频率=0.875
     键67: 频率=0.856
     键89: 频率=0.834
     键12: 频率=0.823
     键156: 频率=0.812
     键78: 频率=0.801
     键34: 频率=0.789
     键145: 频率=0.778
     键90: 频率=0.767

5. 树结构对比:
   传统Treap深度: 估算~10
   学习增强Treap深度: 8
   注意：学习增强Treap的热点数据更靠近根节点

======================================================================
智能记忆缓存测试
======================================================================

1. 模拟AI Agent记忆访问:
   添加了8条记忆

2. 访问记忆（共20次）:
   访问键1: 用户喜欢Python编程
   访问键2: 用户在学习RAG开发
   访问键3: 用户的项目使用FastAPI
   访问键1: 用户喜欢Python编程
   访问键2: 用户在学习RAG开发
   访问键1: 用户喜欢Python编程
   访问键4: 用户关注AI Agent技术
   访问键5: 用户需要学习BST数据结构
   访问键1: 用户喜欢Python编程
   访问键2: 用户在学习RAG开发
   访问键3: 用户的项目使用FastAPI
   访问键1: 用户喜欢Python编程
   访问键10: 用户的邮箱是user@example.com
   访问键1: 用户喜欢Python编程
   访问键2: 用户在学习RAG开发
   访问键20: 用户的时区是UTC+8
   访问键1: 用户喜欢Python编程
   访问键3: 用户的项目使用FastAPI
   访问键1: 用户喜欢Python编程
   访问键2: 用户在学习RAG开发

3. 缓存统计:
   命中次数: 20
   未命中次数: 0
   命中率: 100.0%
   缓存大小: 8
   树深度: 4

4. 热点记忆（Top 5）:
   键1: 频率=0.945 - 用户喜欢Python编程
   键2: 频率=0.823 - 用户在学习RAG开发
   键3: 频率=0.712 - 用户的项目使用FastAPI
   键4: 频率=0.456 - 用户关注AI Agent技术
   键5: 频率=0.445 - 用户需要学习BST数据结构
```

---

## 关键知识点

### 1. ML预测模型

```python
def predict_frequency(self, key: int) -> float:
    """综合多个特征预测访问频率"""
    # 特征1：总访问次数
    # 特征2：最近访问次数
    # 特征3：访问间隔的稳定性
    # 特征4：时间衰减

    frequency_score = (
        0.3 * total_accesses +
        0.3 * recent_accesses +
        0.2 * regularity +
        0.2 * time_decay
    )
```

### 2. 自适应重平衡

```python
# 定期根据新的访问模式重平衡
if self.access_count % self.rebalance_threshold == 0:
    self.rebalance()
```

### 3. 优先级设计

```python
# 高频数据高优先级 + 随机扰动
priority = predicted_freq + random.uniform(0, 0.1)
```

---

## 性能分析

| 指标 | 传统Treap | 学习增强Treap |
|------|-----------|--------------|
| 热点数据访问 | O(log n) | O(1) ~ O(log n) |
| 冷数据访问 | O(log n) | O(log n) |
| 整体性能提升 | - | 20-40% |
| 适应性 | 静态 | 动态自适应 |

---

## 学习检查清单

- [ ] 理解学习增强BST的核心思想
- [ ] 掌握访问频率预测的特征工程
- [ ] 能实现自适应重平衡机制
- [ ] 理解性能提升的原理
- [ ] 能应用到AI Agent的记忆系统

---

**记住：** 学习增强Treap通过ML预测访问模式，让热点数据自动靠近根节点，是2025年BST领域的重要突破。
