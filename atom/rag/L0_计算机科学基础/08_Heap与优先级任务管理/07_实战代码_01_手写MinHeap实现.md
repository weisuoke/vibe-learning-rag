# 实战代码01: 手写MinHeap实现

## 核心目标

**从零实现一个完整的MinHeap类,包括所有核心操作、边界处理、性能优化和测试用例。**

---

## 1. 完整实现

### 1.1 基础MinHeap类

```python
class MinHeap:
    """
    最小堆实现

    性质: heap[i] <= heap[2*i+1] and heap[i] <= heap[2*i+2]

    时间复杂度:
    - push: O(log n)
    - pop: O(log n)
    - peek: O(1)
    - heapify: O(n)

    空间复杂度: O(n)
    """

    def __init__(self):
        """初始化空堆"""
        self.heap = []

    def __len__(self):
        """返回堆大小"""
        return len(self.heap)

    def __bool__(self):
        """检查堆是否非空"""
        return len(self.heap) > 0

    def __repr__(self):
        """字符串表示"""
        return f"MinHeap({self.heap})"

    # ========== 辅助方法 ==========

    def _parent(self, i):
        """获取父节点索引"""
        return (i - 1) // 2

    def _left_child(self, i):
        """获取左子节点索引"""
        return 2 * i + 1

    def _right_child(self, i):
        """获取右子节点索引"""
        return 2 * i + 2

    def _has_parent(self, i):
        """检查是否有父节点"""
        return i > 0

    def _has_left_child(self, i):
        """检查是否有左子节点"""
        return self._left_child(i) < len(self.heap)

    def _has_right_child(self, i):
        """检查是否有右子节点"""
        return self._right_child(i) < len(self.heap)

    # ========== 核心操作 ==========

    def _heapify_up(self, i):
        """
        向上调整heap性质

        Args:
            i: 需要调整的节点索引

        时间复杂度: O(log n)
        """
        while self._has_parent(i):
            parent = self._parent(i)

            if self.heap[i] < self.heap[parent]:
                # 违反heap性质,交换
                self.heap[i], self.heap[parent] = \
                    self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _heapify_down(self, i):
        """
        向下调整heap性质

        Args:
            i: 需要调整的节点索引

        时间复杂度: O(log n)
        """
        while self._has_left_child(i):
            # 找出最小的子节点
            smallest_child_idx = self._left_child(i)

            if self._has_right_child(i):
                right_child_idx = self._right_child(i)
                if self.heap[right_child_idx] < self.heap[smallest_child_idx]:
                    smallest_child_idx = right_child_idx

            # 比较并交换
            if self.heap[i] > self.heap[smallest_child_idx]:
                self.heap[i], self.heap[smallest_child_idx] = \
                    self.heap[smallest_child_idx], self.heap[i]
                i = smallest_child_idx
            else:
                break

    def push(self, value):
        """
        插入元素

        Args:
            value: 要插入的值

        时间复杂度: O(log n)
        """
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """
        删除并返回最小元素

        Returns:
            最小元素

        Raises:
            IndexError: 如果堆为空

        时间复杂度: O(log n)
        """
        if not self.heap:
            raise IndexError("pop from empty heap")

        if len(self.heap) == 1:
            return self.heap.pop()

        # 保存根节点
        min_value = self.heap[0]

        # 末尾元素移到根
        self.heap[0] = self.heap.pop()

        # 向下调整
        self._heapify_down(0)

        return min_value

    def peek(self):
        """
        查看最小元素(不删除)

        Returns:
            最小元素

        Raises:
            IndexError: 如果堆为空

        时间复杂度: O(1)
        """
        if not self.heap:
            raise IndexError("peek from empty heap")

        return self.heap[0]

    def heapify(self, arr):
        """
        从数组建堆

        Args:
            arr: 输入数组

        时间复杂度: O(n)
        """
        self.heap = arr.copy()

        # 从最后一个非叶子节点开始,向前heapify
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

    def is_valid(self):
        """
        验证heap性质

        Returns:
            bool: 是否满足heap性质
        """
        for i in range(len(self.heap)):
            if self._has_left_child(i):
                left = self._left_child(i)
                if self.heap[i] > self.heap[left]:
                    return False

            if self._has_right_child(i):
                right = self._right_child(i)
                if self.heap[i] > self.heap[right]:
                    return False

        return True
```

---

## 2. 优化版本

### 2.1 减少交换次数

```python
class MinHeapOptimized(MinHeap):
    """
    优化版MinHeap

    优化点:
    1. 用赋值代替交换(减少赋值次数)
    2. 提前终止heapify
    """

    def _heapify_up(self, i):
        """
        优化版heapify-up

        标准版:每次交换3次赋值
        优化版:总共n+1次赋值
        """
        value = self.heap[i]

        while self._has_parent(i):
            parent = self._parent(i)

            if value < self.heap[parent]:
                # 父节点下移
                self.heap[i] = self.heap[parent]
                i = parent
            else:
                break

        # 最终位置赋值
        self.heap[i] = value

    def _heapify_down(self, i):
        """
        优化版heapify-down

        减少比较和赋值次数
        """
        value = self.heap[i]
        n = len(self.heap)

        while self._left_child(i) < n:
            # 找出较小的子节点
            child = self._left_child(i)
            right = self._right_child(i)

            if right < n and self.heap[right] < self.heap[child]:
                child = right

            if self.heap[child] < value:
                # 子节点上移
                self.heap[i] = self.heap[child]
                i = child
            else:
                break

        # 最终位置赋值
        self.heap[i] = value
```

---

## 3. 扩展功能

### 3.1 支持自定义比较

```python
from typing import Callable, Any

class CustomHeap:
    """
    支持自定义比较函数的堆

    可以实现min-heap或max-heap
    """

    def __init__(self, key: Callable[[Any], Any] = None, reverse=False):
        """
        Args:
            key: 比较键函数
            reverse: 是否反转(True为max-heap)
        """
        self.heap = []
        self.key = key or (lambda x: x)
        self.reverse = reverse

    def _compare(self, a, b):
        """比较两个元素"""
        key_a = self.key(a)
        key_b = self.key(b)

        if self.reverse:
            return key_a > key_b
        else:
            return key_a < key_b

    def _heapify_up(self, i):
        """向上调整"""
        value = self.heap[i]

        while i > 0:
            parent = (i - 1) // 2

            if self._compare(value, self.heap[parent]):
                self.heap[i] = self.heap[parent]
                i = parent
            else:
                break

        self.heap[i] = value

    def _heapify_down(self, i):
        """向下调整"""
        value = self.heap[i]
        n = len(self.heap)

        while 2 * i + 1 < n:
            left = 2 * i + 1
            right = 2 * i + 2
            child = left

            if right < n and self._compare(self.heap[right], self.heap[left]):
                child = right

            if self._compare(self.heap[child], value):
                self.heap[i] = self.heap[child]
                i = child
            else:
                break

        self.heap[i] = value

    def push(self, value):
        """插入元素"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """删除并返回极值元素"""
        if not self.heap:
            raise IndexError("pop from empty heap")

        if len(self.heap) == 1:
            return self.heap.pop()

        result = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)

        return result

    def peek(self):
        """查看极值元素"""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

# 使用示例

# Min-heap
min_heap = CustomHeap()
for val in [5, 3, 7, 1, 9]:
    min_heap.push(val)
print(min_heap.pop())  # 1

# Max-heap
max_heap = CustomHeap(reverse=True)
for val in [5, 3, 7, 1, 9]:
    max_heap.push(val)
print(max_heap.pop())  # 9

# 自定义key
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
]

heap = CustomHeap(key=lambda x: x["score"])
for student in students:
    heap.push(student)

print(heap.pop())  # {"name": "Charlie", "score": 78}
```

### 3.2 支持decrease-key操作

```python
class IndexedMinHeap:
    """
    支持decrease-key的MinHeap

    应用: Dijkstra算法、Prim算法
    """

    def __init__(self):
        self.heap = []
        self.position = {}  # value -> index in heap

    def _swap(self, i, j):
        """交换两个元素并更新position"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.position[self.heap[i]] = i
        self.position[self.heap[j]] = j

    def _heapify_up(self, i):
        """向上调整"""
        while i > 0:
            parent = (i - 1) // 2

            if self.heap[i] < self.heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _heapify_down(self, i):
        """向下调整"""
        n = len(self.heap)

        while 2 * i + 1 < n:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if self.heap[left] < self.heap[smallest]:
                smallest = left

            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def push(self, value):
        """插入元素"""
        self.heap.append(value)
        self.position[value] = len(self.heap) - 1
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """删除并返回最小元素"""
        if not self.heap:
            raise IndexError("pop from empty heap")

        if len(self.heap) == 1:
            value = self.heap.pop()
            del self.position[value]
            return value

        min_value = self.heap[0]
        del self.position[min_value]

        last_value = self.heap.pop()
        if self.heap:
            self.heap[0] = last_value
            self.position[last_value] = 0
            self._heapify_down(0)

        return min_value

    def decrease_key(self, old_value, new_value):
        """
        减小元素的值

        Args:
            old_value: 旧值
            new_value: 新值(必须 < 旧值)

        时间复杂度: O(log n)
        """
        if new_value > old_value:
            raise ValueError("new value must be smaller than old value")

        if old_value not in self.position:
            raise ValueError("value not in heap")

        # 获取位置
        i = self.position[old_value]

        # 更新值
        del self.position[old_value]
        self.heap[i] = new_value
        self.position[new_value] = i

        # 向上调整
        self._heapify_up(i)

    def contains(self, value):
        """检查元素是否在堆中"""
        return value in self.position
```

---

## 4. 完整测试套件

```python
import unittest

class TestMinHeap(unittest.TestCase):
    """MinHeap测试套件"""

    def setUp(self):
        """每个测试前执行"""
        self.heap = MinHeap()

    def test_empty_heap(self):
        """测试空堆"""
        self.assertEqual(len(self.heap), 0)
        self.assertFalse(self.heap)

        with self.assertRaises(IndexError):
            self.heap.pop()

        with self.assertRaises(IndexError):
            self.heap.peek()

    def test_single_element(self):
        """测试单元素"""
        self.heap.push(5)

        self.assertEqual(len(self.heap), 1)
        self.assertTrue(self.heap)
        self.assertEqual(self.heap.peek(), 5)
        self.assertEqual(self.heap.pop(), 5)
        self.assertEqual(len(self.heap), 0)

    def test_multiple_elements(self):
        """测试多元素"""
        values = [5, 3, 7, 1, 9, 2, 8]

        for val in values:
            self.heap.push(val)

        self.assertEqual(len(self.heap), len(values))
        self.assertTrue(self.heap.is_valid())

        # 验证pop顺序
        sorted_values = sorted(values)
        for expected in sorted_values:
            self.assertEqual(self.heap.pop(), expected)

    def test_duplicate_elements(self):
        """测试重复元素"""
        values = [5, 3, 5, 1, 3, 1]

        for val in values:
            self.heap.push(val)

        self.assertTrue(self.heap.is_valid())

        sorted_values = sorted(values)
        for expected in sorted_values:
            self.assertEqual(self.heap.pop(), expected)

    def test_heapify(self):
        """测试heapify"""
        arr = [5, 3, 7, 1, 9, 2, 8]
        self.heap.heapify(arr)

        self.assertEqual(len(self.heap), len(arr))
        self.assertTrue(self.heap.is_valid())

        sorted_arr = sorted(arr)
        for expected in sorted_arr:
            self.assertEqual(self.heap.pop(), expected)

    def test_heap_property(self):
        """测试heap性质"""
        values = [10, 5, 15, 3, 7, 12, 20]

        for val in values:
            self.heap.push(val)
            self.assertTrue(self.heap.is_valid())

        while self.heap:
            self.heap.pop()
            if self.heap:
                self.assertTrue(self.heap.is_valid())

    def test_large_dataset(self):
        """测试大数据集"""
        import random

        values = [random.randint(1, 1000) for _ in range(1000)]

        for val in values:
            self.heap.push(val)

        self.assertTrue(self.heap.is_valid())

        sorted_values = sorted(values)
        for expected in sorted_values:
            self.assertEqual(self.heap.pop(), expected)

# 运行测试
if __name__ == '__main__':
    unittest.main()
```

---

## 5. 性能基准测试

```python
import time
import random
import heapq

def benchmark_minheap():
    """性能基准测试"""

    sizes = [1000, 10000, 100000]

    for n in sizes:
        values = [random.randint(1, 1000000) for _ in range(n)]

        # 测试自定义MinHeap
        heap1 = MinHeap()
        start = time.time()
        for val in values:
            heap1.push(val)
        time_custom_push = time.time() - start

        start = time.time()
        while heap1:
            heap1.pop()
        time_custom_pop = time.time() - start

        # 测试heapq
        heap2 = []
        start = time.time()
        for val in values:
            heapq.heappush(heap2, val)
        time_heapq_push = time.time() - start

        start = time.time()
        while heap2:
            heapq.heappop(heap2)
        time_heapq_pop = time.time() - start

        print(f"\nn={n}:")
        print(f"  Custom MinHeap:")
        print(f"    Push: {time_custom_push:.4f}s")
        print(f"    Pop:  {time_custom_pop:.4f}s")
        print(f"  heapq:")
        print(f"    Push: {time_heapq_push:.4f}s")
        print(f"    Pop:  {time_heapq_pop:.4f}s")
        print(f"  Ratio:")
        print(f"    Push: {time_custom_push/time_heapq_push:.2f}x")
        print(f"    Pop:  {time_custom_pop/time_heapq_pop:.2f}x")

# 运行基准测试
benchmark_minheap()

# 输出示例:
# n=1000:
#   Custom MinHeap:
#     Push: 0.0012s
#     Pop:  0.0008s
#   heapq:
#     Push: 0.0003s
#     Pop:  0.0002s
#   Ratio:
#     Push: 4.00x
#     Pop:  4.00x
#
# 结论:heapq(C实现)比Python实现快约4倍
```

---

## 6. 可视化工具

```python
def visualize_heap(heap):
    """
    可视化heap结构

    Args:
        heap: MinHeap实例或数组
    """
    if isinstance(heap, MinHeap):
        arr = heap.heap
    else:
        arr = heap

    if not arr:
        print("Empty heap")
        return

    n = len(arr)
    height = 0
    temp = n
    while temp > 0:
        height += 1
        temp //= 2

    print("\nHeap structure:")
    index = 0
    for level in range(height):
        level_size = 2 ** level
        indent = " " * (2 ** (height - level - 1) - 1)

        for _ in range(level_size):
            if index >= n:
                break
            print(f"{indent}{arr[index]:3}", end=" ")
            index += 1
        print()

    print(f"\nArray representation: {arr}")
    print(f"Size: {n}")
    print(f"Height: {height}")

# 使用示例
heap = MinHeap()
for val in [5, 10, 15, 20, 25, 30, 35]:
    heap.push(val)

visualize_heap(heap)

# 输出:
# Heap structure:
#        5
#      10  15
#     20 25 30 35
#
# Array representation: [5, 10, 15, 20, 25, 30, 35]
# Size: 7
# Height: 3
```

---

## 7. RAG应用示例

```python
class RAGDocumentHeap:
    """
    RAG文档检索的MinHeap

    用于维护Top-K最相似文档
    """

    def __init__(self, k):
        self.k = k
        self.heap = MinHeap()
        self.doc_scores = {}  # doc_id -> score

    def add_document(self, doc_id, score):
        """
        添加文档

        Args:
            doc_id: 文档ID
            score: 相似度分数
        """
        if len(self.heap) < self.k:
            self.heap.push((score, doc_id))
            self.doc_scores[doc_id] = score
        elif score > self.heap.peek()[0]:
            # 移除最小分数的文档
            old_score, old_doc_id = self.heap.pop()
            del self.doc_scores[old_doc_id]

            # 添加新文档
            self.heap.push((score, doc_id))
            self.doc_scores[doc_id] = score

    def get_top_k(self):
        """
        获取Top-K文档

        Returns:
            [(score, doc_id), ...] 按分数降序
        """
        results = []
        temp_heap = MinHeap()
        temp_heap.heap = self.heap.heap.copy()

        while temp_heap:
            results.append(temp_heap.pop())

        return sorted(results, reverse=True)

    def get_threshold(self):
        """获取当前门槛分数"""
        if len(self.heap) < self.k:
            return float('-inf')
        return self.heap.peek()[0]

# 使用示例
retriever = RAGDocumentHeap(k=10)

# 模拟文档检索
for i in range(100):
    doc_id = f"doc_{i}"
    score = random.random()
    retriever.add_document(doc_id, score)

# 获取Top-10
top_docs = retriever.get_top_k()
print("Top-10 documents:")
for score, doc_id in top_docs:
    print(f"  {doc_id}: {score:.4f}")
```

---

## 8. 常见陷阱与调试

### 8.1 边界检查

```python
# ❌ 错误:没有检查边界
def heapify_down_wrong(heap, i):
    left = 2 * i + 1
    right = 2 * i + 2

    if heap[i] > heap[right]:  # 可能越界!
        heap[i], heap[right] = heap[right], heap[i]

# ✅ 正确:检查边界
def heapify_down_correct(heap, i):
    n = len(heap)
    left = 2 * i + 1
    right = 2 * i + 2

    if right < n and heap[i] > heap[right]:
        heap[i], heap[right] = heap[right], heap[i]
```

### 8.2 调试工具

```python
def debug_heap(heap, operation=""):
    """调试工具:打印heap状态"""
    print(f"\n=== {operation} ===")
    print(f"Heap: {heap.heap}")
    print(f"Valid: {heap.is_valid()}")
    visualize_heap(heap)

# 使用示例
heap = MinHeap()

debug_heap(heap, "Initial")

heap.push(5)
debug_heap(heap, "After push(5)")

heap.push(3)
debug_heap(heap, "After push(3)")

heap.pop()
debug_heap(heap, "After pop()")
```

---

## 9. 一句话总结

**手写MinHeap实现包括核心的heapify-up/down操作、push/pop/peek接口、heapify建堆方法,以及边界检查、性能优化、自定义比较、decrease-key等扩展功能,是理解heap原理和应用的最佳实践。**
