# 核心概念 05：自定义 Reducer 实现

> 本文档详细讲解如何实现自定义 Reducer 函数，包括自定义逻辑、边界处理、条件合并和最佳实践

---

## 引用来源

**源码分析**:
- `libs/langgraph/langgraph/graph/state.py` (行 149-180)
- `libs/langgraph/langgraph/channels/binop.py` (行 132-153)

**官方文档**:
- Context7 LangGraph 文档 (2026-02-17)

---

## 1. 为什么需要自定义 Reducer？

### 1.1 内置 Reducer 的局限性

内置 Reducer (`operator.add`, `operator.or_`, `add_messages`) 虽然强大，但无法满足所有业务需求：

```python
# 场景 1: 去重追加
# operator.add 会产生重复
[1, 2, 3] + [2, 3, 4] = [1, 2, 3, 2, 3, 4]  # 有重复

# 场景 2: 保留最大值
# operator.add 会累加
5 + 3 = 8  # 但我们想要 max(5, 3) = 5

# 场景 3: 条件合并
# operator.or_ 总是覆盖
{"status": "success"} | {"status": "error"} = {"status": "error"}
# 但我们可能想保留 "success"
```

### 1.2 自定义 Reducer 的优势

- **灵活性**: 实现任意复杂的合并逻辑
- **业务适配**: 符合特定业务规则
- **性能优化**: 针对特定场景优化
- **类型安全**: 处理特定数据类型

---

## 2. 自定义 Reducer 基础

### 2.1 函数签名

```python
def custom_reducer(old_value: T, new_value: T) -> T:
    """
    自定义 Reducer 函数

    Args:
        old_value: 当前状态中的旧值
        new_value: 节点返回的新值

    Returns:
        合并后的值
    """
    # 实现合并逻辑
    return merged_value
```

**关键要求**:
- 必须接收 2 个参数
- 必须返回 1 个值
- 输入和输出类型必须一致

### 2.2 基础示例

```python
from typing import Annotated
from typing_extensions import TypedDict

def unique_list_reducer(old: list, new: list) -> list:
    """去重追加"""
    result = old.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

class State(TypedDict):
    unique_items: Annotated[list, unique_list_reducer]

# 使用
# [1, 2, 3] + [2, 3, 4] = [1, 2, 3, 4]  # 去重
```

---

## 3. 常见自定义 Reducer 模式

### 3.1 去重追加

```python
def unique_list_reducer(old: list, new: list) -> list:
    """去重追加到列表"""
    result = old.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result

# 使用场景：标签列表、ID 列表
class State(TypedDict):
    tags: Annotated[list[str], unique_list_reducer]
```

### 3.2 保留最大值

```python
def max_reducer(old: int, new: int) -> int:
    """保留较大的值"""
    return max(old, new)

# 使用场景：最高分数、最大值
class State(TypedDict):
    max_score: Annotated[int, max_reducer]
```

### 3.3 保留最小值

```python
def min_reducer(old: int, new: int) -> int:
    """保留较小的值"""
    return min(old, new)

# 使用场景：最低价格、最小值
class State(TypedDict):
    min_price: Annotated[float, min_reducer]
```

### 3.4 按 ID 合并对象

```python
def merge_by_id(old: list[dict], new: list[dict]) -> list[dict]:
    """按 ID 合并对象列表"""
    result = {item["id"]: item for item in old}
    for item in new:
        result[item["id"]] = item
    return list(result.values())

# 使用场景：用户列表、产品列表
class State(TypedDict):
    users: Annotated[list[dict], merge_by_id]
```

### 3.5 深度合并字典

```python
def deep_merge_dict(old: dict, new: dict) -> dict:
    """递归合并字典"""
    result = old.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result

# 使用场景：配置合并、嵌套数据
class State(TypedDict):
    config: Annotated[dict, deep_merge_dict]
```

### 3.6 条件合并

```python
def conditional_merge(old: dict, new: dict) -> dict:
    """条件合并：只保留 success 状态"""
    if old.get("status") == "success":
        return old  # 保留成功状态
    return new

# 使用场景：状态管理、错误处理
class State(TypedDict):
    result: Annotated[dict, conditional_merge]
```

### 3.7 限制列表长度

```python
def limited_list_reducer(old: list, new: list, max_len: int = 100) -> list:
    """限制列表长度"""
    result = old + new
    return result[-max_len:]  # 只保留最后 max_len 个

# 使用场景：日志列表、历史记录
class State(TypedDict):
    logs: Annotated[list, lambda old, new: limited_list_reducer(old, new, 50)]
```

### 3.8 排序合并

```python
def sorted_merge(old: list, new: list) -> list:
    """合并后排序"""
    result = old + new
    return sorted(result, key=lambda x: x.get("timestamp", 0))

# 使用场景：时间序列数据、事件列表
class State(TypedDict):
    events: Annotated[list[dict], sorted_merge]
```

---

## 4. 处理边界情况

### 4.1 处理 None 值

```python
def safe_list_reducer(old: list | None, new: list | None) -> list:
    """安全的列表合并：处理 None 值"""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new

class State(TypedDict):
    items: Annotated[list, safe_list_reducer]
```

### 4.2 处理空值

```python
def non_empty_reducer(old: str, new: str) -> str:
    """只保留非空值"""
    if not new:
        return old
    return new

class State(TypedDict):
    name: Annotated[str, non_empty_reducer]
```

### 4.3 处理类型错误

```python
def type_safe_reducer(old: list, new: list | int) -> list:
    """类型安全的 Reducer"""
    if isinstance(new, int):
        new = [new]  # 转换为列表
    return old + new

class State(TypedDict):
    items: Annotated[list, type_safe_reducer]
```

---

## 5. 实际应用场景

### 5.1 文档去重系统

```python
from typing import Annotated
from typing_extensions import TypedDict

class Document:
    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content

def merge_documents(old: list[Document], new: list[Document]) -> list[Document]:
    """按 ID 合并文档，去重"""
    result = {doc.id: doc for doc in old}
    for doc in new:
        result[doc.id] = doc
    return list(result.values())

class RAGState(TypedDict):
    documents: Annotated[list[Document], merge_documents]

def retrieve_docs_a(state: RAGState) -> dict:
    return {"documents": [Document("1", "A"), Document("2", "B")]}

def retrieve_docs_b(state: RAGState) -> dict:
    return {"documents": [Document("2", "C"), Document("3", "D")]}

# 结果: [Document("1", "A"), Document("2", "C"), Document("3", "D")]
# ID "2" 的文档被替换
```

### 5.2 评分系统

```python
def max_score_reducer(old: float, new: float) -> float:
    """保留最高分数"""
    return max(old, new)

class EvaluationState(TypedDict):
    max_score: Annotated[float, max_score_reducer]
    min_score: Annotated[float, lambda old, new: min(old, new)]

def evaluate_a(state: EvaluationState) -> dict:
    return {"max_score": 0.8, "min_score": 0.8}

def evaluate_b(state: EvaluationState) -> dict:
    return {"max_score": 0.9, "min_score": 0.7}

# 结果: {"max_score": 0.9, "min_score": 0.7}
```

### 5.3 配置管理系统

```python
def deep_merge_config(old: dict, new: dict) -> dict:
    """深度合并配置"""
    result = old.copy()
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result

class ConfigState(TypedDict):
    config: Annotated[dict, deep_merge_config]

def load_default(state: ConfigState) -> dict:
    return {
        "config": {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"ttl": 3600}
        }
    }

def load_user(state: ConfigState) -> dict:
    return {
        "config": {
            "database": {"port": 5433},  # 只覆盖 port
            "cache": {"enabled": True}   # 添加新字段
        }
    }

# 结果:
# {
#     "database": {"host": "localhost", "port": 5433},
#     "cache": {"ttl": 3600, "enabled": True}
# }
```

### 5.4 日志系统

```python
def limited_log_reducer(old: list[str], new: list[str]) -> list[str]:
    """限制日志数量，只保留最后 100 条"""
    result = old + new
    return result[-100:]

class LogState(TypedDict):
    logs: Annotated[list[str], limited_log_reducer]

def log_a(state: LogState) -> dict:
    return {"logs": ["[INFO] Process A started"]}

def log_b(state: LogState) -> dict:
    return {"logs": ["[INFO] Process B started"]}

# 结果: 最多保留 100 条日志
```

---

## 6. 性能优化

### 6.1 避免不必要的复制

```python
# ❌ 坏：每次都复制
def bad_reducer(old: list, new: list) -> list:
    result = old.copy()
    result.extend(new)
    return result

# ✅ 好：使用 + 运算符（内部优化）
def good_reducer(old: list, new: list) -> list:
    return old + new
```

### 6.2 使用集合去重

```python
# ❌ 坏：O(n²) 复杂度
def bad_unique_reducer(old: list, new: list) -> list:
    result = old.copy()
    for item in new:
        if item not in result:  # O(n) 查找
            result.append(item)
    return result

# ✅ 好：O(n) 复杂度
def good_unique_reducer(old: list, new: list) -> list:
    seen = set(old)
    result = old.copy()
    for item in new:
        if item not in seen:  # O(1) 查找
            result.append(item)
            seen.add(item)
    return result
```

### 6.3 缓存计算结果

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_merge(old_tuple: tuple, new_tuple: tuple) -> tuple:
    """缓存昂贵的合并操作"""
    # 复杂的合并逻辑
    return merged_tuple

def cached_reducer(old: list, new: list) -> list:
    """使用缓存的 Reducer"""
    result = expensive_merge(tuple(old), tuple(new))
    return list(result)
```

---

## 7. 常见问题

### Q1: 如何在 Reducer 中访问其他状态字段？

**A**: Reducer 只能访问当前字段的旧值和新值，无法访问其他字段。如果需要访问其他字段，在节点中处理。

```python
# ❌ 错误：Reducer 无法访问其他字段
def bad_reducer(old: list, new: list, state: State) -> list:
    # 无法获取 state
    pass

# ✅ 正确：在节点中处理
def node(state: State) -> dict:
    # 访问其他字段
    other_field = state["other_field"]
    # 根据其他字段决定如何更新
    new_items = process(state["items"], other_field)
    return {"items": new_items}
```

### Q2: Reducer 可以有副作用吗？

**A**: 不推荐。Reducer 应该是纯函数，不应该修改输入值或产生副作用。

```python
# ❌ 坏：修改输入值
def bad_reducer(old: list, new: list) -> list:
    old.extend(new)  # 修改了 old
    return old

# ✅ 好：创建新值
def good_reducer(old: list, new: list) -> list:
    return old + new  # 创建新列表
```

### Q3: 如何处理异步操作？

**A**: Reducer 必须是同步函数。如果需要异步操作，在节点中处理。

```python
# ❌ 错误：Reducer 不能是异步函数
async def bad_reducer(old: list, new: list) -> list:
    await some_async_operation()
    return old + new

# ✅ 正确：在节点中处理异步操作
async def node(state: State) -> dict:
    result = await some_async_operation()
    return {"items": result}
```

### Q4: 如何调试自定义 Reducer？

**A**: 添加日志或使用装饰器。

```python
def debug_reducer(func):
    """调试装饰器"""
    def wrapper(old, new):
        print(f"Reducer called: {old} + {new}")
        result = func(old, new)
        print(f"Result: {result}")
        return result
    return wrapper

@debug_reducer
def my_reducer(old: list, new: list) -> list:
    return old + new
```

---

## 8. 最佳实践

### 8.1 保持简单

```python
# ✅ 好：简单明了
def simple_reducer(old: list, new: list) -> list:
    return old + new

# ❌ 坏：过度复杂
def complex_reducer(old: list, new: list) -> list:
    # 100 行复杂逻辑
    pass
```

### 8.2 处理边界情况

```python
def robust_reducer(old: list | None, new: list | None) -> list:
    """处理 None、空值等边界情况"""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new
```

### 8.3 添加类型注解

```python
from typing import List

def typed_reducer(old: List[str], new: List[str]) -> List[str]:
    """带类型注解的 Reducer"""
    return old + new
```

### 8.4 编写单元测试

```python
def test_unique_reducer():
    """测试去重 Reducer"""
    old = [1, 2, 3]
    new = [2, 3, 4]
    result = unique_list_reducer(old, new)
    assert result == [1, 2, 3, 4]
```

### 8.5 文档化

```python
def documented_reducer(old: list, new: list) -> list:
    """
    去重追加到列表

    Args:
        old: 当前列表
        new: 新元素列表

    Returns:
        去重后的列表

    Example:
        >>> documented_reducer([1, 2], [2, 3])
        [1, 2, 3]
    """
    result = old.copy()
    for item in new:
        if item not in result:
            result.append(item)
    return result
```

---

## 9. 与前端开发的类比

### Redux Reducer

**LangGraph 自定义 Reducer** 类似于 **Redux 自定义 Reducer**：

```python
# LangGraph
def custom_reducer(old: list, new: list) -> list:
    return old + new

# Redux (JavaScript)
const customReducer = (state = [], action) => {
    switch (action.type) {
        case 'ADD_ITEMS':
            return [...state, ...action.payload];
        default:
            return state;
    }
};
```

---

## 10. 总结

**自定义 Reducer 是 LangGraph 状态管理的强大工具**：

1. **灵活性**: 实现任意复杂的合并逻辑
2. **业务适配**: 符合特定业务规则
3. **性能优化**: 针对特定场景优化
4. **类型安全**: 处理特定数据类型

**关键要点**:
- 保持 Reducer 简单、纯粹
- 处理边界情况（None、空值）
- 避免副作用和修改输入值
- 添加类型注解和文档
- 编写单元测试

---

## 参考资源

1. **源码**: `libs/langgraph/langgraph/graph/state.py`
2. **官方文档**: https://langchain-ai.github.io/langgraph/
3. **示例**: https://github.com/langchain-ai/langgraph/tree/main/examples

---

**版本**: v1.0
**最后更新**: 2026-02-26
**维护者**: Claude Code
