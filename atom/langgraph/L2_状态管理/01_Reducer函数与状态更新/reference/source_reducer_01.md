---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/channels/binop.py
  - libs/langgraph/langgraph/types.py
analyzed_at: 2026-02-26
knowledge_point: Reducer函数与状态更新
---

# 源码分析：Reducer 函数核心实现

## 分析的文件

### 1. `libs/langgraph/langgraph/graph/state.py`
StateGraph 核心实现,包含 Reducer 的定义和使用方式。

### 2. `libs/langgraph/langgraph/channels/binop.py`
BinaryOperatorAggregate Channel 的完整实现,这是 Reducer 的底层执行机制。

### 3. `libs/langgraph/langgraph/types.py`
类型定义,包含 Overwrite 等相关类型。

## 关键发现

### 1. Reducer 的定义 (state.py:112-180)

```python
class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """A graph whose nodes communicate by reading and writing to a shared state.

    The signature of each node is `State -> Partial<State>`.

    Each state key can optionally be annotated with a reducer function that
    will be used to aggregate the values of that key received from multiple nodes.
    The signature of a reducer function is `(Value, Value) -> Value`.
    """
```

**核心概念**:
- 节点签名: `State -> Partial<State>` (节点可以返回部分状态)
- Reducer 签名: `(Value, Value) -> Value` (接收两个值,返回合并后的值)
- 使用 `Annotated` 绑定 Reducer 到状态字段

**示例代码** (state.py:149-180):
```python
def reducer(a: list, b: int | None) -> list:
    if b is not None:
        return a + [b]
    return a

class State(TypedDict):
    x: Annotated[list, reducer]

class Context(TypedDict):
    r: float

graph = StateGraph(state_schema=State, context_schema=Context)

def node(state: State, runtime: Runtime[Context]) -> dict:
    r = runtime.context.get("r", 1.0)
    x = state["x"][-1]
    next_value = x * r * (1 - x)
    return {"x": next_value}

graph.add_node("A", node)
graph.set_entry_point("A")
graph.set_finish_point("A")
compiled = graph.compile()

step1 = compiled.invoke({"x": 0.5}, context={"r": 3.0})
# {'x': [0.5, 0.75]}
```

### 2. Reducer 验证逻辑 (state.py:1633-1651)

```python
def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if len(params) == 2:
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None
```

**验证规则**:
1. 检查类型是否有 `__metadata__` 属性 (来自 `Annotated`)
2. 检查最后一个元数据是否可调用
3. 检查函数签名是否有恰好 2 个参数
4. 如果不符合,抛出 `ValueError`

### 3. BinaryOperatorAggregate 实现 (binop.py:41-135)

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```
    """

    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        # special forms from typing or collections.abc are not instantiable
        # so we need to replace them with their concrete counterparts
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        try:
            self.value = typ()
        except Exception:
            self.value = MISSING

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        if self.value is MISSING:
            self.value = values[0]
            values = values[1:]
        seen_overwrite: bool = False
        for value in values:
            is_overwrite, overwrite_value = _get_overwrite(value)
            if is_overwrite:
                if seen_overwrite:
                    msg = create_error_message(
                        message="Can receive only one Overwrite value per super-step.",
                        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
                    )
                    raise InvalidUpdateError(msg)
                self.value = overwrite_value
                seen_overwrite = True
                continue
            if not seen_overwrite:
                self.value = self.operator(self.value, value)
        return True
```

**关键实现细节**:

1. **初始化策略**:
   - 尝试调用 `typ()` 创建空值 (如 `list()`, `dict()`)
   - 如果失败,设置为 `MISSING`
   - 处理抽象类型 (如 `Sequence` -> `list`)

2. **更新逻辑** (`update` 方法):
   - 如果当前值是 `MISSING`,第一个值直接赋值,不调用 Reducer
   - 后续值才使用 Reducer 合并
   - 支持 `Overwrite` 机制覆盖当前值
   - 每个 super-step 只能有一个 `Overwrite`

3. **Overwrite 检测** (binop.py:32-38):
```python
def _get_overwrite(value: Any) -> tuple[bool, Any]:
    """Inspects the given value and returns (is_overwrite, overwrite_value)."""
    if isinstance(value, Overwrite):
        return True, value.value
    if isinstance(value, dict) and set(value.keys()) == {OVERWRITE}:
        return True, value[OVERWRITE]
    return False, None
```

### 4. 状态更新流程

**完整流程**:
1. 节点返回部分状态更新 (如 `{"x": 5}`)
2. StateGraph 识别字段是否有 Reducer
3. 如果有 Reducer,创建 `BinaryOperatorAggregate` Channel
4. Channel 的 `update` 方法被调用
5. Reducer 函数合并旧值和新值
6. 更新后的值保存到状态中

**示例**:
```python
# 初始状态
state = {"x": [1, 2, 3]}

# 节点 A 返回
node_a_output = {"x": 4}

# 节点 B 返回
node_b_output = {"x": 5}

# Reducer 合并
# 第一次: reducer([1, 2, 3], 4) -> [1, 2, 3, 4]
# 第二次: reducer([1, 2, 3, 4], 5) -> [1, 2, 3, 4, 5]

# 最终状态
final_state = {"x": [1, 2, 3, 4, 5]}
```

## 技术要点总结

### 1. Reducer 函数签名
- **必须**: 接收 2 个参数
- **返回**: 合并后的值
- **类型**: `(Value, Value) -> Value`

### 2. 初始值处理
- 如果类型可实例化 (如 `list`, `dict`),自动创建空值
- 第一个更新值直接赋值,不调用 Reducer
- 后续值才使用 Reducer 合并

### 3. Overwrite 机制
- 使用 `Overwrite(value)` 或 `{"__overwrite__": value}` 覆盖当前值
- 不调用 Reducer,直接替换
- 每个 super-step 只能有一个 Overwrite

### 4. 常见 Reducer
- `operator.add`: 字符串拼接、列表合并、数值相加
- `operator.or_`: 字典合并
- 自定义函数: 处理复杂逻辑

## 源码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| StateGraph 定义 | state.py | 112-180 |
| Reducer 验证 | state.py | 1633-1651 |
| BinaryOperatorAggregate | binop.py | 41-135 |
| Overwrite 检测 | binop.py | 32-38 |
| 示例代码 | state.py | 149-180 |
