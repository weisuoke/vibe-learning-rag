---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/channels/binop.py
  - libs/langgraph/langgraph/channels/last_value.py
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/pregel/_algo.py
analyzed_at: 2026-02-27
knowledge_point: 08_状态转换函数
---

# 源码分析：状态转换函数核心机制

## 分析的文件

### 1. `channels/binop.py` - Reducer 实现（BinaryOperatorAggregate）

**核心类：`BinaryOperatorAggregate`**

- 存储二元运算符应用于当前值和新值的结果
- 签名：`operator(current_value, new_value) -> updated_value`
- 支持 `Overwrite` 类型绕过 reducer

**关键方法：**

```python
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
                raise InvalidUpdateError("Can receive only one Overwrite value per super-step.")
            self.value = overwrite_value
            seen_overwrite = True
            continue
        if not seen_overwrite:
            self.value = self.operator(self.value, value)
    return True
```

**初始化逻辑：**
- 自动处理 typing 特殊形式（Sequence -> list, Set -> set, Mapping -> dict）
- 尝试用类型的无参构造函数创建初始值
- 如果失败则设为 MISSING

### 2. `channels/last_value.py` - 默认状态更新（LastValue）

**核心类：`LastValue`**

- 存储最后接收的值
- 每步只能接收一个值（否则抛出 InvalidUpdateError）
- 用于没有 reducer 的普通状态字段

```python
def update(self, values: Sequence[Value]) -> bool:
    if len(values) == 0:
        return False
    if len(values) != 1:
        raise InvalidUpdateError(
            f"At key '{self.key}': Can receive only one value per step. "
            "Use an Annotated key to handle multiple values."
        )
    self.value = values[-1]
    return True
```

### 3. `graph/state.py` - 状态 Schema 解析

**关键函数：**

#### `_get_channels(schema)` - 从 Schema 提取 Channel
- 解析 TypedDict/Pydantic 的类型注解
- 为每个字段创建对应的 Channel

#### `_get_channel(name, annotation)` - 为单个字段创建 Channel
- 优先检查是否是 ManagedValue
- 然后检查是否是显式 Channel 注解
- 然后检查是否是 BinaryOperator（reducer）
- 默认回退到 LastValue

#### `_is_field_binop(typ)` - 检测 Reducer 函数
```python
def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params) == 2:
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}")
    return None
```

**关键发现：**
- Reducer 必须是 Annotated 元数据中的最后一个可调用对象
- Reducer 必须恰好有 2 个位置参数
- 签名验证：`(a, b) -> c`

### 4. `pregel/_algo.py` - 状态转换执行

**`apply_writes()` 函数 - 核心状态转换逻辑：**

1. 按路径排序任务（确保确定性顺序）
2. 更新已见版本
3. 消费已读 Channel
4. 按 Channel 分组写入
5. 应用写入到 Channel（调用 channel.update()）

```python
# 按 Channel 分组写入
pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
for task in tasks:
    for chan, val in task.writes:
        if chan in channels:
            pending_writes_by_channel[chan].append(val)

# 应用写入到 Channel
for chan, vals in pending_writes_by_channel.items():
    if chan in channels:
        channels[chan].update(vals)
```

## 状态转换流程总结

```
节点执行 → 返回部分状态更新
    ↓
ChannelWrite 收集写入
    ↓
apply_writes() 按 Channel 分组
    ↓
channel.update(values) 应用更新
    ↓
├── LastValue: 直接覆盖（只允许一个值）
├── BinaryOperatorAggregate: 应用 reducer 函数
├── EphemeralValue: 临时值（步骤结束后清除）
└── Topic: 发布/订阅广播
    ↓
下一步节点根据 Channel 变化触发
```
