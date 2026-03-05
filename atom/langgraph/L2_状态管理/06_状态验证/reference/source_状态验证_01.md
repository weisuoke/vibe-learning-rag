---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/_internal/_pydantic.py
  - libs/langgraph/langgraph/_internal/_fields.py
  - libs/langgraph/langgraph/errors.py
analyzed_at: 2026-02-27
knowledge_point: 06_状态验证
---

# 源码分析：LangGraph 状态验证机制

## 分析的文件

- `libs/langgraph/langgraph/graph/state.py` - StateGraph 核心，状态 schema 处理与 channel 创建
- `libs/langgraph/langgraph/_internal/_pydantic.py` - Pydantic 模型创建工具
- `libs/langgraph/langgraph/_internal/_fields.py` - 字段默认值、类型提示提取
- `libs/langgraph/langgraph/errors.py` - 错误类型定义

## 关键发现

### 1. 状态 Schema 验证警告 (state.py:93-102)

```python
def _warn_invalid_state_schema(schema: type[Any] | Any) -> None:
    if isinstance(schema, type):
        return
    if typing.get_args(schema):
        return
    warnings.warn(
        f"Invalid state_schema: {schema}. Expected a type or Annotated[type, reducer]. "
        "Please provide a valid schema to ensure correct updates.\n"
        " See: https://langchain-ai.github.io/langgraph/reference/graphs/#stategraph"
    )
```

### 2. 状态强制转换 (state.py:1478-1489)

```python
def _pick_mapper(state_keys, schema):
    if state_keys == ["__root__"]:
        return None
    if isclass(schema) and issubclass(schema, dict):
        return None
    return partial(_coerce_state, schema)

def _coerce_state(schema, input):
    return schema(**input)  # 调用 Pydantic BaseModel(**input) 触发验证
```

关键点：当使用 Pydantic BaseModel 作为 state_schema 时，`_coerce_state` 会调用 `schema(**input)`，
这会触发 Pydantic 的完整验证流程（包括 field_validator 和 model_validator）。

### 3. Pydantic 更新处理 (_fields.py:166-188)

```python
def get_update_as_tuples(input, keys):
    if isinstance(input, BaseModel):
        keep = input.model_fields_set
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}
    # 只更新与默认值不同的字段或在 keep 集合中的字段
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (value is not None or defaults.get(k, MISSING) is not None
             or (keep is not None and k in keep))
    ]
```

### 4. 字段默认值判定 (_fields.py:79-122)

支持多种 schema 类型的默认值判定：
- TypedDict: Required/NotRequired, total=False
- Pydantic: model_fields
- dataclass: dataclasses.fields
- Optional 类型自动设为 None

### 5. 动态模型创建 (_pydantic.py:181-249)

```python
def create_model(model_name, *, field_definitions=None, root=None):
    # 使用 LRU 缓存优化性能
    # 处理保留字段名（添加 private_ 前缀）
    # 支持 RootModel 和普通模型
```

### 6. 错误类型 (errors.py)

- `InvalidUpdateError` - 无效状态更新
- `GraphRecursionError` - 图递归限制
- `ErrorCode` 枚举：INVALID_GRAPH_NODE_RETURN_VALUE, INVALID_CONCURRENT_GRAPH_UPDATE 等

### 7. 验证仅在输入时触发

关键发现：Pydantic 验证仅在图的**首个节点输入**时触发（通过 _coerce_state），
节点之间的状态传递不会重新触发完整的 Pydantic 验证。
