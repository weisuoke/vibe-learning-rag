---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/graph/_branch.py
  - libs/langgraph/langgraph/types.py
  - libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py
  - examples/chatbot-simulation-evaluation/simulation_utils.py
analyzed_at: 2026-02-28
knowledge_point: 06_条件分支策略
---

# 源码分析：LangGraph 条件分支核心实现

## 分析的文件

- `libs/langgraph/langgraph/graph/state.py` - StateGraph 类，包含 add_conditional_edges() API
- `libs/langgraph/langgraph/graph/_branch.py` - BranchSpec 类，条件分支内部实现
- `libs/langgraph/langgraph/types.py` - Send 类和 Command 类定义
- `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py` - React Agent 实战用法
- `examples/chatbot-simulation-evaluation/simulation_utils.py` - 模拟评估示例

## 关键发现

### 1. add_conditional_edges() API 签名

```python
def add_conditional_edges(
    self,
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
        | Runnable[Any, Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
) -> Self:
```

**参数说明：**
- `source`: 源节点名称，条件边从此节点出发
- `path`: 路由函数，接收状态返回目标节点名或 Send 对象
- `path_map`: 可选的路径映射，将路由函数返回值映射到节点名

### 2. BranchSpec 内部实现

BranchSpec 是一个 NamedTuple，包含：
- `path`: 路由函数（Runnable）
- `ends`: 路径映射字典
- `input_schema`: 输入 schema（可选）

**路径推断机制：**
- 如果 path_map 是 dict，直接使用
- 如果 path_map 是 list，转换为 {name: name} 字典
- 如果 path_map 为 None，尝试从函数返回类型的 Literal 注解推断

```python
if rtn_type := get_type_hints(func).get("return"):
    if get_origin(rtn_type) is Literal:
        path_map_ = {name: name for name in get_args(rtn_type)}
```

### 3. 路由执行流程

1. `_route()` 或 `_aroute()` 被调用
2. 如果有 reader，读取状态；否则使用输入
3. 调用 path 函数获取路由结果
4. `_finish()` 处理结果：
   - 单个结果包装为列表
   - 通过 ends 映射解析目标节点
   - 验证目标有效性（不能是 None、START，Send 不能到 END）
   - 写入 channel 触发目标节点执行

### 4. Send 类 - 动态路由原语

```python
class Send:
    node: str  # 目标节点名
    arg: Any   # 发送给目标节点的自定义状态
```

**用途：** 在条件边中动态创建多个并行任务，每个任务可以有不同的输入状态。
**典型场景：** Map-Reduce 模式

### 5. Command 类 - 高级路由控制

```python
@dataclass
class Command(Generic[N]):
    graph: str | None = None      # 目标图（None=当前图，PARENT=父图）
    update: Any | None = None     # 状态更新
    resume: Any | None = None     # 中断恢复值
    goto: Send | Sequence[Send | N] | N = ()  # 导航目标
```

**Command 比 Send 更强大：**
- 可以同时更新状态和路由
- 支持跨图导航（父图）
- 支持中断恢复

### 6. React Agent 中的条件分支实战

```python
# should_continue: 根据 LLM 输出决定是否继续
workflow.add_conditional_edges("agent", should_continue, path_map=agent_paths)

# post_model_hook_router: 后处理路由
# - 有工具调用 → 返回 Send 对象列表（并行执行工具）
# - 是 ToolMessage → 返回入口点
# - 有结构化响应 → 返回 "generate_structured_response"
# - 否则 → 返回 END

# route_tool_responses: 工具响应路由
# - 有 return_direct 工具 → END
# - 否则 → 返回入口点继续循环
```

## 五种路由模式总结

| 模式 | 返回值 | 适用场景 |
|------|--------|----------|
| 简单字符串路由 | `"node_name"` | 二选一或多选一 |
| Literal 类型推断 | `Literal["a", "b"]` | 静态已知的分支 |
| path_map 映射 | 自定义键 → 节点名 | 解耦路由逻辑和节点名 |
| Send 动态路由 | `[Send(...)]` | Map-Reduce 并行 |
| Command 高级路由 | `Command(goto=..., update=...)` | 路由+状态更新 |
