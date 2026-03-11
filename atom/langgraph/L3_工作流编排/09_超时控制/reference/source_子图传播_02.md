---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/tests/test_pregel.py
  - libs/langgraph/tests/test_pregel_async.py
analyzed_at: 2026-03-07
knowledge_point: 09_超时控制
---

# 源码分析：子图超时与命令传播

## 分析的文件
- `libs/langgraph/tests/test_pregel.py` - 同步子图、`ParentCommand`、timeout 相关测试
- `libs/langgraph/tests/test_pregel_async.py` - 异步子图与 timeout 传播测试

## 关键发现

### 1. LangGraph 明确测试了“内层子图超时”和“外层父图超时”同时存在的情况

测试参数：

```python
@pytest.mark.parametrize("with_timeout", [False, "inner", "outer", "both"])
```

这说明官方维护者默认承认以下四种组合都要成立：
- 不设置 timeout
- 只给子图设置 timeout
- 只给父图设置 timeout
- 父图和子图都设置 timeout

### 2. 子图和父图都可以单独设置 `step_timeout`

```python
sub_graph = sub_builder.compile(checkpointer=subgraph_persist)
if with_timeout in ("inner", "both"):
    sub_graph.step_timeout = 1

main_graph = main_builder.compile(sync_checkpointer, name="parent")
if with_timeout in ("outer", "both"):
    main_graph.step_timeout = 1
```

这意味着：**超时预算是可以分层配置的。**

### 3. `ParentCommand` 的优先级高于 timeout 误判

测试 `test_timeout_with_parent_command()` 明确断言：

```python
with pytest.raises(ParentCommand) as exc_info:
    graph.invoke({"value": "start"}, thread1)
assert exc_info.value.args[0].goto == "test_cmd"
```

即使设置了 `graph.step_timeout = 1`，只要节点返回的是 `Command(graph=Command.PARENT, ...)`，图运行时应当传播 `ParentCommand`，而不是错误地转成 timeout。

### 4. 超时机制不会吞掉语义性控制信号

结合 `_panic_or_proceed()` 可知：

- `GraphInterrupt` 会被收集并重新抛出；
- `ParentCommand` 在测试中被验证可以继续向上冒泡；
- 只有真正“还存在未完成 inflight 任务”时才被认定为超时。

这说明 timeout 在 LangGraph 里是 **调度层兜底**，而不是 **业务语义覆盖器**。

### 5. 子图 timeout 的设计启示：局部限时，保留全局流程控制

从测试用例可以推导出一个重要设计原则：

1. 子图可以有更严格的局部 timeout；
2. 父图仍保有自己的 step budget；
3. 语义命令（跳转、恢复、冒泡）优先于“误判成超时”。

## 代码片段

### 子图 / 父图双层 timeout

```python
sub_graph = sub_builder.compile(checkpointer=subgraph_persist)
if with_timeout in ("inner", "both"):
    sub_graph.step_timeout = 1

main_graph = main_builder.compile(sync_checkpointer, name="parent")
if with_timeout in ("outer", "both"):
    main_graph.step_timeout = 1
```

### Timeout 场景下仍应传播 `ParentCommand`

```python
if with_timeout:
    graph.step_timeout = 1

with pytest.raises(ParentCommand) as exc_info:
    graph.invoke({"value": "start"}, thread1)
assert exc_info.value.args[0].goto == "test_cmd"
assert exc_info.value.args[0].update == {"key": "value"}
```

## 结论

关于子图和超时，源码测试给出了三个非常稳定的结论：

1. `step_timeout` 可以分层设置在子图和父图上；
2. timeout 不应吞掉 `ParentCommand` 这类控制流信号；
3. 子图 timeout 是“局部执行预算”，父图 timeout 是“外层 orchestration 预算”。

这也是设计复杂 LangGraph 工作流时推荐采用“分层预算”的直接依据。

