---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/pregel/main.py (invoke method)
  - libs/langgraph/langgraph/graph/state.py (compile method)
analyzed_at: 2026-02-25
knowledge_point: 04_图的编译与执行
---

# 源码分析：invoke 方法与执行流程

## 分析的文件

- `libs/langgraph/langgraph/pregel/main.py:3024` - Pregel.invoke() 方法
- `libs/langgraph/langgraph/graph/state.py:1035` - StateGraph.compile() 方法

## invoke 方法详细分析

### 方法签名（pregel/main.py:3024）

```python
def invoke(
    self,
    input: InputT | Command | None,
    config: RunnableConfig | None = None,
    *,
    context: ContextT | None = None,
    stream_mode: StreamMode = "values",
    print_mode: StreamMode | Sequence[StreamMode] = (),
    output_keys: str | Sequence[str] | None = None,
    interrupt_before: All | Sequence[str] | None = None,
    interrupt_after: All | Sequence[str] | None = None,
    durability: Durability | None = None,
    **kwargs: Any,
) -> dict[str, Any] | Any:
```

### 参数说明

1. **input**: 图的输入数据
   - 可以是字典或任何其他类型
   - 也可以是 `Command` 对象（用于控制执行）

2. **config**: 运行配置
   - `RunnableConfig` 类型
   - 包含 `thread_id`、回调等配置

3. **context**: 静态上下文（v0.6.0 新增）
   - 在整个执行过程中保持不变的上下文数据

4. **stream_mode**: 流模式
   - 默认为 `"values"`
   - 控制输出的格式

5. **print_mode**: 打印模式
   - 仅用于调试，打印到控制台
   - 不影响实际输出

6. **output_keys**: 输出键
   - 指定要从图中检索的输出键
   - 如果为 `None`，使用 `self.output_channels`

7. **interrupt_before/after**: 中断点
   - 指定在哪些节点前后中断执行
   - 可以是节点名称列表或 `All`

8. **durability**: 持久化模式
   - `"sync"`: 同步持久化（下一步开始前完成）
   - `"async"`: 异步持久化（下一步执行时并行持久化）
   - `"exit"`: 仅在图退出时持久化

### 执行流程（pregel/main.py:3065-3103）

```python
def invoke(self, input, config=None, **kwargs):
    # 1. 准备输出键
    output_keys = output_keys if output_keys is not None else self.output_channels

    # 2. 初始化变量
    latest: dict[str, Any] | Any = None
    chunks: list[dict[str, Any] | Any] = []
    interrupts: list[Interrupt] = []

    # 3. 通过 stream 方法执行图
    for chunk in self.stream(
        input,
        config,
        context=context,
        stream_mode=["updates", "values"] if stream_mode == "values" else stream_mode,
        print_mode=print_mode,
        output_keys=output_keys,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        durability=durability,
        **kwargs,
    ):
        # 4. 处理流式输出
        if stream_mode == "values":
            if len(chunk) == 2:
                mode, payload = cast(tuple[StreamMode, Any], chunk)
            else:
                _, mode, payload = cast(
                    tuple[tuple[str, ...], StreamMode, Any], chunk
                )

            # 5. 收集中断信息
            if (
                mode == "updates"
                and isinstance(payload, dict)
                and (ints := payload.get(INTERRUPT)) is not None
            ):
                interrupts.extend(ints)
            # 6. 保存最新值
            elif mode == "values":
                latest = payload
        else:
            chunks.append(chunk)

    # 7. 返回结果
    if stream_mode == "values":
        # 返回最新值（如果有中断，抛出异常）
        if interrupts:
            raise GraphInterrupt(interrupts)
        return latest
    else:
        # 返回所有块
        return chunks
```

### 关键发现

1. **invoke 是 stream 的封装**：
   - `invoke` 方法内部调用 `stream` 方法
   - 通过流式执行获取结果
   - 最终返回最后一个值或所有块

2. **流式执行模式**：
   - `stream_mode="values"`: 返回最新的状态值
   - 其他模式: 返回所有流式输出块

3. **中断处理**：
   - 收集执行过程中的中断信息
   - 如果有中断，抛出 `GraphInterrupt` 异常

4. **持久化控制**：
   - 通过 `durability` 参数控制持久化时机
   - 支持同步、异步和退出时持久化

## compile 方法详细分析

### 核心逻辑（graph/state.py:1081-1134）

```python
def compile(self, checkpointer=None, **kwargs):
    # 1. 验证 checkpointer
    checkpointer = ensure_valid_checkpointer(checkpointer)

    # 2. 设置默认值
    interrupt_before = interrupt_before or []
    interrupt_after = interrupt_after or []

    # 3. 验证图结构
    self.validate(
        interrupt=(
            (interrupt_before if interrupt_before != "*" else []) + interrupt_after
            if interrupt_after != "*"
            else []
        )
    )

    # 4. 准备输出通道
    output_channels = (
        "__root__"
        if len(self.schemas[self.output_schema]) == 1
        and "__root__" in self.schemas[self.output_schema]
        else [
            key
            for key, val in self.schemas[self.output_schema].items()
            if not is_managed_value(val)
        ]
    )

    # 5. 准备流通道
    stream_channels = (
        "__root__"
        if len(self.channels) == 1 and "__root__" in self.channels
        else [
            key for key, val in self.channels.items() if not is_managed_value(val)
        ]
    )

    # 6. 创建 CompiledStateGraph 对象
    compiled = CompiledStateGraph[StateT, ContextT, InputT, OutputT](
        builder=self,
        schema_to_mapper={},
        context_schema=self.context_schema,
        nodes={},
        channels={
            **self.channels,
            **self.managed,
            START: EphemeralValue(self.input_schema),
        },
        input_channels=START,
        stream_mode="updates",
        output_channels=output_channels,
        stream_channels=stream_channels,
        checkpointer=checkpointer,
        interrupt_before_nodes=interrupt_before,
        interrupt_after_nodes=interrupt_after,
        auto_validate=False,
        debug=debug,
        store=store,
        # ... 更多配置
    )

    return compiled
```

### 关键步骤

1. **验证阶段**：
   - 验证 checkpointer 的有效性
   - 验证图结构的完整性
   - 验证中断点的有效性

2. **通道准备**：
   - 准备输出通道（output_channels）
   - 准备流通道（stream_channels）
   - 过滤掉托管值（managed values）

3. **对象构建**：
   - 创建 `CompiledStateGraph` 对象
   - 配置所有通道（channels）
   - 配置中断点
   - 配置持久化

4. **返回编译后的图**：
   - 返回实现了 `Runnable` 接口的对象
   - 可以被 invoke、stream、batch 等方法调用

## 执行流程总结

```
用户代码
  ↓
graph.compile(checkpointer=...)
  ↓
CompiledStateGraph (实现 Runnable 接口)
  ↓
compiled_graph.invoke(input, config)
  ↓
内部调用 stream() 方法
  ↓
PregelLoop 执行循环
  ↓
逐步执行节点，更新状态
  ↓
返回最终结果
```

## 关键技术点

1. **编译时配置**：
   - checkpointer（持久化）
   - interrupt_before/after（中断点）
   - cache、store（缓存和存储）

2. **运行时配置**：
   - config（包含 thread_id 等）
   - context（静态上下文）
   - durability（持久化模式）

3. **流式执行**：
   - 所有执行都通过 stream 方法
   - invoke 是 stream 的便捷封装
   - 支持多种流模式

4. **中断与恢复**：
   - 通过 interrupt_before/after 控制
   - 通过 checkpointer 实现断点续传
   - 抛出 GraphInterrupt 异常通知中断
