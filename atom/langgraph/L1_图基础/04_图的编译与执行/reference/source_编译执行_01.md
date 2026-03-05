---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/state.py
  - libs/langgraph/langgraph/pregel/main.py
  - libs/langgraph/langgraph/pregel/_loop.py
analyzed_at: 2026-02-25
knowledge_point: 04_图的编译与执行
---

# 源码分析：图的编译与执行

## 分析的文件

- `libs/langgraph/langgraph/graph/state.py` - StateGraph.compile() 方法实现
- `libs/langgraph/langgraph/pregel/main.py` - Pregel.invoke() 方法实现
- `libs/langgraph/langgraph/pregel/_loop.py` - 执行循环实现

## 关键发现

### 1. compile 方法（state.py:1035）

**方法签名**：
```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    cache: BaseCache | None = None,
    store: BaseStore | None = None,
    interrupt_before: All | list[str] | None = None,
    interrupt_after: All | list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
```

**功能说明**：
- 将 `StateGraph` 编译成 `CompiledStateGraph` 对象
- 编译后的图实现了 `Runnable` 接口
- 可以被 invoked, streamed, batched, 和异步运行

**关键参数**：
1. **checkpointer**: 检查点保存器
   - 作为图的"短期记忆"
   - 允许图被暂停、恢复和重放
   - 需要在 config 中传递 `thread_id`

2. **interrupt_before/after**: 中断点配置
   - 指定在哪些节点前后中断执行

3. **cache**: 缓存配置
4. **store**: 存储配置
5. **debug**: 调试模式
6. **name**: 编译后图的名称

**返回值**：
- `CompiledStateGraph` 对象，实现了 Runnable 接口

### 2. invoke 方法（pregel/main.py:3024）

**方法签名**：
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

**功能说明**：
- 使用单个输入和配置运行图
- 支持多种执行模式和中断控制

**关键参数**：
1. **input**: 图的输入数据
2. **config**: 运行配置（包含 thread_id 等）
3. **context**: 静态上下文（v0.6.0 新增）
4. **stream_mode**: 流模式
5. **print_mode**: 打印模式（调试用）
6. **output_keys**: 要检索的输出键
7. **interrupt_before/after**: 中断点配置
8. **durability**: 持久化模式（默认 "async"）

### 3. 执行流程架构

从源码导入可以看到执行流程涉及：

**核心模块**：
- `langgraph.pregel._loop` - PregelLoop（同步/异步执行循环）
- `langgraph.pregel._algo` - Pregel 算法实现
- `langgraph.pregel._runner` - PregelRunner（执行器）
- `langgraph.pregel._checkpoint` - Checkpoint 机制
- `langgraph.pregel._executor` - 任务执行器

**关键算法函数**（从 _algo 导入）：
- `prepare_next_tasks` - 准备下一批任务
- `apply_writes` - 应用写入操作
- `local_read` - 本地读取
- `_scratchpad` - 临时存储

**Checkpoint 相关**（从 _checkpoint 导入）：
- `channels_from_checkpoint` - 从检查点恢复通道
- `copy_checkpoint` - 复制检查点
- `create_checkpoint` - 创建检查点
- `empty_checkpoint` - 空检查点

### 4. Pregel 算法应用

LangGraph 使用 Pregel 算法来执行图：
- Pregel 是 Google 开发的大规模图处理算法
- 基于 BSP（Bulk Synchronous Parallel）模型
- 适合迭代式图计算

**在 LangGraph 中的应用**：
- 节点作为 Pregel 的顶点
- 边作为消息传递通道
- 状态作为顶点的值
- 支持超步（superstep）迭代

### 5. Runnable 接口集成

CompiledStateGraph 实现了 LangChain 的 Runnable 接口：
- `invoke()` - 同步执行
- `ainvoke()` - 异步执行
- `stream()` - 流式执行
- `batch()` - 批量执行

这使得 LangGraph 可以无缝集成到 LangChain 生态系统中。

## 代码片段

### compile 方法核心逻辑（简化）

```python
def compile(self, checkpointer=None, **kwargs):
    # 1. 验证 checkpointer
    checkpointer = ensure_valid_checkpointer(checkpointer)

    # 2. 设置默认值
    interrupt_before = interrupt_before or []
    interrupt_after = interrupt_after or []

    # 3. 构建 Pregel 对象（CompiledStateGraph 的基类）
    # 4. 配置节点、边、通道
    # 5. 返回 CompiledStateGraph 实例

    return CompiledStateGraph(...)
```

### invoke 方法执行流程（简化）

```python
def invoke(self, input, config=None, **kwargs):
    # 1. 准备配置
    config = ensure_config(config)

    # 2. 初始化执行循环
    loop = SyncPregelLoop(...)

    # 3. 执行图
    for chunk in loop:
        # 处理每个执行步骤
        pass

    # 4. 返回最终结果
    return final_output
```

## 关键技术点总结

1. **编译过程**：
   - 将声明式的图定义转换为可执行的 Pregel 对象
   - 配置持久化、缓存、中断等运行时特性

2. **执行过程**：
   - 基于 Pregel 算法的迭代执行
   - 支持同步/异步、流式/批量等多种模式
   - 集成 Checkpoint 实现状态持久化

3. **Runnable 集成**：
   - 实现 LangChain 的 Runnable 接口
   - 支持链式组合和复杂工作流

4. **中断与恢复**：
   - 通过 interrupt_before/after 控制执行流程
   - 通过 Checkpoint 实现断点续传
